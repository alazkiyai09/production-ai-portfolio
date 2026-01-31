"""
Prompt template management system.

This module provides a comprehensive template system for managing prompt
templates with versioning, variable extraction, validation, and rendering.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
from datetime import datetime
import hashlib
import json
import logging
import re

from jinja2 import Environment, BaseLoader, meta, TemplateSyntaxError, UndefinedError

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PromptTemplate:
    """
    A versioned prompt template.

    Attributes:
        id: Unique template identifier (name_version)
        name: Template name
        description: Human-readable description
        template_string: Jinja2 template content
        variables: List of variable names used in template
        version: Version number
        created_at: Creation timestamp
        metadata: Additional metadata
        category: Template category (qa, code, creative, etc.)
        tags: List of tags for organization
    """

    id: str
    name: str
    description: str
    template_string: str
    variables: List[str] = field(default_factory=list)
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: str = "general"
    tags: List[str] = field(default_factory=list)

    @property
    def hash(self) -> str:
        """
        Generate unique hash for template content.

        Returns:
            First 12 characters of SHA256 hash
        """
        content = self.template_string.strip()
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "template_string": self.template_string,
            "variables": self.variables,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "category": self.category,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create PromptTemplate from dictionary."""
        return cls(**data)


@dataclass
class RenderedPrompt:
    """
    A rendered prompt ready for LLM execution.

    Attributes:
        template_id: ID of the source template
        template_version: Version of the source template
        content: Rendered prompt content
        variables_used: Variables and their values
        rendered_at: Render timestamp
        hash: Hash of rendered content
    """

    template_id: str
    template_version: int
    content: str
    variables_used: Dict[str, Any]
    rendered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def hash(self) -> str:
        """Generate hash of rendered content."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_id": self.template_id,
            "template_version": self.template_version,
            "content": self.content,
            "variables_used": self.variables_used,
            "rendered_at": self.rendered_at,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderedPrompt":
        """Create RenderedPrompt from dictionary."""
        return cls(**data)


@dataclass
class TemplateValidationResult:
    """
    Result of template validation.

    Attributes:
        is_valid: Whether template is valid
        errors: List of validation errors
        warnings: List of warnings
        variables: Extracted variables
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "variables": self.variables,
        }


# ============================================================================
# Template Manager
# ============================================================================

class TemplateManager:
    """
    Manage prompt templates with versioning and rendering.

    Provides:
    - Template creation with versioning
    - Variable extraction from Jinja2 templates
    - Template validation
    - Rendering with variable substitution
    - Template comparison across versions
    - Storage and retrieval
    """

    def __init__(
        self,
        storage_path: str | Path = "./data/prompt_templates",
        auto_load: bool = True,
    ):
        """
        Initialize the template manager.

        Args:
            storage_path: Directory to store templates
            auto_load: Automatically load templates on init
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Jinja2 environment
        self.env = Environment(
            loader=BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Template storage
        self._templates: Dict[str, List[PromptTemplate]] = {}

        # Load existing templates
        if auto_load:
            self._load_templates()

        logger.info(f"TemplateManager initialized with {len(self._templates)} templates")

    # ========================================================================
    # Template Creation
    # ========================================================================

    def create_template(
        self,
        name: str,
        template_string: str,
        description: str = "",
        category: str = "general",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptTemplate:
        """
        Create a new prompt template.

        Args:
            name: Template name (must be unique)
            template_string: Jinja2 template content
            description: Human-readable description
            category: Template category
            tags: List of tags for organization
            metadata: Additional metadata

        Returns:
            Created PromptTemplate

        Raises:
            ValueError: If template is invalid or name already exists
        """
        # Check if template exists
        if name in self._templates and self._templates[name]:
            raise ValueError(f"Template '{name}' already exists")

        # Validate template
        validation = self.validate_template_string(template_string)
        if not validation.is_valid:
            raise ValueError(f"Invalid template: {validation.errors}")

        # Extract variables
        variables = validation.variables

        # Determine version
        version = 1

        # Create template
        template = PromptTemplate(
            id=f"{name}_v{version}",
            name=name,
            description=description,
            template_string=template_string,
            variables=variables,
            version=version,
            category=category,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Store
        if name not in self._templates:
            self._templates[name] = []
        self._templates[name].append(template)

        # Save to disk
        self._save_template(template)

        logger.info(f"Created template '{name}' version {version} with {len(variables)} variables")
        return template

    def create_from_dict(
        self,
        data: Dict[str, Any],
    ) -> PromptTemplate:
        """
        Create template from dictionary (useful for loading from YAML).

        Args:
            data: Template data dictionary

        Returns:
            Created PromptTemplate
        """
        return self.create_template(
            name=data["name"],
            template_string=data["template_string"],
            description=data.get("description", ""),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    # ========================================================================
    # Template Retrieval
    # ========================================================================

    def get_template(
        self,
        name: str,
        version: Optional[int] = None,
    ) -> Optional[PromptTemplate]:
        """
        Get a template by name and optional version.

        Args:
            name: Template name
            version: Specific version (None = latest)

        Returns:
            PromptTemplate or None if not found
        """
        templates = self._templates.get(name)
        if not templates:
            return None

        if version is None:
            return templates[-1]  # Latest version

        for t in templates:
            if t.version == version:
                return t

        return None

    def get_latest_version(self, name: str) -> Optional[int]:
        """
        Get the latest version number for a template.

        Args:
            name: Template name

        Returns:
            Latest version number or None
        """
        templates = self._templates.get(name)
        return templates[-1].version if templates else None

    # ========================================================================
    # Template Rendering
    # ========================================================================

    def render(
        self,
        name: str,
        variables: Dict[str, Any],
        version: Optional[int] = None,
        strict: bool = True,
    ) -> RenderedPrompt:
        """
        Render a template with variables.

        Args:
            name: Template name
            variables: Variables to fill in template
            version: Specific version (None = latest)
            strict: Raise error for undefined variables

        Returns:
            RenderedPrompt with filled content

        Raises:
            ValueError: If template not found or variables missing
        """
        template = self.get_template(name, version)
        if not template:
            raise ValueError(f"Template not found: {name}")

        # Check required variables
        missing = set(template.variables) - set(variables.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Check for extra variables
        defined = set(template.variables)
        provided = set(variables.keys())
        extra = provided - defined
        if extra and strict:
            logger.warning(f"Extra variables provided (not in template): {extra}")

        # Render
        try:
            jinja_template = self.env.from_string(template.template_string, undefined=Strict if strict else Undefined)
            content = jinja_template.render(**variables)
        except Exception as e:
            raise ValueError(f"Template rendering failed: {e}")

        return RenderedPrompt(
            template_id=template.id,
            template_version=template.version,
            content=content,
            variables_used=variables,
            rendered_at=datetime.utcnow().isoformat(),
        )

    # ========================================================================
    # Template Listing
    # ========================================================================

    def list_templates(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all templates with optional filtering.

        Args:
            category: Filter by category
            tags: Filter by tags (must match all)

        Returns:
            List of template summaries
        """
        result = []

        for name, versions in self._templates.items():
            if not versions:
                continue

            latest = versions[-1]

            # Category filter
            if category and latest.category != category:
                continue

            # Tags filter
            if tags:
                if not any(tag in latest.tags for tag in tags):
                    continue

            result.append({
                "name": name,
                "latest_version": latest.version,
                "description": latest.description,
                "category": latest.category,
                "tags": latest.tags,
                "variables": latest.variables,
                "total_versions": len(versions),
                "created_at": latest.created_at,
                "hash": latest.hash,
            })

        return result

    def get_all_versions(self, name: str) -> List[PromptTemplate]:
        """
        Get all versions of a template.

        Args:
            name: Template name

        Returns:
            List of all template versions
        """
        return self._templates.get(name, [])

    # ========================================================================
    # Template Comparison
    # ========================================================================

    def compare_versions(
        self,
        name: str,
        version1: int,
        version2: int,
    ) -> Dict[str, Any]:
        """
        Compare two versions of a template.

        Args:
            name: Template name
            version1: First version number
            version2: Second version number

        Returns:
            Dictionary with comparison results
        """
        t1 = self.get_template(name, version1)
        t2 = self.get_template(name, version2)

        if not t1 or not t2:
            raise ValueError(f"Version(s) not found for template: {name}")

        # Calculate differences
        lines1 = t1.template_string.splitlines()
        lines2 = t2.template_string.splitlines()

        # Find added/removed/modified lines
        added = set(lines2) - set(lines1)
        removed = set(lines1) - set(lines2)
        common = set(lines1) & set(lines2)

        # Variable changes
        vars_added = list(set(t2.variables) - set(t1.variables))
        vars_removed = list(set(t1.variables) - set(t2.variables))

        return {
            "template": name,
            "version1": version1,
            "version2": version2,
            "variables_added": vars_added,
            "variables_removed": vars_removed,
            "line_count_change": len(lines2) - len(lines1),
            "lines_added": len(added),
            "lines_removed": len(removed),
            "lines_modified": len(lines1) - len(common),
            "hash_v1": t1.hash,
            "hash_v2": t2.hash,
            "content_change_percent": (
                abs(len(t2.template_string) - len(t1.template_string)) /
                len(t1.template_string) * 100
                if t1.template_string else 0
            ),
        }

    # ========================================================================
    # Template Validation
    # ========================================================================

    def validate_template_string(
        self,
        template_string: str,
    ) -> TemplateValidationResult:
        """
        Validate a Jinja2 template string.

        Args:
            template_string: Template content to validate

        Returns:
            TemplateValidationResult with validation status
        """
        errors = []
        warnings = []
        variables = []

        try:
            # Parse template
            ast = self.env.parse(template_string)

            # Extract variables
            variables = list(meta.find_undeclared_variables(ast))

            # Check for common issues
            if "{{" in template_string and "}}" not in template_string:
                errors.append("Template contains unclosed variable brackets")

            # Check for potential syntax issues
            if "{%" in template_string and "%}" not in template_string:
                errors.append("Template contains unclosed statement brackets")

            # Check for very long lines
            lines = template_string.splitlines()
            for i, line in enumerate(lines, 1):
                if len(line) > 500:
                    warnings.append(f"Line {i} is very long ({len(line)} chars)")

            # Check for print statements (often accidental)
            if "{{ print(" in template_string:
                warnings.append("Template contains print() statement (may be accidental)")

        except TemplateSyntaxError as e:
            errors.append(f"Syntax error: {e}")
        except Exception as e:
            errors.append(f"Validation error: {e}")

        return TemplateValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            variables=variables,
        )

    # ========================================================================
    # Template Update
    # ========================================================================

    def update_template(
        self,
        name: str,
        template_string: str,
        description: Optional[str] = None,
        create_new_version: bool = True,
    ) -> PromptTemplate:
        """
        Update an existing template or create new version.

        Args:
            name: Template name
            template_string: New template content
            description: New description (optional)
            create_new_version: Create new version (true) or update latest (false)

        Returns:
            Updated or new PromptTemplate

        Raises:
            ValueError: If template not found
        """
        templates = self._templates.get(name)

        if not templates:
            # Create new template
            return self.create_template(
                name=name,
                template_string=template_string,
                description=description or "",
            )

        if create_new_version:
            # Create new version
            latest = templates[-1]
            version = latest.version + 1

            template = PromptTemplate(
                id=f"{name}_v{version}",
                name=name,
                description=description or latest.description,
                template_string=template_string,
                variables=self._extract_variables(template_string),
                version=version,
                created_at=datetime.utcnow().isoformat(),
                category=latest.category,
                tags=latest.tags.copy(),
                metadata=latest.metadata.copy(),
            )
        else:
            # Update latest version
            template = templates[-1]
            template.template_string = template_string
            template.variables = self._extract_variables(template_string)
            if description:
                template.description = description
            template_hash = template.hash

        # Save
        if create_new_version:
            templates.append(template)
        self._save_template(template)

        logger.info(f"Updated template '{name}' (version {template.version})")
        return template

    # ========================================================================
    # Template Deletion
    # ========================================================================

    def delete_template(self, name: str, version: Optional[int] = None):
        """
        Delete a template or specific version.

        Args:
            name: Template name
            version: Specific version (None = delete all versions)
        """
        if version is None:
            # Delete all versions
            if name in self._templates:
                del self._templates[name]

            # Delete files
            template_dir = self.storage_path / name
            if template_dir.exists():
                for file in template_dir.glob("*.json"):
                    file.unlink()
                template_dir.rmdir()
        else:
            # Delete specific version
            templates = self._templates.get(name, [])
            if templates:
                self._templates[name] = [t for t in templates if t.version != version]

                # Delete file
                file_path = self.storage_path / name / f"v{version}.json"
                if file_path.exists():
                    file_path.unlink()

        logger.info(f"Deleted template '{name}' version {version or 'all'}")

    # ========================================================================
    # Import/Export
    # ========================================================================

    def export_template(
        self,
        name: str,
        version: Optional[int] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export template to file.

        Args:
            name: Template name
            version: Template version
            output_path: Output file path

        Returns:
            Path to exported file
        """
        template = self.get_template(name, version)
        if not template:
            raise ValueError(f"Template not found: {name}")

        if output_path is None:
            output_path = self.storage_path / f"{name}_v{template.version}_export.json"

        with open(output_path, 'w') as f:
            json.dump(template.to_dict(), f, indent=2)

        logger.info(f"Exported template '{name}' to {output_path}")
        return output_path

    def import_template(
        self,
        file_path: Path | str,
        override_name: Optional[str] = None,
    ) -> PromptTemplate:
        """
        Import template from file.

        Args:
            file_path: Path to template JSON file
            override_name: Override template name from file

        Returns:
            Imported PromptTemplate
        """
        file_path = Path(file_path)

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Override name if specified
        if override_name:
            data["name"] = override_name

        return self.create_from_dict(data)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _extract_variables(self, template_string: str) -> List[str]:
        """
        Extract variable names from Jinja2 template.

        Args:
            template_string: Template content

        Returns:
            List of variable names
        """
        try:
            ast = self.env.parse(template_string)
            variables = list(meta.find_undeclared_variables(ast))
            return sorted(variables)
        except Exception as e:
            logger.warning(f"Could not extract variables: {e}")
            return []

    def _save_template(self, template: PromptTemplate):
        """Save template to disk."""
        template_dir = self.storage_path / template.name
        template_dir.mkdir(exist_ok=True)

        file_path = template_dir / f"v{template.version}.json"
        with open(file_path, 'w') as f:
            json.dump(template.to_dict(), f, indent=2)

    def _load_templates(self):
        """Load all templates from disk."""
        if not self.storage_path.exists():
            return

        for template_dir in sorted(self.storage_path.iterdir()):
            if template_dir.is_dir():
                for version_file in sorted(template_dir.glob("v*.json")):
                    try:
                        with open(version_file, 'r') as f:
                            data = json.load(f)
                            template = PromptTemplate.from_dict(data)

                            if template.name not in self._templates:
                                self._templates[template.name] = []
                            self._templates[template.name].append(template)
                    except Exception as e:
                        logger.error(f"Error loading template from {version_file}: {e}")


# ============================================================================
# Custom Undefined Behavior
# ============================================================================

class Strict(Undefined):
    """
    Strict undefined behavior - raises error for undefined variables.
    """

    def __getattr__(self, name):
        raise UndefinedError(f"'{name}' is undefined")


# ============================================================================
# Predefined Template Patterns
# ============================================================================

TEMPLATE_PATTERNS = {
    "instruction_first": """{{ instruction }}

{{ context }}

{{ output_format }}""",

    "context_first": """{{ context }}

Based on the above, {{ instruction }}

{{ output_format }}""",

    "cot_explicit": """{{ instruction }}

{{ context }}

Let's think step by step:
1. First, I'll analyze what's being asked
2. Then, I'll consider the relevant information
3. Finally, I'll provide my response

{{ output_format }}""",

    "cot_implicit": """{{ instruction }}

{{ context }}

Think through this carefully before responding.

{{ output_format }}""",

    "few_shot": """{{ instruction }}

Here are some examples:
{% for example in examples %}
Input: {{ example.input }}
Output: {{ example.output }}
{% endfor %}

Now, given:
{{ context }}

{{ output_format }}""",

    "json_format": """{{ instruction }}

{{ context }}

Respond in JSON format with the following structure:
```json
{
    "answer": "...",
    "confidence": 0.0-1.0,
    "reasoning": "..."
}
```""",

    "code_format": """{{ instruction }}

{{ context }}

Provide your response as code:

```{{ language }}
{{ output_format }}
```""",
}


# ============================================================================
# Convenience Functions
# ============================================================================

def create_template_manager(
    storage_path: str | Path = "./data/prompt_templates",
) -> TemplateManager:
    """
    Create a template manager instance.

    Args:
        storage_path: Path to template storage

    Returns:
        Initialized TemplateManager
    """
    return TemplateManager(storage_path=storage_path)


# Export main classes and functions
__all__ = [
    "PromptTemplate",
    "RenderedPrompt",
    "TemplateValidationResult",
    "TemplateManager",
    "create_template_manager",
    "TEMPLATE_PATTERNS",
]
