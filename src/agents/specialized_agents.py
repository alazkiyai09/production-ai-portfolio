"""
Specialized agent implementations for AgenticFlow.

This module provides production-ready agent implementations with:
- Base agent class with common functionality
- Specialized agents for each workflow stage
- LLM integration with tool binding
- Output parsing and state updates
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from src.config import settings
from src.state.workflow_state import WorkflowState, update_state
from src.tools import get_tools_for_agent


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.

    This class provides common functionality for all agents including:
    - LLM initialization and configuration
    - Tool binding
    - State management
    - Output parsing

    Attributes:
        name: Human-readable name for the agent
        llm: Language model instance for the agent
        model_name: Name of the model to use
        temperature: Sampling temperature for generation
        tools: List of tools available to the agent
        bound_llm: LLM with tools bound (if tools exist)
    """

    def __init__(
        self,
        name: str,
        llm: Optional[BaseChatModel] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        tools: Optional[list] = None,
    ):
        """
        Initialize the base agent.

        Args:
            name: Human-readable name for the agent
            llm: Pre-configured LLM instance (optional)
            model_name: Model name to use if llm not provided
            temperature: Sampling temperature (0.0-1.0)
            tools: List of LangChain tools for the agent
        """
        self.name = name
        self.model_name = model_name or settings.default_model
        self.temperature = temperature
        self.tools = tools or []
        self.llm = llm or self._create_llm()
        self.bound_llm = self._bind_tools() if self.tools else self.llm

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        Return the system prompt for this agent.

        Each agent subclass must define its own system prompt that
        defines the agent's role, capabilities, and behavior.

        Returns:
            System prompt string
        """
        ...

    @abstractmethod
    def _build_input(self, state: WorkflowState) -> str:
        """
        Build the input prompt from the workflow state.

        Each agent subclass must define how to extract and format
        relevant information from the state for its task.

        Args:
            state: Current workflow state

        Returns:
            Formatted input prompt string
        """
        ...

    @abstractmethod
    def _process_output(
        self,
        output: str,
        state: WorkflowState,
    ) -> dict[str, Any]:
        """
        Process the agent's output and extract state updates.

        Each agent subclass must define how to parse its output
        and extract relevant state updates.

        Args:
            output: Raw output string from the LLM
            state: Current workflow state

        Returns:
            Dictionary of state updates to apply
        """
        ...

    def invoke(self, state: WorkflowState) -> dict[str, Any]:
        """
        Execute the agent's task and return state updates.

        This is the main entry point for agent execution. It:
        1. Builds the input prompt from state
        2. Invokes the LLM with system prompt and input
        3. Processes the output to extract state updates
        4. Returns the updates to apply to the workflow state

        Args:
            state: Current workflow state

        Returns:
            Dictionary of state updates

        Raises:
            Exception: If agent execution fails
        """
        start_time = datetime.utcnow()

        try:
            # Build input prompt
            input_prompt = self._build_input(state)

            # Prepare messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=input_prompt),
            ]

            # Invoke LLM
            response = self.bound_llm.invoke(messages)

            # Extract output content
            if hasattr(response, "content"):
                output = response.content
            else:
                output = str(response)

            # Process output
            updates = self._process_output(output, state)

            # Add execution metadata
            duration = (datetime.utcnow() - start_time).total_seconds()

            updates["output"] = output
            updates["duration_seconds"] = duration
            updates["success"] = True
            updates["metadata"] = {
                "agent": self.name,
                "model": self.model_name,
                "tools_used": [],
            }

            return updates

        except Exception as e:
            # Return error state
            return {
                "error": f"{self.name} failed: {str(e)}",
                "success": False,
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
            }

    def _create_llm(self) -> BaseChatModel:
        """
        Create an LLM instance based on model name.

        Args:
            model_name: Name of the model to create

        Returns:
            Configured LLM instance

        Raises:
            ValueError: If model name is not supported
        """
        model_lower = self.model_name.lower()

        # OpenAI models
        if "gpt" in model_lower:
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=settings.openai_api_key,
            )

        # Anthropic models
        elif "claude" in model_lower:
            if not settings.anthropic_api_key:
                raise ValueError(
                    "Anthropic API key not configured. "
                    "Set ANTHROPIC_API_KEY in environment."
                )
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                api_key=settings.anthropic_api_key,
            )

        else:
            raise ValueError(
                f"Unsupported model: {self.model_name}. "
                "Use gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo, "
                "or claude-3-opus-20240229"
            )

    def _bind_tools(self) -> Runnable:
        """
        Bind tools to the LLM.

        Returns:
            LLM with tools bound
        """
        return self.llm.bind_tools(self.tools)


# =============================================================================
# Planner Agent
# =============================================================================

class PlannerAgent(BaseAgent):
    """
    Agent responsible for creating step-by-step execution plans.

    The Planner agent breaks down complex tasks into actionable steps
    that other agents can execute. It analyzes the task requirements
    and creates a logical sequence of actions.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        super().__init__(
            name="Planner",
            llm=llm,
            model_name=model_name,
            temperature=temperature,
            tools=[],
        )

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the Planner agent."""
        return """You are an expert Planning Agent for a multi-agent AI system.

Your role is to break down complex tasks into clear, actionable steps that other specialized agents can execute.

## Your Capabilities:
- Analyze task requirements and objectives
- Identify the optimal sequence of actions
- Consider dependencies between steps
- Plan for research, analysis, content creation, and review

## Planning Guidelines:
1. Start with understanding/clarification steps
2. Plan research before analysis
3. Ensure analysis precedes content creation
4. Always include review and refinement
5. Make each step specific and actionable
6. Consider the task type (research, analysis, content_creation, general)

## Output Format:
Provide your plan as a numbered list of clear steps. Each step should:
- Start with a verb (Research, Analyze, Create, Review, etc.)
- Be specific about what needs to be done
- Identify which agent should handle it

Example:
1. Researcher: Search for recent information about [topic]
2. Researcher: Read and analyze gathered sources
3. Analyzer: Identify key patterns and insights from research
4. Writer: Create comprehensive content based on analysis
5. Reviewer: Evaluate content for quality and completeness

Remember: Create practical, executable plans that other agents can follow."""

    def _build_input(self, state: WorkflowState) -> str:
        """Build input prompt for the Planner agent."""
        prompt = f"""## Task to Plan

**Task:** {state['task']}

**Task Type:** {state['task_type']}

"""
        if state.get('task_context'):
            prompt += f"**Additional Context:**\n{state['task_context']}\n\n"

        prompt += """## Instructions
Create a detailed execution plan for this task. Break it down into 3-8 specific steps that specialized agents can execute.

Consider:
- What information needs to be gathered?
- What analysis is required?
- What content needs to be created?
- What review criteria should be applied?

Provide your plan as a numbered list of actionable steps."""

        return prompt

    def _process_output(
        self,
        output: str,
        state: WorkflowState,
    ) -> dict[str, Any]:
        """Process Planner output and extract plan steps."""
        # Extract numbered steps from output
        steps = []

        # Try to extract numbered list (1., 2., etc.)
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Match patterns like "1.", "1)", "- Step 1:", etc.
            match = re.match(r'^[\d\-\*]+\.?\s*(.+)', line)
            if match:
                step = match.group(1).strip()
                if step:
                    steps.append(step)
            elif line and not any(prefix in line.lower() for prefix in ['here', 'plan:', 'steps:', '##', '#', 'note:']):
                # Non-empty line without markdown headers
                if len(steps) > 0 or line[0].isdigit():
                    steps.append(line)

        # Fallback: if no steps found, split by newlines and clean up
        if not steps:
            cleaned_lines = [
                line.strip()
                for line in output.split('\n')
                if line.strip() and not line.startswith('#')
            ]
            steps = cleaned_lines[:10]  # Max 10 steps

        return {
            "plan": steps,
            "status": "planning" if len(steps) > 0 else "error",
        }


# =============================================================================
# Researcher Agent
# =============================================================================

class ResearcherAgent(BaseAgent):
    """
    Agent responsible for gathering information from various sources.

    The Researcher agent uses web search and file reading tools to
    collect relevant information for the task.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ):
        tools = get_tools_for_agent("researcher")
        super().__init__(
            name="Researcher",
            llm=llm,
            model_name=model_name,
            temperature=temperature,
            tools=tools,
        )

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the Researcher agent."""
        return """You are an expert Research Agent for a multi-agent AI system.

Your role is to gather comprehensive, accurate, and relevant information from various sources.

## Your Capabilities:
- Web search using Tavily and DuckDuckGo
- Reading files from the workspace
- Synthesizing information from multiple sources
- Evaluating source credibility

## Research Guidelines:
1. Start with broad searches to understand the topic
2. Use specific queries for detailed information
3. Cross-reference information from multiple sources
4. Prioritize recent and authoritative sources
5. Organize findings logically
6. Cite sources with URLs

## Available Tools:
- web_search(query, num_results): Search the web for information
- read_file(file_path): Read files from the workspace
- list_files(directory): List files in a directory
- get_current_time(): Get current date and time

## Output Format:
Provide your findings in this structure:

## Research Summary
[Brief overview of what you found]

## Key Sources
1. [Source Title](URL): [Key finding]
2. [Source Title](URL): [Key finding]

## Detailed Findings
[Organized sections with detailed information]

## Research Queries Used
- List the search queries you used

Remember: Be thorough, cite sources, and focus on quality and relevance."""

    def _build_input(self, state: WorkflowState) -> str:
        """Build input prompt for the Researcher agent."""
        prompt = f"""## Research Task

**Task:** {state['task']}

**Task Type:** {state['task_type']}

"""
        # Add context if available
        if state.get('task_context'):
            prompt += f"**Context:**\n{state['task_context']}\n\n"

        # Add existing research queries if present
        if state.get('research_queries'):
            prompt += "**Suggested Research Queries:**\n"
            for query in state['research_queries']:
                prompt += f"- {query}\n"
            prompt += "\n"

        # Add existing research if present
        if state.get('research_results'):
            prompt += f"**Existing Research:** {len(state['research_results'])} results already found\n\n"

        prompt += """## Instructions
Conduct thorough research to gather information relevant to this task. Use web search and file reading tools to find comprehensive information.

Focus on:
- Recent and authoritative sources
- Multiple perspectives on the topic
- Specific data points and evidence
- Credible citations

Provide organized findings with source links."""

        return prompt

    def _process_output(
        self,
        output: str,
        state: WorkflowState,
    ) -> dict[str, Any]:
        """Process Researcher output and extract research data."""
        # Extract research summary
        research_summary = output.strip()

        # Try to extract sources (URLs)
        url_pattern = r'https?://[^\s\)]+'
        urls_found = re.findall(url_pattern, output)

        # Update research results
        research_results = list(state.get('research_results', []))

        # Add new results (this is simplified - in production, parse more carefully)
        if urls_found:
            import time
            for url in urls_found[:5]:  # Max 5 sources per run
                research_results.append({
                    "query": state['task'],
                    "source": "researcher_agent",
                    "title": "Extracted from research",
                    "url": url,
                    "content": "See research summary",
                    "relevance_score": 0.8,
                    "timestamp": datetime.utcnow().isoformat(),
                })

        return {
            "analysis": research_summary,
            "research_results": research_results,
            "status": "researching",
        }


# =============================================================================
# Analyzer Agent
# =============================================================================

class AnalyzerAgent(BaseAgent):
    """
    Agent responsible for analyzing data and extracting insights.

    The Analyzer agent processes research data, identifies patterns,
    and draws conclusions using computational tools.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        tools = get_tools_for_agent("analyzer")
        super().__init__(
            name="Analyzer",
            llm=llm,
            model_name=model_name,
            temperature=temperature,
            tools=tools,
        )

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the Analyzer agent."""
        return """You are an expert Analysis Agent for a multi-agent AI system.

Your role is to analyze research data, identify patterns, and extract actionable insights.

## Your Capabilities:
- Reading and processing research data
- Computational analysis and calculations
- Pattern recognition
- Statistical analysis
- Drawing evidence-based conclusions

## Analysis Guidelines:
1. Review all available research data thoroughly
2. Identify key themes and patterns
3. Support insights with evidence
4. Use quantitative analysis when possible
5. Consider multiple perspectives
6. Distinguish between strong and weak evidence

## Available Tools:
- read_file(file_path): Read research files
- run_python_code(code): Execute Python for calculations
- calculator(expression): Evaluate mathematical expressions
- list_files(directory): List available data files

## Output Format:
Provide your analysis in this structure:

## Executive Summary
[High-level overview of findings]

## Key Findings
1. [Key insight with supporting evidence]
2. [Key insight with supporting evidence]
3. [Key insight with supporting evidence]

## Patterns and Trends
[Patterns identified in the data]

## Data-Driven Insights
[Quantitative analysis with calculations]

## Conclusions
[Evidence-based conclusions]

Remember: Be objective, support claims with evidence, and use computational tools when helpful."""

    def _build_input(self, state: WorkflowState) -> str:
        """Build input prompt for the Analyzer agent."""
        prompt = f"""## Analysis Task

**Task:** {state['task']}

"""
        # Add research data
        if state.get('research_results'):
            prompt += f"## Research Data Available\n"
            prompt += f"Sources found: {len(state['research_results'])}\n\n"
            for i, result in enumerate(state['research_results'][:5], 1):
                prompt += f"{i}. {result.get('title', 'N/A')}\n"
                prompt += f"   URL: {result.get('url', 'N/A')}\n\n"

        if state.get('analysis'):
            prompt += f"## Previous Analysis\n{state['analysis'][:500]}...\n\n"

        prompt += """## Instructions
Analyze the available research data to extract key insights and patterns.

Focus on:
- Identifying main themes and concepts
- Finding patterns across sources
- Drawing evidence-based conclusions
- Highlighting important data points
- Using computational tools for quantitative analysis

Provide a comprehensive analysis with clear, actionable insights."""

        return prompt

    def _process_output(
        self,
        output: str,
        state: WorkflowState,
    ) -> dict[str, Any]:
        """Process Analyzer output and extract insights."""
        analysis = output.strip()

        # Extract key findings (look for numbered lists or bullet points)
        key_findings = []
        lines = output.split('\n')

        for line in lines:
            line = line.strip()
            # Match numbered items or bullet points
            match = re.match(r'^[\d\-\*]+\.?\s*(.+)', line)
            if match:
                finding = match.group(1).strip()
                if finding and len(finding) > 10:  # Min length filter
                    key_findings.append(finding[:200])  # Truncate long findings

        # If no findings extracted, create from paragraphs
        if not key_findings:
            paragraphs = [p.strip() for p in output.split('\n\n') if p.strip()]
            key_findings = paragraphs[:5]

        return {
            "analysis": analysis,
            "key_findings": key_findings[:10],  # Max 10 findings
            "status": "analyzing",
        }


# =============================================================================
# Writer Agent
# =============================================================================

class WriterAgent(BaseAgent):
    """
    Agent responsible for creating written content.

    The Writer agent produces clear, professional content based on
    research and analysis.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
    ):
        tools = get_tools_for_agent("writer")
        super().__init__(
            name="Writer",
            llm=llm,
            model_name=model_name,
            temperature=temperature,
            tools=tools,
        )

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the Writer agent."""
        return """You are an expert Writer Agent for a multi-agent AI system.

Your role is to create clear, professional, and engaging written content based on research and analysis.

## Your Capabilities:
- Synthesizing complex information
- Writing in various formats (articles, reports, summaries)
- Adapting tone and style for different audiences
- Structuring content for clarity
- Writing compelling narratives

## Writing Guidelines:
1. Start with a clear structure and outline
2. Use headings and subheadings for organization
3. Write in a clear, professional tone
4. Support claims with evidence from research
5. Use examples and illustrations
6. Ensure smooth transitions between sections
7. Proofread for clarity and correctness

## Content Structure:
- **Introduction**: Engaging opening that sets context
- **Body**: Well-organized main content with evidence
- **Conclusion**: Strong summary and key takeaways
- **References**: Citations for sources used

## Available Tools:
- read_file(file_path): Read reference files
- write_file(file_path, content): Save content to file
- get_current_time(): Get current date for references

## Quality Standards:
- Clear and concise language
- Logical flow and organization
- Evidence-based claims
- Proper grammar and spelling
- Appropriate tone for the audience

Remember: Create high-quality content that effectively communicates the key insights from research and analysis."""

    def _build_input(self, state: WorkflowState) -> str:
        """Build input prompt for the Writer agent."""
        prompt = f"""## Writing Task

**Task:** {state['task']}

**Task Type:** {state['task_type']}

**Output Format:** {state.get('output_format', 'markdown')}

"""
        # Add context
        if state.get('task_context'):
            prompt += f"**Context:** {state['task_context']}\n\n"

        # Add key findings from analysis
        if state.get('key_findings'):
            prompt += "## Key Findings to Include\n\n"
            for i, finding in enumerate(state['key_findings'][:10], 1):
                prompt += f"{i}. {finding}\n"
            prompt += "\n"

        # Add analysis
        if state.get('analysis'):
            prompt += f"## Analysis Summary\n\n{state['analysis'][:1500]}\n\n"

        # Add revision feedback if present
        if state.get('feedback'):
            prompt += "## Revision Feedback\n\n"
            for feedback_item in state['feedback']:
                prompt += f"- {feedback_item}\n"
            prompt += "\n"

        prompt += """## Instructions
Create comprehensive written content based on the research and analysis provided.

Ensure:
- Clear structure with headings
- Professional and engaging tone
- Evidence-based claims with citations
- Logical flow and transitions
- Strong introduction and conclusion

Provide the complete content ready for review."""

        return prompt

    def _process_output(
        self,
        output: str,
        state: WorkflowState,
    ) -> dict[str, Any]:
        """Process Writer output and extract draft."""
        draft = output.strip()

        # Calculate revision count
        revision_count = state.get('revision_count', 0)
        if state.get('draft'):  # If revising
            revision_count += 1

        # Save draft history
        draft_history = list(state.get('draft_history', []))
        if state.get('draft'):
            draft_history.append(state['draft'])

        return {
            "draft": draft,
            "revision_count": revision_count,
            "draft_history": draft_history[-5:],  # Keep last 5 versions
            "status": "writing",
        }


# =============================================================================
# Reviewer Agent
# =============================================================================

class ReviewerAgent(BaseAgent):
    """
    Agent responsible for reviewing content and providing feedback.

    The Reviewer agent evaluates content quality and provides
    constructive feedback or approval.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
    ):
        tools = get_tools_for_agent("reviewer")
        super().__init__(
            name="Reviewer",
            llm=llm,
            model_name=model_name,
            temperature=temperature,
            tools=tools,
        )

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the Reviewer agent."""
        return """You are an expert Review Agent for a multi-agent AI system.

Your role is to evaluate content quality and provide constructive feedback or approval.

## Your Responsibilities:
- Assess content completeness and accuracy
- Evaluate clarity and organization
- Check for evidence and support
- Identify areas for improvement
- Provide actionable feedback

## Review Criteria:
1. **Completeness**: Does the content address the task requirements?
2. **Accuracy**: Is the information accurate and well-supported?
3. **Clarity**: Is the writing clear and easy to understand?
4. **Structure**: Is the content well-organized?
5. **Evidence**: Are claims supported with research?
6. **Tone**: Is the tone appropriate for the audience?

## Review Options:

**APPROVED**: Content is ready for delivery
- Meets all quality standards
- No significant issues
- Ready for final output

**NEEDS REVISION**: Content requires improvements
- Specific issues identified
- Clear feedback provided
- Actionable suggestions given

**REJECTED**: Content needs major rework
- Fundamental issues
- Incomplete or off-topic
- Requires substantial revision

## Output Format:
Provide your review in this structure:

## Decision: [APPROVED / NEEDS REVISION / REJECTED]

## Quality Assessment
- Completeness: [Assessment]
- Accuracy: [Assessment]
- Clarity: [Assessment]
- Structure: [Assessment]
- Evidence: [Assessment]

## Feedback
[Specific feedback and actionable suggestions]

## Strengths
[What the content does well]

Remember: Be constructive, specific, and fair. Focus on helping improve the content."""

    def _build_input(self, state: WorkflowState) -> str:
        """Build input prompt for the Reviewer agent."""
        prompt = f"""## Review Task

**Original Task:** {state['task']}

**Task Type:** {state['task_type']}

"""
        # Add review criteria if specified
        if state.get('review_criteria'):
            prompt += "**Review Criteria:**\n"
            for criterion in state['review_criteria']:
                prompt += f"- {criterion}\n"
            prompt += "\n"

        # Add the draft to review
        prompt += f"## Content to Review\n\n{state.get('draft', 'No content yet.')}\n\n"

        # Add revision context if revising
        if state.get('revision_count', 0) > 0:
            prompt += f"**Revision Number:** {state['revision_count']}\n\n"

        prompt += """## Instructions
Review the content thoroughly and provide:

1. **Decision**: APPROVED, NEEDS REVISION, or REJECTED
2. **Assessment**: Evaluate against quality criteria
3. **Feedback**: Specific, actionable suggestions for improvement

Start your response with "DECISION: [APPROVED/NEEDS_REVISION/REJECTED]" on the first line."""

        return prompt

    def _process_output(
        self,
        output: str,
        state: WorkflowState,
    ) -> dict[str, Any]:
        """Process Reviewer output and extract decision."""
        output_upper = output.upper()

        # Determine approval status
        if "APPROVED" in output_upper[:200]:
            approval_status = "approved"
        elif "REJECTED" in output_upper[:200]:
            approval_status = "rejected"
        else:
            approval_status = "needs_revision"

        # Extract feedback items (look for bullet points or numbered lists)
        feedback = []
        lines = output.split('\n')

        for line in lines:
            line = line.strip()
            # Extract bullet points and numbered items
            match = re.match(r'^[\d\-\*•]+\.\s*(.+)', line)
            if match:
                item = match.group(1).strip()
                if item and len(item) > 5:
                    feedback.append(item[:300])  # Truncate long items

        # If no structured feedback found, extract paragraphs
        if not feedback:
            # Look for feedback section
            if "FEEDBACK" in output_upper:
                feedback_section = output.split("FEEDBACK")[-1]
                feedback = [
                    p.strip()
                    for p in feedback_section.split('\n\n')
                    if p.strip() and len(p.strip()) > 10
                ][:5]

        # Set status and final output based on approval
        if approval_status == "approved":
            final_output = state.get('draft', '')
            status = "complete"
        else:
            final_output = ""
            status = "reviewing"

        return {
            "approval_status": approval_status,
            "feedback": feedback[:10],  # Max 10 feedback items
            "final_output": final_output,
            "status": status,
        }


# =============================================================================
# Agent Factory
# =============================================================================

def create_agent(
    agent_type: str,
    llm: Optional[BaseChatModel] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseAgent:
    """
    Factory function to create agent instances.

    Args:
        agent_type: Type of agent to create
            (planner, researcher, analyzer, writer, reviewer)
        llm: Pre-configured LLM instance (optional)
        model_name: Model name to use
        temperature: Sampling temperature

    Returns:
        Configured agent instance

    Raises:
        ValueError: If agent_type is unknown

    Examples:
        >>> planner = create_agent("planner")
        >>> researcher = create_agent("researcher", model_name="gpt-4")
        >>> writer = create_agent("writer", temperature=0.5)
    """
    agent_type_lower = agent_type.lower().replace("_agent", "").replace(" ", "")

    # Set default model and temperature based on agent type
    if model_name is None:
        model_name = settings.default_model

    if temperature is None:
        temperature_defaults = {
            "planner": 0.1,
            "researcher": 0.2,
            "analyzer": 0.1,
            "writer": 0.3,
            "reviewer": 0.1,
        }
        temperature = temperature_defaults.get(agent_type_lower, 0.2)

    # Create agent based on type
    agent_classes = {
        "planner": PlannerAgent,
        "researcher": ResearcherAgent,
        "analyzer": AnalyzerAgent,
        "writer": WriterAgent,
        "reviewer": ReviewerAgent,
    }

    agent_class = agent_classes.get(agent_type_lower)

    if agent_class is None:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Valid types: {', '.join(agent_classes.keys())}"
        )

    return agent_class(
        llm=llm,
        model_name=model_name,
        temperature=temperature,
    )


def create_all_agents(
    llm: Optional[BaseChatModel] = None,
    model_name: Optional[str] = None,
) -> dict[str, BaseAgent]:
    """
    Create instances of all agent types.

    Args:
        llm: Pre-configured LLM instance (optional)
        model_name: Default model name to use

    Returns:
        Dictionary mapping agent names to instances
    """
    return {
        "planner": create_agent("planner", llm=llm, model_name=model_name),
        "researcher": create_agent("researcher", llm=llm, model_name=model_name),
        "analyzer": create_agent("analyzer", llm=llm, model_name=model_name),
        "writer": create_agent("writer", llm=llm, model_name=model_name),
        "reviewer": create_agent("reviewer", llm=llm, model_name=model_name),
    }
