"""
FastAPI Security Middleware for AIGuard.

Intercepts and filters HTTP requests/responses to protect LLM applications.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Import guardrails
import sys
from pathlib import Path as PathLib

sys.path.insert(0, str(PathLib(__file__).parent.parent))
from guardrails.prompt_injection.prompt_injection import PromptInjectionDetector, ThreatType
from guardrails.pii.pii_detector import PIIDetector
from guardrails.output_filter.output_guard import OutputGuard

from .config import GuardrailsConfig, get_config


# Configure logging
def setup_logging(log_file: Optional[str]) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_file: Path to log file

    Returns:
        Configured logger
    """
    logger = logging.getLogger("aiguard")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


class BlockedRequestException(Exception):
    """Exception raised when a request is blocked."""

    def __init__(
        self,
        message: str,
        threat_type: str,
        confidence: float,
        details: str,
        status_code: int = 400,
    ):
        self.message = message
        self.threat_type = threat_type
        self.confidence = confidence
        self.details = details
        self.status_code = status_code


class GuardrailsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for LLM security guardrails.

    Intercepts requests/responses and runs security checks.
    """

    def __init__(
        self,
        app: ASGIApp,
        config: Optional[GuardrailsConfig] = None,
    ):
        """
        Initialize middleware.

        Args:
            app: ASGI application
            config: Configuration object (uses default if None)
        """
        super().__init__(app)
        self.config = config or get_config()

        # Setup logging
        self.logger = setup_logging(self.config.log_file_path)

        # Initialize detectors
        self._init_detectors()

        # Rate limiting store
        self._rate_limit_store: Dict[str, List[float]] = {}

        # Statistics
        self._stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "sanitized_requests": 0,
            "passed_requests": 0,
            "threats_by_type": {},
        }

    def _init_detectors(self) -> None:
        """Initialize detection modules based on configuration."""
        self.logger.info("Initializing AIGuard detectors...")

        # Prompt injection detector
        if self.config.enable_injection_detection:
            self.injection_detector = PromptInjectionDetector(
                embedding_model=self.config.embedding_model,
                similarity_threshold=self.config.injection_threshold,
            )
            self.logger.info(f"✓ Injection detection enabled (threshold: {self.config.injection_threshold})")
        else:
            self.injection_detector = None
            self.logger.info("✗ Injection detection disabled")

        # PII detector
        if self.config.enable_pii_detection:
            self.pii_detector = PIIDetector(
                redaction_mode=self.config.pii_redaction_mode,
                spacy_model=self.config.spacy_model,
            )
            self.logger.info(f"✓ PII detection enabled (mode: {self.config.pii_redaction_mode})")
        else:
            self.pii_detector = None
            self.logger.info("✗ PII detection disabled")

        # Output guard
        if self.config.enable_output_filtering:
            self.output_guard = OutputGuard(
                pii_detector=self.pii_detector,
                strict_mode=False,
            )
            self.logger.info("✓ Output filtering enabled")
        else:
            self.output_guard = None
            self.logger.info("✗ Output filtering disabled")

        # Encoding detection
        self.encoding_enabled = self.config.enable_encoding_detection
        self.logger.info(f"✓ Encoding detection: {'enabled' if self.encoding_enabled else 'disabled'}")

        self.logger.info("AIGuard middleware initialized successfully")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process incoming request and outgoing response.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response (possibly modified or error)
        """
        start_time = time.time()
        client_ip = self._get_client_ip(request)

        # Update statistics
        self._stats["total_requests"] += 1

        # Check IP blocklist
        if client_ip in self.config.get_blocked_ips_list():
            self.logger.warning(f"Blocked IP attempted access: {client_ip}")
            return self._create_error_response(
                "Access denied",
                "blocked_ip",
                1.0,
                f"IP address {client_ip} is blocked",
                status_code=403,
            )

        # Check rate limiting
        if self.config.enable_rate_limiting:
            if not self._check_rate_limit(client_ip):
                self.logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return self._create_error_response(
                    "Rate limit exceeded",
                    "rate_limit",
                    1.0,
                    f"More than {self.config.rate_limit_requests} requests per {self.config.rate_limit_window}s",
                    status_code=429,
                )

        # Read request body for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()

                # Parse JSON
                try:
                    request_data = json.loads(body.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Not JSON, skip content inspection
                    request_data = None

                if request_data:
                    # Extract text content from common fields
                    text_content = self._extract_text_content(request_data)

                    if text_content:
                        # Check length limits
                        if len(text_content) > self.config.max_prompt_length:
                            if self.config.truncate_long_inputs:
                                self.logger.warning(
                                    f"Truncated long input from {client_ip}: {len(text_content)} chars"
                                )
                                text_content = text_content[: self.config.max_prompt_length]
                                # Update request data
                                request_data = self._update_text_content(
                                    request_data, text_content
                                )
                                body = json.dumps(request_data).encode()
                            else:
                                self.logger.warning(
                                    f"Rejected long input from {client_ip}: {len(text_content)} chars"
                                )
                                return self._create_error_response(
                                    "Input too long",
                                    "length_limit",
                                    1.0,
                                    f"Input exceeds maximum length of {self.config.max_prompt_length}",
                                    status_code=413,
                                )

                        # Process input through guardrails
                        is_safe, processed_content, guard_result = self.process_input(
                            text_content
                        )

                        if not is_safe:
                            if self.config.block_on_detection:
                                self._stats["blocked_requests"] += 1
                                threat_type_str = (
                                    guard_result.threat_type.value
                                    if guard_result
                                    else "unknown"
                                )

                                self.log_blocked_request(
                                    request,
                                    threat_type_str,
                                    guard_result.confidence if guard_result else 0.0,
                                    guard_result.details if guard_result else "",
                                )

                                return self._create_error_response(
                                    "Request blocked by security policy",
                                    threat_type_str,
                                    guard_result.confidence if guard_result else 0.0,
                                    guard_result.details if guard_result else "Threat detected",
                                    status_code=400,
                                )
                            else:
                                # Just sanitize and continue
                                self._stats["sanitized_requests"] += 1
                                request_data = self._update_text_content(
                                    request_data, processed_content
                                )
                                body = json.dumps(request_data).encode()

                        # Update request body
                        request._body = body

            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                # Continue with original request if processing fails

        # Call next middleware/route
        try:
            response = await call_next(request)

            # Optionally filter response
            if self.config.enable_output_filtering and request.method in [
                "POST",
                "PUT",
            ]:
                response = await self._process_response(request, response)

            # Log if enabled
            if self.config.log_all_requests:
                duration = time.time() - start_time
                self.logger.info(
                    f"Request: {request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Duration: {duration:.3f}s"
                )

            self._stats["passed_requests"] += 1
            return response

        except BlockedRequestException as e:
            # Re-raise blocked request exceptions
            self._stats["blocked_requests"] += 1
            self.log_blocked_request(request, e.threat_type, e.confidence, e.details)
            return self._create_error_response(e.message, e.threat_type, e.confidence, e.details, e.status_code)

    async def _process_response(self, request: Request, response: Response) -> Response:
        """
        Process and filter response content.

        Args:
            request: Original request
            response: Response to filter

        Returns:
            Filtered response
        """
        try:
            # Only process JSON responses
            if "application/json" not in response.headers.get("content-type", ""):
                return response

            # Read response body
            body_bytes = b""
            async for chunk in response.body_iterator:
                body_bytes += chunk

            try:
                response_data = json.loads(body_bytes.decode())

                # Extract text content
                text_content = self._extract_text_content(response_data)

                if text_content and self.output_guard:
                    # Filter output
                    filtered_content, results = self.output_guard.filter_output(text_content)

                    # Check if any issues were found
                    has_issues = any(not r.is_safe for r in results)

                    if has_issues:
                        self.logger.warning(
                            f"Output filtering detected issues: {sum(1 for r in results if not r.is_safe)} threats"
                        )

                        # Update response data
                        response_data = self._update_text_content(
                            response_data, filtered_content
                        )

                        # Create new response
                        new_body = json.dumps(response_data).encode()
                        new_response = Response(
                            content=new_body,
                            status_code=response.status_code,
                            headers=dict(response.headers),
                            media_type="application/json",
                        )

                        # Update content-length
                        new_response.headers["content-length"] = str(len(new_body))

                        return new_response

            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            # Return original response if processing fails
            return Response(
                content=body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

        except Exception as e:
            self.logger.error(f"Error processing response: {e}")
            return response

    def process_input(self, text: str) -> Tuple[bool, str, Optional[Any]]:
        """
        Process input text through guardrails.

        Args:
            text: Input text to process

        Returns:
            Tuple of (is_safe, processed_text, guard_result)
        """
        processed_text = text

        # Check for prompt injection
        if self.injection_detector:
            result = self.injection_detector.detect(text)
            if not result.is_safe:
                if self.config.sanitize_on_detection:
                    processed_text = result.sanitized_input or processed_text
                else:
                    return False, processed_text, result

        # Check for PII
        if self.pii_detector and self.config.enable_pii_redaction:
            has_pii, _ = self.pii_detector.detect(processed_text)
            if has_pii:
                processed_text = self.pii_detector.redact(processed_text)
                # Don't block for PII if just redacting

        return True, processed_text, None

    def process_output(self, text: str) -> Tuple[str, List[Any]]:
        """
        Process output text through guardrails.

        Args:
            text: Output text to process

        Returns:
            Tuple of (filtered_text, list_of_guard_results)
        """
        if self.output_guard:
            return self.output_guard.filter_output(text)
        return text, []

    def _extract_text_content(self, data: Dict) -> Optional[str]:
        """
        Extract text content from request/response data.

        Looks for common field names like 'prompt', 'message', 'content', 'text', etc.

        Args:
            data: Request/response data dictionary

        Returns:
            Extracted text content or None
        """
        text_fields = [
            "prompt",
            "message",
            "messages",
            "content",
            "text",
            "input",
            "query",
            "question",
            "user_message",
            "user_input",
            "completion",
            "response",
            "output",
            "answer",
        ]

        for field in text_fields:
            if field in data:
                value = data[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, list) and len(value) > 0:
                    # Handle messages array format
                    if isinstance(value[0], dict):
                        # Extract content from message objects
                        contents = []
                        for msg in value:
                            if isinstance(msg, dict) and "content" in msg:
                                contents.append(str(msg["content"]))
                        return " ".join(contents)
                    return " ".join(str(v) for v in value)

        return None

    def _update_text_content(self, data: Dict, new_text: str) -> Dict:
        """
        Update text content in request/response data.

        Args:
            data: Request/response data dictionary
            new_text: New text content

        Returns:
            Updated data dictionary
        """
        text_fields = [
            "prompt",
            "message",
            "content",
            "text",
            "input",
            "query",
            "question",
            "user_message",
            "user_input",
        ]

        import copy
        data = copy.deepcopy(data)

        for field in text_fields:
            if field in data:
                if isinstance(data[field], str):
                    data[field] = new_text
                    return data

        return data

    def _check_rate_limit(self, client_ip: str) -> bool:
        """
        Check if client has exceeded rate limit.

        Args:
            client_ip: Client IP address

        Returns:
            True if within limit, False otherwise
        """
        now = time.time()

        # Clean old entries
        if client_ip in self._rate_limit_store:
            self._rate_limit_store[client_ip] = [
                ts
                for ts in self._rate_limit_store[client_ip]
                if now - ts < self.config.rate_limit_window
            ]
        else:
            self._rate_limit_store[client_ip] = []

        # Check limit
        if len(self._rate_limit_store[client_ip]) >= self.config.rate_limit_requests:
            return False

        # Add current request
        self._rate_limit_store[client_ip].append(now)
        return True

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.

        Args:
            request: FastAPI request

        Returns:
            Client IP address
        """
        # Check for forwarded IP (proxy/load balancer)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct IP
        return request.client.host if request.client else "unknown"

    def _create_error_response(
        self,
        message: str,
        threat_type: str,
        confidence: float,
        details: str,
        status_code: int = 400,
    ) -> JSONResponse:
        """
        Create error response for blocked requests.

        Args:
            message: Error message
            threat_type: Type of threat detected
            confidence: Detection confidence
            details: Additional details
            status_code: HTTP status code

        Returns:
            JSONResponse with error details
        """
        error_data = {
            "error": message,
            "threat_detected": True,
            "threat_type": threat_type,
            "confidence": confidence,
        }

        if self.config.return_details:
            error_data["details"] = details

        return JSONResponse(
            content=error_data,
            status_code=status_code,
        )

    def log_blocked_request(
        self,
        request: Request,
        threat_type: str,
        confidence: float,
        details: str,
    ) -> None:
        """
        Log blocked request details.

        Args:
            request: Blocked request
            threat_type: Type of threat
            confidence: Detection confidence
            details: Additional details
        """
        # Update threat statistics
        self._stats["threats_by_type"][threat_type] = (
            self._stats["threats_by_type"].get(threat_type, 0) + 1
        )

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "client_ip": self._get_client_ip(request),
            "threat_type": threat_type,
            "confidence": confidence,
            "details": details,
        }

        self.logger.warning(f"Blocked request: {json.dumps(log_entry)}")

    def get_statistics(self) -> Dict:
        """
        Get middleware statistics.

        Returns:
            Dictionary with statistics
        """
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "sanitized_requests": 0,
            "passed_requests": 0,
            "threats_by_type": {},
        }


# Convenience function for easy middleware addition
def add_guardrails(
    app,
    config: Optional[GuardrailsConfig] = None,
) -> GuardrailsMiddleware:
    """
    Add guardrails middleware to FastAPI app.

    Args:
        app: FastAPI application
        config: Optional configuration

    Returns:
        GuardrailsMiddleware instance

    Example:
        from fastapi import FastAPI
        from src.middleware.security_middleware import add_guardrails, GuardrailsConfig

        app = FastAPI()
        config = GuardrailsConfig(
            enable_injection_detection=True,
            enable_pii_detection=True,
            block_on_detection=True,
        )
        add_guardrails(app, config)
    """
    middleware = GuardrailsMiddleware(app, config)
    app.add_middleware(GuardrailsMiddleware, config=config)
    return middleware
