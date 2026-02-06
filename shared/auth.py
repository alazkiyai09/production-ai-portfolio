# ============================================================
# Authentication and Authorization for All Projects
# ============================================================
"""
JWT-based authentication module for FastAPI applications.

This module provides comprehensive authentication functionality including:
- JWT token generation and validation
- Password hashing with bcrypt
- OAuth2 Bearer token dependencies for FastAPI
- User model with role-based access control
- Decorator-based authentication requirements
- Token refresh mechanism
- Password reset functionality

Example:
    >>> from shared.auth import (
    ...     create_access_token,
    ...     verify_token,
    ...     get_current_user,
    ...     require_role,
    ...     hash_password,
    ...     verify_password,
    ... )
    >>> from fastapi import FastAPI, Depends
    >>>
    >>> app = FastAPI()
    >>>
    >>> @app.post("/login")
    >>> async def login(username: str, password: str):
    ...     user = authenticate_user(username, password)
    ...     token = create_access_token(user)
    ...     return {"access_token": token}
    >>>
    >>> @app.get("/protected")
    >>> async def protected(current_user: User = Depends(get_current_user)):
    ...     return {"user": current_user.username}
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Callable, Awaitable
from functools import wraps
from enum import Enum

import bcrypt
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr, validator

logger = logging.getLogger(__name__)


# ============================================================
# Configuration Constants
# ============================================================

# JWT Configuration
SECRET_KEY: str = secrets.token_urlsafe(32)
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
REFRESH_TOKEN_EXPIRE_DAYS: int = 7

# Password Configuration
MIN_PASSWORD_LENGTH: int = 8
PASSWORD_REQUIRE_UPPERCASE: bool = True
PASSWORD_REQUIRE_LOWERCASE: bool = True
PASSWORD_REQUIRE_DIGIT: bool = True
PASSWORD_REQUIRE_SPECIAL: bool = False

# Token Configuration
RESET_TOKEN_EXPIRE_HOURS: int = 1
VERIFICATION_TOKEN_EXPIRE_HOURS: int = 24


# ============================================================
# User Role Enum
# ============================================================

class Role(str, Enum):
    """User roles for access control."""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


# Role hierarchy - higher roles inherit lower role permissions
ROLE_HIERARCHY: dict[Role, int] = {
    Role.VIEWER: 1,
    Role.USER: 2,
    Role.ADMIN: 3,
}


def get_role_level(role: Role) -> int:
    """
    Get the permission level for a role.

    Args:
        role: Role to get level for

    Returns:
        Integer permission level (higher = more permissions)

    Example:
        >>> get_role_level(Role.ADMIN)
        3
        >>> get_role_level(Role.VIEWER)
        1
    """
    return ROLE_HIERARCHY.get(role, 0)


def role_has_permission(required_role: Role, user_role: Role) -> bool:
    """
    Check if a user role has permission for the required role.

    Args:
        required_role: The role required for the operation
        user_role: The user's current role

    Returns:
        True if user_role has permission for required_role

    Example:
        >>> role_has_permission(Role.USER, Role.ADMIN)
        True
        >>> role_has_permission(Role.ADMIN, Role.USER)
        False
    """
    return get_role_level(user_role) >= get_role_level(required_role)


# ============================================================
# Pydantic Models
# ============================================================

class UserBase(BaseModel):
    """Base user model with common fields."""

    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    is_active: bool = True


class UserCreate(UserBase):
    """Model for user registration."""

    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH)
    role: Role = Role.USER

    @validator("password")
    def validate_password(cls, v: str) -> str:
        """Validate password strength requirements."""
        if PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if PASSWORD_REQUIRE_DIGIT and not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if PASSWORD_REQUIRE_SPECIAL:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in v):
                raise ValueError("Password must contain at least one special character")
        return v


class UserUpdate(BaseModel):
    """Model for updating user information."""

    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    role: Optional[Role] = None
    password: Optional[str] = Field(None, min_length=MIN_PASSWORD_LENGTH)


class User(UserBase):
    """Complete user model with all fields."""

    id: str
    role: Role
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        """Pydantic config."""

        from_attributes = True


class UserInDB(User):
    """User model with hashed password (internal use only)."""

    hashed_password: str


class Token(BaseModel):
    """JWT token response model."""

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: User


class TokenData(BaseModel):
    """Data extracted from JWT token."""

    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[Role] = None


class PasswordResetRequest(BaseModel):
    """Model for password reset request."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Model for password reset confirmation."""

    token: str
    new_password: str = Field(..., min_length=MIN_PASSWORD_LENGTH)


class PasswordChange(BaseModel):
    """Model for password change (authenticated)."""

    old_password: str
    new_password: str = Field(..., min_length=MIN_PASSWORD_LENGTH)


class RefreshTokenRequest(BaseModel):
    """Model for token refresh request."""

    refresh_token: str


# ============================================================
# OAuth2 Scheme
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


# ============================================================
# Password Utilities
# ============================================================

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password string

    Example:
        >>> hashed = hash_password("mypassword123")
        >>> isinstance(hashed, str)
        True
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.

    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password to compare against

    Returns:
        True if password matches, False otherwise

    Example:
        >>> hashed = hash_password("mypassword123")
        >>> verify_password("mypassword123", hashed)
        True
        >>> verify_password("wrongpassword", hashed)
        False
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """
    Validate password strength and return any issues.

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, list_of_errors)

    Example:
        >>> is_valid, errors = validate_password_strength("weak")
        >>> if not is_valid:
        ...     print(errors)
    """
    errors = []

    if len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")

    if PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")

    if PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")

    if PASSWORD_REQUIRE_DIGIT and not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")

    if PASSWORD_REQUIRE_SPECIAL:
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            errors.append("Password must contain at least one special character")

    return len(errors) == 0, errors


# ============================================================
# JWT Token Utilities
# ============================================================

def create_access_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    custom_secret: Optional[str] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload data to encode in the token
        expires_delta: Optional custom expiration time
        custom_secret: Optional custom secret key

    Returns:
        Encoded JWT token string

    Example:
        >>> token = create_access_token(
        ...     {"sub": "john_doe", "role": "user"},
        ...     expires_delta=timedelta(hours=1)
        ... )
    """
    to_encode = data.copy()

    # Set expiration
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    })

    # Encode token
    secret = custom_secret or SECRET_KEY
    encoded_jwt = jwt.encode(to_encode, secret, algorithm=ALGORITHM)

    return encoded_jwt


def create_refresh_token(
    user_id: str,
    username: str,
    custom_secret: Optional[str] = None,
) -> str:
    """
    Create a JWT refresh token.

    Args:
        user_id: User ID to embed in token
        username: Username to embed in token
        custom_secret: Optional custom secret key

    Returns:
        Encoded JWT refresh token string

    Example:
        >>> token = create_refresh_token("user123", "john_doe")
    """
    expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    return create_access_token(
        data={
            "sub": username,
            "user_id": user_id,
            "type": "refresh",
        },
        expires_delta=expires_delta,
        custom_secret=custom_secret,
    )


def decode_token(
    token: str,
    custom_secret: Optional[str] = None,
) -> dict[str, Any]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string to decode
        custom_secret: Optional custom secret key

    Returns:
        Decoded token payload

    Raises:
        JWTError: If token is invalid or expired

    Example:
        >>> token = create_access_token({"sub": "john_doe"})
        >>> payload = decode_token(token)
        >>> payload["sub"]
        'john_doe'
    """
    secret = custom_secret or SECRET_KEY
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"Token decode failed: {e}")
        raise


def verify_token(token: str) -> TokenData:
    """
    Verify a JWT token and extract user data.

    Args:
        token: JWT token string to verify

    Returns:
        TokenData object with extracted user information

    Raises:
        HTTPException: If token is invalid or expired

    Example:
        >>> token = create_access_token(
        ...     {"sub": "john_doe", "user_id": "123", "role": "user"}
        ... )
        >>> token_data = verify_token(token)
        >>> token_data.username
        'john_doe'
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(token)

        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        role: str = payload.get("role", Role.USER)

        token_type: str = payload.get("type", "access")
        if token_type != "access":
            raise credentials_exception

        if username is None:
            raise credentials_exception

        return TokenData(
            username=username,
            user_id=user_id,
            role=Role(role) if role else Role.USER,
        )

    except (JWTError, ValueError) as e:
        logger.warning(f"Token verification failed: {e}")
        raise credentials_exception


def refresh_access_token(refresh_token: str) -> tuple[str, datetime]:
    """
    Create a new access token from a refresh token.

    Args:
        refresh_token: Valid refresh token

    Returns:
        Tuple of (new_access_token, expiration_time)

    Raises:
        HTTPException: If refresh token is invalid

    Example:
        >>> refresh = create_refresh_token("user123", "john_doe")
        >>> access_token, expires = refresh_access_token(refresh)
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(refresh_token)

        token_type: str = payload.get("type")
        if token_type != "refresh":
            raise credentials_exception

        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        role: str = payload.get("role", Role.USER)

        if username is None or user_id is None:
            raise credentials_exception

        # Create new access token
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_token = create_access_token(
            data={
                "sub": username,
                "user_id": user_id,
                "role": role,
            },
            expires_delta=expires_delta,
        )

        expires = datetime.now(timezone.utc) + expires_delta

        return new_token, expires

    except (JWTError, ValueError) as e:
        logger.warning(f"Refresh token verification failed: {e}")
        raise credentials_exception


# ============================================================
# Password Reset Utilities
# ============================================================

def generate_reset_token() -> str:
    """
    Generate a secure password reset token.

    Returns:
        URL-safe random token

    Example:
        >>> token = generate_reset_token()
        >>> len(token)
        64
    """
    return secrets.token_urlsafe(48)


def generate_password_reset_link(
    reset_token: str,
    base_url: str = "https://example.com",
) -> str:
    """
    Generate a password reset link.

    Args:
        reset_token: Reset token from generate_reset_token()
        base_url: Base URL for the application

    Returns:
        Full password reset URL

    Example:
        >>> token = generate_reset_token()
        >>> link = generate_password_reset_link(
        ...     token,
        ...     base_url="https://myapp.com"
        ... )
    """
    return f"{base_url}/auth/reset-password?token={reset_token}"


def create_reset_token_jwt(
    email: str,
    custom_secret: Optional[str] = None,
) -> str:
    """
    Create a JWT token for password reset.

    Args:
        email: User email
        custom_secret: Optional custom secret key

    Returns:
        Encoded JWT reset token

    Example:
        >>> token = create_reset_token_jwt("user@example.com")
    """
    expires_delta = timedelta(hours=RESET_TOKEN_EXPIRE_HOURS)

    return create_access_token(
        data={
            "sub": email,
            "type": "reset",
        },
        expires_delta=expires_delta,
        custom_secret=custom_secret,
    )


def verify_reset_token(token: str) -> Optional[str]:
    """
    Verify a password reset token and return the email.

    Args:
        token: Reset token to verify

    Returns:
        Email if token is valid, None otherwise

    Example:
        >>> token = create_reset_token_jwt("user@example.com")
        >>> email = verify_reset_token(token)
        >>> email
        'user@example.com'
    """
    try:
        payload = decode_token(token)

        token_type: str = payload.get("type")
        if token_type != "reset":
            return None

        email: str = payload.get("sub")
        return email

    except (JWTError, ValueError):
        return None


# ============================================================
# FastAPI Dependencies
# ============================================================

async def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
) -> TokenData:
    """
    FastAPI dependency to get current authenticated user from token.

    Args:
        request: FastAPI request object
        token: JWT token from Authorization header

    Returns:
        TokenData with user information

    Raises:
        HTTPException: If token is invalid

    Example:
        >>> @app.get("/me")
        >>> async def get_profile(
        ...     current: TokenData = Depends(get_current_user)
        ... ):
        ...     return {"username": current.username}
    """
    token_data = verify_token(token)

    # Store user info in request state for access in other dependencies
    request.state.user_id = token_data.user_id
    request.state.username = token_data.username
    request.state.role = token_data.role

    return token_data


async def get_current_user_required(
    current: TokenData = Depends(get_current_user),
) -> TokenData:
    """
    Dependency that requires authentication (always raises if no user).

    This is the same as get_current_user but semantically indicates
    that the endpoint cannot be accessed without authentication.

    Example:
        >>> @app.get("/protected")
        >>> async def protected(
        ...     current: TokenData = Depends(get_current_user_required)
        ... ):
        ...     return {"user": current.username}
    """
    return current


async def require_role(required_role: Role):
    """
    Create a dependency that requires a specific role.

    Args:
        required_role: Minimum role required

    Returns:
        Dependency function

    Example:
        >>> @app.post("/admin")
        >>> async def admin_endpoint(
        ...     current: TokenData = Depends(require_role(Role.ADMIN))
        ... ):
        ...     return {"message": "Welcome admin"}
    """
    async def role_checker(current: TokenData = Depends(get_current_user)) -> TokenData:
        if current.role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not determine user role",
            )

        if not role_has_permission(required_role, current.role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Role '{required_role.value}' required.",
            )

        return current

    return role_checker


def require_admin():
    """
    Dependency that requires admin role.

    Example:
        >>> @app.delete("/users/{user_id}")
        >>> async def delete_user(
        ...     current: TokenData = Depends(require_admin())
        ... ):
        ...     return {"message": "User deleted"}
    """
    return require_role(Role.ADMIN)


def require_user():
    """
    Dependency that requires at least user role (not viewer).

    Example:
        >>> @app.post("/create")
        >>> async def create_resource(
        ...     current: TokenData = Depends(require_user())
        ... ):
        ...     return {"message": "Resource created"}
    """
    return require_role(Role.USER)


# ============================================================
# Decorators
# ============================================================

def auth_required(
    role: Optional[Role] = None,
    allow_refresh: bool = False,
):
    """
    Decorator to require authentication for an endpoint.

    Args:
        role: Optional minimum role required
        allow_refresh: Whether to allow token refresh

    Example:
        >>> @app.get("/protected")
        >>> @auth_required(role=Role.USER)
        >>> async def protected():
        ...     return {"message": "Protected resource"}
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if request is in kwargs
            request: Optional[Request] = kwargs.get("request")

            if request is None:
                # Try to get request from args
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                )

            # Get token from Authorization header
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication header",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            token = auth_header.split(" ")[1]

            # Verify token
            try:
                token_data = verify_token(token)

                # Check role if specified
                if role and not role_has_permission(role, token_data.role):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Insufficient permissions. Role '{role.value}' required.",
                    )

                # Add user info to request state
                request.state.user_id = token_data.user_id
                request.state.username = token_data.username
                request.state.role = token_data.role

            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Auth verification failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Call the original function
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def roles_allowed(*allowed_roles: Role):
    """
    Decorator to allow only specific roles.

    Args:
        *allowed_roles: Roles that are allowed to access the endpoint

    Example:
        >>> @app.get("/moderate")
        >>> @roles_allowed(Role.ADMIN, Role.USER)
        >>> async def moderate():
        ...     return {"message": "Moderation endpoint"}
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request: Optional[Request] = kwargs.get("request")

            if request is None:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                )

            # Get user role from request state
            user_role = getattr(request.state, "role", None)

            if user_role is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if user_role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{user_role.value}' not allowed. Allowed roles: {[r.value for r in allowed_roles]}",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================
# User Store (Abstract Interface)
# ============================================================

class UserStore:
    """
    Abstract interface for user storage and authentication.

    Implement this class for your specific storage backend
    (database, ORM, etc.) and use it with the auth functions.

    Example:
        >>> class MyUserStore(UserStore):
        ...     async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        ...         # Load from database
        ...         return user_from_db
        ...
        >>> user_store = MyUserStore()
    """

    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """
        Retrieve a user by username.

        Args:
            username: Username to look up

        Returns:
            UserInDB if found, None otherwise
        """
        raise NotImplementedError("Implement get_user_by_username in your UserStore")

    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """
        Retrieve a user by email.

        Args:
            email: Email to look up

        Returns:
            UserInDB if found, None otherwise
        """
        raise NotImplementedError("Implement get_user_by_email in your UserStore")

    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """
        Retrieve a user by ID.

        Args:
            user_id: User ID to look up

        Returns:
            UserInDB if found, None otherwise
        """
        raise NotImplementedError("Implement get_user_by_id in your UserStore")

    async def create_user(self, user: UserCreate) -> User:
        """
        Create a new user.

        Args:
            user: User data to create

        Returns:
            Created user without password
        """
        raise NotImplementedError("Implement create_user in your UserStore")

    async def update_user(self, user_id: str, user_update: UserUpdate) -> User:
        """
        Update an existing user.

        Args:
            user_id: ID of user to update
            user_update: Updated user data

        Returns:
            Updated user without password
        """
        raise NotImplementedError("Implement update_user in your UserStore")

    async def update_password(
        self,
        user_id: str,
        new_hashed_password: str,
    ) -> bool:
        """
        Update a user's password.

        Args:
            user_id: ID of user
            new_hashed_password: New hashed password

        Returns:
            True if successful
        """
        raise NotImplementedError("Implement update_password in your UserStore")

    async def store_reset_token(
        self,
        user_id: str,
        reset_token: str,
        expires_at: datetime,
    ) -> bool:
        """
        Store a password reset token.

        Args:
            user_id: ID of user
            reset_token: Reset token to store
            expires_at: Expiration time

        Returns:
            True if successful
        """
        raise NotImplementedError("Implement store_reset_token in your UserStore")

    async def verify_reset_token(
        self,
        reset_token: str,
    ) -> Optional[UserInDB]:
        """
        Verify a reset token and return the associated user.

        Args:
            reset_token: Reset token to verify

        Returns:
            User if token is valid, None otherwise
        """
        raise NotImplementedError("Implement verify_reset_token in your UserStore")

    async def delete_reset_token(self, reset_token: str) -> bool:
        """
        Delete a used reset token.

        Args:
            reset_token: Token to delete

        Returns:
            True if successful
        """
        raise NotImplementedError("Implement delete_reset_token in your UserStore")

    async def store_refresh_token(
        self,
        user_id: str,
        refresh_token: str,
        expires_at: datetime,
    ) -> bool:
        """
        Store a refresh token for a user.

        Args:
            user_id: ID of user
            refresh_token: Refresh token to store
            expires_at: Expiration time

        Returns:
            True if successful
        """
        raise NotImplementedError("Implement store_refresh_token in your UserStore")

    async def verify_refresh_token(
        self,
        refresh_token: str,
    ) -> Optional[UserInDB]:
        """
        Verify a refresh token and return the associated user.

        Args:
            refresh_token: Refresh token to verify

        Returns:
            User if token is valid, None otherwise
        """
        raise NotImplementedError("Implement verify_refresh_token in your UserStore")

    async def revoke_refresh_token(self, refresh_token: str) -> bool:
        """
        Revoke a refresh token.

        Args:
            refresh_token: Token to revoke

        Returns:
            True if successful
        """
        raise NotImplementedError("Implement revoke_refresh_token in your UserStore")

    async def revoke_all_user_tokens(self, user_id: str) -> bool:
        """
        Revoke all refresh tokens for a user.

        Args:
            user_id: ID of user

        Returns:
            True if successful
        """
        raise NotImplementedError("Implement revoke_all_user_tokens in your UserStore")


# ============================================================
# Authentication Functions
# ============================================================

async def authenticate_user(
    username: str,
    password: str,
    user_store: UserStore,
) -> Optional[UserInDB]:
    """
    Authenticate a user with username and password.

    Args:
        username: Username or email
        password: Plain text password
        user_store: UserStore instance

    Returns:
        UserInDB if authentication successful, None otherwise

    Example:
        >>> user = await authenticate_user(
        ...     "john_doe",
        ...     "password123",
        ...     user_store
        ... )
        >>> if user:
        ...     print(f"Authenticated: {user.username}")
    """
    # Try username first, then email
    user = await user_store.get_user_by_username(username)
    if user is None:
        # Check if username is actually an email
        if "@" in username:
            user = await user_store.get_user_by_email(username)

    if user is None:
        logger.warning(f"Authentication failed: User not found - {username}")
        return None

    if not verify_password(password, user.hashed_password):
        logger.warning(f"Authentication failed: Invalid password - {username}")
        return None

    if not user.is_active:
        logger.warning(f"Authentication failed: User inactive - {username}")
        return None

    return user


async def login_user(
    username: str,
    password: str,
    user_store: UserStore,
) -> Optional[Token]:
    """
    Login a user and return access token.

    Args:
        username: Username or email
        password: Plain text password
        user_store: UserStore instance

    Returns:
        Token object if successful, None otherwise

    Example:
        >>> token = await login_user(
        ...     "john_doe",
        ...     "password123",
        ...     user_store
        ... )
        >>> if token:
        ...     print(f"Access token: {token.access_token}")
    """
    user = await authenticate_user(username, password, user_store)

    if user is None:
        return None

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "role": user.role.value,
        },
        expires_delta=access_token_expires,
    )

    # Create refresh token
    refresh_token = create_refresh_token(
        user_id=user.id,
        username=user.username,
    )
    refresh_expires = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    # Store refresh token
    await user_store.store_refresh_token(
        user_id=user.id,
        refresh_token=refresh_token,
        expires_at=refresh_expires,
    )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=User(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
        ),
    )


async def register_user(
    user_data: UserCreate,
    user_store: UserStore,
) -> User:
    """
    Register a new user.

    Args:
        user_data: User registration data
        user_store: UserStore instance

    Returns:
        Created user

    Raises:
        HTTPException: If user already exists

    Example:
        >>> new_user = UserCreate(
        ...     username="john_doe",
        ...     email="john@example.com",
        ...     password="SecurePass123",
        ...     role=Role.USER
        ... )
        >>> user = await register_user(new_user, user_store)
    """
    # Check if user already exists
    existing = await user_store.get_user_by_username(user_data.username)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    if user_data.email:
        existing_email = await user_store.get_user_by_email(user_data.email)
        if existing_email is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

    # Create user
    user = await user_store.create_user(user_data)

    logger.info(f"New user registered: {user.username}")

    return user


async def refresh_user_token(
    refresh_token: str,
    user_store: UserStore,
) -> Optional[Token]:
    """
    Refresh an access token using a refresh token.

    Args:
        refresh_token: Valid refresh token
        user_store: UserStore instance

    Returns:
        New Token object if successful, None otherwise

    Example:
        >>> token = await refresh_user_token(refresh_token, user_store)
    """
    # Verify refresh token with user store
    user = await user_store.verify_refresh_token(refresh_token)

    if user is None:
        return None

    # Create new tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "role": user.role.value,
        },
        expires_delta=access_token_expires,
    )

    new_refresh_token = create_refresh_token(
        user_id=user.id,
        username=user.username,
    )
    refresh_expires = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    # Revoke old refresh token and store new one
    await user_store.revoke_refresh_token(refresh_token)
    await user_store.store_refresh_token(
        user_id=user.id,
        refresh_token=new_refresh_token,
        expires_at=refresh_expires,
    )

    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=User(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
        ),
    )


async def initiate_password_reset(
    email: str,
    user_store: UserStore,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """
    Initiate password reset process.

    Args:
        email: User email
        user_store: UserStore instance
        base_url: Optional base URL for reset link

    Returns:
        Reset link if user found, None otherwise

    Example:
        >>> reset_link = await initiate_password_reset(
        ...     "user@example.com",
        ...     user_store,
        ...     base_url="https://myapp.com"
        ... )
        >>> if reset_link:
        ...     # Send email with reset_link
        ...     send_email(email, reset_link)
    """
    user = await user_store.get_user_by_email(email)

    if user is None:
        # Don't reveal whether email exists
        logger.info(f"Password reset requested for non-existent email: {email}")
        return None

    # Create reset token
    reset_token = create_reset_token_jwt(email)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=RESET_TOKEN_EXPIRE_HOURS)

    # Store token
    await user_store.store_reset_token(
        user_id=user.id,
        reset_token=reset_token,
        expires_at=expires_at,
    )

    logger.info(f"Password reset initiated for: {email}")

    if base_url:
        return generate_password_reset_link(reset_token, base_url)

    return reset_token


async def complete_password_reset(
    token: str,
    new_password: str,
    user_store: UserStore,
) -> bool:
    """
    Complete password reset with token.

    Args:
        token: Reset token
        new_password: New plain text password
        user_store: UserStore instance

    Returns:
        True if password reset successful

    Example:
        >>> success = await complete_password_reset(
        ...     reset_token,
        ...     "NewSecurePass123",
        ...     user_store
        ... )
    """
    # Verify token and get user
    email = verify_reset_token(token)

    if email is None:
        logger.warning(f"Invalid password reset token used")
        return False

    user = await user_store.get_user_by_email(email)
    if user is None:
        return False

    # Verify token in user store
    user_from_token = await user_store.verify_reset_token(token)
    if user_from_token is None:
        logger.warning(f"Expired or invalid reset token for: {email}")
        return False

    # Hash new password
    hashed_password = hash_password(new_password)

    # Update password
    await user_store.update_password(user.id, hashed_password)

    # Delete used token
    await user_store.delete_reset_token(token)

    # Revoke all refresh tokens for security
    await user_store.revoke_all_user_tokens(user.id)

    logger.info(f"Password reset completed for: {email}")

    return True


async def change_user_password(
    user_id: str,
    old_password: str,
    new_password: str,
    user_store: UserStore,
) -> bool:
    """
    Change user password (authenticated).

    Args:
        user_id: ID of user
        old_password: Current password
        new_password: New password
        user_store: UserStore instance

    Returns:
        True if password changed successfully

    Raises:
        HTTPException: If old password is incorrect

    Example:
        >>> success = await change_user_password(
        ...     "user123",
        ...     "OldPass123",
        ...     "NewPass456",
        ...     user_store
        ... )
    """
    user = await user_store.get_user_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if not verify_password(old_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password",
        )

    # Hash new password
    hashed_password = hash_password(new_password)

    # Update password
    await user_store.update_password(user_id, hashed_password)

    # Revoke all refresh tokens for security
    await user_store.revoke_all_user_tokens(user_id)

    logger.info(f"Password changed for user: {user.username}")

    return True


async def logout_user(
    refresh_token: str,
    user_store: UserStore,
) -> bool:
    """
    Logout a user by revoking their refresh token.

    Args:
        refresh_token: Refresh token to revoke
        user_store: UserStore instance

    Returns:
        True if logout successful

    Example:
        >>> success = await logout_user(refresh_token, user_store)
    """
    result = await user_store.revoke_refresh_token(refresh_token)
    if result:
        logger.info("User logged out successfully")
    return result


async def logout_all_devices(
    user_id: str,
    user_store: UserStore,
) -> bool:
    """
    Logout a user from all devices by revoking all tokens.

    Args:
        user_id: ID of user
        user_store: UserStore instance

    Returns:
        True if logout successful

    Example:
        >>> success = await logout_all_devices("user123", user_store)
    """
    result = await user_store.revoke_all_user_tokens(user_id)
    if result:
        logger.info(f"User {user_id} logged out from all devices")
    return result


# ============================================================
# In-Memory User Store (for development/testing)
# ============================================================

class InMemoryUserStore(UserStore):
    """
    In-memory implementation of UserStore for development and testing.

    This is NOT suitable for production use as all data is lost on restart.
    Use a proper database-backed implementation in production.

    Example:
        >>> user_store = InMemoryUserStore()
        >>> user = await user_store.create_user(UserCreate(
        ...     username="test",
        ...     password="TestPass123",
        ...     email="test@example.com"
        ... ))
    """

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._users: dict[str, UserInDB] = {}
        self._users_by_email: dict[str, UserInDB] = {}
        self._users_by_username: dict[str, UserInDB] = {}
        self._reset_tokens: dict[str, tuple[str, datetime]] = {}
        self._refresh_tokens: dict[str, tuple[str, datetime]] = {}

    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        return self._users_by_username.get(username)

    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email."""
        return self._users_by_email.get(email)

    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID."""
        return self._users.get(user_id)

    async def create_user(self, user: UserCreate) -> User:
        """Create a new user."""
        user_id = f"user_{secrets.token_hex(8)}"
        now = datetime.now(timezone.utc)

        hashed_password = hash_password(user.password)

        user_in_db = UserInDB(
            id=user_id,
            username=user.username,
            email=user.email,
            hashed_password=hashed_password,
            role=user.role,
            is_active=True,
            created_at=now,
            updated_at=now,
        )

        self._users[user_id] = user_in_db
        self._users_by_username[user.username] = user_in_db
        if user.email:
            self._users_by_email[user.email] = user_in_db

        return User(
            id=user_in_db.id,
            username=user_in_db.username,
            email=user_in_db.email,
            role=user_in_db.role,
            is_active=user_in_db.is_active,
            created_at=user_in_db.created_at,
            updated_at=user_in_db.updated_at,
        )

    async def update_user(self, user_id: str, user_update: UserUpdate) -> User:
        """Update an existing user."""
        user = self._users.get(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        if user_update.email is not None:
            # Remove from email index if email changed
            if user.email and user.email != user_update.email:
                self._users_by_email.pop(user.email, None)
            user.email = user_update.email
            self._users_by_email[user_update.email] = user

        if user_update.is_active is not None:
            user.is_active = user_update.is_active

        if user_update.role is not None:
            user.role = user_update.role

        if user_update.password is not None:
            user.hashed_password = hash_password(user_update.password)

        user.updated_at = datetime.now(timezone.utc)

        return User(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    async def update_password(self, user_id: str, new_hashed_password: str) -> bool:
        """Update user password."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.hashed_password = new_hashed_password
        user.updated_at = datetime.now(timezone.utc)
        return True

    async def store_reset_token(
        self,
        user_id: str,
        reset_token: str,
        expires_at: datetime,
    ) -> bool:
        """Store a password reset token."""
        self._reset_tokens[reset_token] = (user_id, expires_at)
        return True

    async def verify_reset_token(self, reset_token: str) -> Optional[UserInDB]:
        """Verify a reset token."""
        data = self._reset_tokens.get(reset_token)
        if data is None:
            return None

        user_id, expires_at = data
        if datetime.now(timezone.utc) > expires_at:
            del self._reset_tokens[reset_token]
            return None

        return self._users.get(user_id)

    async def delete_reset_token(self, reset_token: str) -> bool:
        """Delete a reset token."""
        self._reset_tokens.pop(reset_token, None)
        return True

    async def store_refresh_token(
        self,
        user_id: str,
        refresh_token: str,
        expires_at: datetime,
    ) -> bool:
        """Store a refresh token."""
        self._refresh_tokens[refresh_token] = (user_id, expires_at)
        return True

    async def verify_refresh_token(self, refresh_token: str) -> Optional[UserInDB]:
        """Verify a refresh token."""
        data = self._refresh_tokens.get(refresh_token)
        if data is None:
            return None

        user_id, expires_at = data
        if datetime.now(timezone.utc) > expires_at:
            del self._refresh_tokens[refresh_token]
            return None

        return self._users.get(user_id)

    async def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke a refresh token."""
        self._refresh_tokens.pop(refresh_token, None)
        return True

    async def revoke_all_user_tokens(self, user_id: str) -> bool:
        """Revoke all refresh tokens for a user."""
        to_delete = [
            token
            for token, (uid, _) in self._refresh_tokens.items()
            if uid == user_id
        ]
        for token in to_delete:
            self._refresh_tokens.pop(token, None)
        return True


# ============================================================
# Configuration Helper
# ============================================================

def configure_auth(
    secret_key: Optional[str] = None,
    access_token_expire_minutes: Optional[int] = None,
    refresh_token_expire_days: Optional[int] = None,
    min_password_length: Optional[int] = None,
) -> None:
    """
    Configure authentication settings globally.

    Args:
        secret_key: Custom secret key for JWT
        access_token_expire_minutes: Access token expiration in minutes
        refresh_token_expire_days: Refresh token expiration in days
        min_password_length: Minimum password length

    Example:
        >>> configure_auth(
        ...     secret_key="my-secret-key",
        ...     access_token_expire_minutes=60
        ... )
    """
    global SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
    global MIN_PASSWORD_LENGTH

    if secret_key is not None:
        SECRET_KEY = secret_key

    if access_token_expire_minutes is not None:
        ACCESS_TOKEN_EXPIRE_MINUTES = access_token_expire_minutes

    if refresh_token_expire_days is not None:
        REFRESH_TOKEN_EXPIRE_DAYS = refresh_token_expire_days

    if min_password_length is not None:
        MIN_PASSWORD_LENGTH = min_password_length


# ============================================================
# Export
# ============================================================

__all__ = [
    # Enums
    "Role",
    # Models
    "User",
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserInDB",
    "UserStore",
    "InMemoryUserStore",
    "Token",
    "TokenData",
    "PasswordResetRequest",
    "PasswordResetConfirm",
    "PasswordChange",
    "RefreshTokenRequest",
    # JWT Functions
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "verify_token",
    "refresh_access_token",
    # Password Functions
    "hash_password",
    "verify_password",
    "validate_password_strength",
    "generate_reset_token",
    "generate_password_reset_link",
    "create_reset_token_jwt",
    "verify_reset_token",
    # OAuth2
    "oauth2_scheme",
    # Dependencies
    "get_current_user",
    "get_current_user_required",
    "require_role",
    "require_admin",
    "require_user",
    # Decorators
    "auth_required",
    "roles_allowed",
    # Auth Functions
    "authenticate_user",
    "login_user",
    "register_user",
    "refresh_user_token",
    "initiate_password_reset",
    "complete_password_reset",
    "change_user_password",
    "logout_user",
    "logout_all_devices",
    # Configuration
    "configure_auth",
    # Constants
    "SECRET_KEY",
    "ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "REFRESH_TOKEN_EXPIRE_DAYS",
    "MIN_PASSWORD_LENGTH",
    "ROLE_HIERARCHY",
    "role_has_permission",
    "get_role_level",
]
