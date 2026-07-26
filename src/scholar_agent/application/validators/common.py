"""Small validation helpers for application inputs."""


def require_non_blank(value: str, field_name: str) -> str:
    """Return a stripped value or raise when it is blank."""
    normalized_value = value.strip()
    if not normalized_value:
        message = f"{field_name} must not be blank."
        raise ValueError(message)
    return normalized_value


def require_positive(value: int, field_name: str) -> int:
    """Return a positive integer or raise when it is not positive."""
    if value < 1:
        message = f"{field_name} must be greater than zero."
        raise ValueError(message)
    return value
