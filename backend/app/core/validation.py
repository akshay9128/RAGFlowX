import re
from fastapi import HTTPException, status

ID_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_id(val: str, name: str = "ID") -> str:
    if not isinstance(val, str) or not ID_REGEX.fullmatch(val):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {name} format. Must start with an alphanumeric character, contain only letters, numbers, underscores, or hyphens, and be at most 64 characters long."
        )
    return val
