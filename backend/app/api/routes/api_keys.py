from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List
from backend.app.core.security import api_key_store, require_scope, APIKeyRecord

router = APIRouter(prefix="/keys", tags=["API Key Management"])


class KeyCreateRequest(BaseModel):
    name: str = Field(..., description="Name or owner of the API key")
    scopes: List[str] = Field(default=["read", "write"], description="Allowed scopes: read, write, admin")


class KeyCreateResponse(BaseModel):
    key_id: str
    name: str
    raw_key: str = Field(..., description="The raw API key. Shown ONLY ONCE at creation.")
    prefix: str
    scopes: List[str]
    created_at: str


class KeyItem(BaseModel):
    key_id: str
    name: str
    prefix: str
    scopes: List[str]
    created_at: str
    revoked: bool


@router.post(
    "",
    response_model=KeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new API key (Admin required)"
)
async def create_api_key(payload: KeyCreateRequest, _record: APIKeyRecord = Depends(require_scope("admin"))):
    """Creates a new API key. The raw key is returned only once."""
    raw_key, record = api_key_store.create_key(name=payload.name, scopes=payload.scopes)
    return KeyCreateResponse(
        key_id=record.key_id,
        name=record.name,
        raw_key=raw_key,
        prefix=record.prefix,
        scopes=record.scopes,
        created_at=record.created_at
    )


@router.get(
    "",
    response_model=List[KeyItem],
    status_code=status.HTTP_200_OK,
    summary="List all API keys (Admin required)"
)
async def list_api_keys(_record: APIKeyRecord = Depends(require_scope("admin"))):
    """Lists all stored API keys with masked prefixes."""
    keys = api_key_store.list_keys()
    return keys


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_200_OK,
    summary="Revoke an API key (Admin required)"
)
async def revoke_api_key(key_id: str, _record: APIKeyRecord = Depends(require_scope("admin"))):
    """Revokes an API key by ID."""
    success = api_key_store.revoke_key(key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key '{key_id}' not found."
        )
    return {"status": "success", "message": f"API key '{key_id}' revoked successfully."}
