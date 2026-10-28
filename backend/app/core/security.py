import hmac
import hashlib
import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from fastapi import Request, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from backend.app.config import settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyRecord:
    def __init__(
        self,
        key_id: str,
        name: str,
        hashed_key: str,
        prefix: str,
        scopes: List[str],
        created_at: str,
        revoked: bool = False
    ):
        self.key_id = key_id
        self.name = name
        self.hashed_key = hashed_key
        self.prefix = prefix
        self.scopes = scopes
        self.created_at = created_at
        self.revoked = revoked

    def to_dict(self) -> Dict:
        return {
            "key_id": self.key_id,
            "name": self.name,
            "hashed_key": self.hashed_key,
            "prefix": self.prefix,
            "scopes": self.scopes,
            "created_at": self.created_at,
            "revoked": self.revoked
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "APIKeyRecord":
        return cls(
            key_id=data["key_id"],
            name=data["name"],
            hashed_key=data["hashed_key"],
            prefix=data["prefix"],
            scopes=data.get("scopes", ["read", "write"]),
            created_at=data.get("created_at", datetime.datetime.utcnow().isoformat()),
            revoked=data.get("revoked", False)
        )


class APIKeyStore:
    """Manages persistent API keys stored hashed on disk."""
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.store_path = self.data_dir / "api_keys.json"
        self._keys: Dict[str, APIKeyRecord] = {}
        self._load()
        self._bootstrap_default_key()

    def _load(self):
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        record = APIKeyRecord.from_dict(item)
                        self._keys[record.key_id] = record
            except Exception:
                pass

    def _save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = [record.to_dict() for record in self._keys.values()]
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _bootstrap_default_key(self):
        # If bootstrap key is provided in settings and not already stored
        if settings.BOOTSTRAP_API_KEY:
            raw_bootstrap = settings.BOOTSTRAP_API_KEY.strip()
            hashed = hash_api_key(raw_bootstrap)
            # Check if this exact hash already exists
            exists = any(r.hashed_key == hashed for r in self._keys.values())
            if not exists:
                key_id = str(uuid.uuid4())
                prefix = raw_bootstrap[:6] + "..." if len(raw_bootstrap) >= 6 else "boot..."
                record = APIKeyRecord(
                    key_id=key_id,
                    name="Bootstrap Admin Key",
                    hashed_key=hashed,
                    prefix=prefix,
                    scopes=["read", "write", "admin"],
                    created_at=datetime.datetime.utcnow().isoformat(),
                    revoked=False
                )
                self._keys[key_id] = record
                self._save()

    def hash_key(self, raw_key: str) -> str:
        return hash_api_key(raw_key)

    def create_key(self, name: str, scopes: List[str]) -> tuple[str, APIKeyRecord]:
        raw_key = f"rwx_{uuid.uuid4().hex}{uuid.uuid4().hex}"
        hashed = self.hash_key(raw_key)
        key_id = str(uuid.uuid4())
        prefix = raw_key[:10] + "..."
        record = APIKeyRecord(
            key_id=key_id,
            name=name,
            hashed_key=hashed,
            prefix=prefix,
            scopes=scopes,
            created_at=datetime.datetime.utcnow().isoformat(),
            revoked=False
        )
        self._keys[key_id] = record
        self._save()
        return raw_key, record

    def list_keys(self) -> List[Dict]:
        return [
            {
                "key_id": r.key_id,
                "name": r.name,
                "prefix": r.prefix,
                "scopes": r.scopes,
                "created_at": r.created_at,
                "revoked": r.revoked
            }
            for r in self._keys.values()
        ]

    def revoke_key(self, key_id: str) -> bool:
        if key_id in self._keys:
            self._keys[key_id].revoked = True
            self._save()
            return True
        return False

    def verify_raw_key(self, raw_key: str) -> Optional[APIKeyRecord]:
        hashed = self.hash_key(raw_key)
        for record in self._keys.values():
            if not record.revoked and hmac.compare_digest(record.hashed_key, hashed):
                return record
        return None


def hash_api_key(raw_key: str) -> str:
    pepper = settings.API_KEY_PEPPER.encode("utf-8")
    key_bytes = raw_key.encode("utf-8")
    return hashlib.sha256(pepper + key_bytes).hexdigest()


api_key_store = APIKeyStore()


# FastAPI Dependency for Authentication & Scopes
async def verify_api_key_dependency(request: Request, api_key: Optional[str] = Security(API_KEY_HEADER)) -> APIKeyRecord:
    if not settings.AUTH_ENABLED:
        # Return a dummy admin record when auth is disabled
        return APIKeyRecord("dev-bypass", "Development Bypass", "", "dev...", ["read", "write", "admin"], "", False)

    # Bypass paths
    path = request.url.path
    bypass_prefixes = ["/health", "/docs", "/redoc", "/openapi.json", "/static", "/ui"]
    if path == "/" or any(path.startswith(p) for p in bypass_prefixes):
        return APIKeyRecord("public", "Public Endpoint", "", "pub...", ["read"], "", False)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key in 'X-API-Key' header."
        )

    record = api_key_store.verify_raw_key(api_key)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key."
        )

    # Store authenticated key in request state for downstream rate limiting / logging
    request.state.api_key_record = record
    return record


def require_scope(required_scope: str):
    async def scope_dependency(record: APIKeyRecord = Security(verify_api_key_dependency)) -> APIKeyRecord:
        if not settings.AUTH_ENABLED:
            return record
        if record.key_id in ("dev-bypass", "public"):
            return record
        if required_scope not in record.scopes and "admin" not in record.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient scope. Required: '{required_scope}'."
            )
        return record
    return scope_dependency
