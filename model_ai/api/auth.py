import os
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str = Security(_header)) -> str:
    expected = os.getenv("API_KEY", "")
    if not expected or key != expected:
        raise HTTPException(status_code=403, detail="API түлхүүр буруу эсвэл байхгүй")
    return key
