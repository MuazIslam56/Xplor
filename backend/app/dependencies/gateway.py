import os
from fastapi import Header, HTTPException

# Set GATEWAY_ENABLED=true in .env when IntelliRate is in front of this server.
# Leave it false (the default) for local development / direct access.
_GATEWAY_ENABLED = os.getenv("GATEWAY_ENABLED", "false").lower() == "true"


def get_gateway_user(
    x_intellirate_verified: str | None = Header(default=None, alias="X-IntelliRate-Verified"),
    x_intellirate_user_id: str | None = Header(default=None, alias="X-IntelliRate-User-ID"),
    x_forwarded_for: str | None = Header(default=None, alias="X-Forwarded-For"),
) -> dict:
    """
    Validates that the request arrived through the IntelliRate gateway.

    When GATEWAY_ENABLED=true:
      - X-IntelliRate-Verified must be "true"  → 403 if not
      - X-IntelliRate-User-ID must be present  → 400 if missing

    When GATEWAY_ENABLED=false (default for local dev):
      - Validation is skipped; requests reach routes directly.
    """
    if not _GATEWAY_ENABLED:
        return {"user_id": x_intellirate_user_id, "forwarded_for": x_forwarded_for}

    if x_intellirate_verified != "true":
        raise HTTPException(
            status_code=403,
            detail="Direct access not allowed. Use IntelliRate gateway.",
        )
    if not x_intellirate_user_id:
        raise HTTPException(
            status_code=400,
            detail="Missing X-IntelliRate-User-ID header",
        )
    return {"user_id": x_intellirate_user_id, "forwarded_for": x_forwarded_for}
