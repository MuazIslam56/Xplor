import json
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.security import require_roles
from app.models.models import SecurityEvent, User

router = APIRouter()


def _event_out(e: SecurityEvent) -> dict:
    return {
        "id":         e.id,
        "event_type": e.event_type,
        "severity":   e.severity,
        "username":   e.username,
        "user_id":    e.user_id,
        "ip_address": e.ip_address,
        "detail":     json.loads(e.detail) if e.detail else {},
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }


@router.get("/events")
def list_events(
    limit: int = 100,
    severity: str | None = None,
    event_type: str | None = None,
    db: Session = Depends(get_db),
    current=Depends(require_roles("admin")),
):
    q = db.query(SecurityEvent).order_by(SecurityEvent.created_at.desc())
    if severity:
        q = q.filter(SecurityEvent.severity == severity.upper())
    if event_type:
        q = q.filter(SecurityEvent.event_type == event_type)
    return [_event_out(e) for e in q.limit(limit).all()]


@router.get("/stats")
def security_stats(
    db: Session = Depends(get_db),
    current=Depends(require_roles("admin")),
):
    total       = db.query(func.count(SecurityEvent.id)).scalar() or 0
    criticals   = db.query(func.count(SecurityEvent.id)).filter(SecurityEvent.severity == "CRITICAL").scalar() or 0
    warnings    = db.query(func.count(SecurityEvent.id)).filter(SecurityEvent.severity == "WARNING").scalar() or 0

    since_24h   = datetime.utcnow() - timedelta(hours=24)
    recent      = db.query(func.count(SecurityEvent.id)).filter(SecurityEvent.created_at >= since_24h).scalar() or 0

    by_type = (
        db.query(SecurityEvent.event_type, func.count(SecurityEvent.id))
        .group_by(SecurityEvent.event_type)
        .all()
    )

    return {
        "total_events":    total,
        "critical_count":  criticals,
        "warning_count":   warnings,
        "info_count":      total - criticals - warnings,
        "last_24h":        recent,
        "by_type":         {t: c for t, c in by_type},
    }


@router.get("/users-snapshot")
def users_snapshot(
    db: Session = Depends(get_db),
    current=Depends(require_roles("admin")),
):
    """Show users table with hashed passwords to demonstrate bcrypt storage."""
    users = db.query(User).order_by(User.created_at).all()
    return [
        {
            "id":          u.id,
            "username":    u.username,
            "email":       u.email,
            "role":        u.role,
            "hashed_pw":   u.hashed_pw,
            "pw_algorithm": "bcrypt (rounds=12)" if u.hashed_pw and u.hashed_pw.startswith("$2b$") else "unknown",
            "created_at":  u.created_at.isoformat() if u.created_at else None,
        }
        for u in users
    ]
