"""
FastAPI email simulation server.

Provides a REST API to manage a simulated email inbox backed by SQLite.
Pre-loads sample emails on startup and resets them via /reset_database.

Run with:
    uvicorn email_server.email_service:app --port 5000 --reload
"""

import random
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import delete
from sqlalchemy.orm import Session

from .email_database import Base, SessionLocal, engine
from .email_models import Email
from .email_schema import EmailCreate, EmailOut

app = FastAPI(title="Email Simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

SAMPLE_EMAILS = [
    {"sender": "boss@email.com",    "recipient": "you@email.com", "subject": "Quarterly Report",  "body": "Please finalize the report ASAP."},
    {"sender": "alice@work.com",    "recipient": "you@email.com", "subject": "Lunch?",             "body": "Free for lunch today?"},
    {"sender": "bob@work.com",      "recipient": "you@email.com", "subject": "Code Review",        "body": "I left some comments on your PR."},
    {"sender": "charlie@work.com",  "recipient": "you@email.com", "subject": "Meeting",            "body": "Can we reschedule?"},
    {"sender": "eric@work.com",     "recipient": "you@email.com", "subject": "Happy Hour",         "body": "We're planning drinks this Friday!"},
    {"sender": "you@email.com",     "recipient": "boss@email.com","subject": "Days off",           "body": "Can I get some days off the coming week?"},
]


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _seed_emails(db: Session) -> None:
    """Delete all emails and insert fresh sample data."""
    db.execute(delete(Email))
    db.commit()
    now = datetime.utcnow()
    rows = [Email(**e, timestamp=now, read=False) for e in SAMPLE_EMAILS]
    random.shuffle(rows)
    db.add_all(rows)
    db.commit()


@app.on_event("startup")
def startup():
    db = SessionLocal()
    try:
        _seed_emails(db)
    finally:
        db.close()


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/reset_database")
def reset_database():
    db = SessionLocal()
    try:
        _seed_emails(db)
    finally:
        db.close()
    return {"message": "Database reset and emails reloaded"}


@app.post("/send", response_model=EmailOut)
def send_email(email: EmailCreate, db: Session = Depends(get_db)):
    new_email = Email(
        recipient=email.recipient,
        subject=email.subject,
        body=email.body,
        sender="you@email.com",
    )
    db.add(new_email)
    db.commit()
    db.refresh(new_email)
    return new_email


@app.get("/emails", response_model=List[EmailOut])
def list_emails(db: Session = Depends(get_db)):
    return db.query(Email).order_by(Email.timestamp.desc()).all()


@app.get("/emails/unread", response_model=List[EmailOut])
def list_unread_emails(db: Session = Depends(get_db)):
    return (
        db.query(Email)
        .filter(Email.read == False)  # noqa: E712
        .order_by(Email.timestamp.desc())
        .all()
    )


@app.get("/emails/search", response_model=List[EmailOut])
def search_emails(
    q: str = Query(..., description="Keyword to search in subject/body/sender"),
    db: Session = Depends(get_db),
):
    return (
        db.query(Email)
        .filter(
            Email.subject.ilike(f"%{q}%")
            | Email.body.ilike(f"%{q}%")
            | Email.sender.ilike(f"%{q}%")
        )
        .order_by(Email.timestamp.desc())
        .all()
    )


@app.get("/emails/filter", response_model=List[EmailOut])
def filter_emails(
    recipient:  str | None = Query(None),
    date_from:  str | None = Query(None, description="YYYY-MM-DD"),
    date_to:    str | None = Query(None, description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
):
    q = db.query(Email)
    if recipient:
        q = q.filter(Email.recipient == recipient)
    if date_from:
        try:
            q = q.filter(Email.timestamp >= datetime.strptime(date_from, "%Y-%m-%d"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_from; use YYYY-MM-DD")
    if date_to:
        try:
            q = q.filter(Email.timestamp <= datetime.strptime(date_to, "%Y-%m-%d"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_to; use YYYY-MM-DD")
    return q.order_by(Email.timestamp.desc()).all()


@app.get("/emails/{email_id}", response_model=EmailOut)
def get_email(email_id: int, db: Session = Depends(get_db)):
    email = db.query(Email).filter(Email.id == email_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    return email


@app.patch("/emails/{email_id}/read", response_model=EmailOut)
def mark_email_as_read(email_id: int, db: Session = Depends(get_db)):
    email = db.query(Email).filter(Email.id == email_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    email.read = True
    db.commit()
    db.refresh(email)
    return email


@app.patch("/emails/{email_id}/unread", response_model=EmailOut)
def mark_email_as_unread(email_id: int, db: Session = Depends(get_db)):
    email = db.query(Email).filter(Email.id == email_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    email.read = False
    db.commit()
    db.refresh(email)
    return email


@app.delete("/emails/{email_id}")
def delete_email(email_id: int, db: Session = Depends(get_db)):
    email = db.query(Email).filter(Email.id == email_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    db.delete(email)
    db.commit()
    return {"message": "Email deleted"}
