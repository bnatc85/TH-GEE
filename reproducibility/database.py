from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


SQLITE_DEFAULT_URL = "sqlite:///data/thgee.db"


def resolve_database_url() -> str:
    """Return the database URL to use.

    SQLite is used locally by default. A PostgreSQL database is only used when
    DATABASE_URL is explicitly set to a valid PostgreSQL URL. Anything else
    (unset, blank, or a non-PostgreSQL/non-SQLite value) falls back to the local
    SQLite database so the app never fails to start on a bad or empty env var.
    """
    raw = os.getenv("DATABASE_URL", "").strip()

    if not raw:
        return SQLITE_DEFAULT_URL

    scheme = urlsplit(raw).scheme.lower()

    # Normalize the legacy "postgres://" scheme (rejected by SQLAlchemy) to the
    # "postgresql://" form it expects.
    if scheme == "postgres" or scheme.startswith("postgres+"):
        raw = "postgresql" + raw[len("postgres"):]
        scheme = urlsplit(raw).scheme.lower()

    if scheme == "postgresql" or scheme.startswith("postgresql+"):
        return raw

    # An explicit local SQLite URL is honored; anything else is not a valid
    # PostgreSQL URL, so fall back to the local SQLite default.
    if scheme.startswith("sqlite"):
        return raw

    return SQLITE_DEFAULT_URL


DATABASE_URL = resolve_database_url()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

if DATABASE_URL.startswith("sqlite:///") and ":memory:" not in DATABASE_URL:
    # Ensure the parent directory for the SQLite file exists so create_engine /
    # connections don't fail with "unable to open database file".
    db_path = Path(DATABASE_URL[len("sqlite:///"):])
    db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class TrackedQuery(Base):
    __tablename__ = "tracked_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(String(300), nullable=False)
    query_type: Mapped[str] = mapped_column(
        String(30),
        default="keyword",
        nullable=False,
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    posts: Mapped[list["TrackedPost"]] = relationship(
        back_populates="tracked_query",
        cascade="all, delete-orphan",
    )


class TrackedPost(Base):
    __tablename__ = "tracked_posts"

    uri: Mapped[str] = mapped_column(String(500), primary_key=True)
    cid: Mapped[str | None] = mapped_column(String(200))
    author_did: Mapped[str] = mapped_column(String(200), nullable=False)
    text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    query_id: Mapped[int] = mapped_column(
        ForeignKey("tracked_queries.id"),
        nullable=False,
    )

    tracked_query: Mapped[TrackedQuery] = relationship(back_populates="posts")


class EngagementEvent(Base):
    __tablename__ = "engagement_events"
    __table_args__ = (
        UniqueConstraint("event_uri", "event_type", name="uq_event_uri_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query_id: Mapped[int] = mapped_column(
        ForeignKey("tracked_queries.id"),
        nullable=False,
    )
    event_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    target_uri: Mapped[str | None] = mapped_column(String(500))
    actor_did: Mapped[str] = mapped_column(String(200), nullable=False)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Engagement magnitude carried by the event (e.g. a video's like or comment
    # count at its publication time). Defaults to 1.0 so a bare event counts once.
    magnitude: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


def _ensure_magnitude_column() -> None:
    """Additive migration: add engagement_events.magnitude to older databases."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "engagement_events" not in inspector.get_table_names():
        return
    columns = {col["name"] for col in inspector.get_columns("engagement_events")}
    if "magnitude" not in columns:
        with engine.begin() as connection:
            connection.execute(text(
                "ALTER TABLE engagement_events "
                "ADD COLUMN magnitude FLOAT NOT NULL DEFAULT 1.0"
            ))


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_magnitude_column()


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized: {DATABASE_URL}")