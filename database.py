from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/thgee.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

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
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized: {DATABASE_URL}")