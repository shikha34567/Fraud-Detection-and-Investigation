"""
Database connection and session lifecycle management using SQLAlchemy.
Supports SQLite out of the box and seamlessly switches to PostgreSQL via DATABASE_URL.
"""

from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from config.settings import DB_CONFIG, PATHS
from src.utils.logging_config import logger

Base = declarative_base()

# Configure SQLite pragmas for concurrency/write performance if sqlite
is_sqlite = DB_CONFIG.DB_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

# Ensure data directory exists
PATHS.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DB_CONFIG.DB_URL,
    connect_args=connect_args,
    echo=False,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initializes tables in database."""
    # Import models here to ensure they register on Base.metadata
    from src.database import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database schema initialized successfully at {DB_CONFIG.DB_URL}")


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for safe SQLAlchemy session execution with commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database transaction error, rolling back: {e}", exc_info=True)
        raise
    finally:
        session.close()
