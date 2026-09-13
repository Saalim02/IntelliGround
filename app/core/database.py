import os
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import BASE_DIR


DEFAULT_DATABASE_URL = (
    f"sqlite:///{Path(BASE_DIR) / 'storage' / 'app.db'}"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    DEFAULT_DATABASE_URL,
)

connect_args = {}

if DATABASE_URL.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False,
    }

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.models.auth_models import User, PasswordResetToken

    Base.metadata.create_all(bind=engine)

    # create_all() does not alter an existing SQLite table. Add profile
    # columns separately so existing users and authentication data survive.
    existing_columns = {
        column["name"]
        for column in inspect(engine).get_columns(User.__tablename__)
    }
    with engine.begin() as connection:
        for column_name in ("first_name", "last_name"):
            if column_name not in existing_columns:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN {column_name} VARCHAR(80)"
                    )
                )
