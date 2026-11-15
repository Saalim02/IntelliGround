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
    from app.models.auth_models import User, PasswordResetToken, VoicePreference
    from app.models.conversation_models import Conversation, ConversationTurn

    Base.metadata.create_all(bind=engine)

    # create_all() does not alter existing SQLite tables. Add new nullable/
    # defaulted columns separately so existing users and conversations survive.
    existing_columns = {
        column["name"]
        for column in inspect(engine).get_columns(User.__tablename__)
    }
    boolean_default = "FALSE" if engine.dialect.name == "postgresql" else "0"
    with engine.begin() as connection:
        for column_name in ("first_name", "last_name"):
            if column_name not in existing_columns:
                connection.execute(
                    text(
                        f"ALTER TABLE users ADD COLUMN {column_name} VARCHAR(80)"
                    )
                )
        if "onboarding_completed" not in existing_columns:
            connection.execute(text(
                f"ALTER TABLE users ADD COLUMN onboarding_completed BOOLEAN NOT NULL DEFAULT {boolean_default}"
            ))
        voice_preference_columns = {
            column["name"]
            for column in inspect(engine).get_columns("voice_preferences")
        }
        if "response_emotion" not in voice_preference_columns:
            connection.execute(text(
                "ALTER TABLE voice_preferences ADD COLUMN response_emotion VARCHAR(24) DEFAULT 'AUTO'"
            ))
        conversation_columns = {
            column["name"]
            for column in inspect(engine).get_columns("conversations")
        }
        conversation_additions = {
            "title": "VARCHAR(512)",
            "document_path": "VARCHAR(2048)",
            "document_sha256": "VARCHAR(64)",
            "document_input_language": "VARCHAR(32)",
            "document_evidence_language": "VARCHAR(32)",
            "document_total_pages": "INTEGER",
            "response_length": "VARCHAR(16) NOT NULL DEFAULT 'medium'",
            "summary_json": "TEXT",
            "topics_json": "TEXT",
            "questions_json": "TEXT",
        }
        for column_name, definition in conversation_additions.items():
            if column_name not in conversation_columns:
                connection.execute(text(f"ALTER TABLE conversations ADD COLUMN {column_name} {definition}"))
