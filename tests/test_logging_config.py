import logging

from app.core.logging_config import configure_application_logging


def test_application_logging_is_centralized_and_idempotent(caplog):
    configure_application_logging()
    configure_application_logging()

    application_logger = logging.getLogger("app.logging-config-test")
    caplog.set_level(logging.INFO, logger=application_logger.name)
    application_logger.info("privacy-safe logging configuration test")

    root_handlers = logging.getLogger().handlers
    application_handlers = [
        handler
        for handler in root_handlers
        if getattr(handler, "_intelliground_application_handler", False)
    ]

    assert len(application_handlers) == 1
    assert application_logger.handlers == []
    assert application_logger.propagate is True
    assert any(
        record.message == "privacy-safe logging configuration test"
        for record in caplog.records
    )
