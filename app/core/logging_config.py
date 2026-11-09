"""Central logging configuration for application modules."""

import logging
import sys


_APPLICATION_HANDLER_MARKER = "_intelliground_application_handler"


def configure_application_logging() -> None:
    """Route ``app.*`` INFO records to the container's standard error stream.

    Application loggers intentionally propagate to the root logger and do not
    own handlers.  The marker makes this setup idempotent when the FastAPI app
    is imported more than once by tests or a development server.  Uvicorn's
    separately configured ``uvicorn`` and ``uvicorn.access`` loggers are left
    unchanged.
    """

    application_logger = logging.getLogger("app")
    application_logger.setLevel(logging.INFO)
    application_logger.propagate = True

    root_logger = logging.getLogger()
    if not any(
        getattr(handler, _APPLICATION_HANDLER_MARKER, False)
        for handler in root_logger.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        setattr(handler, _APPLICATION_HANDLER_MARKER, True)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
        )
        root_logger.addHandler(handler)


__all__ = ["configure_application_logging"]
