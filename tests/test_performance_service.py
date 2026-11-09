import logging

from app.services import performance_service


def test_record_timing_logs_sanitized_percentiles(caplog):
    performance_service._samples.clear()

    with caplog.at_level(logging.INFO, logger="app.services.performance_service"):
        performance_service.record_timing("qa.test_stage", 12.5)
        performance_service.record_timing("qa.test_stage", 25.0)

    message = " ".join(record.getMessage() for record in caplog.records)
    assert "stage=qa.test_stage" in message
    assert "p50_ms=" in message
    assert "p95_ms=" in message
    assert "duration_ms=" in message
    assert "password" not in message.lower()
