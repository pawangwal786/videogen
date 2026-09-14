import logging
from uuid import uuid4

import structlog

from app.logging import bind_correlation, clear_correlation, get_logger, setup_logging


def test_setup_logging_development():
    setup_logging(env="development", log_level="DEBUG")
    logger = get_logger("dev_logger")
    assert logger is not None
    # Verify standard logging root level was configured
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_production():
    setup_logging(env="production", log_level="WARNING")
    logger = get_logger("prod_logger")
    assert logger is not None
    assert logging.getLogger().level == logging.WARNING


def test_bind_and_clear_correlation():
    wf_id = uuid4()
    task_id = uuid4()

    # Bind UUIDs
    bind_correlation(workflow_id=wf_id, task_id=task_id)
    ctx = structlog.contextvars.get_contextvars()
    assert ctx["workflow_id"] == str(wf_id)
    assert ctx["task_id"] == str(task_id)

    # Clear
    clear_correlation()
    ctx_cleared = structlog.contextvars.get_contextvars()
    assert "workflow_id" not in ctx_cleared
    assert "task_id" not in ctx_cleared

    # Bind string format
    bind_correlation(workflow_id="custom-wf", task_id="custom-task")
    ctx_str = structlog.contextvars.get_contextvars()
    assert ctx_str["workflow_id"] == "custom-wf"
    assert ctx_str["task_id"] == "custom-task"

    # Calling with None does not overwrite or raise
    clear_correlation()
    bind_correlation(workflow_id=None, task_id=None)
    assert structlog.contextvars.get_contextvars() == {}
