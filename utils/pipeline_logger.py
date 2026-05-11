"""
Pipeline Structured Logger
============================
Replaces bare print() calls with structured JSON logging.
Provides stage-aware logging with patient context.

Usage:
    from utils.pipeline_logger import get_logger
    logger = get_logger("segmentation")
    logger.log_stage_start("patient_001")
    logger.info("Processing complete", tumor_voxels=12345)
    logger.log_stage_error("patient_001", error)
"""
import os
import sys
import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Format log records as structured JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "stage": getattr(record, "stage", "pipeline"),
            "message": record.getMessage(),
        }
        # Add extra context fields
        context = getattr(record, "context", None)
        if context:
            entry["context"] = context
        patient_id = getattr(record, "patient_id", None)
        if patient_id:
            entry["patient_id"] = patient_id
        if record.exc_info and record.exc_info[1]:
            entry["error"] = str(record.exc_info[1])
            entry["traceback"] = traceback.format_exception(*record.exc_info)
        return json.dumps(entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console format that matches existing print() style."""

    LEVEL_PREFIXES = {
        "DEBUG": "  [DEBUG]",
        "INFO": " ",
        "WARNING": "  [WARNING]",
        "ERROR": "  [ERROR]",
        "CRITICAL": "  [CRITICAL]",
    }

    def format(self, record: logging.LogRecord) -> str:
        prefix = self.LEVEL_PREFIXES.get(record.levelname, "")
        stage = getattr(record, "stage", "")
        patient_id = getattr(record, "patient_id", "")

        if stage and patient_id and record.levelname == "INFO":
            header = f"[{stage}] {patient_id}"
        elif stage:
            header = f"[{stage}]"
        else:
            header = ""

        msg = record.getMessage()

        # For stage start messages, format like existing print output
        if "Starting" in msg and header:
            return f"{header} {msg}"

        if header and prefix.strip():
            return f"{prefix} {msg}"
        elif prefix.strip():
            return f"{prefix} {msg}"
        else:
            return f"  {msg}"


class PipelineLogger:
    """
    Stage-aware structured logger for pipeline nodes.

    Outputs to both console (human-readable) and optional file (JSON structured).
    """

    def __init__(self, stage: str, log_level: str = "INFO",
                 log_file: Optional[str] = None):
        self.stage = stage
        self._logger = logging.getLogger(f"pipeline.{stage}")
        self._logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self._logger.propagate = False

        # Avoid duplicate handlers on reload
        if not self._logger.handlers:
            # Console handler (human-readable)
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(ConsoleFormatter())
            self._logger.addHandler(console)

            # File handler (structured JSON) — optional
            if log_file:
                try:
                    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
                    fh = logging.FileHandler(log_file, mode="a")
                    fh.setFormatter(StructuredFormatter())
                    self._logger.addHandler(fh)
                except Exception:
                    pass  # Don't fail pipeline over logging setup

    def _log(self, level: int, msg: str, patient_id: str = None,
             context: Dict[str, Any] = None, exc_info=None, **kwargs):
        extra = {
            "stage": self.stage,
            "patient_id": patient_id or "",
            "context": {**(context or {}), **kwargs} if (context or kwargs) else None,
        }
        self._logger.log(level, msg, extra=extra, exc_info=exc_info)

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)

    def log_stage_start(self, patient_id: str):
        """Log the beginning of a pipeline stage for a patient."""
        self._log(logging.INFO, f"Processing patient: {patient_id}",
                  patient_id=patient_id)

    def log_stage_complete(self, patient_id: str, **metrics):
        """Log successful completion of a pipeline stage."""
        self._log(logging.INFO, f"Completed patient: {patient_id}",
                  patient_id=patient_id, **metrics)

    def log_stage_error(self, patient_id: str, error: Exception,
                        context: Dict[str, Any] = None):
        """Log a pipeline stage error with full context."""
        msg = f"Error for {patient_id}: {error}"
        self._log(logging.ERROR, msg, patient_id=patient_id,
                  context=context, exc_info=(type(error), error, error.__traceback__))


# ─── Logger Factory ──────────────────────────────────────────────────

_loggers: Dict[str, PipelineLogger] = {}


def get_logger(stage: str, log_level: str = None,
               log_file: str = None) -> PipelineLogger:
    """
    Get or create a PipelineLogger for the given stage.

    Args:
        stage: Pipeline stage name (e.g. "preprocessing", "segmentation")
        log_level: Override log level (default: from config)
        log_file: Override log file path (default: from config)
    """
    if stage not in _loggers:
        # Try to load from config, but don't fail if config isn't available
        effective_level = log_level or "INFO"
        effective_file = log_file
        try:
            from config.config_loader import get_config
            cfg = get_config()
            effective_level = log_level or cfg.error_handling.log_level
            effective_file = log_file or cfg.error_handling.structured_log_file
        except Exception:
            pass

        _loggers[stage] = PipelineLogger(
            stage=stage,
            log_level=effective_level,
            log_file=effective_file,
        )
    return _loggers[stage]
