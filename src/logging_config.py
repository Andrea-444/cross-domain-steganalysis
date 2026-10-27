"""Shared logging configuration.

Adapted from the project-standard configuration with two changes that this
repository needs.

Colour is written only when the stream is a terminal. Runs here are started
from an agent session and through `subprocess`, and both capture the output.
Escape codes in a captured log make the result hard to read.

`ProgressReporter` reports progress during a long loop. Embedding, preparation
and extraction each process many thousands of images and previously printed
nothing until the end, so a slow run looked the same as a stopped one.
"""

from __future__ import annotations

import logging
import os
import sys
import time

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"

LOG_COLORS = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}
RESET_COLOR = "\033[0m"
SUCCESS_COLOR = LOG_COLORS["green"]
ERROR_COLOR = LOG_COLORS["red"]

PROJECT_LOGGER = "src"
DEFAULT_LEVEL = "INFO"
DEFAULT_COLOR = "cyan"

# Third-party loggers that report at INFO or DEBUG and would hide our own lines.
NOISY_LOGGERS = [
    "PIL",
    "matplotlib",
    "urllib3",
    "filelock",
    "fsspec",
    "numba",
    "huggingface_hub",
    "datasets",
]


class ColorFormatter(logging.Formatter):
    """Format log records with a configured terminal color."""

    def __init__(self, color_name: str, use_color: bool = True) -> None:
        super().__init__(LOG_FORMAT)
        self._color = LOG_COLORS.get(color_name.lower(), LOG_COLORS["cyan"])
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Return the formatted log line, with color only when enabled."""
        message = super().format(record)
        if not self._use_color:
            return message
        if record.levelno >= logging.WARNING:
            return f"{ERROR_COLOR}{message}{RESET_COLOR}"
        return f"{self._color}{message}{RESET_COLOR}"


# Set by configure_logging. The summary helpers embed color in the message
# itself, so they cannot use the formatter's check and need their own.
_COLOR_ENABLED: bool | None = None


def color_enabled() -> bool:
    """Report whether terminal color is in use."""
    if _COLOR_ENABLED is not None:
        return _COLOR_ENABLED
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def configure_logging(
    log_level: str | None = None,
    log_color: str | None = None,
    stream=None,
) -> None:
    """Configure the project logger from arguments or environment variables.

    `LOG_LEVEL` and `LOG_COLOR` set the defaults when no argument is given.
    """
    log_level = log_level or os.environ.get("LOG_LEVEL", DEFAULT_LEVEL)
    log_color = log_color or os.environ.get("LOG_COLOR", DEFAULT_COLOR)
    stream = stream if stream is not None else sys.stderr

    level = logging.getLevelName(log_level.upper())
    if not isinstance(level, int):
        level = logging.INFO

    use_color = bool(getattr(stream, "isatty", lambda: False)())
    if os.environ.get("NO_COLOR"):
        use_color = False
    global _COLOR_ENABLED
    _COLOR_ENABLED = use_color

    handler = logging.StreamHandler(stream)
    handler.setFormatter(ColorFormatter(log_color, use_color=use_color))

    project_logger = logging.getLogger(PROJECT_LOGGER)
    project_logger.handlers.clear()
    project_logger.addHandler(handler)
    project_logger.setLevel(level)
    project_logger.propagate = False

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)

    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named project logger.

    A module started with `python -m` has `__name__` set to `"__main__"`, and
    that name sits outside the `src` namespace where `configure_logging`
    attaches its handler. Such a logger reaches no project handler, so its
    INFO lines disappear. Warnings still appear, through the last-resort
    handler of the standard library, which makes the loss silent and easy to
    miss: a command prints its warnings and none of its results.

    Python records the real dotted name of a module started this way in
    `__spec__`, so resolve `"__main__"` back to it.
    """
    if name == "__main__":
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        resolved = getattr(spec, "name", None)
        if resolved:
            name = resolved
    return logging.getLogger(name)


def success_line(message: str) -> str:
    """Return a green terminal summary line, or plain text when captured."""
    if not color_enabled():
        return message
    return f"{SUCCESS_COLOR}{message}{RESET_COLOR}"


def error_line(message: str) -> str:
    """Return a red terminal summary line, or plain text when captured."""
    if not color_enabled():
        return message
    return f"{ERROR_COLOR}{message}{RESET_COLOR}"


class ProgressReporter:
    """Report progress at a fixed interval during a long loop.

    A progress bar is not used. Runs are started through captured output as
    often as through a terminal, and a bar writes one line for each redraw,
    which fills a captured log. A line at a fixed interval reads the same in
    both places.
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        label: str,
        every: int | None = None,
    ) -> None:
        self._logger = logger
        self._total = max(0, int(total))
        self._label = label
        # Aim for about twenty reports. A fixed interval gives no output on a
        # short run and too much output on a very long one.
        if every is None:
            every = min(1000, max(50, self._total // 20)) if self._total else 500
        self._every = max(1, int(every))
        self._count = 0
        self._started = time.monotonic()
        self._last_reported = 0

    def advance(self, step: int = 1) -> None:
        """Count completed items and report when the interval is reached."""
        self._count += step
        if self._count - self._last_reported >= self._every:
            self._report()

    def finish(self) -> None:
        """Report the final count, even when it is not on the interval."""
        if self._count != self._last_reported or self._count == 0:
            self._report()

    def _report(self) -> None:
        self._last_reported = self._count
        elapsed = time.monotonic() - self._started
        parts = [f"{self._label}: {self._count}/{self._total}"]
        if self._total:
            parts.append(f"{100 * self._count // self._total}%")
        if elapsed > 0 and self._count:
            rate = self._count / elapsed
            parts.append(f"{rate:.1f}/s")
            remaining = self._total - self._count
            if remaining > 0 and rate > 0:
                parts.append(f"{remaining / rate / 60:.1f} min left")
        self._logger.info(" | ".join(parts))
