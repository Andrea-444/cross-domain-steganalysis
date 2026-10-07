from __future__ import annotations

import io
import logging
import unittest
from unittest.mock import patch

from src.logging_config import (
    ERROR_COLOR,
    SUCCESS_COLOR,
    LOG_COLORS,
    PROJECT_LOGGER,
    RESET_COLOR,
    ColorFormatter,
    ProgressReporter,
    configure_logging,
    error_line,
    get_logger,
    success_line,
)


def record(level: int, message: str = "hello") -> logging.LogRecord:
    return logging.LogRecord(
        name="src.test", level=level, pathname=__file__, lineno=1,
        msg=message, args=(), exc_info=None,
    )


class ColorFormatterTests(unittest.TestCase):
    def test_wraps_message_in_the_configured_color(self) -> None:
        formatted = ColorFormatter("green", use_color=True).format(record(logging.INFO))
        self.assertTrue(formatted.startswith(LOG_COLORS["green"]))
        self.assertTrue(formatted.endswith(RESET_COLOR))

    def test_warning_and_above_use_the_error_color(self) -> None:
        for level in (logging.WARNING, logging.ERROR, logging.CRITICAL):
            formatted = ColorFormatter("green", use_color=True).format(record(level))
            self.assertTrue(formatted.startswith(ERROR_COLOR), level)

    def test_unknown_color_name_falls_back_to_cyan(self) -> None:
        formatted = ColorFormatter("nonexistent", use_color=True).format(
            record(logging.INFO)
        )
        self.assertTrue(formatted.startswith(LOG_COLORS["cyan"]))

    def test_no_escape_codes_when_color_is_disabled(self) -> None:
        """Captured output must stay readable, so colour is opt-in on a TTY."""
        formatted = ColorFormatter("green", use_color=False).format(record(logging.INFO))
        self.assertNotIn("\033[", formatted)
        self.assertIn("hello", formatted)


class ConfigureLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        logging.getLogger(PROJECT_LOGGER).handlers.clear()

    def test_sets_the_requested_level(self) -> None:
        configure_logging(log_level="DEBUG")
        self.assertEqual(logging.getLogger(PROJECT_LOGGER).level, logging.DEBUG)

    def test_unknown_level_falls_back_to_info(self) -> None:
        configure_logging(log_level="NOT_A_LEVEL")
        self.assertEqual(logging.getLogger(PROJECT_LOGGER).level, logging.INFO)

    def test_reads_the_environment_when_no_argument_is_given(self) -> None:
        with patch.dict("os.environ", {"LOG_LEVEL": "WARNING"}, clear=False):
            configure_logging()
        self.assertEqual(logging.getLogger(PROJECT_LOGGER).level, logging.WARNING)

    def test_repeated_calls_do_not_add_handlers(self) -> None:
        configure_logging(log_level="INFO")
        configure_logging(log_level="INFO")
        self.assertEqual(len(logging.getLogger(PROJECT_LOGGER).handlers), 1)

    def test_project_logger_does_not_propagate(self) -> None:
        configure_logging(log_level="INFO")
        self.assertFalse(logging.getLogger(PROJECT_LOGGER).propagate)

    def test_get_logger_returns_a_child_of_the_project_logger(self) -> None:
        self.assertTrue(get_logger("src.datasets.audit").name.startswith(PROJECT_LOGGER))


class SummaryLineTests(unittest.TestCase):
    def tearDown(self) -> None:
        logging.getLogger(PROJECT_LOGGER).handlers.clear()

    def test_lines_carry_color_when_the_stream_is_a_terminal(self) -> None:
        class Tty(io.StringIO):
            def isatty(self) -> bool:
                return True

        configure_logging(log_level="INFO", stream=Tty())
        self.assertIn(SUCCESS_COLOR, success_line("done"))
        self.assertIn(ERROR_COLOR, error_line("failed"))

    def test_lines_carry_no_escape_codes_when_output_is_captured(self) -> None:
        """A captured log must not contain terminal escape codes."""
        configure_logging(log_level="INFO", stream=io.StringIO())
        self.assertEqual(success_line("done"), "done")
        self.assertEqual(error_line("failed"), "failed")

    def test_message_text_survives_either_way(self) -> None:
        configure_logging(log_level="INFO", stream=io.StringIO())
        self.assertIn("done", success_line("done"))


class ProgressReporterTests(unittest.TestCase):
    def build(self, total: int, every: int):
        logger = logging.getLogger("src.test.progress")
        logger.handlers.clear()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return ProgressReporter(logger, total=total, label="embed", every=every), stream

    def build_auto(self, total: int):
        reporter, stream = self.build(total=total, every=1)
        return ProgressReporter(reporter._logger, total=total, label="embed"), stream

    def test_logs_at_the_interval_not_every_item(self) -> None:
        reporter, stream = self.build(total=1000, every=250)
        for _ in range(1000):
            reporter.advance()
        lines = [line for line in stream.getvalue().splitlines() if line]
        self.assertEqual(len(lines), 4)

    def test_reports_count_and_percentage(self) -> None:
        reporter, stream = self.build(total=100, every=50)
        for _ in range(50):
            reporter.advance()
        self.assertIn("50/100", stream.getvalue())
        self.assertIn("50%", stream.getvalue())

    def test_finish_reports_the_total_even_off_interval(self) -> None:
        reporter, stream = self.build(total=10, every=100)
        for _ in range(10):
            reporter.advance()
        reporter.finish()
        self.assertIn("10/10", stream.getvalue())

    def test_interval_adapts_to_a_small_total(self) -> None:
        """A fixed 500-item interval gives no output at all on a short run."""
        reporter, stream = self.build_auto(total=200)
        for _ in range(200):
            reporter.advance()
        lines = [line for line in stream.getvalue().splitlines() if line]
        self.assertGreaterEqual(len(lines), 2)

    def test_interval_adapts_to_a_large_total(self) -> None:
        reporter, stream = self.build_auto(total=12000)
        for _ in range(12000):
            reporter.advance()
        lines = [line for line in stream.getvalue().splitlines() if line]
        self.assertGreaterEqual(len(lines), 10)
        self.assertLessEqual(len(lines), 40)

    def test_finish_is_safe_on_an_empty_run(self) -> None:
        reporter, stream = self.build(total=0, every=100)
        reporter.finish()
        self.assertIn("embed", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
