from datetime import datetime
from unittest.mock import patch

from rss_summary.last_run import get_last_run_date, set_last_run_date


class TestLastRunDate:
    def test_set_and_get_roundtrip(self, tmp_path):
        last_run_file = tmp_path / ".last-run"
        with patch("rss_summary.last_run.LAST_RUN_FILE", last_run_file):
            set_last_run_date()
            result = get_last_run_date()
        assert isinstance(result, datetime)
        assert result.date() == datetime.today().date()

    def test_get_returns_midnight_when_file_missing(self, tmp_path):
        last_run_file = tmp_path / ".last-run"
        with patch("rss_summary.last_run.LAST_RUN_FILE", last_run_file):
            result = get_last_run_date()
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        assert result == today

    def test_get_returns_midnight_when_file_corrupted(self, tmp_path):
        last_run_file = tmp_path / ".last-run"
        last_run_file.write_text("not-a-valid-date")
        with patch("rss_summary.last_run.LAST_RUN_FILE", last_run_file):
            result = get_last_run_date()
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        assert result == today
