from datetime import datetime
from unittest.mock import patch

from rss_summary.last_run import (
    DATE_FMT,
    get_last_run_date,
    restore_last_run_date,
    set_last_run_date,
)


class TestLastRunDate:
    def test_set_and_get_roundtrip(self, tmp_path):
        last_run_file = tmp_path / ".last-run"
        backup_file = tmp_path / ".last-run.bak"
        with patch("rss_summary.last_run.LAST_RUN_FILE", last_run_file), \
             patch("rss_summary.last_run.LAST_RUN_BACKUP", backup_file):
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


class TestBackupAndRestore:
    def test_set_creates_backup(self, tmp_path):
        last_run_file = tmp_path / ".last-run"
        backup_file = tmp_path / ".last-run.bak"
        last_run_file.write_text("01-Jan-2025 (10:00:00.000000)")
        with patch("rss_summary.last_run.LAST_RUN_FILE", last_run_file), \
             patch("rss_summary.last_run.LAST_RUN_BACKUP", backup_file):
            set_last_run_date()
        assert backup_file.exists()
        assert backup_file.read_text() == "01-Jan-2025 (10:00:00.000000)"

    def test_set_without_existing_file_no_backup(self, tmp_path):
        last_run_file = tmp_path / ".last-run"
        backup_file = tmp_path / ".last-run.bak"
        with patch("rss_summary.last_run.LAST_RUN_FILE", last_run_file), \
             patch("rss_summary.last_run.LAST_RUN_BACKUP", backup_file):
            set_last_run_date()
        assert not backup_file.exists()

    def test_restore_from_backup(self, tmp_path):
        last_run_file = tmp_path / ".last-run"
        backup_file = tmp_path / ".last-run.bak"
        original_content = "01-Jan-2025 (10:00:00.000000)"
        backup_file.write_text(original_content)
        last_run_file.write_text("25-Feb-2026 (19:00:00.000000)")
        with patch("rss_summary.last_run.LAST_RUN_FILE", last_run_file), \
             patch("rss_summary.last_run.LAST_RUN_BACKUP", backup_file):
            restore_last_run_date()
        assert last_run_file.read_text() == original_content

    def test_restore_without_backup(self, tmp_path, caplog):
        last_run_file = tmp_path / ".last-run"
        backup_file = tmp_path / ".last-run.bak"
        last_run_file.write_text("25-Feb-2026 (19:00:00.000000)")
        with patch("rss_summary.last_run.LAST_RUN_FILE", last_run_file), \
             patch("rss_summary.last_run.LAST_RUN_BACKUP", backup_file):
            restore_last_run_date()
        assert last_run_file.read_text() == "25-Feb-2026 (19:00:00.000000)"
        assert "No backup found" in caplog.text
