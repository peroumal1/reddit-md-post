from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from rss_summary.aggregate import main


def _mock_entry(title="Article", published=(2025, 1, 2, 10, 0, 0, 0, 0, 0)):
    entry = MagicMock()
    entry.title = title
    entry.published_parsed = published
    entry.link = "https://example.com/article"
    entry.get.return_value = published
    entry.summary_detail.value = "<p>Summary</p>"
    return entry


@pytest.fixture
def rss_file(tmp_path):
    f = tmp_path / "rss_list.txt"
    f.write_text("https://example.com/feed\n")
    return str(f)


@pytest.fixture
def output_file(tmp_path):
    return str(tmp_path / "feed.md")


def _run(rss_file, output_file, entries=None, extra_args=None):
    runner = CliRunner()
    fake_model = MagicMock()
    fake_feed = MagicMock()
    fake_feed.entries = entries if entries is not None else [_mock_entry()]

    with patch("rss_summary.aggregate.SentenceTransformer", return_value=fake_model), \
         patch("rss_summary.aggregate.feedparser.parse", return_value=fake_feed), \
         patch("rss_summary.aggregate.get_last_run_date", return_value=datetime(2025, 1, 1)), \
         patch("rss_summary.aggregate.set_last_run_date") as mock_set, \
         patch("rss_summary.aggregate.encode_text", return_value=np.array([0.1, 0.2])), \
         patch("rss_summary.aggregate.is_duplicate", return_value=False), \
         patch("rss_summary.aggregate.title_is_duplicate", return_value=False):
        args = [rss_file, output_file] + (extra_args or [])
        result = runner.invoke(main, args)
        return result, mock_set


class TestAggregateCLI:
    def test_writes_feed_when_entries_exist(self, rss_file, output_file):
        result, _ = _run(rss_file, output_file)
        assert result.exit_code == 0
        assert Path(output_file).exists()

    def test_no_file_written_when_no_entries(self, rss_file, output_file):
        result, _ = _run(rss_file, output_file, entries=[])
        assert result.exit_code == 0
        assert not Path(output_file).exists()

    def test_normal_run_updates_last_run(self, rss_file, output_file):
        _, mock_set = _run(rss_file, output_file)
        mock_set.assert_called_once()

    def test_dry_run_does_not_update_last_run(self, rss_file, output_file):
        _, mock_set = _run(rss_file, output_file, extra_args=["--dry-run"])
        mock_set.assert_not_called()

    def test_until_excludes_entries_after_bound(self, rss_file, output_file):
        # Entry published 2025-01-02, until bound is 2025-01-01 — should be excluded
        result, _ = _run(rss_file, output_file, extra_args=["--until", "2025-01-01 00:00:00"])
        assert result.exit_code == 0
        assert not Path(output_file).exists()

    def test_until_includes_entries_before_bound(self, rss_file, output_file):
        result, _ = _run(rss_file, output_file, extra_args=["--until", "2025-01-03 00:00:00"])
        assert result.exit_code == 0
        assert Path(output_file).exists()

    def test_invalid_until_exits_with_error(self, rss_file, output_file):
        result, _ = _run(rss_file, output_file, extra_args=["--until", "not-a-date"])
        assert result.exit_code != 0

    def test_missing_rss_file_exits_with_error(self, tmp_path, output_file):
        result, _ = _run(str(tmp_path / "missing.txt"), output_file)
        assert result.exit_code != 0

    def test_duplicate_entries_deduplicated(self, rss_file, output_file):
        runner = CliRunner()
        fake_model = MagicMock()
        fake_feed = MagicMock()
        fake_feed.entries = [_mock_entry("A"), _mock_entry("B")]

        with patch("rss_summary.aggregate.SentenceTransformer", return_value=fake_model), \
             patch("rss_summary.aggregate.feedparser.parse", return_value=fake_feed), \
             patch("rss_summary.aggregate.get_last_run_date", return_value=datetime(2025, 1, 1)), \
             patch("rss_summary.aggregate.set_last_run_date"), \
             patch("rss_summary.aggregate.encode_text", return_value=np.array([0.1, 0.2])), \
             patch("rss_summary.aggregate.is_duplicate", side_effect=[False, True]), \
             patch("rss_summary.aggregate.title_is_duplicate", return_value=False):
            result = runner.invoke(main, [rss_file, output_file])

        assert result.exit_code == 0
        content = Path(output_file).read_text()
        assert "[A]" in content
        assert "[B]" not in content

    def test_restore_flag_calls_restore(self):
        runner = CliRunner()
        with patch("rss_summary.aggregate.restore_last_run_date") as mock_restore:
            result = runner.invoke(main, ["data/rss_list.txt", "data/feed.md", "--restore"])
        mock_restore.assert_called_once()
        assert result.exit_code == 0
