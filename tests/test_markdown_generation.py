from datetime import datetime
from pathlib import Path

from py_markdown_table.markdown_table import markdown_table

from rss_summary.formatting import format_feed_entries


class TestMarkdownGeneration:
    def _build_rows(self, entries, with_images=False):
        """Mirror what main() does: sort then format."""
        sorted_entries = sorted(
            entries, key=lambda item: item["published_date"], reverse=True
        )
        return format_feed_entries(sorted_entries, with_images)

    def test_generates_valid_markdown_table(self, sample_feed_entry):
        rows = self._build_rows([sample_feed_entry])
        md = (
            markdown_table(rows)
            .set_params(row_sep="markdown", quote=False)
            .get_markdown()
        )
        assert "|" in md
        assert "Titre" in md
        assert "Résumé" in md
        assert "Date de publication" in md
        assert "[Test Title](https://example.com)" in md
        lines = md.strip().splitlines()
        assert len(lines) == 3  # header, separator, 1 data row

    def test_markdown_with_images_has_apercu_column(self, sample_feed_entry):
        rows = self._build_rows([sample_feed_entry], with_images=True)
        md = (
            markdown_table(rows)
            .set_params(row_sep="markdown", quote=False)
            .get_markdown()
        )
        assert "Aperçu" in md
        assert "![media](https://example.com/img.jpg)" in md

    def test_markdown_entries_sorted_most_recent_first(self, sample_feed_entry):
        older = {**sample_feed_entry, "title": "Older", "published_date": datetime(2025, 1, 1)}
        newer = {**sample_feed_entry, "title": "Newer", "published_date": datetime(2025, 1, 2)}
        rows = self._build_rows([older, newer])
        md = (
            markdown_table(rows)
            .set_params(row_sep="markdown", quote=False)
            .get_markdown()
        )
        newer_pos = md.index("Newer")
        older_pos = md.index("Older")
        assert newer_pos < older_pos, "Newer entry should appear before older entry"

    def test_markdown_written_to_file(self, sample_feed_entry, tmp_path):
        rows = self._build_rows([sample_feed_entry])
        md = (
            markdown_table(rows)
            .set_params(row_sep="markdown", quote=False)
            .get_markdown()
        )
        output_file = tmp_path / "feed.md"
        Path(output_file).write_text(md)
        content = output_file.read_text()
        assert content == md
        assert "Test Title" in content

    def test_multiple_entries_produce_correct_row_count(self, sample_feed_entry):
        entries = [
            {**sample_feed_entry, "title": f"Title {i}", "published_date": datetime(2025, 1, i + 1)}
            for i in range(5)
        ]
        rows = self._build_rows(entries)
        md = (
            markdown_table(rows)
            .set_params(row_sep="markdown", quote=False)
            .get_markdown()
        )
        lines = md.strip().splitlines()
        assert len(lines) == 7  # header + separator + 5 data rows
