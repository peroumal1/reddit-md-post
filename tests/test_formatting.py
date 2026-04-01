from rss_summary.formatting import UNCLASSIFIED, format_feed_entries, format_feed_entries_classified


class TestFormatFeedEntries:
    def test_without_images(self, sample_feed_entry):
        rows = format_feed_entries([sample_feed_entry], with_images=False)
        assert len(rows) == 1
        assert "Aperçu" not in rows[0]
        assert rows[0]["Titre"] == "[Test Title](https://example.com)"
        assert rows[0]["Résumé"] == "A summary"

    def test_with_images(self, sample_feed_entry):
        rows = format_feed_entries([sample_feed_entry], with_images=True)
        assert rows[0]["Aperçu"] == "![media](https://example.com/img.jpg)"

    def test_empty_list(self):
        assert format_feed_entries([]) == []

    def test_multiple_entries_preserve_order(self, sample_feed_entry):
        e1 = {**sample_feed_entry, "title": "First"}
        e2 = {**sample_feed_entry, "title": "Second"}
        rows = format_feed_entries([e1, e2])
        assert "First" in rows[0]["Titre"]
        assert "Second" in rows[1]["Titre"]


class TestFormatFeedEntriesClassified:
    def test_sections_grouped_by_theme(self, sample_feed_entry):
        entries = [
            {**sample_feed_entry, "title": "Pol 1", "theme": "Politique"},
            {**sample_feed_entry, "title": "Spo 1", "theme": "Sport"},
            {**sample_feed_entry, "title": "Pol 2", "theme": "Politique"},
        ]
        output = format_feed_entries_classified(entries, ["Politique", "Sport"])
        assert "## Politique" in output
        assert "## Sport" in output

    def test_taxonomy_order_preserved(self, sample_feed_entry):
        entries = [
            {**sample_feed_entry, "theme": "Sport"},
            {**sample_feed_entry, "theme": "Politique"},
        ]
        output = format_feed_entries_classified(entries, ["Politique", "Sport"])
        assert output.index("## Politique") < output.index("## Sport")

    def test_unclassified_at_end(self, sample_feed_entry):
        entries = [
            {**sample_feed_entry, "title": "Unknown", "theme": UNCLASSIFIED},
            {**sample_feed_entry, "title": "Known", "theme": "Politique"},
        ]
        output = format_feed_entries_classified(entries, ["Politique"])
        assert output.index("## Politique") < output.index(f"## {UNCLASSIFIED}")

    def test_missing_theme_falls_back_to_unclassified(self, sample_feed_entry):
        entries = [{**sample_feed_entry, "title": "No theme"}]  # no "theme" key
        output = format_feed_entries_classified(entries, ["Politique"])
        assert f"## {UNCLASSIFIED}" in output

    def test_empty_theme_section_omitted(self, sample_feed_entry):
        entries = [{**sample_feed_entry, "theme": "Sport"}]
        output = format_feed_entries_classified(entries, ["Politique", "Sport"])
        assert "## Politique" not in output
        assert "## Sport" in output
