from rss_summary.formatting import format_feed_entries


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
