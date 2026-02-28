from unittest.mock import MagicMock

from rss_summary.similarity import encode_text, is_duplicate, title_is_duplicate


class TestIsDuplicate:
    def test_empty_list_returns_false(self):
        model = MagicMock()
        mock_embedding = MagicMock()
        assert is_duplicate(model, mock_embedding, []) is False
        model.similarity.assert_not_called()

    def test_high_similarity_returns_true(self, make_model):
        mock_emb = MagicMock()
        assert is_duplicate(make_model(0.85), mock_emb, [mock_emb]) is True

    def test_low_similarity_returns_false(self, make_model):
        mock_emb = MagicMock()
        assert is_duplicate(make_model(0.3), mock_emb, [mock_emb]) is False

    def test_exact_threshold_returns_false(self, make_model):
        mock_emb = MagicMock()
        assert is_duplicate(make_model(0.82), mock_emb, [mock_emb], threshold=0.82) is False

    def test_custom_threshold(self, make_model):
        mock_emb = MagicMock()
        assert is_duplicate(make_model(0.5), mock_emb, [mock_emb], threshold=0.4) is True


class TestTitleIsDuplicate:
    def test_empty_list_returns_false(self):
        assert title_is_duplicate("Some Title", []) is False

    def test_near_identical_title_returns_true(self):
        # "en hausse" vs "à la hausse" — real-world near-duplicate
        t1 = "Carburants : les prix en hausse au 1er mars 2026 en Guadeloupe"
        t2 = "Carburants : les prix à la hausse au 1er mars 2026 en Guadeloupe"
        assert title_is_duplicate(t1, [t2]) is True

    def test_different_title_returns_false(self):
        assert title_is_duplicate("Séisme en Guadeloupe", ["Élections municipales à Pointe-à-Pitre"]) is False

    def test_case_insensitive(self):
        assert title_is_duplicate("GUADELOUPE. Hausse des hydrocarbures", ["guadeloupe. hausse des hydrocarbures"]) is True

    def test_custom_threshold(self):
        # At threshold=1.0 nothing matches except identical strings
        assert title_is_duplicate("Foo bar", ["Foo baz"], threshold=1.0) is False

    def test_exact_match_returns_true(self):
        title = "Guadeloupe. Hausse des hydrocarbures le 1er mars"
        assert title_is_duplicate(title, [title]) is True

    def test_checks_all_existing_titles(self):
        # Match is in the middle of the list, not just first or last
        candidate = "Carburants : les prix en hausse au 1er mars 2026 en Guadeloupe"
        existing = [
            "Séisme en Guadeloupe",
            "Carburants : les prix à la hausse au 1er mars 2026 en Guadeloupe",
            "Élections municipales 2026",
        ]
        assert title_is_duplicate(candidate, existing) is True


class TestEncodeText:
    def test_strips_html_before_encoding(self):
        model = MagicMock()
        model.encode.side_effect = lambda x: [x[0]]
        encode_text(model, "<p>Hello <b>world</b></p>")
        model.encode.assert_called_once_with(["Hello world"])

    def test_returns_single_embedding_vector(self):
        model = MagicMock()
        sentinel = object()
        model.encode.return_value = [sentinel]
        result = encode_text(model, "some text")
        assert result is sentinel
