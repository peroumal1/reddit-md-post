from unittest.mock import MagicMock

from rss_summary.similarity import is_duplicate


class TestIsDuplicate:
    def test_empty_list_returns_false(self):
        model = MagicMock()
        assert is_duplicate(model, "some text", []) is False
        model.encode.assert_not_called()

    def test_high_similarity_returns_true(self, make_model):
        assert is_duplicate(make_model(0.85), "text", ["similar text"]) is True

    def test_low_similarity_returns_false(self, make_model):
        assert is_duplicate(make_model(0.3), "text", ["different text"]) is False

    def test_exact_threshold_returns_false(self, make_model):
        assert is_duplicate(make_model(0.6), "text", ["other"]) is False

    def test_custom_threshold(self, make_model):
        assert is_duplicate(make_model(0.5), "text", ["other"], threshold=0.4) is True
