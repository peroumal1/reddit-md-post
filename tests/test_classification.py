import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from rss_summary.classification import (
    UNCLASSIFIED,
    classify_article,
    classify_article_scored,
    encode_for_classification,
    load_classifier_head,
    load_taxonomy,
)


class TestLoadTaxonomy:
    def test_returns_theme_list(self, tmp_path):
        toml = tmp_path / "taxonomy.toml"
        toml.write_text('themes = ["Politique", "Économie", "Sport"]')
        assert load_taxonomy(str(toml)) == ["Politique", "Économie", "Sport"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_taxonomy(str(tmp_path / "missing.toml"))


class TestLoadClassifierHead:
    def test_missing_file_raises_with_hint(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="classifier/train.py"):
            load_classifier_head(str(tmp_path / "missing.joblib"))

    def test_loads_head(self, tmp_path):
        import joblib
        path = tmp_path / "head.joblib"
        head = {"clf": "stub", "label_encoder": "stub"}
        joblib.dump(head, path)
        result = load_classifier_head(str(path))
        assert result["clf"] == "stub"


class TestEncodeForClassification:
    def test_output_shape_and_normalized(self):
        model_bge = MagicMock()
        model_bge.encode.return_value = np.array([1.0, 0.0])
        model_e5 = MagicMock()
        model_e5.encode.return_value = np.array([0.0, 1.0])
        result = encode_for_classification("test text", model_bge, model_e5)
        assert result.shape == (4,)
        assert pytest.approx(np.linalg.norm(result), abs=1e-6) == 1.0

    def test_zero_vector_returned_as_is(self):
        model_bge = MagicMock()
        model_bge.encode.return_value = np.zeros(2)
        model_e5 = MagicMock()
        model_e5.encode.return_value = np.zeros(2)
        result = encode_for_classification("test", model_bge, model_e5)
        assert np.all(result == 0)


class TestClassifyArticleScored:
    def _make_head(self, proba, classes, label_to_theme):
        clf = MagicMock()
        clf.predict_proba.return_value = [proba]
        le = MagicMock()
        le.classes_ = classes
        le.inverse_transform.side_effect = lambda idx: [classes[i] for i in idx]
        return {"clf": clf, "label_encoder": le, "label_to_theme": label_to_theme}

    def test_returns_top_theme_above_threshold(self):
        head = self._make_head(
            proba=[0.1, 0.8, 0.1],
            classes=["eco", "pol", "spo"],
            label_to_theme={"eco": "Économie", "pol": "Politique", "spo": "Sport"},
        )
        result = classify_article_scored(np.array([1.0, 0.0]), head, threshold=0.15)
        assert result["theme"] == "Politique"
        assert result["top_score"] == pytest.approx(0.8)

    def test_returns_unclassified_below_threshold(self):
        head = self._make_head(
            proba=[0.04, 0.05, 0.91],
            classes=["eco", "pol", "spo"],
            label_to_theme={"eco": "Économie", "pol": "Politique", "spo": "Sport"},
        )
        result = classify_article_scored(np.array([1.0, 0.0]), head, threshold=0.95)
        assert result["theme"] == UNCLASSIFIED

    def test_runner_up_populated(self):
        head = self._make_head(
            proba=[0.3, 0.6, 0.1],
            classes=["eco", "pol", "spo"],
            label_to_theme={"eco": "Économie", "pol": "Politique", "spo": "Sport"},
        )
        result = classify_article_scored(np.array([1.0, 0.0]), head, threshold=0.15)
        assert result["runner_up"] == "Économie"
        assert result["runner_up_score"] == pytest.approx(0.3)

    def test_normalizes_embedding(self):
        head = self._make_head(
            proba=[0.0, 1.0],
            classes=["a", "b"],
            label_to_theme={"a": "A", "b": "B"},
        )
        result = classify_article_scored(np.array([5.0, 0.0]), head, threshold=0.15)
        assert result["theme"] == "B"


class TestClassifyArticle:
    def test_returns_theme_string(self):
        with patch("rss_summary.classification.classify_article_scored") as mock_scored:
            mock_scored.return_value = {
                "theme": "Politique", "top_score": 0.8,
                "runner_up": None, "runner_up_score": None,
            }
            result = classify_article(np.array([1.0]), {}, threshold=0.15)
        assert result == "Politique"
