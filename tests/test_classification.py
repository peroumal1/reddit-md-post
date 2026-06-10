import numpy as np
import pytest
from unittest.mock import MagicMock, call, patch

from mistralai.client.errors.sdkerror import SDKError

from rss_summary.classification import (
    THEME_INTERNATIONAL,
    THEME_OUTREMER,
    UNCLASSIFIED,
    classify_article,
    classify_article_scored,
    encode_for_classification,
    geo_theme,
    load_classifier_head,
    load_taxonomy,
    mistral_chat_with_retry,
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


class TestMistralChatWithRetry:
    def _make_client(self, side_effects):
        client = MagicMock()
        client.chat.complete.side_effect = side_effects
        return client

    def _sdk_error(self, status_code):
        raw = MagicMock()
        raw.status_code = status_code
        raw.text = ""
        raw.headers = {}
        return SDKError("", raw)

    def test_success_on_first_try(self):
        response = MagicMock()
        client = self._make_client([response])
        with patch("rss_summary.classification.time.sleep"):
            result = mistral_chat_with_retry(client, "model", [{"role": "user", "content": "hi"}])
        assert result is response

    def test_retries_on_429_then_succeeds(self):
        response = MagicMock()
        client = self._make_client([self._sdk_error(429), self._sdk_error(429), response])
        with patch("rss_summary.classification.time.sleep") as mock_sleep:
            result = mistral_chat_with_retry(client, "model", [{"role": "user", "content": "hi"}], base_delay=1)
        assert result is response
        assert client.chat.complete.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    def test_raises_after_max_retries(self):
        client = self._make_client([self._sdk_error(429)] * 5)
        with patch("rss_summary.classification.time.sleep"):
            with pytest.raises(SDKError):
                mistral_chat_with_retry(client, "model", [], retries=5, base_delay=1)
        assert client.chat.complete.call_count == 5

    def test_non_429_raises_immediately(self):
        client = self._make_client([self._sdk_error(500)])
        with patch("rss_summary.classification.time.sleep") as mock_sleep:
            with pytest.raises(SDKError):
                mistral_chat_with_retry(client, "model", [], retries=5, base_delay=1)
        assert client.chat.complete.call_count == 1
        mock_sleep.assert_not_called()


class TestGeoTheme:
    @pytest.mark.parametrize("title,expected", [
        # Sovereign Caribbean states → International
        ("Haïti. Artibonite : manifestation populaire contre l'insécurité", THEME_INTERNATIONAL),
        ("Cuba : pas de kérosène dans les aéroports jusqu'à avril", THEME_INTERNATIONAL),
        ("À la Dominique, d'importantes pluies ont provoqué des inondations", THEME_INTERNATIONAL),
        ("La Jamaïque plongée dans le noir après une panne électrique", THEME_INTERNATIONAL),
        ("République dominicaine. 2,4 milliards de dollars investis", THEME_INTERNATIONAL),
        ("En Haïti, la situation sécuritaire se dégrade", THEME_INTERNATIONAL),
        # French / non-sovereign territories → Outre-mer & Caraïbes
        ("Martinique. Une grève paralyse le port de Fort-de-France", THEME_OUTREMER),
        ("Guyane. Pêche illégale : déroutement d'un navire", THEME_OUTREMER),
        ("En Martinique, le carnaval bat son plein", THEME_OUTREMER),
        ("La Réunion : une éruption du Piton de la Fournaise", THEME_OUTREMER),
        # No explicit prefix → None (classifier decides)
        ("Guadeloupe. Quatre parcelles de cannes en feu", None),
        ("Un navire dérouté vers Cuba après une avarie", None),
        ("Le chef de l'ONU en visite en Haïti la semaine prochaine", None),
        # Person name and common noun must not match
        ("Dominique Théophile interpelle le Premier ministre", None),
        ("La réunion, prévue à 18h, a été annulée", None),
        ("La réunion publique du conseil municipal reportée", None),
    ])
    def test_routing(self, title, expected):
        assert geo_theme(title) == expected


class TestClassifyArticle:
    def test_returns_theme_string(self):
        with patch("rss_summary.classification.classify_article_scored") as mock_scored:
            mock_scored.return_value = {
                "theme": "Politique", "top_score": 0.8,
                "runner_up": None, "runner_up_score": None,
            }
            result = classify_article(np.array([1.0]), {}, threshold=0.15)
        assert result == "Politique"
