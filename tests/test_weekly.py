from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from rss_summary.classification import UNCLASSIFIED
from rss_summary.weekly import (
    cluster_articles,
    extract_source,
    get_most_read_urls,
    parse_feed_file,
    pick_representative_article,
    render_suggestions,
    representative_embedding,
    score_cluster,
)


class TestGetMostReadUrls:
    def _mock_response(self, html="", ok=True):
        r = MagicMock()
        r.ok = ok
        r.text = html
        return r

    def test_returns_empty_set_when_both_requests_fail(self):
        with patch("rss_summary.weekly.requests.get", side_effect=requests.RequestException("network error")):
            result = get_most_read_urls()
        assert result == set()

    def test_returns_empty_set_when_both_responses_not_ok(self):
        with patch("rss_summary.weekly.requests.get", return_value=self._mock_response(ok=False)):
            result = get_most_read_urls()
        assert isinstance(result, set)
        assert len(result) == 0

    def test_extracts_rci_links(self):
        html = """
        <div id="block-views-block-block-articles-les-plus-lus-teaser-short-block-1">
            <a href="/guadeloupe/article-1">Article 1</a>
            <a href="/guadeloupe/article-2">Article 2</a>
        </div>
        """
        responses = [self._mock_response(html=html), self._mock_response(ok=False)]
        with patch("rss_summary.weekly.requests.get", side_effect=responses):
            result = get_most_read_urls()
        assert "/guadeloupe/article-1" in result
        assert "/guadeloupe/article-2" in result

    def test_extracts_france_antilles_links(self):
        html = """
        <div>
            <span>Articles les plus Lus</span>
            <a href="/article-a">A</a>
            <a href="/article-b">B</a>
            <a href="/article-c">C</a>
        </div>
        """
        responses = [self._mock_response(ok=False), self._mock_response(html=html)]
        with patch("rss_summary.weekly.requests.get", side_effect=responses):
            result = get_most_read_urls()
        assert "/article-a" in result

    def test_returns_set_on_empty_pages(self):
        with patch("rss_summary.weekly.requests.get", return_value=self._mock_response(html="<html></html>")):
            result = get_most_read_urls()
        assert isinstance(result, set)


def _make_article(title="T", url="https://rci.fm/article", source="RCI", date=None):
    return {
        "title": title,
        "url": url,
        "summary": "summary",
        "source": source,
        "date": date or datetime(2025, 1, 1),
    }


def _make_cluster(articles):
    return [{"article": a, "embedding": np.array([1.0, 0.0])} for a in articles]


class TestExtractSource:
    def test_known_domain(self):
        assert extract_source("https://rci.fm/article") == "RCI"
        assert extract_source("https://karibinfo.com/news") == "Karibinfo"

    def test_unknown_domain_returns_hostname(self):
        assert extract_source("https://unknown.example.com/article") == "unknown.example.com"

    def test_empty_url_returns_empty(self):
        assert extract_source("") == ""


class TestParseFeedFile:
    def test_adds_source_to_articles(self, tmp_path):
        md = (
            "| Titre | Résumé | Date de publication |\n"
            "|---|---|---|\n"
            "| [Article](https://rci.fm/a) | Summary | 2025-01-01T10:00:00 |\n"
        )
        f = tmp_path / "feed-2025-01-01.md"
        f.write_text(md)
        articles = parse_feed_file(f)
        assert len(articles) == 1
        assert articles[0]["source"] == "RCI"


class TestClusterArticles:
    def _make_model(self, sim_value):
        model = MagicMock()
        tensor = MagicMock()
        tensor.item.return_value = sim_value
        model.similarity.return_value = [[tensor]]
        return model

    def test_single_article_forms_one_cluster(self):
        model = self._make_model(0.9)
        articles = [_make_article()]
        with patch("rss_summary.weekly.encode_text", return_value=np.array([1.0, 0.0])):
            clusters = cluster_articles(articles, model)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    @pytest.mark.parametrize("sim_value,expected_clusters", [(0.9, 1), (0.3, 2)])
    def test_cluster_count_by_similarity(self, sim_value, expected_clusters):
        model = self._make_model(sim_value)
        articles = [_make_article("A"), _make_article("B")]
        with patch("rss_summary.weekly.encode_text", return_value=np.array([1.0, 0.0])):
            clusters = cluster_articles(articles, model)
        assert len(clusters) == expected_clusters


class TestScoreCluster:
    def test_base_score_days_times_sources(self):
        cluster = _make_cluster([
            _make_article(url="https://rci.fm/a", source="RCI", date=datetime(2025, 1, 1)),
            _make_article(url="https://karibinfo.com/a", source="Karibinfo", date=datetime(2025, 1, 2)),
        ])
        score = score_cluster(cluster, most_read_paths=set())
        assert score == 2 * 2  # 2 days × 2 sources

    def test_most_read_bonus_added(self):
        cluster = _make_cluster([
            _make_article(url="https://rci.fm/article", source="RCI", date=datetime(2025, 1, 1)),
        ])
        score = score_cluster(cluster, most_read_paths={"/article"})
        assert score == 1 * 1 + 1  # 1 day × 1 source + 1 bonus

    def test_same_source_same_day_no_double_count(self):
        cluster = _make_cluster([
            _make_article(url="https://rci.fm/a", source="RCI", date=datetime(2025, 1, 1)),
            _make_article(url="https://rci.fm/b", source="RCI", date=datetime(2025, 1, 1)),
        ])
        score = score_cluster(cluster, most_read_paths=set())
        assert score == 1 * 1  # 1 unique day × 1 unique source


class TestRepresentativeEmbedding:
    def test_centroid_of_two_vectors(self):
        cluster = [
            {"article": _make_article(), "embedding": np.array([1.0, 0.0])},
            {"article": _make_article(), "embedding": np.array([0.0, 1.0])},
        ]
        centroid = representative_embedding(cluster)
        np.testing.assert_allclose(centroid, [0.5, 0.5])


class TestPickRepresentativeArticle:
    def test_returns_closest_to_centroid(self):
        a1 = _make_article("Near")
        a2 = _make_article("Far")
        centroid = np.array([1.0, 0.0])
        cluster = [
            {"article": a1, "embedding": np.array([1.0, 0.0])},   # identical to centroid
            {"article": a2, "embedding": np.array([0.0, 1.0])},   # orthogonal
        ]
        rep = pick_representative_article(cluster, centroid)
        assert rep["title"] == "Near"



class TestRenderSuggestions:
    def _make_scored(self, theme, top_score, runner_up="Autre thème", runner_up_score=0.05, title="T"):
        article = _make_article(title=title)
        return {
            "theme": theme,
            "top_score": top_score,
            "runner_up": runner_up,
            "runner_up_score": runner_up_score,
            "articles": [article],
        }

    def test_unclassified_listed(self):
        scored = [self._make_scored(UNCLASSIFIED, top_score=0.05, title="Mystery")]
        output = render_suggestions(1, scored, threshold=0.15)
        assert "Mystery" in output
        assert "non classifiés" in output

    def test_low_confidence_listed(self):
        scored = [self._make_scored("Politique", top_score=0.20, runner_up_score=0.10, title="Iffy")]
        output = render_suggestions(1, scored, threshold=0.15, low_confidence_margin=0.10)
        assert "Iffy" in output
        assert "faible confiance" in output

    def test_ambiguous_listed(self):
        scored = [self._make_scored("Politique", top_score=0.50, runner_up_score=0.48, title="Ambig")]
        output = render_suggestions(1, scored, threshold=0.15, low_confidence_margin=0.10, ambiguity_margin=0.05)
        assert "Ambig" in output
        assert "ambigus" in output

    def test_clean_cluster_not_listed(self):
        scored = [self._make_scored("Politique", top_score=0.90, runner_up_score=0.05, title="Clear")]
        output = render_suggestions(1, scored, threshold=0.15, low_confidence_margin=0.10, ambiguity_margin=0.05)
        assert "Clear" not in output
