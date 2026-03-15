from unittest.mock import MagicMock, patch

import requests

from rss_summary.weekly import get_most_read_urls


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
