from unittest.mock import patch

from rss_summary.parsing import extract_first_paragraph, get_default_image_link


class TestExtractFirstParagraph:
    def test_simple_html_with_newlines(self):
        html = "<p>First paragraph.</p>\n<p>Second paragraph.</p>"
        assert extract_first_paragraph(html) == "First paragraph."

    def test_inline_html_concatenates(self):
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        assert extract_first_paragraph(html) == "First paragraph.Second paragraph."

    def test_nested_html_with_newlines(self):
        html = "<div><b>Bold text</b> and more</div>\n<p>Other</p>"
        assert extract_first_paragraph(html) == "Bold text and more"

    def test_empty_html(self):
        assert extract_first_paragraph("") == ""

    def test_multiline_html(self):
        html = "<p>Line one</p>\n<p>Line two</p>"
        assert extract_first_paragraph(html) == "Line one"

    def test_strips_whitespace(self):
        html = "   <p>  Hello world  </p>  "
        assert extract_first_paragraph(html) == "Hello world"


class TestGetDefaultImageLink:
    def test_returns_media_content_when_present(self):
        resource = {"media_content": [{"url": "https://img.com/photo.jpg"}]}
        result = get_default_image_link(resource, "https://example.com")
        assert result == [{"url": "https://img.com/photo.jpg"}]

    @patch("rss_summary.parsing.requests.get")
    def test_fetches_from_page_when_no_media_content(self, mock_get):
        mock_get.return_value.text = (
            '<html><body><img src="/images/logo.png"></body></html>'
        )
        result = get_default_image_link({}, "https://example.com/article")
        assert len(result) == 1
        assert "/images/logo.png" in result[0]["url"]

    @patch("rss_summary.parsing.requests.get")
    def test_returns_empty_url_when_no_img_on_page(self, mock_get):
        mock_get.return_value.text = "<html><body>No images here</body></html>"
        result = get_default_image_link({}, "https://example.com/article")
        assert result == [{"url": ""}]

    @patch("rss_summary.parsing.requests.get")
    def test_returns_empty_url_on_request_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("timeout")
        result = get_default_image_link({}, "https://example.com/article")
        assert result == [{"url": ""}]
