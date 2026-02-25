from datetime import datetime
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_feed_entry():
    """A single feed entry dict as produced by the main loop."""
    return {
        "title": "Test Title",
        "link": "https://example.com",
        "summary": "A summary",
        "published_date": datetime(2025, 1, 1, 12, 0),
        "media_content": [{"url": "https://example.com/img.jpg"}],
    }


@pytest.fixture
def make_model():
    """Factory fixture: returns a mock SentenceTransformer with a given max similarity."""
    def _make(max_similarity):
        model = MagicMock()
        model.encode.side_effect = lambda x: x
        tensor = MagicMock()
        tensor.max.return_value.item.return_value = max_similarity
        model.similarity.return_value = [tensor]
        return model
    return _make
