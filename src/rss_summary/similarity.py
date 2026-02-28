import difflib
import logging

from rss_summary.parsing import strip_html

SIMILARITY_THRESHOLD = 0.75
TITLE_SIMILARITY_THRESHOLD = 0.85


def encode_text(model, text):
    """Strip HTML from text and encode it to a single embedding vector."""
    return model.encode([strip_html(text)])[0]


def is_duplicate(model, embedding, existing_embeddings, threshold=SIMILARITY_THRESHOLD):
    """Check if embedding is semantically similar to any entry in existing_embeddings."""
    if not existing_embeddings:
        return False
    similarities = model.similarity([embedding], existing_embeddings)
    max_score = similarities[0].max().item()
    if max_score > threshold:
        logging.debug("Similarity detected (score: %.4f), skipping duplicate.", max_score)
    return max_score > threshold


def title_is_duplicate(title, existing_titles, threshold=TITLE_SIMILARITY_THRESHOLD):
    """Fast fuzzy title check to avoid expensive semantic encoding for obvious duplicates."""
    if not existing_titles:
        return False
    normalized = title.strip().lower()
    for existing in existing_titles:
        ratio = difflib.SequenceMatcher(None, normalized, existing.strip().lower()).ratio()
        if ratio > threshold:
            logging.debug("Fuzzy title match (ratio: %.4f), skipping duplicate.", ratio)
            return True
    return False
