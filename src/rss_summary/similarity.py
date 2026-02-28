import difflib

from bs4 import BeautifulSoup

SIMILARITY_THRESHOLD = 0.75
TITLE_SIMILARITY_THRESHOLD = 0.85


def encode_text(model, text):
    """Strip HTML from text and encode it to a single embedding vector."""
    plain = BeautifulSoup(text, features="html.parser").get_text()
    return model.encode([plain])[0]


def is_duplicate(model, embedding, existing_embeddings, threshold=SIMILARITY_THRESHOLD):
    """Check if embedding is semantically similar to any entry in existing_embeddings."""
    if not existing_embeddings:
        return False
    similarities = model.similarity([embedding], existing_embeddings)
    max_score = similarities[0].max().item()
    if max_score > threshold:
        print(f"Similarity detected (score: {max_score:.4f}), skipping duplicate.")
    return max_score > threshold


def title_is_duplicate(title, existing_titles, threshold=TITLE_SIMILARITY_THRESHOLD):
    """Fast fuzzy title check to avoid expensive semantic encoding for obvious duplicates."""
    if not existing_titles:
        return False
    normalized = title.strip().lower()
    for existing in existing_titles:
        ratio = difflib.SequenceMatcher(None, normalized, existing.strip().lower()).ratio()
        if ratio > threshold:
            print(f"Fuzzy title match (ratio: {ratio:.4f}), skipping duplicate.")
            return True
    return False
