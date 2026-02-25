SIMILARITY_THRESHOLD = 0.6


def is_duplicate(model, text, existing_summaries, threshold=SIMILARITY_THRESHOLD):
    """Check if text is semantically similar to any entry in existing_summaries."""
    if not existing_summaries:
        return False
    embeddings = model.encode([text])
    existing_embeddings = model.encode(existing_summaries)
    similarities = model.similarity(embeddings, existing_embeddings)
    max_score = similarities[0].max().item()
    if max_score > threshold:
        print(f"Similarity detected (score: {max_score:.4f}), skipping duplicate.")
    return max_score > threshold
