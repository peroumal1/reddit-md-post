import tomllib
from pathlib import Path

DEFAULT_TAXONOMY_PATH = Path("data/taxonomy.toml")
DEFAULT_HEAD_PATH = Path("data/classifier_head.joblib")
UNCLASSIFIED = "Autres"


def load_taxonomy(path=None):
    """Load theme definitions from a TOML config file."""
    p = Path(path) if path else DEFAULT_TAXONOMY_PATH
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return data["themes"]


def load_classifier_head(path=None):
    """Load the trained classifier head if available. Returns None if not found."""
    p = Path(path) if path else DEFAULT_HEAD_PATH
    if not p.exists():
        return None
    try:
        import joblib
        return joblib.load(p)
    except Exception:
        return None


def encode_themes(model, themes):
    """Encode theme descriptions into embeddings using the given model."""
    descriptions = [t["description"] for t in themes]
    return model.encode(descriptions)


def classify_article(model, embedding, theme_embeddings, theme_names, threshold=0.15, head=None):
    """Return the theme name for the given article embedding.

    Uses the trained classifier head when available, falls back to zero-shot
    cosine similarity against theme description embeddings.
    """
    result = classify_article_scored(model, embedding, theme_embeddings, theme_names, threshold, head)
    return result["theme"]


def classify_article_scored(model, embedding, theme_embeddings, theme_names, threshold=0.15, head=None):
    """Return classification result with full score breakdown.

    When a trained head is provided, uses logistic regression probabilities.
    Falls back to cosine similarity against theme descriptions otherwise.

    Returns a dict with:
      theme        — winning theme name (or UNCLASSIFIED)
      top_score    — confidence score of the winning theme
      runner_up    — name of the second-best theme (or None)
      runner_up_score — score of the second-best theme (or None)
    """
    if head is not None:
        return _classify_with_head(embedding, head, threshold)
    return _classify_zero_shot(model, embedding, theme_embeddings, theme_names, threshold)


def _classify_with_head(embedding, head, threshold):
    """Classify using the trained logistic regression head."""
    import numpy as np
    clf = head["clf"]
    le = head["label_encoder"]
    label_to_theme = head["label_to_theme"]

    # Normalize to match training distribution (train.py uses normalize_embeddings=True)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    proba = clf.predict_proba([embedding])[0]
    sorted_idx = np.argsort(proba)[::-1]

    top_idx = int(sorted_idx[0])
    top_score = float(proba[top_idx])
    top_label = le.inverse_transform([top_idx])[0]
    top_theme = label_to_theme[top_label] if top_score >= threshold else UNCLASSIFIED

    runner_up = None
    runner_up_score = None
    if len(sorted_idx) > 1:
        ru_idx = int(sorted_idx[1])
        runner_up_score = float(proba[ru_idx])
        runner_up_label = le.inverse_transform([ru_idx])[0]
        runner_up = label_to_theme[runner_up_label]

    return {
        "theme": top_theme,
        "top_score": top_score,
        "runner_up": runner_up,
        "runner_up_score": runner_up_score,
    }


def _classify_zero_shot(model, embedding, theme_embeddings, theme_names, threshold):
    """Zero-shot classification via cosine similarity against theme descriptions."""
    if theme_embeddings is None or len(theme_embeddings) == 0:
        return {"theme": UNCLASSIFIED, "top_score": 0.0, "runner_up": None, "runner_up_score": None}
    sims = model.similarity([embedding], theme_embeddings)[0]
    scores = [(theme_names[i], sims[i].item()) for i in range(len(theme_names))]
    scores.sort(key=lambda x: x[1], reverse=True)
    winner_name, winner_score = scores[0]
    runner_name, runner_score = scores[1] if len(scores) > 1 else (None, None)
    theme = winner_name if winner_score >= threshold else UNCLASSIFIED
    return {
        "theme": theme,
        "top_score": winner_score,
        "runner_up": runner_name,
        "runner_up_score": runner_score,
    }
