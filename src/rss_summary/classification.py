import tomllib
from pathlib import Path

DEFAULT_TAXONOMY_PATH = Path("data/taxonomy.toml")
DEFAULT_HEAD_PATH = Path("data/classifier_head.joblib")
UNCLASSIFIED = "Autres"

E5_MODEL_ID = "intfloat/multilingual-e5-large-instruct"
E5_PROMPT = "Instruct: Classify the following French news headline into a thematic category.\nQuery: "


def load_taxonomy(path=None):
    """Load ordered theme names from taxonomy TOML. Returns a list of strings."""
    p = Path(path) if path else DEFAULT_TAXONOMY_PATH
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return data["themes"]


def load_e5_model():
    """Load the e5-instruct model used for the second half of the classification embedding."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(E5_MODEL_ID)


def encode_for_classification(text: str, model_bge, model_e5) -> "np.ndarray":
    """Encode text with bge-m3 + e5-instruct and return a concatenated 2048-dim vector.

    Both individual embeddings are L2-normalized before concatenation.
    The result is also L2-normalized to match the training distribution.
    """
    import numpy as np
    from rss_summary.parsing import strip_html

    clean = strip_html(text)
    emb_bge = model_bge.encode(clean, normalize_embeddings=True)
    emb_e5 = model_e5.encode(E5_PROMPT + clean, normalize_embeddings=True)
    concat = np.concatenate([emb_bge, emb_e5])
    norm = np.linalg.norm(concat)
    return concat / norm if norm > 0 else concat


def load_classifier_head(path=None):
    """Load the trained classifier head. Raises FileNotFoundError if not found."""
    p = Path(path) if path else DEFAULT_HEAD_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Classifier head not found at '{p}'. Run: pdm run python classifier/train.py"
        )
    import joblib
    return joblib.load(p)


def classify_article(embedding, head, threshold=0.15):
    """Return the theme name for the given article embedding."""
    return classify_article_scored(embedding, head, threshold)["theme"]


def classify_article_scored(embedding, head, threshold=0.15):
    """Return classification result with full score breakdown.

    Returns a dict with:
      theme        — winning theme name (or UNCLASSIFIED)
      top_score    — confidence score of the winning theme
      runner_up    — name of the second-best theme (or None)
      runner_up_score — score of the second-best theme (or None)
    """
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
