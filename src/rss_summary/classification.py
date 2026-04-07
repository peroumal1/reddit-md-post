import tomllib
from pathlib import Path

import numpy as np

from rss_summary.parsing import strip_html

DEFAULT_TAXONOMY_PATH = Path("data/taxonomy.toml")
DEFAULT_HEAD_PATH = Path("data/classifier_head.joblib")
UNCLASSIFIED = "Autres"

BGE_MODEL_ID = "BAAI/bge-m3"
E5_MODEL_ID = "intfloat/multilingual-e5-large-instruct"
E5_PROMPT = "Instruct: Classify the following French news headline into a thematic category.\nQuery: "
MISTRAL_MODEL = "mistral-small-latest"
CLASSIFICATION_THRESHOLD = 0.15


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


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def encode_for_classification(text: str, model_bge, model_e5) -> np.ndarray:
    """Encode text with bge-m3 + e5-instruct and return a concatenated 2048-dim vector.

    Both individual embeddings are L2-normalized before concatenation.
    The result is also L2-normalized to match the training distribution.
    """
    clean = strip_html(text)
    emb_bge = model_bge.encode(clean, normalize_embeddings=True)
    emb_e5 = model_e5.encode(E5_PROMPT + clean, normalize_embeddings=True)
    return _l2_normalize(np.concatenate([emb_bge, emb_e5]))


def batch_encode_e5(texts: list, model_e5) -> np.ndarray:
    """Batch-encode a list of texts with e5-instruct. Returns a 2D array (N, 1024).

    Strips HTML and prepends the instruction prompt before encoding.
    """
    prompts = [E5_PROMPT + strip_html(t) for t in texts]
    return model_e5.encode(prompts, normalize_embeddings=True)


def build_cls_embedding(bge_embedding: np.ndarray, e5_embedding: np.ndarray) -> np.ndarray:
    """Build a 2048-dim classification embedding from pre-computed components.

    bge_embedding may be unnormalized (as produced by encode_text in similarity.py);
    it is L2-normalized here to match the training distribution before concatenation.
    """
    return _l2_normalize(np.concatenate([_l2_normalize(bge_embedding), e5_embedding]))


def load_classifier_head(path=None):
    """Load the trained classifier head. Raises FileNotFoundError if not found."""
    import joblib
    p = Path(path) if path else DEFAULT_HEAD_PATH
    try:
        return joblib.load(p)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Classifier head not found at '{p}'. Run: pdm run python classifier/train.py"
        )


def classify_article(embedding, head, threshold=CLASSIFICATION_THRESHOLD):
    """Return the theme name for the given article embedding."""
    return classify_article_scored(embedding, head, threshold)["theme"]


def classify_article_scored(embedding, head, threshold=CLASSIFICATION_THRESHOLD):
    """Return classification result with full score breakdown.

    Returns a dict with:
      theme        — winning theme name (or UNCLASSIFIED)
      top_score    — confidence score of the winning theme
      runner_up    — name of the second-best theme (or None)
      runner_up_score — score of the second-best theme (or None)
    """
    clf = head["clf"]
    le = head["label_encoder"]
    label_to_theme = head["label_to_theme"]

    # Normalize to match training distribution (train.py uses normalize_embeddings=True)
    embedding = _l2_normalize(embedding)

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
