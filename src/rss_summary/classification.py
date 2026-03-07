import tomllib
from pathlib import Path

DEFAULT_TAXONOMY_PATH = Path("data/taxonomy.toml")
UNCLASSIFIED = "Autres"


def load_taxonomy(path=None):
    """Load theme definitions from a TOML config file."""
    p = Path(path) if path else DEFAULT_TAXONOMY_PATH
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return data["themes"]


def encode_themes(model, themes):
    """Encode theme descriptions into embeddings using the given model."""
    descriptions = [t["description"] for t in themes]
    return model.encode(descriptions)


def classify_article(model, embedding, theme_embeddings, theme_names, threshold=0.35):
    """Return the theme name with highest cosine similarity, or UNCLASSIFIED."""
    result = classify_article_scored(model, embedding, theme_embeddings, theme_names, threshold)
    return result["theme"]


def classify_article_scored(model, embedding, theme_embeddings, theme_names, threshold=0.35):
    """Return classification result with full score breakdown.

    Returns a dict with:
      theme        — winning theme name (or UNCLASSIFIED)
      top_score    — similarity score of the winning theme
      runner_up    — name of the second-best theme (or None)
      runner_up_score — score of the second-best theme (or None)
    """
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
