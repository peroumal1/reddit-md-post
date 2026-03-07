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
    if not theme_embeddings:
        return UNCLASSIFIED
    sims = model.similarity([embedding], theme_embeddings)[0]
    best_idx = int(sims.argmax())
    if sims[best_idx].item() >= threshold:
        return theme_names[best_idx]
    return UNCLASSIFIED
