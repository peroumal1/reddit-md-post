"""
Run batch inference on a daily feed file using the trained classifier head.

Usage:
    pdm run python classifier/infer.py data/feed-2026-03-12.md
    pdm run python classifier/infer.py data/feed-2026-03-12.md --head data/classifier_head.joblib

Prints a classified summary grouped by theme, useful for evaluating model quality
on real articles before enabling --classify in the main pipeline.
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from rss_summary.parsing import parse_daily_feed_md


def parse_feed_md(path: str) -> list[dict]:
    """Extract articles from a daily feed markdown table."""
    return [{"title": a["title"], "summary": a["summary"]} for a in parse_daily_feed_md(path)]


def load_head(path: str) -> dict:
    return joblib.load(path)


def classify_batch(
    model: SentenceTransformer,
    articles: list[dict],
    head: dict,
    threshold: float = 0.5,
) -> list[dict]:
    texts = [f"{a['title']} | {a['summary']}" for a in articles]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    clf = head["clf"]
    le = head["label_encoder"]
    label_to_theme = head["label_to_theme"]

    proba = clf.predict_proba(embeddings)
    results = []
    for i, article in enumerate(articles):
        score = float(np.max(proba[i]))
        pred_idx = int(np.argmax(proba[i]))
        label = le.inverse_transform([pred_idx])[0]
        theme = label_to_theme[label] if score >= threshold else "Autres"
        results.append({**article, "theme": theme, "label": label, "score": score})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify articles from a feed file")
    parser.add_argument("feed", help="Path to a feed-YYYY-MM-DD.md file")
    parser.add_argument("--head", default="data/classifier_head.joblib")
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()

    if not Path(args.head).exists():
        print(f"Error: classifier head not found at {args.head}")
        print("Run: pdm run python classifier/train.py")
        raise SystemExit(1)

    print(f"Loading articles from {args.feed}...")
    articles = parse_feed_md(args.feed)
    print(f"Found {len(articles)} articles")

    print("Loading model and head...")
    model = SentenceTransformer("BAAI/bge-m3")
    head = load_head(args.head)

    print("Classifying...")
    t0 = time.time()
    results = classify_batch(model, articles, head, args.threshold)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s\n")

    # Group by theme
    by_theme = defaultdict(list)
    for r in results:
        by_theme[r["theme"]].append(r)

    total = len(results)
    autres = len(by_theme.get("Autres", []))
    print(f"Results: {total} articles, {autres} unclassified (Autres)\n")

    for theme, items in sorted(by_theme.items()):
        print(f"── {theme} ({len(items)})")
        for item in items:
            score_str = f"{item['score']:.2f}"
            title = item["title"][:80]
            print(f"   [{score_str}] {title}")
        print()


if __name__ == "__main__":
    main()
