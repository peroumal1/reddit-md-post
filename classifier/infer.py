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
import numpy as np
from sentence_transformers import SentenceTransformer

from rss_summary.classification import BGE_MODEL_ID, UNCLASSIFIED, encode_for_classification, load_classifier_head, load_e5_model
from rss_summary.parsing import parse_daily_feed_md


def classify_batch(
    model_bge: SentenceTransformer,
    model_e5: SentenceTransformer,
    articles: list[dict],
    head: dict,
    threshold: float = 0.5,
) -> list[dict]:
    clf = head["clf"]
    le = head["label_encoder"]
    label_to_theme = head["label_to_theme"]

    embeddings = np.array([
        encode_for_classification(f"{a['title']}. {a['summary']}", model_bge, model_e5)
        for a in articles
    ])

    proba = clf.predict_proba(embeddings)
    results = []
    for i, article in enumerate(articles):
        score = float(np.max(proba[i]))
        pred_idx = int(np.argmax(proba[i]))
        label = le.inverse_transform([pred_idx])[0]
        theme = label_to_theme[label] if score >= threshold else UNCLASSIFIED
        results.append({**article, "theme": theme, "label": label, "score": score})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify articles from a feed file")
    parser.add_argument("feed", help="Path to a feed-YYYY-MM-DD.md file")
    parser.add_argument("--head", default="data/classifier_head.joblib")
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()

    print(f"Loading articles from {args.feed}...")
    articles = [{"title": a["title"], "summary": a["summary"]} for a in parse_daily_feed_md(args.feed)]
    print(f"Found {len(articles)} articles")

    print("Loading models and head...")
    model_bge = SentenceTransformer(BGE_MODEL_ID)
    model_e5 = load_e5_model()
    try:
        head = load_classifier_head(args.head)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise SystemExit(1)

    print("Classifying...")
    t0 = time.time()
    results = classify_batch(model_bge, model_e5, articles, head, args.threshold)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s\n")

    by_theme = defaultdict(list)
    for r in results:
        by_theme[r["theme"]].append(r)

    total = len(results)
    autres = len(by_theme.get(UNCLASSIFIED, []))
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
