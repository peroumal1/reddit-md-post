#!/usr/bin/env python3
"""Interactive labeler for unclassified articles from a weekly review file.

For each unclassified article, shows the title, URL, and classifier suggestion,
then lets you assign a theme with a single keypress. Chosen articles are appended
to data/themes.json as new training examples.

Usage:
    pdm run python classifier/label.py data/weekly-w11-review.md
"""

import json
import re
import sys
import termios
import tty
from pathlib import Path

import tomllib

TAXONOMY_PATH = Path("data/taxonomy.toml")
THEMES_JSON_PATH = Path("data/themes.json")


def load_theme_names():
    with open(TAXONOMY_PATH, "rb") as f:
        data = tomllib.load(f)
    return [t["name"] for t in data["themes"]]


def parse_unclassified(review_path):
    """Parse unclassified articles from a weekly review markdown file."""
    text = Path(review_path).read_text()
    section = re.search(
        r"## Articles non classifiés.*?\n\|[^\n]+\|\n\|[^\n]+\|\n(.*?)(?=\n## |\Z)",
        text,
        re.DOTALL,
    )
    if not section:
        return []
    articles = []
    for line in section.group(1).strip().splitlines():
        m = re.match(r"\|\s*\[([^\]]+)\]\(([^)]+)\)\s*\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            articles.append({
                "title": m.group(1).strip(),
                "url": m.group(2).strip(),
                "top_theme": m.group(3).strip(),
                "top_score": float(m.group(4)),
            })
    return articles


def get_keypress():
    """Read a single keypress without requiring Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def append_example(theme_name, title, themes_path=THEMES_JSON_PATH):
    """Append a training example to the matching theme in themes.json."""
    data = json.loads(themes_path.read_text())
    for entry in data:
        if entry["theme"] == theme_name:
            entry["examples"].append(title)
            themes_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            return
    print(f"  ERROR: theme '{theme_name}' not found in themes.json", file=sys.stderr)


def theme_key(index):
    """Map theme index (0-based) to display key: 1-9 for first 9, 0 for 10th."""
    return str(index + 1) if index < 9 else "0"


def main():
    if len(sys.argv) < 2:
        print("Usage: pdm run python classifier/label.py <review-file>")
        sys.exit(1)

    review_path = sys.argv[1]
    themes = load_theme_names()
    articles = parse_unclassified(review_path)

    if not articles:
        print("No unclassified articles found in the review file.")
        return

    print(f"Found {len(articles)} unclassified articles.")
    print("Keys: 1-9, 0 = theme | s = skip (leave as Autres) | q = quit\n")

    labeled = 0
    skipped = 0

    for i, article in enumerate(articles, 1):
        print("=" * 72)
        print(f"[{i}/{len(articles)}]  {article['title']}")
        print(f"         {article['url']}")
        print(f"  Suggestion: {article['top_theme']} ({article['top_score']:.2f})")
        print()
        for j, theme in enumerate(themes):
            print(f"  {theme_key(j)}) {theme}")
        print("  s) Skip  |  q) Quit")
        print()
        print("  Choice: ", end="", flush=True)

        ch = get_keypress().lower()
        print(ch)

        if ch == "q":
            print("Quitting.")
            break
        elif ch == "s":
            skipped += 1
            continue

        # Map key to theme index
        key_to_idx = {theme_key(j): j for j in range(len(themes))}
        if ch in key_to_idx:
            chosen = themes[key_to_idx[ch]]
            append_example(chosen, article["title"])
            print(f"  -> Added to '{chosen}'")
            labeled += 1
        else:
            print("  -> Invalid key, skipping.")
            skipped += 1

        print()

    print(f"\nDone: {labeled} example(s) added, {skipped} skipped.")
    if labeled > 0:
        print("Retrain with:  pdm run python classifier/train.py")


if __name__ == "__main__":
    main()
