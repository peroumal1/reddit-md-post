import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import click
import requests
from bs4 import BeautifulSoup
from py_markdown_table.markdown_table import markdown_table as _markdown_table
from sentence_transformers import SentenceTransformer

from rss_summary.classification import classify_article, encode_themes, load_taxonomy
from rss_summary.similarity import encode_text

MOIS = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "août",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre",
}

SOURCE_MAP = {
    "karibinfo.com": "Karibinfo",
    "la1ere.franceinfo.fr": "La 1ère",
    "rci.fm": "RCI",
    "guadeloupe.franceantilles.fr": "France-Antilles",
}

CLUSTER_THRESHOLD = 0.70


def extract_source(url):
    host = urlparse(url).hostname or ""
    for domain, name in SOURCE_MAP.items():
        if domain in host:
            return name
    return host


def parse_feed_file(path):
    """Parse a daily feed markdown table into a list of article dicts."""
    articles = []
    lines = Path(path).read_text().splitlines()
    for line in lines:
        line = line.strip()
        # Skip header and separator rows
        if not line.startswith("|") or re.match(r"^\|[-| ]+\|$", line):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 3:
            continue
        m = re.match(r"\[(.+?)\]\((.+?)\)", cols[0])
        if not m:
            continue
        title, url = m.group(1), m.group(2)
        summary = cols[1]
        try:
            date = datetime.fromisoformat(cols[2])
        except ValueError:
            continue
        articles.append({
            "title": title,
            "url": url,
            "summary": summary,
            "date": date,
            "source": extract_source(url),
        })
    return articles


def get_most_read_urls():
    """Scrape most-read article URL paths from RCI and France-Antilles homepages."""
    paths = set()

    try:
        r = requests.get("https://rci.fm/guadeloupe", timeout=10)
        if r.ok:
            soup = BeautifulSoup(r.text, "html.parser")
            block = soup.find(id="block-views-block-block-articles-les-plus-lus-teaser-short-block-1")
            if block:
                for a in block.find_all("a", href=True):
                    paths.add(urlparse(a["href"]).path)
    except requests.RequestException:
        logging.warning("Could not fetch RCI most-read section.")

    try:
        r = requests.get("https://www.guadeloupe.franceantilles.fr", timeout=10)
        if r.ok:
            soup = BeautifulSoup(r.text, "html.parser")
            marker = soup.find(string=re.compile(r"Articles les plus [Ll]us"))
            if marker:
                container = marker.find_parent()
                for _ in range(6):
                    if container is None:
                        break
                    links = container.find_all("a", href=True)
                    if len(links) >= 3:
                        for a in links:
                            paths.add(urlparse(a["href"]).path)
                        break
                    container = container.parent
    except requests.RequestException:
        logging.warning("Could not fetch France-Antilles most-read section.")

    return paths


def cluster_articles(articles, model):
    """Greedy semantic clustering across all articles. Returns list of clusters."""
    embeddings = [encode_text(model, f"{a['title']}. {a['summary']}") for a in articles]
    assigned = [False] * len(articles)
    clusters = []

    for i in range(len(articles)):
        if assigned[i]:
            continue
        cluster = [{"article": articles[i], "embedding": embeddings[i]}]
        assigned[i] = True
        for j in range(i + 1, len(articles)):
            if assigned[j]:
                continue
            sim = model.similarity([embeddings[i]], [embeddings[j]])[0][0].item()
            if sim >= CLUSTER_THRESHOLD:
                cluster.append({"article": articles[j], "embedding": embeddings[j]})
                assigned[j] = True
        clusters.append(cluster)

    return clusters


def score_cluster(cluster, most_read_paths):
    days = len({item["article"]["date"].date() for item in cluster})
    sources = len({item["article"]["source"] for item in cluster})
    most_read_bonus = sum(
        1 for item in cluster
        if urlparse(item["article"]["url"]).path in most_read_paths
    )
    return days * sources + most_read_bonus


def representative_embedding(cluster):
    """Return the centroid embedding of a cluster."""
    import numpy as np
    vecs = [item["embedding"] for item in cluster]
    return np.mean(vecs, axis=0)


def render_weekly(week_num, week_start, week_end, clusters_by_theme, theme_names):
    start_str = f"{week_start.day} {MOIS[week_start.month]}"
    end_str = f"{week_end.day} {MOIS[week_end.month]} {week_end.year}"
    lines = [f"# Semaine W{week_num:02d} — {start_str} au {end_str}", "", "## Sujets marquants"]

    ordered_themes = list(theme_names) + ["Autres"]
    for theme in ordered_themes:
        theme_clusters = clusters_by_theme.get(theme, [])
        if not theme_clusters:
            continue
        lines.append(f"\n## {theme}")
        for cluster in theme_clusters:
            rep = cluster["articles"][0]
            score = cluster["score"]
            sources = {item["article"]["source"] for item in cluster["raw"]}
            days = len({item["article"]["date"].date() for item in cluster["raw"]})
            most_read_tags = cluster.get("most_read_tags", [])

            score_parts = [f"{days} jour{'s' if days > 1 else ''}",
                           f"{len(sources)} source{'s' if len(sources) > 1 else ''}"]
            if most_read_tags:
                score_parts.append("🔥 plus lu (" + ", ".join(most_read_tags) + ")")

            lines.append(f"\n### {rep['title']}")
            lines.append(f"**{' · '.join(score_parts)}**\n")

            rows = [
                {
                    "Titre": f"[{item['article']['title']}]({item['article']['url']})",
                    "Date": item["article"]["date"].strftime("%d/%m"),
                    "Source": item["article"]["source"],
                }
                for item in cluster["raw"]
            ]
            table = (
                _markdown_table(rows)
                .set_params(row_sep="markdown", quote=False)
                .get_markdown()
            )
            lines.append(table)
            lines.append("\n---")

    return "\n".join(lines)


@click.command()
@click.option("--data-dir", default="data", show_default=True, help="Directory containing daily feed files")
@click.option("--output-dir", default="data", show_default=True, help="Directory to write the weekly digest")
@click.option("--week", default=None, type=int, help="ISO week number (default: current week)")
@click.option("--year", default=None, type=int, help="Year for --week (default: current year)")
@click.option("--taxonomy", default="data/taxonomy.toml", show_default=True, help="Path to taxonomy TOML config")
def main(data_dir, output_dir, week, year, taxonomy):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    today = datetime.now()
    iso = today.isocalendar()
    week_num = week if week is not None else iso.week
    year_num = year if year is not None else iso.year

    week_start = datetime.fromisocalendar(year_num, week_num, 1)
    week_end = datetime.fromisocalendar(year_num, week_num, 7)

    # Collect daily feed files for the week
    data_path = Path(data_dir)
    articles = []
    for day_offset in range(7):
        day = week_start + timedelta(days=day_offset)
        feed_file = data_path / f"feed-{day.strftime('%Y-%m-%d')}.md"
        if feed_file.exists():
            day_articles = parse_feed_file(feed_file)
            logging.info("Loaded %d articles from %s", len(day_articles), feed_file.name)
            articles.extend(day_articles)

    if not articles:
        logging.info("No articles found for week W%02d.", week_num)
        return

    logging.info("Total articles to cluster: %d", len(articles))

    # Fetch most-read signals
    logging.info("Fetching most-read signals…")
    most_read_paths = get_most_read_urls()
    logging.info("Found %d most-read paths.", len(most_read_paths))

    # Load model and taxonomy
    model = SentenceTransformer("BAAI/bge-m3")
    try:
        themes = load_taxonomy(taxonomy)
    except FileNotFoundError:
        raise click.ClickException(f"Taxonomy file not found: {taxonomy}")
    theme_names = [t["name"] for t in themes]
    theme_embeddings = encode_themes(model, themes)

    # Cluster articles
    logging.info("Clustering articles…")
    raw_clusters = cluster_articles(articles, model)
    logging.info("Found %d clusters.", len(raw_clusters))

    # Score, classify, and annotate clusters
    scored = []
    for raw_cluster in raw_clusters:
        score = score_cluster(raw_cluster, most_read_paths)
        centroid = representative_embedding(raw_cluster)
        theme = classify_article(model, centroid, theme_embeddings, theme_names)

        most_read_tags = []
        for item in raw_cluster:
            path = urlparse(item["article"]["url"]).path
            if path in most_read_paths:
                src = item["article"]["source"]
                if src not in most_read_tags:
                    most_read_tags.append(src)

        scored.append({
            "raw": raw_cluster,
            "score": score,
            "theme": theme,
            "most_read_tags": most_read_tags,
            "articles": sorted(
                [item["article"] for item in raw_cluster],
                key=lambda a: a["date"],
                reverse=True,
            ),
        })

    # Group by theme, sort each group by score descending
    clusters_by_theme = {}
    for cluster in sorted(scored, key=lambda c: c["score"], reverse=True):
        clusters_by_theme.setdefault(cluster["theme"], []).append(cluster)

    # Render and write
    markdown = render_weekly(week_num, week_start, week_end, clusters_by_theme, theme_names)
    output_file = Path(output_dir) / f"weekly-w{week_num:02d}.md"
    try:
        output_file.write_text(markdown)
    except OSError as e:
        raise click.ClickException(f"Could not write weekly digest to '{output_file}': {e}") from e

    logging.info("Weekly digest written to %s", output_file)


if __name__ == "__main__":
    main()
