import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import click
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from rss_summary.classification import classify_article_scored, encode_for_classification, load_classifier_head, load_e5_model, load_taxonomy
from rss_summary.parsing import parse_daily_feed_md
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
MISTRAL_MODEL = "mistral-small-latest"
_CLUSTER_SORT_KEY = lambda c: (bool(c["most_read_tags"]), c["score"])


def extract_source(url):
    host = urlparse(url).hostname or ""
    for domain, name in SOURCE_MAP.items():
        if domain in host:
            return name
    return host


def parse_feed_file(path):
    """Parse a daily feed markdown table into a list of article dicts."""
    articles = parse_daily_feed_md(path)
    for a in articles:
        a["source"] = extract_source(a["url"])
    return articles


def get_most_read_urls():
    """Scrape most-read article URL paths from RCI and France-Antilles homepages."""
    def fetch_rci():
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
        return paths

    def fetch_fa():
        paths = set()
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

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_rci, f_fa = ex.submit(fetch_rci), ex.submit(fetch_fa)
        return f_rci.result() | f_fa.result()


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


def pick_representative_article(raw_cluster, centroid):
    """Return the article whose embedding is closest to the cluster centroid.

    This tends to be the most generic/encompassing article rather than a
    specific local variant, making it a better section heading.
    """
    import numpy as np
    norm = np.linalg.norm(centroid)
    c = centroid / norm if norm > 0 else centroid
    best, best_sim = raw_cluster[0]["article"], -1.0
    for item in raw_cluster:
        e = item["embedding"]
        n = np.linalg.norm(e)
        e_norm = e / n if n > 0 else e
        sim = float(np.dot(c, e_norm))
        if sim > best_sim:
            best_sim = sim
            best = item["article"]
    return best


def _format_week_range(week_start, week_end):
    start_str = f"{week_start.day} {MOIS[week_start.month]}"
    end_str = f"{week_end.day} {MOIS[week_end.month]} {week_end.year}"
    return start_str, end_str


def _cluster_sections(clusters):
    """Build numbered article sections for the Mistral prompt."""
    sections = []
    for i, cluster in enumerate(clusters, 1):
        rep = pick_representative_article(cluster["raw"], cluster["centroid"])
        article_lines = []
        for item in cluster["raw"]:
            a = item["article"]
            summary = a.get("summary", "").strip()[:300]
            line = f'- [{a["source"]}, {a["date"].strftime("%d/%m")}] [{a["title"]}]({a["url"]})'
            if summary:
                line += f" : {summary}"
            article_lines.append(line)
        sections.append(f"[{i}] {rep['title']}\n" + "\n".join(article_lines))
    return sections


def generate_stitched_narrative(clusters, week_num, week_start, week_end, client):
    """Single Mistral call producing one flowing editorial text covering all clusters."""
    start_str, end_str = _format_week_range(week_start, week_end)
    sections = _cluster_sections(clusters)
    prompt = (
        f"Tu es journaliste et rédiges le résumé de l'actualité hebdomadaire en Guadeloupe "
        f"pour un digest en ligne (semaine W{week_num:02d}, {start_str} au {end_str}).\n\n"
        "Voici les principaux sujets de la semaine, classés par importance :\n\n"
        + "\n\n".join(sections)
        + "\n\nRédige un texte fluide et cohérent de 400 à 600 mots qui passe naturellement "
        "d'un sujet à l'autre. Règles strictes à respecter :\n"
        "- Ton strictement factuel et neutre. Aucune opinion, aucun jugement, aucune recommandation aux autorités ou à la population.\n"
        "- Ne rédige pas de paragraphe d'introduction générale. Commence directement par le premier sujet.\n"
        "- Termine par un court paragraphe de synthèse factuelle (\"En bref\" ou similaire), sans appel à l'action.\n"
        "- Aucun titre, sous-titre, texte en gras ou en italique, lien, référence entre parenthèses, ni section sources : uniquement des paragraphes de prose.\n"
        "- N'invente aucun fait absent des articles fournis."
    )
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def render_prose_digest(week_num, week_start, week_end, clusters, stitched):
    """Flat MD prose digest ordered by importance score with programmatic sources list."""
    start_str, end_str = _format_week_range(week_start, week_end)
    ordered = sorted(clusters, key=_CLUSTER_SORT_KEY, reverse=True)
    lines = [f"# Semaine W{week_num:02d} — {start_str} au {end_str}", ""]
    lines.append(stitched)
    lines.append("")
    lines.append("**Sources**")
    lines.append("")
    for c in ordered:
        rep = pick_representative_article(c["raw"], c["centroid"])
        lines.append(f"- [{rep['title']}]({rep['url']})")
    lines.append("")
    lines += [
        "---",
        f"*Ce digest a été généré automatiquement à partir des sujets les plus couverts "
        f"dans les flux RSS et des articles les plus lus sur RCI et France-Antilles "
        f"(semaine W{week_num:02d}, {start_str} au {end_str}).*",
    ]
    return "\n".join(lines)


def render_suggestions(week_num, scored, threshold=0.15, low_confidence_margin=0.10, ambiguity_margin=0.05):
    """Build a taxonomy review report from scored clusters."""
    UNCLASSIFIED = "Autres"
    unclassified, low_confidence, ambiguous = [], [], []

    for cluster in scored:
        rep = cluster["articles"][0]
        title = f"[{rep['title']}]({rep['url']})"
        top = cluster["top_score"]
        runner = cluster["runner_up"]
        runner_score = cluster["runner_up_score"]

        if cluster["theme"] == UNCLASSIFIED:
            unclassified.append((title, runner, top))  # runner is best theme even if below threshold
        elif top < threshold + low_confidence_margin:
            low_confidence.append((title, cluster["theme"], top, runner, runner_score))
        elif runner_score is not None and (top - runner_score) < ambiguity_margin:
            ambiguous.append((title, cluster["theme"], top, runner, runner_score))

    lines = [
        f"# Revue taxonomique — Semaine W{week_num:02d}",
        "",
        "> Ce fichier est généré automatiquement par `weekly-digest --suggest`.",
        "> Il liste les classements problématiques pour aider à affiner `data/taxonomy.toml`.",
    ]

    lines += [
        "",
        f"## Articles non classifiés ({len(unclassified)})",
        f"Aucun thème n'a atteint le seuil de confiance ({threshold:.2f}). "
        "Envisager un nouveau thème ou ajouter des exemples dans `data/themes.json`.",
        "",
    ]
    if unclassified:
        lines.append("| Titre | Thème le plus proche | Score |")
        lines.append("|---|---|---|")
        for title, best_theme, score in unclassified:
            lines.append(f"| {title} | {best_theme or '—'} | {score:.2f} |")
    else:
        lines.append("_Aucun article non classifié._")

    lines += [
        "",
        f"## Classements à faible confiance ({len(low_confidence)})",
        f"Classifiés mais avec un score entre {threshold:.2f} et {threshold + low_confidence_margin:.2f}. "
        "Renforcer la description du thème gagnant pourrait améliorer la précision.",
        "",
    ]
    if low_confidence:
        lines.append("| Titre | Thème retenu | Score | Thème concurrent | Score concurrent |")
        lines.append("|---|---|---|---|---|")
        for title, theme, score, runner, runner_score in low_confidence:
            rs = f"{runner_score:.2f}" if runner_score is not None else "—"
            lines.append(f"| {title} | {theme} | {score:.2f} | {runner or '—'} | {rs} |")
    else:
        lines.append("_Aucun classement à faible confiance._")

    lines += [
        "",
        f"## Classements ambigus ({len(ambiguous)})",
        f"Écart < {ambiguity_margin:.2f} entre les deux premiers thèmes. "
        "Différencier les descriptions de ces thèmes réduirait l'ambiguïté.",
        "",
    ]
    if ambiguous:
        lines.append("| Titre | Thème 1 | Score 1 | Thème 2 | Score 2 |")
        lines.append("|---|---|---|---|---|")
        for title, theme, score, runner, runner_score in ambiguous:
            lines.append(f"| {title} | {theme} | {score:.2f} | {runner} | {runner_score:.2f} |")
    else:
        lines.append("_Aucun classement ambigu._")

    return "\n".join(lines)


@click.command()
@click.option("--data-dir", default="data", show_default=True, help="Directory containing daily feed files")
@click.option("--output-dir", default="data", show_default=True, help="Directory to write the weekly digest")
@click.option("--week", default=None, type=int, help="ISO week number (default: current week)")
@click.option("--year", default=None, type=int, help="Year for --week (default: current year)")
@click.option("--taxonomy", default="data/taxonomy.toml", show_default=True, help="Path to taxonomy TOML config")
@click.option("--top-per-theme", default=2, show_default=True, help="Max clusters per theme used for prose")
@click.option("--suggest", is_flag=True, help="Write a taxonomy review report alongside the digest")
@click.option("--min-days", default=7, show_default=True, help="Minimum number of daily feed files required before generating")
def main(data_dir, output_dir, week, year, taxonomy, top_per_theme, suggest, min_days):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise click.ClickException("MISTRAL_API_KEY is required.")
    from mistralai.client import Mistral
    mistral_client = Mistral(api_key=api_key)

    today = datetime.now()
    iso = today.isocalendar()
    week_num = week if week is not None else iso.week
    year_num = year if year is not None else iso.year

    week_start = datetime.fromisocalendar(year_num, week_num, 1)
    week_end = datetime.fromisocalendar(year_num, week_num, 7)

    data_path = Path(data_dir)
    articles = []
    days_found = 0
    for day_offset in range(7):
        day = week_start + timedelta(days=day_offset)
        feed_file = data_path / f"feed-{day.strftime('%Y-%m-%d')}.md"
        if feed_file.exists():
            days_found += 1
            day_articles = parse_feed_file(feed_file)
            logging.info("Loaded %d articles from %s", len(day_articles), feed_file.name)
            articles.extend(day_articles)

    if days_found < min_days:
        logging.info(
            "Only %d/%d daily files found for W%02d — need %d before generating. Skipping.",
            days_found, 7, week_num, min_days,
        )
        return

    if not articles:
        logging.info("No articles found for week W%02d.", week_num)
        return

    logging.info("Total articles to cluster: %d", len(articles))

    logging.info("Fetching most-read signals…")
    most_read_paths = get_most_read_urls()
    logging.info("Found %d most-read paths.", len(most_read_paths))

    model = SentenceTransformer("BAAI/bge-m3")
    try:
        theme_names = load_taxonomy(taxonomy)
        head = load_classifier_head()
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    model_e5 = load_e5_model()

    logging.info("Clustering articles…")
    raw_clusters = cluster_articles(articles, model)
    logging.info("Found %d clusters.", len(raw_clusters))

    scored = []
    for raw_cluster in raw_clusters:
        score = score_cluster(raw_cluster, most_read_paths)
        centroid = representative_embedding(raw_cluster)
        rep = pick_representative_article(raw_cluster, centroid)
        rep_text = f"{rep['title']}. {rep['summary']}"
        cls_embedding = encode_for_classification(rep_text, model, model_e5)
        classification = classify_article_scored(cls_embedding, head)

        most_read_tags = set()
        for item in raw_cluster:
            if urlparse(item["article"]["url"]).path in most_read_paths:
                most_read_tags.add(item["article"]["source"])

        scored.append({
            "raw": raw_cluster,
            "centroid": centroid,
            "score": score,
            "theme": classification["theme"],
            "top_score": classification["top_score"],
            "runner_up": classification["runner_up"],
            "runner_up_score": classification["runner_up_score"],
            "most_read_tags": most_read_tags,
            "articles": sorted(
                [item["article"] for item in raw_cluster],
                key=lambda a: a["date"],
                reverse=True,
            ),
        })

    clusters_by_theme = {}
    for cluster in sorted(scored, key=lambda c: c["score"], reverse=True):
        clusters_by_theme.setdefault(cluster["theme"], []).append(cluster)

    ordered_themes = list(theme_names) + ["Autres"]
    clusters_to_render = []
    for theme in ordered_themes:
        top = sorted(clusters_by_theme.get(theme, []), key=_CLUSTER_SORT_KEY, reverse=True)[:top_per_theme]
        clusters_to_render.extend(top)

    logging.info("Generating stitched narrative for %d clusters via Mistral…", len(clusters_to_render))
    stitched_text = generate_stitched_narrative(clusters_to_render, week_num, week_start, week_end, mistral_client)

    prose_text = render_prose_digest(week_num, week_start, week_end, clusters_to_render, stitched_text)
    prose_file = Path(output_dir) / f"weekly-w{week_num:02d}-prose.md"
    try:
        prose_file.write_text(prose_text)
    except OSError as e:
        raise click.ClickException(f"Could not write prose digest to '{prose_file}': {e}") from e
    logging.info("Prose digest written to %s", prose_file)

    if suggest:
        review = render_suggestions(week_num, scored)
        review_file = Path(output_dir) / f"weekly-w{week_num:02d}-review.md"
        try:
            review_file.write_text(review)
        except OSError as e:
            raise click.ClickException(f"Could not write review to '{review_file}': {e}") from e
        logging.info("Taxonomy review written to %s", review_file)


if __name__ == "__main__":
    main()
