import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import click
import numpy as np
import requests
from bs4 import BeautifulSoup
from mistralai.client import Mistral
from sentence_transformers import SentenceTransformer

from rss_summary.classification import BGE_MODEL_ID, CLASSIFICATION_THRESHOLD, MISTRAL_MODEL, UNCLASSIFIED, batch_encode_e5, build_cls_embedding, classify_article_scored, load_classifier_head, load_e5_model, load_taxonomy
from rss_summary.parsing import format_article_text, parse_daily_feed_md
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
_FAITS_DIVERS = "Faits divers"
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
    embeddings = [encode_text(model, format_article_text(a)) for a in articles]
    sim_matrix = model.similarity(embeddings, embeddings)
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
            if sim_matrix[i][j].item() >= CLUSTER_THRESHOLD:
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
    vecs = [item["embedding"] for item in cluster]
    return np.mean(vecs, axis=0)


def pick_representative_article(raw_cluster, centroid):
    """Return the article whose embedding is closest to the cluster centroid.

    This tends to be the most generic/encompassing article rather than a
    specific local variant, making it a better section heading.
    """
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
        rep = cluster["rep"]
        theme = cluster.get("theme", "")
        theme_prefix = f"[{theme}] " if theme and theme != UNCLASSIFIED else ""
        article_lines = []
        for item in cluster["raw"]:
            a = item["article"]
            summary = a.get("summary", "").strip()[:300]
            line = f'- [{a["source"]}, {a["date"].strftime("%d/%m")}] [{a["title"]}]({a["url"]})'
            if summary:
                line += f" : {summary}"
            article_lines.append(line)
        sections.append(f"[{i}] {theme_prefix}{rep['title']}\n" + "\n".join(article_lines))
    return sections


def split_mixed_clusters(raw_clusters, model_bge, model_e5, head):
    """Split clusters that mix 'Faits divers' articles with other themes.

    After semantic clustering, articles about the same venue (e.g. a school)
    can end up together even when they cover unrelated events (an accident vs.
    a ranking). This pass classifies each article individually and splits any
    cluster where 'Faits divers' is mixed with another theme.

    Reuses bge-m3 embeddings cached in item["embedding"] from cluster_articles,
    and batch-encodes all e5 inputs in a single model call.
    """
    flat_items = []   # (cluster_idx, item) in order
    flat_texts = []
    for cluster_idx, cluster in enumerate(raw_clusters):
        if len(cluster) == 1:
            continue
        for item in cluster:
            flat_texts.append(format_article_text(item["article"]))
            flat_items.append((cluster_idx, item))

    if not flat_items:
        return raw_clusters

    e5_embs = batch_encode_e5(flat_texts, model_e5)

    article_themes = {}  # cluster_idx → [(item, theme), …] in cluster order
    for (cluster_idx, item), e5_emb in zip(flat_items, e5_embs):
        cls_emb = build_cls_embedding(item["embedding"], e5_emb)
        theme = classify_article_scored(cls_emb, head)["theme"]
        article_themes.setdefault(cluster_idx, []).append((item, theme))

    result = []
    for cluster_idx, cluster in enumerate(raw_clusters):
        if cluster_idx not in article_themes:
            result.append(cluster)
            continue

        items_and_themes = article_themes[cluster_idx]
        themes = [t for _, t in items_and_themes]
        unique_themes = set(themes)
        has_faits_divers = _FAITS_DIVERS in unique_themes
        has_other = bool(unique_themes - {_FAITS_DIVERS, UNCLASSIFIED})

        if has_faits_divers and has_other:
            by_theme = {}
            for item, theme in items_and_themes:
                by_theme.setdefault(theme, []).append(item)
            result.extend(by_theme.values())
            logging.info(
                "Split cluster '%s…' (%d articles) → %s",
                cluster[0]["article"]["title"][:50],
                len(cluster),
                list(by_theme.keys()),
            )
        else:
            result.append(cluster)

    return result


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
        "- Ne rédige pas de paragraphe de conclusion générale.\n"
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
        rep = c["rep"]
        lines.append(f"- [{rep['title']}]({rep['url']})")
    lines.append("")
    lines += [
        "---",
        f"*Ce digest a été généré automatiquement à partir des sujets les plus couverts "
        f"dans les flux RSS et des articles les plus lus sur RCI et France-Antilles "
        f"(semaine W{week_num:02d}, {start_str} au {end_str}).*",
    ]
    return "\n".join(lines)


def _cluster_status(cluster, threshold=CLASSIFICATION_THRESHOLD, low_confidence_margin=0.10, ambiguity_margin=0.05):
    """Return the problematic status of a cluster, or None if clean."""
    top = cluster["top_score"]
    runner_score = cluster["runner_up_score"]
    if cluster["theme"] == UNCLASSIFIED:
        return "unclassified"
    if top < threshold + low_confidence_margin:
        return "low_confidence"
    if runner_score is not None and (top - runner_score) < ambiguity_margin:
        return "ambiguous"
    return None


def _problematic_clusters(scored, threshold=CLASSIFICATION_THRESHOLD, low_confidence_margin=0.10, ambiguity_margin=0.05):
    """Return clusters that are unclassified, low-confidence, or ambiguous."""
    return [c for c in scored if _cluster_status(c, threshold, low_confidence_margin, ambiguity_margin) is not None]


def enrich_review_with_suggestions(problematic, theme_names, client):
    """Single Mistral call suggesting themes for problematic clusters.

    Returns (markdown_section, structured_suggestions) where each suggestion is
    a dict with keys 'theme', 'example', 'reason' (all may be None on parse failure).
    """
    if not problematic:
        return "", []

    themes_str = "\n".join(f"- {t}" for t in theme_names)
    cluster_lines = []
    for i, cluster in enumerate(problematic, 1):
        rep = cluster["articles"][0]
        summary = rep.get("summary", "").strip()[:300]
        status = ("non classifié" if cluster["theme"] == UNCLASSIFIED
                  else f"classifié '{cluster['theme']}' avec score {cluster['top_score']:.2f}")
        cluster_lines.append(
            f"[{i}]\nTitre: {rep['title']}\nRésumé: {summary}\nStatut classifieur: {status}"
        )

    prompt = (
        "Tu es expert en classification d'articles de presse pour un digest Guadeloupe.\n\n"
        "RÈGLE GÉOGRAPHIQUE (prioritaire sur le sujet) :\n"
        "- Si l'article porte principalement sur la Guyane, Martinique, La Réunion, Mayotte, "
        "Saint-Martin, Saint-Barthélemy ou tout autre territoire ultramarin non guadeloupéen → "
        "suggère 'Outre-mer & Caraïbes', même si le sujet est environnemental, politique ou économique.\n"
        "- Si l'article porte principalement sur Haïti, Cuba, la République dominicaine ou tout "
        "autre État souverain caribéen → suggère 'International', sauf si la CARICOM ou une "
        "coopération régionale impliquant un territoire français est au cœur du sujet.\n"
        "- N'assigne un thème de contenu (Eau, Politique, Économie, Santé…) "
        "que si l'article est ancré en Guadeloupe.\n\n"
        f"Thèmes disponibles :\n{themes_str}\n\n"
        "Notes de périmètre :\n"
        "- 'Eau & environnement' : eau potable, sécheresse, sargasses, pollution, "
        "biodiversité guadeloupéenne. Pas les chantiers routiers ni les opérations de secours.\n"
        "- 'Faits divers' : accidents, crimes, secours d'urgence, disparitions, incidents — "
        "même si l'élément déclencheur est naturel (inondation, noyade, montée des eaux).\n"
        "- 'Outre-mer & Caraïbes' : thème de localisation — couvre tout article dont la "
        "géographie principale est un territoire non guadeloupéen (DOM, COM, îles caribéennes).\n\n"
        "Pour chaque article ci-dessous, dont la classification automatique est incertaine, réponds :\n"
        "- Le thème le plus approprié (ou 'Nouveau thème: <nom>' si aucun ne convient)\n"
        "- Une ligne d'exemple au format themes.json : \"Titre de l'article | Extrait du résumé\"\n"
        "- Une justification courte (1 phrase)\n\n"
        "Format de réponse strict :\n"
        "[1]\nThème: <theme>\nExemple: \"<titre> | <résumé>\"\nRaison: <justification>\n\n"
        "[2]\n...\n\n"
        "=== Articles ===\n\n"
        + "\n\n".join(cluster_lines)
    )

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()

    raw_blocks = {}
    parts = re.split(r"\[(\d+)\]", text)
    for j in range(1, len(parts) - 1, 2):
        idx = int(parts[j]) - 1
        if 0 <= idx < len(problematic):
            raw_blocks[idx] = parts[j + 1].strip()

    structured = []
    for i in range(len(problematic)):
        block = raw_blocks.get(i, "")
        theme_m = re.search(r"Thème\s*:\s*(.+)", block)
        example_m = re.search(r'Exemple\s*:\s*["\u201c](.+?)["\u201d]', block)
        reason_m = re.search(r"Raison\s*:\s*(.+)", block)
        structured.append({
            "theme": theme_m.group(1).strip().strip("*") if theme_m else None,
            "example": example_m.group(1).strip() if example_m else None,
            "reason": reason_m.group(1).strip() if reason_m else None,
            "_raw": block,
        })

    section = [
        "",
        "## Suggestions Mistral",
        "",
        "> Suggestions générées automatiquement — à valider avant d'ajouter dans `data/themes.json`.",
        "",
    ]
    for i, cluster in enumerate(problematic):
        rep = cluster["articles"][0]
        section.append(f"### [{rep['title']}]({rep['url']})")
        section.append(structured[i]["_raw"] or "_(pas de suggestion disponible)_")
        section.append("")

    return "\n".join(section), structured


def apply_suggestions_to_themes(suggestions, themes_path):
    """Add Mistral-suggested examples to themes.json.

    Skips unknown themes and duplicates. Collects 'Nouveau thème' suggestions
    separately for human review.

    Returns (added, new_theme_suggestions) where new_theme_suggestions is a list
    of suggestion dicts for themes not in the current taxonomy.
    """
    with open(themes_path) as f:
        themes = json.load(f)

    theme_map = {t["theme"]: t for t in themes}
    added = 0
    new_themes = []
    for i, s in enumerate(suggestions):
        theme_name = s.get("theme") or ""
        example = s.get("example") or ""
        if not theme_name or not example:
            logging.warning("Could not parse suggestion %d — skipping.", i)
            continue
        if theme_name.startswith("Nouveau thème"):
            new_themes.append(s)
            logging.info("New theme suggested (manual action required): %s", theme_name)
            continue
        if theme_name not in theme_map:
            logging.warning("Unknown theme '%s' — skipping.", theme_name)
            continue
        if example not in set(theme_map[theme_name]["examples"]):
            theme_map[theme_name]["examples"].append(example)
            added += 1
            logging.info("Added example to '%s': %.60s…", theme_name, example)

    if added:
        with open(themes_path, "w") as f:
            json.dump(themes, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %d new examples to %s.", added, themes_path)

    return added, new_themes


def render_suggestions(week_num, scored, threshold=CLASSIFICATION_THRESHOLD, low_confidence_margin=0.10, ambiguity_margin=0.05):
    """Build a taxonomy review report from scored clusters."""
    unclassified, low_confidence, ambiguous = [], [], []

    for cluster in scored:
        status = _cluster_status(cluster, threshold, low_confidence_margin, ambiguity_margin)
        if status is None:
            continue
        rep = cluster["articles"][0]
        title = f"[{rep['title']}]({rep['url']})"
        top = cluster["top_score"]
        runner = cluster["runner_up"]
        runner_score = cluster["runner_up_score"]
        if status == "unclassified":
            unclassified.append((title, runner, top))
        elif status == "low_confidence":
            low_confidence.append((title, cluster["theme"], top, runner, runner_score))
        else:
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


def _signal_new_themes(new_themes, week_num, body_path="/tmp/new-themes-issue.md"):
    """Write issue body to a temp file and flag GITHUB_OUTPUT for the workflow."""
    lines = [
        f"Mistral a suggéré {len(new_themes)} nouveau(x) thème(s) lors du digest W{week_num:02d}.\n",
    ]
    for s in new_themes:
        lines.append(f"**{s['theme']}**")
        if s.get("example"):
            lines.append(f"Exemple : `{s['example']}`")
        if s.get("reason"):
            lines.append(f"Raison : {s['reason']}")
        lines.append("")
    lines.append(
        "Pour appliquer : ajouter le thème dans `data/taxonomy.toml` et `data/themes.json`, "
        "puis relancer `pdm run python classifier/train.py`."
    )
    Path(body_path).write_text("\n".join(lines))
    logging.info("New theme issue body written to %s.", body_path)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write("has_new_themes=true\n")


@click.command()
@click.option("--data-dir", default="data", show_default=True, help="Directory containing daily feed files")
@click.option("--output-dir", default="data", show_default=True, help="Directory to write the weekly digest")
@click.option("--week", default=None, type=int, help="ISO week number (default: current week)")
@click.option("--year", default=None, type=int, help="Year for --week (default: current year)")
@click.option("--taxonomy", default="data/taxonomy.toml", show_default=True, help="Path to taxonomy TOML config")
@click.option("--top-per-theme", default=2, show_default=True, help="Max clusters per theme used for prose")
@click.option("--suggest", is_flag=True, help="Write a taxonomy review report alongside the digest")
@click.option("--enrich-review", is_flag=True, help="With --suggest: append Mistral theme suggestions for problematic clusters")
@click.option("--apply-suggestions", is_flag=True, help="With --enrich-review: write Mistral suggestions into themes.json")
@click.option("--min-days", default=7, show_default=True, help="Minimum number of daily feed files required before generating")
def main(data_dir, output_dir, week, year, taxonomy, top_per_theme, suggest, enrich_review, apply_suggestions, min_days):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if enrich_review and not suggest:
        raise click.ClickException("--enrich-review requires --suggest.")
    if apply_suggestions and not enrich_review:
        raise click.ClickException("--apply-suggestions requires --enrich-review.")

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise click.ClickException("MISTRAL_API_KEY is required.")
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

    model = SentenceTransformer(BGE_MODEL_ID)
    try:
        theme_names = load_taxonomy(taxonomy)
        head = load_classifier_head()
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    model_e5 = load_e5_model()

    logging.info("Clustering articles…")
    raw_clusters = cluster_articles(articles, model)
    logging.info("Found %d clusters.", len(raw_clusters))

    raw_clusters = split_mixed_clusters(raw_clusters, model, model_e5, head)
    logging.info("After splitting mixed clusters: %d clusters.", len(raw_clusters))

    cluster_meta = []
    for raw_cluster in raw_clusters:
        centroid = representative_embedding(raw_cluster)
        rep = pick_representative_article(raw_cluster, centroid)
        rep_bge = next(
            (item["embedding"] for item in raw_cluster if item["article"] is rep),
            centroid,
        )
        score = score_cluster(raw_cluster, most_read_paths)
        most_read_tags = {
            item["article"]["source"]
            for item in raw_cluster
            if urlparse(item["article"]["url"]).path in most_read_paths
        }
        cluster_meta.append((raw_cluster, rep, rep_bge, score, most_read_tags))

    rep_texts = [format_article_text(rep) for _, rep, _, _, _ in cluster_meta]
    rep_e5_embs = batch_encode_e5(rep_texts, model_e5)

    scored = []
    for (raw_cluster, rep, rep_bge, score, most_read_tags), e5_emb in zip(cluster_meta, rep_e5_embs):
        cls_embedding = build_cls_embedding(rep_bge, e5_emb)
        classification = classify_article_scored(cls_embedding, head)

        scored.append({
            "raw": raw_cluster,
            "rep": rep,
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

    ordered_themes = list(theme_names) + [UNCLASSIFIED]
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
        if enrich_review:
            problematic = _problematic_clusters(scored)
            if problematic:
                logging.info("Enriching review with Mistral suggestions for %d clusters…", len(problematic))
                enrichment, structured = enrich_review_with_suggestions(problematic, theme_names, mistral_client)
                review += enrichment
                if apply_suggestions:
                    themes_path = Path(data_dir) / "themes.json"
                    _, new_themes = apply_suggestions_to_themes(structured, themes_path)
                    if new_themes:
                        _signal_new_themes(new_themes, week_num)
            else:
                logging.info("No problematic clusters — skipping Mistral enrichment.")
        review_file = Path(output_dir) / f"weekly-w{week_num:02d}-review.md"
        try:
            review_file.write_text(review)
        except OSError as e:
            raise click.ClickException(f"Could not write review to '{review_file}': {e}") from e
        logging.info("Taxonomy review written to %s", review_file)


if __name__ == "__main__":
    main()
