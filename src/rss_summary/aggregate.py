import logging
import os
from datetime import datetime
from pathlib import Path

import click
import feedparser
from py_markdown_table.markdown_table import markdown_table
from sentence_transformers import SentenceTransformer

from rss_summary.classification import BGE_MODEL_ID, MISTRAL_MODEL, classify_article, encode_for_classification, load_classifier_head, load_e5_model, load_taxonomy
from rss_summary.formatting import format_feed_entries, format_feed_entries_classified
from rss_summary.last_run import get_last_run_date, restore_last_run_date, set_last_run_date
from rss_summary.parsing import extract_first_paragraph, format_article_text, get_default_image_link
from rss_summary.similarity import encode_text, is_duplicate, title_is_duplicate


def generate_daily_summary(articles, client):
    """Ask Mistral for a short prose summary of the day's articles."""
    lines = []
    for a in articles:
        summary = a.get("summary", "").strip()[:300]
        line = f"- [{a['title']}]({a['link']})"
        if summary:
            line += f" : {summary}"
        lines.append(line)

    prompt = (
        "Tu es journaliste et rédiges un bref résumé de l'actualité du jour en Guadeloupe "
        "pour un digest en ligne.\n\n"
        "Voici les articles du jour :\n\n"
        + "\n".join(lines)
        + "\n\nRédige un texte de 100 à 150 mots qui résume les principales informations du jour. "
        "Règles strictes :\n"
        "- Ton strictement factuel et neutre. Aucune opinion, aucun jugement.\n"
        "- Commence directement par les faits. Pas de phrase d'introduction générale.\n"
        "- Pas de titre, pas de liste, pas de texte en gras : uniquement de la prose.\n"
        "- Si des informations positives ou neutres sont présentes, intègre-les au texte sans te limiter aux faits les plus dramatiques.\n"
        "- N'invente aucun fait absent des articles fournis.\n"
        "- N'utilise jamais l'expression 'En Guadeloupe' dans le texte, ni en début de phrase ni en milieu : le lecteur est déjà en contexte guadeloupéen et cette précision est superflue."
    )
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


@click.command()
@click.argument("rss_links", default="data/rss_list.txt")
@click.argument("feed_output", default="data/feed.md")
@click.option("--with-images", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Run without updating .last-run")
@click.option("--restore", is_flag=True, help="Restore .last-run from backup and exit")
@click.option("--until", default=None, help="Upper date bound for articles (ISO format: YYYY-MM-DD HH:MM:SS)")
@click.option("--classify", is_flag=True, help="Group output by thematic taxonomy")
@click.option("--taxonomy", default="data/taxonomy.toml", show_default=True, help="Path to taxonomy TOML config")
@click.option("--summarize", is_flag=True, help="Prepend a Mistral-generated prose summary to the digest (requires MISTRAL_API_KEY)")
def main(rss_links, feed_output, with_images, dry_run, restore, until, classify, taxonomy, summarize):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    mistral_client = None
    if summarize:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise click.ClickException("--summarize requires MISTRAL_API_KEY to be set.")
        from mistralai.client import Mistral
        mistral_client = Mistral(api_key=api_key)

    if restore:
        restore_last_run_date()
        return

    date_midnight = get_last_run_date()
    try:
        date_until = datetime.fromisoformat(until) if until else None
    except ValueError:
        raise click.ClickException(f"Invalid --until value '{until}'. Expected ISO format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")

    feed_list = []
    seen_titles = []
    seen_embeddings = []

    model = SentenceTransformer(BGE_MODEL_ID)

    try:
        rss_list_file = open(rss_links)
    except FileNotFoundError:
        raise click.ClickException(f"RSS links file not found: {rss_links}")
    with rss_list_file as rss_list:
        for line in rss_list:
            feed = feedparser.parse(line)
            for entry in feed.entries:
                if not entry.get("published_parsed"):
                    continue
                feed_date = datetime(*entry.published_parsed[:6])
                if feed_date <= date_midnight:
                    continue
                if date_until is not None and feed_date > date_until:
                    continue
                title = entry.title
                if title_is_duplicate(title, seen_titles):
                    continue
                summary_detail = getattr(entry, "summary_detail", None)
                summary_text = summary_detail.value if summary_detail else ""
                embedding = encode_text(model, summary_text)
                if not is_duplicate(model, embedding, seen_embeddings):
                    seen_titles.append(title)
                    seen_embeddings.append(embedding)
                    feed_list.append(
                        {
                            "published_date": feed_date,
                            "title": title,
                            "summary": extract_first_paragraph(summary_text),
                            "link": entry.link,
                            "media_content": get_default_image_link(
                                entry, entry.link
                            ),
                            "embedding": embedding,
                        }
                    )

    sorted_list = sorted(feed_list, key=lambda item: item["published_date"], reverse=True)

    if not sorted_list:
        logging.info("No new entries.")
    else:
        if classify:
            try:
                theme_names = load_taxonomy(taxonomy)
                head = load_classifier_head()
            except FileNotFoundError as e:
                raise click.ClickException(str(e))
            model_e5 = load_e5_model()
            for item in sorted_list:
                cls_embedding = encode_for_classification(
                    format_article_text(item), model, model_e5
                )
                item["theme"] = classify_article(cls_embedding, head)
            markdown = format_feed_entries_classified(sorted_list, theme_names, with_images)
        else:
            rows = format_feed_entries(sorted_list, with_images)
            markdown = (
                markdown_table(rows)
                .set_params(row_sep="markdown", quote=False)
                .get_markdown()
            )

        if summarize:
            logging.info("Generating daily summary via Mistral…")
            summary_text = generate_daily_summary(sorted_list, mistral_client)
            markdown = (
                "## En bref\n\n"
                + summary_text
                + "\n\n## Plus en détails\n\n"
                + markdown
            )

        try:
            Path(feed_output).write_text(markdown)
        except OSError as e:
            raise click.ClickException(f"Could not write feed to '{feed_output}': {e}") from e

    if not dry_run:
        set_last_run_date()


if __name__ == "__main__":
    main()
