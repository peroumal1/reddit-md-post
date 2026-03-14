import logging
from datetime import datetime
from pathlib import Path
from time import mktime

import click
import feedparser
from py_markdown_table.markdown_table import markdown_table
from sentence_transformers import SentenceTransformer

from rss_summary.classification import classify_article, encode_themes, load_classifier_head, load_taxonomy
from rss_summary.formatting import format_feed_entries, format_feed_entries_classified
from rss_summary.last_run import get_last_run_date, restore_last_run_date, set_last_run_date
from rss_summary.parsing import extract_first_paragraph, get_default_image_link
from rss_summary.similarity import encode_text, is_duplicate, title_is_duplicate


@click.command()
@click.argument("rss_links", default="data/rss_list.txt")
@click.argument("feed_output", default="data/feed.md")
@click.option("--with-images", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Run without updating .last-run")
@click.option("--restore", is_flag=True, help="Restore .last-run from backup and exit")
@click.option("--until", default=None, help="Upper date bound for articles (ISO format: YYYY-MM-DD HH:MM:SS)")
@click.option("--classify", is_flag=True, help="Group output by thematic taxonomy")
@click.option("--taxonomy", default="data/taxonomy.toml", show_default=True, help="Path to taxonomy TOML config")
def main(rss_links, feed_output, with_images, dry_run, restore, until, classify, taxonomy):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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

    model = SentenceTransformer("BAAI/bge-m3")

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
                feed_date = datetime.fromtimestamp(mktime(entry.published_parsed))
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
                themes = load_taxonomy(taxonomy)
            except FileNotFoundError:
                raise click.ClickException(f"Taxonomy file not found: {taxonomy}")
            theme_names = [t["name"] for t in themes]
            theme_embeddings = encode_themes(model, themes)
            head = load_classifier_head()
            if head:
                logging.info("Using trained classifier head for classification.")
            for item in sorted_list:
                item["theme"] = classify_article(model, item["embedding"], theme_embeddings, theme_names, head=head)
            markdown = format_feed_entries_classified(sorted_list, theme_names, with_images)
        else:
            rows = format_feed_entries(sorted_list, with_images)
            markdown = (
                markdown_table(rows)
                .set_params(row_sep="markdown", quote=False)
                .get_markdown()
            )
        try:
            Path(feed_output).write_text(markdown)
        except OSError as e:
            raise click.ClickException(f"Could not write feed to '{feed_output}': {e}") from e

    if not dry_run:
        set_last_run_date()


if __name__ == "__main__":
    main()
