from datetime import datetime
from pathlib import Path
from time import mktime

import click
import feedparser
from py_markdown_table.markdown_table import markdown_table
from sentence_transformers import SentenceTransformer

from rss_summary.formatting import format_feed_entries
from rss_summary.last_run import get_last_run_date, set_last_run_date
from rss_summary.parsing import extract_first_paragraph, get_default_image_link
from rss_summary.similarity import is_duplicate


@click.command()
@click.argument("rss_links", default="data/rss_list.txt")
@click.argument("feed_output", default="data/feed.md")
@click.option("--with-images", is_flag=True)
def main(rss_links, feed_output, with_images):
    date_midnight = get_last_run_date()
    feed_list = []
    seen_summaries = []

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    with open(rss_links) as rss_list:
        for line in rss_list:
            feed = feedparser.parse(line)
            for entry in feed.entries:
                feed_date = datetime.fromtimestamp(mktime(entry.published_parsed))
                if feed_date > date_midnight:
                    summary_text = entry.summary_detail.value
                    if not is_duplicate(model, summary_text, seen_summaries):
                        seen_summaries.append(summary_text)
                        feed_list.append(
                            {
                                "published_date": feed_date,
                                "title": entry.title,
                                "summary": extract_first_paragraph(summary_text),
                                "link": entry.link,
                                "media_content": get_default_image_link(
                                    entry, entry.link
                                ),
                            }
                        )

    sorted_list = sorted(feed_list, key=lambda item: item["published_date"], reverse=True)
    rows = format_feed_entries(sorted_list, with_images)

    if not rows:
        print("No new entries!")
    else:
        markdown = (
            markdown_table(rows)
            .set_params(row_sep="markdown", quote=False)
            .get_markdown()
        )
        Path(feed_output).write_text(markdown)

    set_last_run_date()


if __name__ == "__main__":
    main()
