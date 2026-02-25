from datetime import datetime
from pathlib import Path
from time import mktime
from urllib.parse import urlsplit

import click
import feedparser
import requests
from bs4 import BeautifulSoup
from py_markdown_table.markdown_table import markdown_table
from sentence_transformers import SentenceTransformer

LAST_RUN_FILE = Path(".last-run")
DATE_FMT = "%d-%b-%Y (%H:%M:%S.%f)"
SIMILARITY_THRESHOLD = 0.6


def is_duplicate(model, text, existing_summaries, threshold=SIMILARITY_THRESHOLD):
    """Check if text is semantically similar to any entry in existing_summaries."""
    if not existing_summaries:
        return False
    embeddings = model.encode([text])
    existing_embeddings = model.encode(existing_summaries)
    similarities = model.similarity(embeddings, existing_embeddings)
    max_score = similarities[0].max().item()
    if max_score > threshold:
        print(f"Similarity detected (score: {max_score:.4f}), skipping duplicate.")
    return max_score > threshold


def set_last_run_date():
    LAST_RUN_FILE.write_text(datetime.today().strftime(DATE_FMT))


def get_last_run_date():
    try:
        return datetime.strptime(LAST_RUN_FILE.read_text(), DATE_FMT)
    except (FileNotFoundError, ValueError):
        return datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)


def format_feed_entries(entries, with_images=False):
    """Transform feed entries into a list of dicts ready for markdown table rendering."""
    rows = []
    for item in entries:
        row = {
            "Titre": f"[{item['title']}]({item['link']})",
            "Résumé": item["summary"],
            "Date de publication": item["published_date"],
        }
        if with_images:
            row["Aperçu"] = f"![media]({item['media_content'][0]['url']})"
        rows.append(row)
    return rows


def get_default_image_link(resource, origin_link):
    img_link = resource.get("media_content", None)
    if not img_link:
        try:
            r = requests.get(origin_link)
            soup = BeautifulSoup(r.text, features="html.parser")
            if soup.img:
                o = urlsplit(origin_link)
                default_img = soup.img.get("src")
                default_url = o._replace(path=default_img).geturl()
                return [{"url": default_url}]
        except requests.RequestException:
            pass
        return [{"url": ""}]
    return img_link


def extract_first_paragraph(html_blob):
    """Extract the first paragraph of text from an HTML blob."""
    soup = BeautifulSoup(html_blob, features="html.parser")
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    paragraphs = "\n".join(chunk for chunk in chunks if chunk).splitlines()
    return paragraphs[0] if paragraphs else ""


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
