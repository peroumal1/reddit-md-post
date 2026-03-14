import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlsplit

import requests
from bs4 import BeautifulSoup


def strip_html(html_blob: str) -> str:
    return BeautifulSoup(html_blob, features="html.parser").get_text()


def extract_first_paragraph(html_blob):
    """Extract the first paragraph of text from an HTML blob."""
    text = strip_html(html_blob)
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    paragraphs = "\n".join(chunk for chunk in chunks if chunk).splitlines()
    return paragraphs[0] if paragraphs else ""


def parse_daily_feed_md(path):
    """Parse a daily feed markdown table into a list of article dicts.

    Returns a list of dicts with keys: title, url, summary, date, source.
    Skips header, separator, and section-header rows automatically.
    """
    articles = []
    lines = Path(path).read_text().splitlines()
    for line in lines:
        line = line.strip()
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
        host = urlparse(url).hostname or ""
        articles.append({
            "title": title,
            "url": url,
            "summary": summary,
            "date": date,
            "source": host,
        })
    return articles


def get_default_image_link(resource, origin_link):
    img_link = resource.get("media_content", None)
    if not img_link:
        try:
            r = requests.get(origin_link)
            if not r.ok:
                return [{"url": ""}]
            soup = BeautifulSoup(r.text, features="html.parser")
            if soup.img:
                default_img = soup.img.get("src")
                if default_img:
                    o = urlsplit(origin_link)
                    default_url = o._replace(path=default_img).geturl()
                    return [{"url": default_url}]
        except requests.RequestException:
            pass
        return [{"url": ""}]
    return img_link
