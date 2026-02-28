from urllib.parse import urlsplit

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
