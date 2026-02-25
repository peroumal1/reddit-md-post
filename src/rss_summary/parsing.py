from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup


def extract_first_paragraph(html_blob):
    """Extract the first paragraph of text from an HTML blob."""
    soup = BeautifulSoup(html_blob, features="html.parser")
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    paragraphs = "\n".join(chunk for chunk in chunks if chunk).splitlines()
    return paragraphs[0] if paragraphs else ""


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
