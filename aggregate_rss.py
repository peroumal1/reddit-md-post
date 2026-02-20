from datetime import datetime
from time import mktime

import click
import feedparser
from bs4 import BeautifulSoup
from py_markdown_table.markdown_table import markdown_table
from sentence_transformers import SentenceTransformer


# compare similarities between a string and a list
# model is passed as an argument to avoid re-instanciating it
# on every call
def is_not_similar(model, string, list2):
    # if the list to which we want to compare is empty there is no similarities
    list1 = [string]
    if not list2:
        return True
    # Compute embeddings for both lists

    embeddings1 = model.encode(list1)
    embeddings2 = model.encode(list2)

    # Compute cosine similarities
    similarities = model.similarity(embeddings1, embeddings2)

    # Output the pairs with their score
    for idx_i, sentence1 in enumerate(list1):
        for idx_j, sentence2 in enumerate(list2):
            if similarities[idx_i][idx_j] > 0.6:
                print("Similarity detected!!")
                print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")
                # if we have a similarity score higher than 0.6 we have at least one
                # similar entry so we can exit safely
                return False
    # if we processed the whole list without a high similarity score we can exit
    return True


def set_last_run_date():
    with open(".last-run", "w") as f:
        last_run_date = datetime.today()
        f.write(last_run_date.strftime("%d-%b-%Y (%H:%M:%S.%f)"))


def get_last_run_date():
    try:
        with open(".last-run", "r") as f:
            last_run_date = datetime.strptime(f.read(), "%d-%b-%Y (%H:%M:%S.%f)")

    except FileNotFoundError:
        last_run_date = datetime.today().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    except ValueError:
        last_run_date = datetime.today().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    return last_run_date


def transform_list(list_sorted):
    list = []
    for item in list_sorted:
        list.append(
            {
                "Titre": f"[{item['title']}]({item['link']})",
                "Résumé": item["summary"],
                "Date de publication": item["published_date"],
            }
        )

    return list


# cleaning the HTML within the summaries
def clean_text(html_blob):
    soup = BeautifulSoup(html_blob, features="html.parser")
    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text_multilines = "\n".join(chunk for chunk in chunks if chunk).splitlines()
    # cleaning for Karibinfo. The second paragraph contains l'article ... KARIBINFO
    # so we clean it
    text = text_multilines[0]
    return text


@click.command()
@click.argument("rss_links", default="data/rss_list.txt")
@click.argument("feed_output", default="data/feed.md")
def main(rss_links, feed_output):
    date_midnight = get_last_run_date()
    feed_list = []

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    # TODO: "last run variable"
    with open(rss_links) as rss_list:
        for line in rss_list:
            d = feedparser.parse(line)
            for entry in d.entries:
                feed_date = datetime.fromtimestamp(mktime(entry.published_parsed))
                # filter and only display the entries that were created today
                if feed_date > date_midnight:
                    summaries = [d["summary"] for d in feed_list]
                    # add only entries that are not similar to others
                    if is_not_similar(model, entry.summary_detail.value, summaries):
                        feed_list.append(
                            {
                                "published_date": feed_date,
                                "title": entry.title,
                                "summary": clean_text(entry.summary_detail.value),
                                "link": entry.link,
                                "media_content": entry.get("media_content", ""),
                            }
                        )
    # sort from most recent to older
    sorted_list = sorted(feed_list, key=lambda item: item["published_date"])
    sorted_list.reverse()
    new_list = transform_list(sorted_list)
    if not new_list:
        print("No new entries!")
    else:
        # generate a markdown table and write it to a file
        markdown = (
            markdown_table(new_list)
            .set_params(row_sep="markdown", quote=False)
            .get_markdown()
        )

        with open(feed_output, "w") as f:
            f.write(markdown)

    set_last_run_date()


if __name__ == "__main__":
    main()
