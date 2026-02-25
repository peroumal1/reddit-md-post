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
