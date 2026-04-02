from py_markdown_table.markdown_table import markdown_table as _markdown_table

from rss_summary.classification import UNCLASSIFIED


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


def format_feed_entries_classified(entries, theme_names, with_images=False):
    """Render entries grouped by theme as a markdown document with section headers."""
    ordered_themes = list(theme_names) + [UNCLASSIFIED]
    by_theme = {}
    for item in entries:
        by_theme.setdefault(item.get("theme", UNCLASSIFIED), []).append(item)

    sections = []
    for theme in ordered_themes:
        theme_entries = by_theme.get(theme, [])
        if not theme_entries:
            continue
        rows = format_feed_entries(theme_entries, with_images)
        table = (
            _markdown_table(rows)
            .set_params(row_sep="markdown", quote=False)
            .get_markdown()
        )
        sections.append(f"## {theme}\n\n{table}")

    return "\n\n".join(sections)
