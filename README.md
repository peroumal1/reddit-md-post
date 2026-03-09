# reddit-md-post

RSS aggregator that fetches feeds, deduplicates them using semantic similarity, and generates a markdown digest posted daily to r/Guadeloupe.

## Setup

[mise](https://mise.jdx.dev/installing-mise.html) is used for tool version management (Python + PDM).

```zsh
mise install       # install Python and PDM
pdm install        # install Python dependencies
```

## Project structure

```
src/rss_summary/          # main package
  aggregate.py            # CLI: pdm run aggregate-rss
  weekly.py               # CLI: pdm run weekly-digest
  post_to_reddit.py       # CLI: pdm run post-to-reddit
  similarity.py           # semantic deduplication via sentence-transformers
  classification.py       # zero-shot thematic classification via BAAI/bge-m3
  formatting.py           # markdown table generation
  parsing.py              # HTML parsing and image extraction
  last_run.py             # .last-run timestamp persistence
tests/
data/
  rss_list.txt            # RSS feed URLs (one per line)
  taxonomy.toml           # thematic taxonomy (9 themes)
  feed.md                 # latest daily digest
  feed-YYYY-MM-DD.md      # dated archive copies
  weekly-wXX.md           # weekly digest
  weekly-wXX-review.md    # taxonomy review report
.last-run                 # last successful run timestamp (committed)
```

## Commands

### `pdm run aggregate-rss`

Fetches RSS feeds and writes a deduplicated markdown digest.

```
pdm run aggregate-rss [RSS_FILE] [OUTPUT_FILE] [OPTIONS]

Arguments (optional):
  RSS_FILE      RSS feed list         [default: data/rss_list.txt]
  OUTPUT_FILE   Output markdown file  [default: data/feed.md]

Options:
  --with-images       Add image preview column to the table
  --classify          Group output by thematic taxonomy
  --taxonomy PATH     Taxonomy TOML config  [default: data/taxonomy.toml]
  --until DATETIME    Upper date bound for articles (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
  --dry-run           Run without updating .last-run
  --restore           Restore .last-run from backup and exit
```

**Deduplication** uses a two-stage pipeline:
1. Fuzzy title match via `difflib.SequenceMatcher` (threshold 0.85)
2. Semantic similarity via `BAAI/bge-m3` (threshold 0.75)

The model is downloaded automatically on first run and cached in `~/.cache/huggingface`.

**Last-run tracking**: the date of last execution is stored in `.last-run`. Only entries published since the previous run are fetched.

---

### `pdm run weekly-digest`

Clusters and classifies a full week of daily digests into a thematic report.

```
pdm run weekly-digest [OPTIONS]

Options:
  --week INT          ISO week number              [default: current week]
  --year INT          Year for --week              [default: current year]
  --data-dir PATH     Directory with daily files   [default: data]
  --output-dir PATH   Output directory             [default: data]
  --taxonomy PATH     Taxonomy TOML config         [default: data/taxonomy.toml]
  --top-per-theme INT Max clusters per section     [default: 2]
  --min-days INT      Min daily files required     [default: 7]
  --suggest           Also write taxonomy review report
```

Requires all 7 daily `feed-YYYY-MM-DD.md` files for the target week (controlled by `--min-days`). Outputs `data/weekly-wXX.md` and, with `--suggest`, `data/weekly-wXX-review.md`.

---

### `pdm run post-to-reddit`

Posts a markdown feed to r/Guadeloupe via Playwright (Firefox). Runs locally only — not suitable for CI due to IP/fingerprint detection.

```
pdm run post-to-reddit [OPTIONS]

Options:
  --feed-file PATH   Local markdown file to post   [default: data/feed.md]
  --feed-url TEXT    URL to fetch the markdown from (e.g. raw GitHub URL)
```

`--feed-file` and `--feed-url` are mutually exclusive.

Requires a `.env` file with:

```
REDDIT_LOGIN=your_username
REDDIT_PASSWORD=your_password
REDDIT_OTP_SECRET=your_totp_secret
```

## Tests

```zsh
pdm run pytest
```
