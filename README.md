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
  classification.py       # thematic classification via trained LinearSVC head
  formatting.py           # markdown table generation
  parsing.py              # HTML parsing and image extraction
  last_run.py             # .last-run timestamp persistence
classifier/
  train.py                # offline: train LinearSVC head on data/themes.json
  infer.py                # offline: batch classify a daily feed file for evaluation
tests/
data/
  rss_list.txt            # RSS feed URLs (one per line)
  taxonomy.toml           # ordered theme names (10 themes) used for digest section ordering
  themes.json             # labeled training examples (one per theme)
  classifier_head.joblib  # trained classifier head (committed, ~800KB)
  classifier_eval.json    # last cross-validation evaluation results
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

**Classification** (enabled with `--classify`): uses a trained LinearSVC head on concatenated `BAAI/bge-m3` + `multilingual-e5-large-instruct` embeddings (2048-dim, ~81% accuracy, 10 themes). The head is stored in `data/classifier_head.joblib` and committed — no retraining needed on first clone.

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

---

## Retraining the classifier

The trained head is committed and ready to use. Retrain only when you update `data/themes.json`.

`data/taxonomy.toml` controls the order of sections in the weekly digest output — changing it does **not** require retraining.

**1. Update training examples** (`data/themes.json`) — add labeled examples for any new or changed theme. Format:

```json
[
  {
    "theme": "Theme display name",
    "label": "theme_slug",
    "description": "...",
    "examples": ["Article title | Summary excerpt", ...]
  }
]
```

**2. Train:**

```zsh
pdm run python classifier/train.py
# outputs: data/classifier_head.joblib, data/classifier_eval.json
```

**3. Evaluate on a real feed file:**

```zsh
pdm run python classifier/infer.py data/feed-YYYY-MM-DD.md
```

**4. Commit the updated head:**

```zsh
git add data/classifier_head.joblib data/classifier_eval.json data/themes.json
git commit -m "feat(classifier): retrain with updated examples"
```

The head is ~800KB and safe to commit. Both `BAAI/bge-m3` and `multilingual-e5-large-instruct` are frozen encoders — only the LinearSVC head is trained. Both models are cached in CI.

---

## Tests

```zsh
pdm run pytest
pdm run pytest --cov=rss_summary   # with coverage report
```
