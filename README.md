# Running the scripts

## Python and dependencies

mise is used for the Python setup in the folder (python and pdm)
- install mise: https://mise.jdx.dev/installing-mise.html
- in the folder run `mise install` to install the requirements
- run `mise use` to activate the requirements 

Once mise is installed run `pdm add` to add all the python dependencies

## Running the aggregation script

`pdm run aggregate_rss.py [rss_file] [output_md_file] --with-images`

The aggregation script reads RSS feeds from the links provided in `rss_file`
and aggregates them into a Markdown table written to `output_md_file`.
Both parameters are optional, defaulting to `data/rss_list.txt` and
`data/feed.md` respectively.

### Duplicate detection

The script uses a sentence transformer model
(`paraphrase-multilingual-mpnet-base-v2`) to detect semantically similar
summaries and exclude duplicates. The model is downloaded locally on first run.
The similarity threshold is configurable via the `SIMILARITY_THRESHOLD` constant
(default: `0.6`).

### Images

The `--with-images` flag adds a preview column to the output table. It uses the
`media_content` from the RSS feed when available, or falls back to the first
`<img>` found on the article page.

### Last run tracking

The date of last execution is stored in `.last-run` so that only entries
published since the previous run are fetched. If the file is missing or
corrupted, the script defaults to midnight of the current day.


## Running the post_to_reddit script

XXX TODO
