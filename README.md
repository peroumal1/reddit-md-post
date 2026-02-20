# Running the scripts

## Python and dependencies

mise is used for the Python setup in the folder (python and pdm)
- install mise: https://mise.jdx.dev/installing-mise.html
- in the folder run `mise install` to install the requirements
- run `mise use` to activate the requirements 

Once mise is installed run `pdm add` to add all the python dependencies

## Running the aggregation script

`pdm run aggregate_rss.py [rss_file] [output_md_file]`

The aggregation script will read the RSS feeds on the links provided in `rss file`
and will aggregate them in a Markdown table which will be output in `output_md_file`.
Both parameters are optional, as this will default to resp. `data/rss_list.txt` and
`data/feed.md` if the script is called without arguments.

When ran, the aggregation script will also try and detect similarities in summaries,
to exclude similar links and avoid duplication: this is done thanks to a sentence transformer
which is downloaded locally whenever called for the first time.

Last, the date of last execution is stored locally in `.last-run` to avoid re-fetching multiple
times the same links from the RSS feeds.


## Running the post_to_reddit script

XXX TODO
