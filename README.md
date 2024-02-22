# Kronos

## Executive Summary
Kronos is an application which aims to build the backend for an organisation intelligence platform. In the scope of this iteration, Kronos connects to various information sources, organises information into a structured format before storing the information in a database.

## Code Examples

1. Download a google sheet which contains timetable information as a csv file

```sh
poetry run python -m kronos.pipelines.download_timetable_df -psj credentials/kronos-408821-b353d5af55b8.json -ptd data/01_raw/timetable_df.csv
```

2. Parse timetable dataframe into graph elements of a layout graph
```sh
poetry run python -m kronos.pipelines.df_to_layout_graph -ptd data/01_raw/timetable_df.csv/ -pnd data/02_intermeidate/layout_node_dfs.json -ped data/02_intermediate/layout_edge_dfs.json
```
