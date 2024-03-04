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
    poetry run python -m kronos.pipelines.df_to_layout_graph -ptd data/01_raw/timetable_df.csv/ -pnd data/02_intermediate/layout_node_dfs.json -ped data/02_intermediate/layout_edge_dfs.json
    ```

3. (Optional) Construct a layout graph from parsed graph elements
    Networkx graphs are inefficient data structures that are only used for analysis.

    ```sh
    poetry run python -m kronos.pipelines.assemble_kg -pnd data/02_intermediate/layout_node_dfs.json -ped data/02_intermediate/layout_edge_dfs.json -png data/03_primary/layout_nx_g.json
    ```

4. Contract sheet cell nodes with identical text

    ```sh
    poetry run python -m kronos.pipelines.contract_sheet_cell_nodes -plnd data/02_intermediate/layout_node_dfs.json -pled data/02_intermediate/layout_edge_dfs.json -pcnd data/02_intermediate/contracted_node_dfs.json -pced data/02_intermediate/contracted_edge_dfs.json
    ```

5. (Optional) Construct a contracted graph from parsed graph elements

    ```sh
    poetry run python -m kronos.pipelines.assemble_kg -pnd data/02_intermediate/contracted_node_dfs.json -ped data/02_intermediate/contracted_edge_dfs.json -png data/03_primary/contracted_nx_g.json
    ```

6. Augments layout graph elements with semantics features

    ```sh
    poetry run python -m kronos.pipelines.add_nlp_feats -plnd data/02_intermediate/contracted_node_dfs.json -pled data/02_intermediate/contracted_edge_dfs.json -psp local_dependencies/en_core_web_lg-3.7.1/en_core_web_lg/en_core_web_lg-3.7.1/ -psnd data/02_intermediate/semantics_node_dfs.json -psed data/02_intermediate/semantics_edge_dfs.json 
    ```

7. (Optional) Construct a semantics graph from parsed graph elements

    ```sh
    poetry run python -m kronos.pipelines.assemble_kg -pnd data/02_intermediate/semantics_node_dfs.json -ped data/02_intermediate/semantics_edge_dfs.json -png data/03_primary/semantics_nx_g.json
    ```

8. Vectorise text features of certain types of nodes with average word vectors

    ```sh
    poetry run python -m kronos.pipelines.vectorise_with_word_vector_local -psnd data/02_intermediate/semantics_node_dfs.json -psp local_dependencies/model-best/ -pte data/04_feature/word_vec_emb.npz
    ```

9. Upload average word vector embedding to a database

    ```sh
    poetry run python -m kronos.pipelines.emb_to_db -pte data/04_feature/word_vec_emb.npz -cn Word
    ```

10. Vectorise text features of certain types of nodes with sentence transformer embeddings

    ```sh
    poetry run python -m kronos.pipelines.vectorise_with_sent_tx_local -psnd data/02_intermediate/semantics_node_dfs.json -pst local_dependencies/all-MiniLM-L6-v2/ -pte data/04_feature/sent_tx_emb.npz
    ```

11. Upload sentence transformer embedding to a database

    ```sh
    poetry run python -m kronos.pipelines.emb_to_db -pte data/04_feature/sent_tx_emb.npz -cn Sentence
    ```
