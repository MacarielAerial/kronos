from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from kronos.data_interfaces.node_dfs_data_interface import NodeDFsDataInterface
from kronos.data_interfaces.text_emb_local_data_interface import (
    TextEmbLocalDataInterface,
)
from kronos.nodes.vectorise_text_feats import _vectorise_with_sentence_transformer


def vectorise_with_sent_tx_local(
    path_semantics_node_dfs: Path, path_sentence_transformer: Path, path_text_emb: Path
) -> None:
    # Data Access - Input
    semantics_node_dfs_data_interface = NodeDFsDataInterface(
        filepath=path_semantics_node_dfs
    )
    semantics_node_dfs = semantics_node_dfs_data_interface.load()

    sentence_transformer = SentenceTransformer(
        model_name_or_path=path_sentence_transformer.resolve()
    )

    # Task Processing
    list_text: List[str] = []
    list_emb: List[np.ndarray] = []
    for single_text, single_emb in _vectorise_with_sentence_transformer(
        node_dfs=semantics_node_dfs, sentence_transformer=sentence_transformer
    ):
        list_text.append(single_text)
        list_emb.append(single_emb)
    text = np.array(list_text)
    emb = np.array(list_emb)

    # Data Access - Output
    text_emb_local_data_interface = TextEmbLocalDataInterface(filepath=path_text_emb)
    text_emb_local_data_interface.save(text=text, emb=emb)


if __name__ == "__main__":
    import argparse

    from kronos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Vectorises text features of nodes of selected node types "
        "with their sentence embeddings"
    )
    parser.add_argument(
        "-psnd",
        "--path_semantics_node_dfs",
        type=Path,
        required=True,
        help="Path from which node dataframes with semantics features are loaded",
    )
    parser.add_argument(
        "-pst",
        "--path_sentence_transformer",
        type=Path,
        required=True,
        help="Path to a direcotry from which a sentence transformer is instantiated",
    )
    parser.add_argument(
        "-pte",
        "--path_text_emb",
        type=Path,
        required=True,
        help="Path to which text array and embedding array are saved",
    )

    args = parser.parse_args()

    vectorise_with_sent_tx_local(
        path_semantics_node_dfs=args.path_semantics_node_dfs,
        path_sentence_transformer=args.path_sentence_transformer,
        path_text_emb=args.path_text_emb,
    )
