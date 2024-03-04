from pathlib import Path
from typing import List

import numpy as np

from kronos.data_interfaces.node_dfs_data_interface import NodeDFsDataInterface
from kronos.data_interfaces.spacy_pipeline_data_interface import (
    SpacyPipelineDataInterface,
)
from kronos.data_interfaces.text_emb_local_data_interface import (
    TextEmbLocalDataInterface,
)
from kronos.nodes.vectorise_text_feats import _vectorise_with_word_vector


def vectorise_with_word_vector_local(
    path_semantics_node_dfs: Path, path_spacy_pipeline: Path, path_text_emb: Path
) -> None:
    # Data Access - Input
    semantics_node_dfs_data_interface = NodeDFsDataInterface(
        filepath=path_semantics_node_dfs
    )
    semantics_node_dfs = semantics_node_dfs_data_interface.load()

    spacy_pipeline_data_interface = SpacyPipelineDataInterface(
        filepath=path_spacy_pipeline
    )
    spacy_pipeline = spacy_pipeline_data_interface.load()

    # Task Processing
    list_text: List[str] = []
    list_emb: List[np.ndarray] = []
    for single_text, single_emb in _vectorise_with_word_vector(
        node_dfs=semantics_node_dfs, spacy_pipeline=spacy_pipeline
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
        "with their average word vectors"
    )
    parser.add_argument(
        "-psnd",
        "--path_semantics_node_dfs",
        type=Path,
        required=True,
        help="Path from which node dataframes with semantics features are loaded",
    )
    parser.add_argument(
        "-psp",
        "--path_spacy_pipeline",
        type=Path,
        required=True,
        help="Path to a direcotry from which a spacy pipeline is instantiated",
    )
    parser.add_argument(
        "-pte",
        "--path_text_emb",
        type=Path,
        required=True,
        help="Path to which text array and embedding array are saved",
    )

    args = parser.parse_args()

    vectorise_with_word_vector_local(
        path_semantics_node_dfs=args.path_semantics_node_dfs,
        path_spacy_pipeline=args.path_spacy_pipeline,
        path_text_emb=args.path_text_emb,
    )
