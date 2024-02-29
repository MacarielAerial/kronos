from pathlib import Path

from kronos.data_interfaces.edge_dfs_data_interface import EdgeDFsDataInterface
from kronos.data_interfaces.node_dfs_data_interface import NodeDFsDataInterface
from kronos.data_interfaces.spacy_pipeline_data_interface import (
    SpacyPipelineDataInterface,
)
from kronos.nodes.add_nlp_feats import _add_nlp_feats


def add_nlp_feats(
    path_layout_node_dfs: Path,
    path_layout_edge_dfs: Path,
    path_spacy_pipeline: Path,
    path_semantics_node_dfs: Path,
    path_semantics_edge_dfs: Path,
) -> None:
    # Data Access - Input
    layout_node_dfs_data_interface = NodeDFsDataInterface(filepath=path_layout_node_dfs)
    layout_node_dfs = layout_node_dfs_data_interface.load()
    layout_node_dfs.validate()

    layout_edge_dfs_data_interface = EdgeDFsDataInterface(filepath=path_layout_edge_dfs)
    layout_edge_dfs = layout_edge_dfs_data_interface.load()
    layout_node_dfs.validate()

    spacy_pipeline_data_interface = SpacyPipelineDataInterface(
        filepath=path_spacy_pipeline
    )
    spacy_pipeline = spacy_pipeline_data_interface.load(
        enable=["tok2vec", "ner"]  # NER component cannot work without tok2vec
    )

    # Task Processing
    semantics_node_dfs, semantics_edge_dfs = _add_nlp_feats(
        node_dfs=layout_node_dfs,
        edge_dfs=layout_edge_dfs,
        spacy_pipeline=spacy_pipeline,
    )

    # Data Access - Output
    semantics_node_dfs.validate()
    semantics_node_dfs.report()

    semantics_edge_dfs.validate()
    semantics_edge_dfs.report()

    semantics_node_dfs_data_interface = NodeDFsDataInterface(
        filepath=path_semantics_node_dfs
    )
    semantics_node_dfs_data_interface.save(node_dfs=semantics_node_dfs)

    semantics_edge_dfs_data_interface = EdgeDFsDataInterface(
        filepath=path_semantics_edge_dfs
    )
    semantics_edge_dfs_data_interface.save(edge_dfs=semantics_edge_dfs)


if __name__ == "__main__":
    import argparse

    from kronos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Leverages a spacy pipeline to parse semantics "
        "graph elements from a layout graph"
    )
    parser.add_argument(
        "-plnd",
        "--path_layout_node_dfs",
        type=Path,
        required=True,
        help="Path from which node dataframes of a layout graph are loaded",
    )
    parser.add_argument(
        "-pled",
        "--path_layout_edge_dfs",
        type=Path,
        required=True,
        help="Path from which edge dataframes of a layout graph are loaded",
    )
    parser.add_argument(
        "-psp",
        "--path_spacy_pipeline",
        type=Path,
        required=True,
        help="Path from which a spacy pipeline is loaded",
    )
    parser.add_argument(
        "-psnd",
        "--path_semantics_node_dfs",
        type=Path,
        required=True,
        help="Path to which node dataframes of a semantics graph are saved",
    )
    parser.add_argument(
        "-psed",
        "--path_semantics_edge_dfs",
        type=Path,
        required=True,
        help="Path to which edge dataframes of a semantics graph are saved",
    )

    args = parser.parse_args()

    add_nlp_feats(
        path_layout_node_dfs=args.path_layout_node_dfs,
        path_layout_edge_dfs=args.path_layout_edge_dfs,
        path_spacy_pipeline=args.path_spacy_pipeline,
        path_semantics_node_dfs=args.path_semantics_node_dfs,
        path_semantics_edge_dfs=args.path_semantics_edge_dfs,
    )
