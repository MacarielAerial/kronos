from pathlib import Path

from kronos.data_interfaces.edge_dfs_data_interface import EdgeDFsDataInterface
from kronos.data_interfaces.node_dfs_data_interface import NodeDFsDataInterface
from kronos.nodes.contract_sheet_cell_nodes import _contract_sheet_cell_nodes


def contract_sheet_cell_nodes(
    path_layout_node_dfs: Path,
    path_layout_edge_dfs: Path,
    path_contracted_node_dfs: Path,
    path_contracted_edge_dfs: Path,
) -> None:
    # Data Access - Input
    layout_node_dfs_data_interface = NodeDFsDataInterface(filepath=path_layout_node_dfs)
    layout_node_dfs = layout_node_dfs_data_interface.load()
    layout_node_dfs.validate()

    layout_edge_dfs_data_interface = EdgeDFsDataInterface(filepath=path_layout_edge_dfs)
    layout_edge_dfs = layout_edge_dfs_data_interface.load()
    layout_edge_dfs.validate()

    # Task Processing
    contracted_node_dfs, contracted_edge_dfs = _contract_sheet_cell_nodes(
        node_dfs=layout_node_dfs, edge_dfs=layout_edge_dfs
    )

    # Data Access - Output
    contracted_node_dfs.validate()
    contracted_node_dfs.report()

    contracted_edge_dfs.validate()
    contracted_edge_dfs.report()

    contracted_node_dfs_data_interface = NodeDFsDataInterface(
        filepath=path_contracted_node_dfs
    )
    contracted_node_dfs_data_interface.save(node_dfs=contracted_node_dfs)

    contracted_edge_dfs_data_interface = EdgeDFsDataInterface(
        filepath=path_contracted_edge_dfs
    )
    contracted_edge_dfs_data_interface.save(edge_dfs=contracted_edge_dfs)


if __name__ == "__main__":
    import argparse

    from kronos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Contracts sheet cell nodes based on common text attribute value"
    )
    parser.add_argument(
        "-plnd",
        "--path_layout_node_dfs",
        type=Path,
        required=True,
        help="Path from which layout node dataframes are loaded",
    )
    parser.add_argument(
        "-pled",
        "--path_layout_edge_dfs",
        type=Path,
        required=True,
        help="Path from which layout edge dataframes are loaded",
    )
    parser.add_argument(
        "-pcnd",
        "--path_contracted_node_dfs",
        type=Path,
        required=True,
        help="Path to which contracted node dataframes are saved",
    )
    parser.add_argument(
        "-pced",
        "--path_contracted_edge_dfs",
        type=Path,
        required=True,
        help="Path to which contracted edge dataframes are saved",
    )

    args = parser.parse_args()

    contract_sheet_cell_nodes(
        path_layout_node_dfs=args.path_layout_node_dfs,
        path_layout_edge_dfs=args.path_layout_edge_dfs,
        path_contracted_node_dfs=args.path_contracted_node_dfs,
        path_contracted_edge_dfs=args.path_contracted_edge_dfs,
    )
