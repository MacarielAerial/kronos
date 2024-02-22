from pathlib import Path

from kronos.data_interfaces.edge_dfs_data_interface import EdgeDFsDataInterface
from kronos.data_interfaces.node_dfs_data_interface import NodeDFsDataInterface
from kronos.data_interfaces.nx_g_data_interface import NXGDataInterface
from kronos.nodes.assemble_kg import _assemble_kg


def assemble_kg(path_node_dfs: Path, path_edge_dfs: Path, path_nx_g: Path) -> None:
    # Data Access - Input
    node_dfs_data_interface = NodeDFsDataInterface(filepath=path_node_dfs)
    node_dfs = node_dfs_data_interface.load()
    node_dfs.validate()

    edge_dfs_data_interface = EdgeDFsDataInterface(filepath=path_edge_dfs)
    edge_dfs = edge_dfs_data_interface.load()
    edge_dfs.validate()

    # Task Processing
    nx_g = _assemble_kg(node_dfs=node_dfs, edge_dfs=edge_dfs)

    # Data Access - Output
    nx_g_data_interface = NXGDataInterface(filepath=path_nx_g)
    nx_g_data_interface.save(nx_g=nx_g)


if __name__ == "__main__":
    import argparse

    from kronos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Assembles a networkx graph object from graph element dataframes"
    )
    parser.add_argument(
        "-pnd",
        "--path_node_dfs",
        type=Path,
        required=True,
        help="Path from which node dataframes are loaded",
    )
    parser.add_argument(
        "-ped",
        "--path_edge_dfs",
        type=Path,
        required=True,
        help="Path from which edge dataframes are loaded",
    )
    parser.add_argument(
        "-png",
        "--path_nx_g",
        type=Path,
        required=True,
        help="Path to which a constructed networkx graph is saved",
    )

    args = parser.parse_args()

    assemble_kg(
        path_node_dfs=args.path_node_dfs,
        path_edge_dfs=args.path_edge_dfs,
        path_nx_g=args.path_nx_g,
    )
