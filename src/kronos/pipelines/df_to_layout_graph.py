from pathlib import Path

from kronos.data_interfaces.edge_dfs_data_interface import EdgeDFsDataInterface
from kronos.data_interfaces.node_dfs_data_interface import NodeDFs, NodeDFsDataInterface
from kronos.data_interfaces.timetable_df_data_interface import TimeTableDFDataInterface
from kronos.nodes.df_to_layout_graph import _df_to_layout_graph


def df_to_layout_graph(
    path_timetable_df: Path, path_node_dfs: Path, path_edge_dfs: Path
) -> None:
    # Data Access - Input
    timetable_df_data_interface = TimeTableDFDataInterface(filepath=path_timetable_df)
    timetable_df = timetable_df_data_interface.load()

    # Task Processing
    sheet_cell_node_df, traversal_edge_dfs = _df_to_layout_graph(df=timetable_df)

    # Data Access - Output
    node_dfs = NodeDFs(members=[sheet_cell_node_df])

    node_dfs.validate()
    node_dfs.report()

    traversal_edge_dfs.validate()
    traversal_edge_dfs.report()

    node_dfs_data_interface = NodeDFsDataInterface(filepath=path_node_dfs)
    node_dfs_data_interface.save(node_dfs=node_dfs)

    edge_dfs_data_interface = EdgeDFsDataInterface(filepath=path_edge_dfs)
    edge_dfs_data_interface.save(edge_dfs=traversal_edge_dfs)


if __name__ == "__main__":
    import argparse

    from kronos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Parses a timetable dataframe into a layout graph "
        "represented by their graph elements"
    )
    parser.add_argument(
        "-ptd",
        "--path_timetable_df",
        required=True,
        type=Path,
        help="Path from which the timetable dataframe is loaded",
    )
    parser.add_argument(
        "-pnd",
        "--path_node_dfs",
        type=Path,
        required=True,
        help="Path to which layout node dataframes are saved",
    )
    parser.add_argument(
        "-ped",
        "--path_edge_dfs",
        type=Path,
        required=True,
        help="Path to which layout edge dataframes are saved",
    )

    args = parser.parse_args()

    df_to_layout_graph(
        path_timetable_df=args.path_timetable_df,
        path_node_dfs=args.path_node_dfs,
        path_edge_dfs=args.path_edge_dfs,
    )
