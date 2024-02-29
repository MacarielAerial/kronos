from kronos.pipelines.df_to_layout_graph import df_to_layout_graph
from tests.conftest import TestDataPaths


def test_df_to_layout_graph(test_data_paths: TestDataPaths) -> None:
    df_to_layout_graph(
        path_timetable_df=test_data_paths.path_mock_timetable_df,
        path_node_dfs=test_data_paths.path_integration_node_dfs,
        path_edge_dfs=test_data_paths.path_integration_edge_dfs,
    )

    assert test_data_paths.path_integration_node_dfs.is_file()
    assert test_data_paths.path_integration_edge_dfs.is_file()
