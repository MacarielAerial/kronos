import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kronos.data_interfaces.edge_dfs_data_interface import (
    EdgeDF,
    EdgeDFs,
    EdgeDFsDataInterface,
    EdgeType,
)
from tests.conftest import TestDataPaths

#
# Dataclass tests
#


def test_edge_dfs_validate_unique() -> None:
    # Arrange
    df = pd.DataFrame({"col1": [1, 2]})
    edge_df1 = EdgeDF(etype=EdgeType.up, df=df)
    edge_df2 = EdgeDF(etype=EdgeType.down, df=df)
    edge_dfs = EdgeDFs(members=[edge_df1, edge_df2])

    # Act & Assert
    try:
        edge_dfs.validate()
    except ValueError:
        pytest.fail("validate() raised ValueError unexpectedly!")


def test_edge_dfs_validate_non_unique() -> None:
    # Arrange
    df = pd.DataFrame({"col1": [1, 2]})
    edge_df1 = EdgeDF(etype=EdgeType.up, df=df)
    edge_df2 = EdgeDF(etype=EdgeType.up, df=df)
    edge_dfs = EdgeDFs(members=[edge_df1, edge_df2])

    # Act & Assert
    with pytest.raises(ValueError):
        edge_dfs.validate()


def test_edge_dfs_to_dict() -> None:
    # Arrange
    df1 = pd.DataFrame({"col1": [1, 2]})
    df2 = pd.DataFrame({"col2": [3, 4]})
    edge_df1 = EdgeDF(etype=EdgeType.left, df=df1)
    edge_df2 = EdgeDF(etype=EdgeType.right, df=df2)
    edge_dfs = EdgeDFs(members=[edge_df1, edge_df2])

    # Act
    result_dict = edge_dfs.to_dict()

    # Assert
    assert list(result_dict.keys()) == [EdgeType.left, EdgeType.right]
    assert_frame_equal(result_dict[EdgeType.left], df1)
    assert_frame_equal(result_dict[EdgeType.right], df2)


def test_save(mock_edge_dfs: EdgeDFs, test_data_paths: TestDataPaths) -> None:
    edge_dfs_data_interface = EdgeDFsDataInterface(
        filepath=test_data_paths.path_saved_edge_dfs
    )
    edge_dfs_data_interface.save(mock_edge_dfs)

    assert test_data_paths.path_saved_edge_dfs.is_file()


def test_load(test_data_paths: TestDataPaths) -> None:
    edge_dfs_data_interface = EdgeDFsDataInterface(
        filepath=test_data_paths.path_mock_edge_dfs
    )
    edge_dfs = edge_dfs_data_interface.load()

    assert len(edge_dfs.members) == 2
