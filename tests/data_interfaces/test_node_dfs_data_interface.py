import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kronos.data_interfaces.node_dfs_data_interface import (
    NodeAttrKey,
    NodeDF,
    NodeDFs,
    NodeDFsDataInterface,
    NodeType,
)
from tests.conftest import TestDataPaths


def test_node_dfs_validate_unique_types_and_ids() -> None:
    # Arrange
    df1 = pd.DataFrame(
        {NodeAttrKey.nid.value: [1, 2], NodeAttrKey.text.value: ["text1", "text2"]}
    )
    df2 = pd.DataFrame(
        {NodeAttrKey.nid.value: [3, 4], NodeAttrKey.text.value: ["text3", "text4"]}
    )
    node_df1 = NodeDF(ntype=NodeType.sheet_cell, df=df1)
    node_df2 = NodeDF(ntype=NodeType.token, df=df2)
    node_dfs = NodeDFs(members=[node_df1, node_df2])

    # Act & Assert
    try:
        node_dfs.validate()
    except ValueError:
        pytest.fail("validate() raised ValueError unexpectedly!")


def test_node_dfs_validate_non_unique_ids() -> None:
    # Arrange
    df = pd.DataFrame(
        {NodeAttrKey.nid.value: [1, 1], NodeAttrKey.text.value: ["text1", "text2"]}
    )
    node_df = NodeDF(ntype=NodeType.sheet_cell, df=df)
    node_dfs = NodeDFs(members=[node_df])

    # Act & Assert
    with pytest.raises(ValueError):
        node_dfs.validate()


def test_node_dfs_to_dict() -> None:
    # Arrange
    df1 = pd.DataFrame(
        {NodeAttrKey.nid.value: [1, 2], NodeAttrKey.text.value: ["text1", "text2"]}
    )
    df2 = pd.DataFrame(
        {NodeAttrKey.nid.value: [3, 4], NodeAttrKey.text.value: ["text3", "text4"]}
    )
    node_df1 = NodeDF(ntype=NodeType.sheet_cell, df=df1)
    node_df2 = NodeDF(ntype=NodeType.token, df=df2)
    node_dfs = NodeDFs(members=[node_df1, node_df2])

    # Act
    result_dict = node_dfs.to_dict()

    # Assert
    assert list(result_dict.keys()) == [NodeType.sheet_cell, NodeType.token]
    assert_frame_equal(result_dict[NodeType.sheet_cell], df1)
    assert_frame_equal(result_dict[NodeType.token], df2)


def test_node_dfs_ntypes_property() -> None:
    # Arrange
    df1 = pd.DataFrame(
        {NodeAttrKey.nid.value: [1, 2], NodeAttrKey.text.value: ["text1", "text2"]}
    )
    df2 = pd.DataFrame(
        {NodeAttrKey.nid.value: [3, 4], NodeAttrKey.text.value: ["text3", "text4"]}
    )
    node_df1 = NodeDF(ntype=NodeType.sheet_cell, df=df1)
    node_df2 = NodeDF(ntype=NodeType.token, df=df2)
    node_dfs = NodeDFs(members=[node_df1, node_df2])

    # Act & Assert
    assert node_dfs.ntypes == [
        NodeType.sheet_cell,
        NodeType.token,
    ], "ntypes property should list all node types correctly"


def test_save(mock_node_dfs: NodeDFs, test_data_paths: TestDataPaths) -> None:
    node_dfs_data_interface = NodeDFsDataInterface(
        filepath=test_data_paths.path_saved_node_dfs
    )
    node_dfs_data_interface.save(mock_node_dfs)

    assert test_data_paths.path_saved_node_dfs.is_file()


def test_load(test_data_paths: TestDataPaths) -> None:
    node_dfs_data_interface = NodeDFsDataInterface(
        filepath=test_data_paths.path_mock_node_dfs
    )
    node_dfs = node_dfs_data_interface.load()

    assert len(node_dfs.members) == 2
