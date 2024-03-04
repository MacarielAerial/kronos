from pathlib import Path
from unittest.mock import Mock, mock_open, patch

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

#
# Dataclass tests
#


def test_node_dfs_validate_unique_types_and_ids() -> None:
    # Arrange
    df1 = pd.DataFrame(
        {
            NodeAttrKey.nid.value: [1, 2],
            NodeAttrKey.ntype.value: NodeType.sheet_cell.value,
            NodeAttrKey.text.value: ["text1", "text2"],
        }
    )
    df2 = pd.DataFrame(
        {
            NodeAttrKey.nid.value: [3, 4],
            NodeAttrKey.ntype.value: NodeType.token.value,
            NodeAttrKey.text.value: ["text3", "text4"],
        }
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
        {
            NodeAttrKey.nid.value: [1, 1],
            NodeAttrKey.ntype.value: NodeType.sheet_cell.value,
            NodeAttrKey.text.value: ["text1", "text2"],
        }
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


#
# Data interface tests
#


@patch("pathlib.Path.mkdir")
@patch("builtins.open", new_callable=mock_open)
@patch("orjson.dumps")
def test_save(
    mock_dumps, mock_file: Mock, mock_mkdir: Mock, mock_node_dfs: NodeDFs
) -> None:
    # Arrange
    filepath = Path("/fake/path/nodes.json")
    mock_dumps.return_value = b"{}"  # Mock the JSON bytes returned by orjson.dumps

    # Act
    interface = NodeDFsDataInterface(filepath=filepath)
    with patch.object(Path, "exists") as mock_exists:
        mock_exists.return_value = False
        interface.save(mock_node_dfs)

    # Assert
    mock_exists.assert_called_once()
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_dumps.assert_called_once()
    mock_file.assert_called_once_with(filepath, "wb")


@patch("builtins.open", new_callable=mock_open, read_data=b"{}")
@patch("orjson.loads")
@patch("dacite.from_dict")
def test_load(mock_from_dict: Mock, mock_loads: Mock, mock_file: Mock) -> None:
    # Arrange
    filepath = Path("/fake/path/nodes.json")
    mock_loads.return_value = {}  # Mock the dictionary returned by orjson.loads
    mock_from_dict.return_value = NodeDFs(
        members=[]
    )  # Mock the NodeDFs object returned by dacite.from_dict

    # Act
    interface = NodeDFsDataInterface(filepath=filepath)
    result = interface.load()

    # Assert
    mock_file.assert_called_once_with(filepath, "rb")
    mock_loads.assert_called_once()
    mock_from_dict.assert_called_once()
    assert isinstance(result, NodeDFs)
