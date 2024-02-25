from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kronos.data_interfaces.edge_dfs_data_interface import (
    EdgeDF,
    EdgeDFs,
    EdgeDFsDataInterface,
    EdgeType,
)

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


#
# Data interface tests
#


@patch("pathlib.Path.mkdir")
@patch("builtins.open", new_callable=mock_open)
@patch("orjson.dumps")
def test_save(
    mock_dumps: Mock, mock_file: Mock, mock_mkdir: Mock, mock_edge_dfs: EdgeDFs
) -> None:
    # Arrange
    filepath = Path("/fake/path/edges.json")
    mock_dumps.return_value = b"{}"  # Mock the JSON bytes returned by orjson.dumps

    # Act
    interface = EdgeDFsDataInterface(filepath=filepath)
    with patch.object(Path, "exists") as mock_exists:
        mock_exists.return_value = False
        interface.save(mock_edge_dfs)

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
    filepath = Path("/fake/path/edges.json")
    mock_loads.return_value = {}  # Mock the dictionary returned by orjson.loads
    # Mock the EdgeDFs object returned by dacite.from_dict
    mock_from_dict.return_value = EdgeDFs(members=[])

    # Act
    interface = EdgeDFsDataInterface(filepath=filepath)
    result = interface.load()

    # Assert
    mock_file.assert_called_once_with(filepath, "rb")
    mock_loads.assert_called_once()
    mock_from_dict.assert_called_once()
    assert isinstance(result, EdgeDFs)
