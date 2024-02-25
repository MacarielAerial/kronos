from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

from kronos.pipelines.assemble_kg import assemble_kg


@patch("kronos.pipelines.assemble_kg.NXGDataInterface")
@patch("kronos.pipelines.assemble_kg._assemble_kg")
@patch("kronos.pipelines.assemble_kg.EdgeDFsDataInterface")
@patch("kronos.pipelines.assemble_kg.NodeDFsDataInterface")
@patch("builtins.open", new_callable=mock_open)
def test_assemble_kg(
    mock_open: Mock,
    mock_node_interface: Mock,
    mock_edge_interface: Mock,
    mock_assemble_kg: Mock,
    mock_nxg_interface: Mock,
) -> None:
    # Arrange
    mock_paths = {
        "path_node_dfs": Path("/fake/node_dfs"),
        "path_edge_dfs": Path("/fake/edge_dfs"),
        "path_nx_g": Path("/fake/nx_g"),
    }

    mock_node_dfs = MagicMock()
    mock_edge_dfs = MagicMock()
    mock_nx_g = MagicMock()

    mock_node_interface.return_value.load.return_value = mock_node_dfs
    mock_edge_interface.return_value.load.return_value = mock_edge_dfs
    mock_assemble_kg.return_value = mock_nx_g

    # Act
    assemble_kg(
        path_node_dfs=mock_paths["path_node_dfs"],
        path_edge_dfs=mock_paths["path_edge_dfs"],
        path_nx_g=mock_paths["path_nx_g"],
    )

    # Assert
    mock_node_interface.assert_called_once_with(filepath=mock_paths["path_node_dfs"])
    mock_node_interface.return_value.load.assert_called_once()
    mock_node_dfs.validate.assert_called_once()

    mock_edge_interface.assert_called_once_with(filepath=mock_paths["path_edge_dfs"])
    mock_edge_interface.return_value.load.assert_called_once()
    mock_edge_dfs.validate.assert_called_once()

    mock_assemble_kg.assert_called_once_with(
        node_dfs=mock_node_dfs, edge_dfs=mock_edge_dfs
    )

    mock_nxg_interface.assert_called_once_with(filepath=mock_paths["path_nx_g"])
    mock_nxg_interface.return_value.save.assert_called_once_with(nx_g=mock_nx_g)
