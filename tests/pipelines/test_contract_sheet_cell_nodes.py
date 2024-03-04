from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

from kronos.pipelines.contract_sheet_cell_nodes import contract_sheet_cell_nodes


@patch("kronos.pipelines.contract_sheet_cell_nodes._contract_sheet_cell_nodes")
@patch("kronos.pipelines.contract_sheet_cell_nodes.EdgeDFsDataInterface")
@patch("kronos.pipelines.contract_sheet_cell_nodes.NodeDFsDataInterface")
@patch("builtins.open", new_callable=mock_open)
def test_contracted_sheet_cell_nodes(
    mock_open: Mock,
    mock_node_interface: Mock,
    mock_edge_interface: Mock,
    mock_contracted_sheet_cell_nodes: Mock,
) -> None:
    # Arrange
    mock_paths = {
        "path_layout_node_dfs": Path("/fake/layout_node_dfs"),
        "path_layout_edge_dfs": Path("/fake/layout_edge_dfs"),
        "path_contracted_node_dfs": Path("/fake/contracted_nx_g"),
        "path_contracted_edge_dfs": Path("/fake/contracted_nx_g"),
    }

    mock_layout_node_dfs = MagicMock()
    mock_layout_edge_dfs = MagicMock()
    mock_contracted_node_dfs = MagicMock()
    mock_contracted_edge_dfs = MagicMock()

    mock_node_interface.return_value.load.return_value = mock_layout_node_dfs
    mock_edge_interface.return_value.load.return_value = mock_layout_edge_dfs

    mock_contracted_sheet_cell_nodes.return_value = (
        mock_contracted_node_dfs,
        mock_contracted_edge_dfs,
    )

    # Act
    contract_sheet_cell_nodes(
        path_layout_node_dfs=mock_paths["path_layout_node_dfs"],
        path_layout_edge_dfs=mock_paths["path_layout_edge_dfs"],
        path_contracted_node_dfs=mock_paths["path_contracted_node_dfs"],
        path_contracted_edge_dfs=mock_paths["path_contracted_edge_dfs"],
    )

    # Assert
    mock_node_interface.return_value.load.assert_called_once()
    mock_layout_node_dfs.validate.assert_called_once()
    mock_contracted_node_dfs.validate.assert_called_once()

    mock_edge_interface.return_value.load.assert_called_once()
    mock_layout_edge_dfs.validate.assert_called_once()
    mock_contracted_edge_dfs.validate.assert_called_once()

    mock_contracted_sheet_cell_nodes.assert_called_once_with(
        node_dfs=mock_layout_node_dfs, edge_dfs=mock_layout_edge_dfs
    )

    mock_node_interface.return_value.save.assert_called_once()
    mock_edge_interface.return_value.save.assert_called_once()
