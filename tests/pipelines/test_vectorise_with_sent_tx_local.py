from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np

from kronos.pipelines.vectorise_with_sent_tx_local import vectorise_with_sent_tx_local


@patch(
    "kronos.pipelines.vectorise_with_sent_tx_local._vectorise_with_sentence_transformer"
)
@patch("kronos.pipelines.vectorise_with_sent_tx_local.NodeDFsDataInterface")
@patch("kronos.pipelines.vectorise_with_sent_tx_local.SentenceTransformer")
@patch("kronos.pipelines.vectorise_with_sent_tx_local.TextEmbLocalDataInterface")
@patch("builtins.open", new_callable=mock_open)
def test_contracted_sheet_cell_nodes(
    mock_open: Mock,
    mock_text_emb_interface: Mock,
    mock_sent_tx: Mock,
    mock_node_interface: Mock,
    mock_vectorise_with_sent_tx: Mock,
) -> None:
    # Arrange
    mock_paths = {
        "path_semantics_node_dfs": Path("/fake/semantics_node_dfs"),
        "path_sentence_transformer": Path("/fake/sentence_transformer/"),
        "path_text_emb": Path("/fake/text__emb"),
    }

    mock_semantics_node_dfs = MagicMock()
    mock_sent_tx_instance = MagicMock()

    mock_node_interface.return_value.load.return_value = mock_semantics_node_dfs
    mock_sent_tx.return_value = mock_sent_tx_instance
    mock_vectorise_with_sent_tx.return_value = iter(
        [("some_text", np.array([0.1, 0.2]))]
    )

    # Act
    vectorise_with_sent_tx_local(
        path_semantics_node_dfs=mock_paths["path_semantics_node_dfs"],
        path_sentence_transformer=mock_paths["path_sentence_transformer"],
        path_text_emb=mock_paths["path_text_emb"],
    )

    # Assert
    mock_node_interface.assert_called_once_with(
        filepath=mock_paths["path_semantics_node_dfs"]
    )
    mock_node_interface.return_value.load.assert_called_once()
    mock_sent_tx.assert_called_once_with(
        model_name_or_path=mock_paths["path_sentence_transformer"]
    )

    mock_vectorise_with_sent_tx.assert_called_once_with(
        node_dfs=mock_semantics_node_dfs, sentence_transformer=mock_sent_tx_instance
    )

    mock_text_emb_interface.assert_called_once_with(
        filepath=mock_paths["path_text_emb"]
    )
    mock_text_emb_interface.return_value.save.assert_called_once()
