from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np

from kronos.pipelines.vectorise_with_word_vector_local import (
    vectorise_with_word_vector_local,
)


@patch("kronos.pipelines.vectorise_with_word_vector_local.NodeDFsDataInterface")
@patch("kronos.pipelines.vectorise_with_word_vector_local.SpacyPipelineDataInterface")
@patch("kronos.pipelines.vectorise_with_word_vector_local._vectorise_with_word_vector")
@patch("kronos.pipelines.vectorise_with_word_vector_local.TextEmbLocalDataInterface")
@patch("builtins.open", new_callable=mock_open)
def test_vectorise_with_word_vector_local(
    mock_open: Mock,
    mock_text_emb_interface: Mock,
    mock_vectorise_with_word_vector: Mock,
    mock_spacy_pipeline_interface: Mock,
    mock_node_interface: Mock,
) -> None:
    # Arrange
    mock_paths = {
        "path_semantics_node_dfs": Path("/fake/semantics_node_dfs.json"),
        "path_spacy_pipeline": Path("/fake/spacy_pipeline/"),
        "path_text_emb": Path("/fake/text_emb.json"),
    }

    mock_semantics_node_dfs = MagicMock()
    mock_spacy_pipeline = MagicMock()

    mock_node_interface.return_value.load.return_value = mock_semantics_node_dfs
    mock_spacy_pipeline_interface.return_value.load.return_value = mock_spacy_pipeline
    mock_vectorise_with_word_vector.return_value = iter(
        [("fake text", np.array([0.1, 0.2]))]
    )

    # Act
    vectorise_with_word_vector_local(
        path_semantics_node_dfs=mock_paths["path_semantics_node_dfs"],
        path_spacy_pipeline=mock_paths["path_spacy_pipeline"],
        path_text_emb=mock_paths["path_text_emb"],
    )

    # Assert
    mock_node_interface.assert_called_once_with(
        filepath=mock_paths["path_semantics_node_dfs"]
    )
    mock_node_interface.return_value.load.assert_called_once()

    mock_spacy_pipeline_interface.assert_called_once_with(
        filepath=mock_paths["path_spacy_pipeline"]
    )
    mock_spacy_pipeline_interface.return_value.load.assert_called_once()

    mock_vectorise_with_word_vector.assert_called_once_with(
        node_dfs=mock_semantics_node_dfs, spacy_pipeline=mock_spacy_pipeline
    )

    mock_text_emb_interface.assert_called_once_with(
        filepath=mock_paths["path_text_emb"]
    )
    mock_text_emb_interface.return_value.save.assert_called_once()
