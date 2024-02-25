from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from kronos.pipelines.add_nlp_feats import add_nlp_feats


@patch("kronos.pipelines.add_nlp_feats.NodeDFsDataInterface")
@patch("kronos.pipelines.add_nlp_feats.EdgeDFsDataInterface")
@patch("kronos.pipelines.add_nlp_feats.SpacyPipelineDataInterface")
@patch("kronos.pipelines.add_nlp_feats._add_nlp_feats")
def test_add_nlp_feats(
    mock_add_nlp_feats: Mock,
    mock_spacy_pipeline_interface: Mock,
    mock_edge_interface: Mock,
    mock_node_interface: Mock,
) -> None:
    # Arrange
    mock_paths = {
        "path_layout_node_dfs": Path("/fake/layout_node_dfs"),
        "path_layout_edge_dfs": Path("/fake/layout_edge_dfs"),
        "path_spacy_pipeline": Path("/fake/spacy_pipeline"),
        "path_semantics_node_dfs": Path("/fake/semantics_node_dfs"),
        "path_semantics_edge_dfs": Path("/fake/semantics_edge_dfs"),
    }

    mock_layout_node_dfs = MagicMock()
    mock_layout_edge_dfs = MagicMock()
    mock_spacy_pipeline = MagicMock()
    mock_semantics_node_dfs = MagicMock()
    mock_semantics_edge_dfs = MagicMock()

    mock_node_interface.return_value.load.return_value = mock_layout_node_dfs
    mock_edge_interface.return_value.load.return_value = mock_layout_edge_dfs
    mock_spacy_pipeline_interface.return_value.load.return_value = mock_spacy_pipeline
    mock_add_nlp_feats.return_value = (mock_semantics_node_dfs, mock_semantics_edge_dfs)

    # Act
    add_nlp_feats(
        path_layout_node_dfs=mock_paths["path_layout_node_dfs"],
        path_layout_edge_dfs=mock_paths["path_layout_edge_dfs"],
        path_spacy_pipeline=mock_paths["path_spacy_pipeline"],
        path_semantics_node_dfs=mock_paths["path_semantics_node_dfs"],
        path_semantics_edge_dfs=mock_paths["path_semantics_edge_dfs"],
    )

    # Assert
    mock_node_interface.return_value.load.assert_called_once()
    mock_edge_interface.return_value.load.assert_called_once()
    mock_spacy_pipeline_interface.return_value.load.assert_called_once_with(
        enable=["tok2vec", "ner"]
    )
    mock_add_nlp_feats.assert_called_once_with(
        node_dfs=mock_layout_node_dfs,
        edge_dfs=mock_layout_edge_dfs,
        spacy_pipeline=mock_spacy_pipeline,
    )
    mock_node_interface.return_value.save.assert_called_once_with(
        node_dfs=mock_semantics_node_dfs
    )
    mock_edge_interface.return_value.save.assert_called_once_with(
        edge_dfs=mock_semantics_edge_dfs
    )
