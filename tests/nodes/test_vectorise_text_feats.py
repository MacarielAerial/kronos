from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from pandas import DataFrame
from pytest import fixture

from kronos.data_interfaces.node_dfs_data_interface import (
    NodeAttrKey,
    NodeDF,
    NodeDFs,
    NodeType,
)
from kronos.nodes.vectorise_text_feats import (
    _vectorise_with_word_vector,
    embed_with_avg_word_vec,
    embed_with_sent_tx,
    prep_emb_input,
)

#
# Input Text Preparation
#


def test_prep_emb_input_unique_texts() -> None:
    # Arrange
    mock_list_node_df = [
        NodeDF(
            ntype=NodeType.token,
            df=DataFrame(
                {NodeAttrKey.text.value: ["Non overlapping text", "Overlapping text"]}
            ),
        ),
        NodeDF(
            ntype=NodeType.sheet_cell,
            df=DataFrame({NodeAttrKey.text.value: ["Overlapping text"]}),
        ),
    ]

    # Act
    unique_text = prep_emb_input(mock_list_node_df)

    # Assert
    assert isinstance(unique_text, list)
    assert len(unique_text) == 2  # Number of unique texts across both NodeDFs
    assert "Non overlapping text" in unique_text
    assert "Overlapping text" in unique_text


def test_prep_emb_input_empty_input() -> None:
    with pytest.raises(ValueError):
        _ = prep_emb_input([])


#
# Average Word Vectors
#


@fixture
def mock_spacy_pipeline_empty() -> MagicMock:
    pipeline = MagicMock()
    pipeline.pipe.return_value = iter([])

    return pipeline


@fixture
def mock_spacy_pipeline() -> MagicMock:
    pipeline = MagicMock()
    doc_mock = MagicMock()
    doc_mock.text = "mock_text"
    doc_mock.vector = np.array([1, 2, 3])
    pipeline.pipe.return_value = [doc_mock]

    return pipeline


def test_embed_with_avg_word_vec(mock_spacy_pipeline: MagicMock) -> None:
    # Arrange
    text = ["mock_text"]

    # Act
    vectors = list(
        embed_with_avg_word_vec(text=text, spacy_pipeline=mock_spacy_pipeline)
    )

    # Assert
    assert len(vectors) == 1
    assert vectors[0][0] == "mock_text"
    np.testing.assert_array_equal(vectors[0][1], np.array([1, 2, 3]))


def test_embed_with_avg_word_vec_empty(mock_spacy_pipeline_empty: MagicMock) -> None:
    vectors = list(
        embed_with_avg_word_vec(text=[], spacy_pipeline=mock_spacy_pipeline_empty)
    )

    assert len(vectors) == 0


@patch("kronos.nodes.vectorise_text_feats.prep_emb_input")
def test_vectorise_with_word_vector(
    mock_prep_emb_input: Mock, mock_spacy_pipeline: MagicMock
) -> None:
    # Arrange
    mock_node_dfs = NodeDFs(
        members=[
            NodeDF(
                ntype=NodeType.token,
                df=DataFrame({NodeAttrKey.text.value: ["mock_text"]}),
            )
        ]
    )
    mock_prep_emb_input.return_value = ["mock_text"]

    # Act
    res = list(
        _vectorise_with_word_vector(
            node_dfs=mock_node_dfs, spacy_pipeline=mock_spacy_pipeline
        )
    )

    # Assert
    assert len(res) == 1
    assert res[0][0] == "mock_text"
    np.testing.assert_array_equal(res[0][1], np.array([1, 2, 3]))
    mock_prep_emb_input.assert_called_once()


#
# Sentence Transformer Embeddings
#


@pytest.fixture
def mock_sentence_transformer() -> MagicMock:
    st = MagicMock()
    st.start_multi_process_pool.return_value = "mock_pool"
    st.encode_multi_process.return_value = [np.array([1, 2, 3]) for _ in range(600)]
    st.encode.return_value = [np.array([1, 2, 3]) for _ in range(100)]

    return st


def test_embed_with_sent_tx_single_process(
    mock_sentence_transformer: MagicMock,
) -> None:
    # Arrange
    texts = ["text" for _ in range(100)]

    # Act
    res = list(
        embed_with_sent_tx(
            text=texts, sent_tx=mock_sentence_transformer, use_multi=False
        )
    )

    # Assert
    assert len(res) == 100
    for text, emb in res:
        assert text == "text"
        np.testing.assert_array_equal(emb, np.array([1, 2, 3]))


def test_embed_with_sent_tx_single_batch(mock_sentence_transformer: MagicMock) -> None:
    # Arrange
    texts = ["text" for _ in range(100)]

    # Act
    res = list(
        embed_with_sent_tx(
            text=texts, sent_tx=mock_sentence_transformer, use_multi=True
        )
    )

    # Assert
    assert len(res) == 100
    for text, emb in res:
        assert text == "text"
        np.testing.assert_array_equal(emb, np.array([1, 2, 3]))


def test_embed_with_sent_tx_multiple_batches(
    mock_sentence_transformer: MagicMock,
) -> None:
    # Arrange
    texts = ["text" for _ in range(600)]  # More than one batch

    # Act
    emb = list(
        embed_with_sent_tx(
            text=texts, sent_tx=mock_sentence_transformer, use_multi=True
        )
    )

    # Assert
    assert len(emb) == 600
