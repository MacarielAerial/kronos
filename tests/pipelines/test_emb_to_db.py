from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from kronos.nodes.operate_vector_db import CollectionName, VectorDBEndPoint
from kronos.pipelines.emb_to_db import emb_to_db


@patch("kronos.pipelines.emb_to_db.TextEmbLocalDataInterface")
@patch("kronos.pipelines.emb_to_db.instantiate_client")
@patch("kronos.pipelines.emb_to_db.collection_exists")
@patch("kronos.pipelines.emb_to_db.del_collection")
@patch("kronos.pipelines.emb_to_db.add_collections")
@patch("kronos.pipelines.emb_to_db.batch_import")
@patch("numpy.load")
def test_emb_to_db(
    mock_np_load: Mock,
    mock_batch_import: Mock,
    mock_add_collections: Mock,
    mock_del_collection: Mock,
    mock_collection_exists: Mock,
    mock_instantiate_client: Mock,
    mock_text_emb_interface: Mock,
) -> None:
    # Arrange
    mock_path_text_emb = Path("/fake/text_emb.npz")
    collection_name = CollectionName.word
    end_point = VectorDBEndPoint()
    mock_text_emb_interface.return_value.load.return_value = (
        np.array(["a_text"]),
        np.array([[0.1, 0.2]]),
    )

    mock_client = Mock()
    mock_instantiate_client.return_value.__enter__.return_value = mock_client
    mock_collection_exists.return_value = True

    # Act
    emb_to_db(
        path_text_emb=mock_path_text_emb,
        collection_name=collection_name,
        end_point=end_point,
    )

    # Assert
    mock_text_emb_interface.assert_called_once_with(filepath=mock_path_text_emb)
    mock_text_emb_interface.return_value.load.assert_called_once()
    mock_instantiate_client.assert_called_once_with(end_point=end_point)
    mock_collection_exists.assert_called_once_with(
        client=mock_client, collection_name=collection_name
    )
    mock_del_collection.assert_called_once_with(
        client=mock_client, collection_name=collection_name
    )
    mock_add_collections.assert_called_once()
    mock_batch_import.assert_called_once()
