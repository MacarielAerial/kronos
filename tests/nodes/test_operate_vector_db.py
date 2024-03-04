from unittest.mock import ANY, MagicMock, Mock, patch

import numpy as np
from pydantic_core import Url
from weaviate.collections.classes.config import DataType

from kronos.nodes.operate_vector_db import (
    CollectionName,
    CollectionSchema,
    VectorDBEndPoint,
    add_collections,
    batch_import,
    collection_exists,
    del_collection,
    instantiate_client,
)


@patch("kronos.nodes.operate_vector_db.WeaviateClient")
def test_instantiate_client_default_endpoint(mock_weaviate_client: Mock) -> None:
    # Arrange
    mock_client = Mock()
    mock_client.__enter__ = Mock()
    mock_client.__exit__ = Mock()
    mock_weaviate_client.return_value = mock_client

    # Act
    with instantiate_client() as client:
        # Assert
        assert client == mock_client


@patch("kronos.nodes.operate_vector_db.WeaviateClient")
def test_instantiate_client_custom_endpoint(mock_weaviate_client: Mock) -> None:
    # Arrange
    mock_client = Mock()
    mock_client.__enter__ = Mock()
    mock_client.__exit__ = Mock()
    mock_weaviate_client.return_value = mock_client

    custom_endpoint = VectorDBEndPoint(url=Url("http://custom:8080"))

    # Act
    with instantiate_client(end_point=custom_endpoint) as client:
        # Assert
        mock_weaviate_client.assert_called_once_with(connection_params=ANY)
        assert client == mock_client


def test_add_collections() -> None:
    mock_client = Mock()

    add_collections(
        client=mock_client,
        collections_schema=[
            CollectionSchema(name=CollectionName.word, description="", properties=[])
        ],
    )

    assert mock_client.collections.create.call_count == 1


def test_collection_exists() -> None:
    mock_client = Mock()
    mock_client.collections.list_all.return_value = {}
    collection_name = CollectionName.word

    a_bool = collection_exists(client=mock_client, collection_name=collection_name)

    assert a_bool is False


def test_del_collection() -> None:
    mock_client = Mock()
    collection_name = CollectionName.word

    del_collection(client=mock_client, collection_name=collection_name)

    assert mock_client.collections.delete.call_args == (
        {"name": collection_name.value},
    )


def test_batch_import() -> None:
    # Arrange
    mock_client = Mock()
    collection_name = CollectionName.word
    text = np.array(["a_text"])
    emb = np.array([[0.1, 0.2]])
    mock_batch = Mock()
    mock_context_manager = MagicMock()
    mock_client.batch.dynamic.return_value = mock_context_manager
    mock_context_manager.__enter__.return_value = mock_batch
    mock_client.collections.get.return_value.aggregate.overall.return_value = MagicMock(
        total_count=1
    )

    expected_add_object = [
        (
            {
                "collection": collection_name.value,
                "properties": {"name": "a_text", "data_type": DataType.TEXT},
                "vector": ANY,  # TODO: Test numpy array equality inside call_args_list
            },
        )
    ]

    # Act
    batch_import(
        client=mock_client, collection_name=collection_name, text=text, emb=emb
    )

    # Assert
    assert mock_batch.add_object.call_args_list == expected_add_object
    mock_client.collections.get.assert_called_once_with(collection_name.value)
