from typing import Iterable
from unittest.mock import ANY, Mock, patch

import pytest
from pydantic_core import Url

from kronos.nodes.operate_vector_db import (
    NameToSchema,
    VectorDBEndPoint,
    add_collections,
    instantiate_client,
)


@pytest.fixture
def mock_weaviate_client() -> Iterable[Mock]:
    with patch("kronos.nodes.operate_vector_db.WeaviateClient") as mock:
        yield mock


def test_instantiate_client_default_endpoint(mock_weaviate_client: Mock) -> None:
    _ = instantiate_client()

    mock_weaviate_client.assert_called_once_with(connection_params=ANY)


def test_instantiate_client_custom_endpoint(mock_weaviate_client: Mock) -> None:
    custom_endpoint = VectorDBEndPoint(url=Url("http://custom:8080"))

    _ = instantiate_client(end_point=custom_endpoint)

    mock_weaviate_client.assert_called_once_with(connection_params=ANY)

@patch("kronos.nodes.operate_vector_db.instantiate_client")
def test_add_collections(
    mock_instantiate_client: Mock, mock_weaviate_client: Mock
) -> None:
    mock_client = mock_instantiate_client.return_value.__enter__.return_value
    mock_client.collections.list_all.return_value = []

    add_collections(list(NameToSchema.values()))

    assert mock_client.collections.create.call_count == 2
    mock_client.collections.list_all.assert_called_once()
