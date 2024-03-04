import logging
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, HttpUrl
from pydantic_core import Url
from weaviate import WeaviateClient
from weaviate.classes.config import DataType, Property
from weaviate.connect import ConnectionParams

logger = logging.getLogger(__name__)

# TODO: Refactor to centralise logging configuration in a file
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
httpx_logger = logging.getLogger("httpcore")
httpx_logger.setLevel(logging.WARNING)
httpx_logger = logging.getLogger("grpc")
httpx_logger.setLevel(logging.WARNING)
httpx_logger = logging.getLogger("asyncio")
httpx_logger.setLevel(logging.WARNING)


class VectorDBEndPoint(BaseModel):
    url: HttpUrl = Url("http://weaviate:8080")  # Address within the same network


def instantiate_client(end_point: Optional[VectorDBEndPoint] = None) -> WeaviateClient:  # type: ignore[no-any-unimported]
    if end_point is None:
        end_point = VectorDBEndPoint()

    connection_params = ConnectionParams.from_url(
        url=str(end_point.url),  # Adjust based on docker compose setup,
        grpc_port=50051,
    )

    return WeaviateClient(connection_params=connection_params)


#
# Schema
#


class CollectionName(str, Enum):
    word = "Word"
    sent = "Sentence"


class PropertyName(str, Enum):
    text = "text"
    # ntype = "ntype"


class PropertySchema(BaseModel):  # type: ignore[no-any-unimported]
    name: PropertyName
    description: str
    data_type: DataType  # type: ignore[no-any-unimported]


class CollectionSchema(BaseModel):
    name: CollectionName
    description: str
    properties: List[PropertySchema]


def get_word_collection_schema() -> CollectionSchema:
    word_collection = CollectionSchema(
        name=CollectionName.word,
        description="Text which is encoded with its average word vector",
        properties=[
            PropertySchema(
                name=PropertyName.text,
                description="Text from which average word vector is derived",
                data_type=DataType.TEXT,
            )
        ],
    )

    return word_collection


def get_sent_collection_schema() -> CollectionSchema:
    sent_collection = CollectionSchema(
        name=CollectionName.sent,
        description="Text which is encoded with its sentence transformer embedding",
        properties=[
            PropertySchema(
                name=PropertyName.text,
                description="Text from which sentence transformer embedding is derived",
                data_type=DataType.TEXT,
            ),
        ],
    )

    return sent_collection


NameToSchema: Dict[CollectionName, CollectionSchema] = {
    CollectionName.word: get_word_collection_schema(),
    CollectionName.sent: get_sent_collection_schema(),
}


#
# Declaratives
#


def add_collections(
    collections_schema: List[CollectionSchema],
    end_point: Optional[VectorDBEndPoint] = None,
) -> None:
    if end_point is not None:
        end_point = VectorDBEndPoint()

    with instantiate_client(end_point=end_point) as client:
        if len(client.collections.list_all()) > 0:
            raise KeyError("Collections already exist in the Weaviate instance")

        for schema in collections_schema:
            client.collections.create(
                name=schema.name.value,
                description=schema.description,
                properties=[
                    Property(**property_schema.model_dump())
                    for property_schema in schema.properties
                ],
            )

            logger.info(f"Added schema for collection {schema.name.value}")


# TODO: Add a test for this function once the dev understands how batch.dynamic() works
def batch_import(
    collection_name: CollectionName,
    text: np.ndarray,
    emb: np.ndarray,
    end_point: Optional[VectorDBEndPoint] = None,
) -> None:
    if end_point is None:
        end_point = VectorDBEndPoint()

    logger.info(
        f"Batch importing text array shaped {text.shape} and "
        f"embedding array shaped {emb.shape} to a vector database"
    )

    with instantiate_client(end_point=end_point) as client:
        with client.batch.dynamic() as batch:
            for i in range(len(text)):
                _ = batch.add_object(
                    collection=collection_name.value,
                    properties={"name": text[i], "data_type": DataType.TEXT},
                    vector=emb[i],
                )

        dict_n_imported = client.collections.get(
            collection_name.value
        ).aggregate.over_all(total_count=True)
        logger.info(
            f"After import {dict_n_imported.total_count} objects exist "
            f"in collection {collection_name.value}"
        )


def collection_exists(
    collection_name: CollectionName, end_point: Optional[VectorDBEndPoint] = None
) -> bool:
    with instantiate_client(end_point=end_point) as client:
        if len(client.collections.list_all()) < 1:
            logger.info("No collection exists in this vector database")
            return False
        elif len(client.collections.get(collection_name.value)) < 1:
            logger.info(f"Collection named {collection_name.value} does not exist")
            return False

        return True


def del_collection(
    collection_name: CollectionName, end_point: Optional[VectorDBEndPoint] = None
) -> None:
    with instantiate_client(end_point=end_point) as client:
        client.collections.delete(name=collection_name.value)

        logger.info(f"Deleted collection {collection_name.value}")
