import logging
from enum import Enum
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, HttpUrl
from pydantic_core import Url
from weaviate import WeaviateClient
from weaviate.classes.config import DataType, Property
from weaviate.connect import ConnectionParams

logger = logging.getLogger(__name__)


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


def get_collections_schema() -> List[CollectionSchema]:
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

    return [word_collection, sent_collection]


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

        logger.info(f"Added schema for {len(collections_schema)} collections")


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
            for single_text, single_emb in zip(text, emb):
                uuid = batch.add_object(
                    collection=collection_name.value,
                    properties=Property(name=single_text, data_type=DataType.TEXT),
                    vector=single_emb,
                )
                logger.debug(f"Added object with uuid {uuid}")
