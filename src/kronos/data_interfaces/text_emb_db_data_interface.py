from enum import Enum
from typing import List, TypedDict
import logging

from weaviate import WeaviateClient
from weaviate.classes.config import DataType, Property, ReferenceProperty
from weaviate.connect import ConnectionParams

logger = logging.getLogger(__name__)

def instantiate_client() -> WeaviateClient:
    connection_params = ConnectionParams.from_url(
    url="http://weaviate:8080",  # Adjust based on cocker compose setup,
    grpc_port=50051
    )

    return WeaviateClient(connection_params=connection_params)

#
# Schema
#

class CollectionName(str, Enum):
    word = "Word"
    sent = "Sentence"

class RefName(str, Enum):
    composes_sent = "ComposesSent"
    has_word = "HasWord"

class PropertyName(str, Enum):
    text = "text"
    ntype = "ntype"

class PropertySchema(TypedDict):
    name: str
    description: str
    data_type: DataType

class RefPropertySchema(TypedDict):
    name: str
    target_collection: str

class CollectionSchema(TypedDict):
    name: str
    description: str
    properties: List[PropertySchema]
    references: List[RefPropertySchema]

def get_collections_schema() -> List[CollectionSchema]:
    word_text_schema: PropertySchema = {"name": PropertyName.text.value, "description": "Text from which average word vector is derived", "data_type": DataType.TEXT}
    ntype_schema: PropertySchema = {"name": PropertyName.ntype.value, "description": "Node type of the node of which text is an attribute", "data_type": DataType.TEXT}
    compose_sent_schema: RefPropertySchema = {"name": RefName.composes_sent.value, "target_collection": CollectionName.sent.value}
    word_collection: CollectionSchema = {"name": CollectionName.word.value, "description": "Text which is encoded with its average word vector", "properties": [
        word_text_schema,
        ntype_schema
    ],
    "references": [
        compose_sent_schema
    ]
    }

    sent_text_schema: PropertySchema = {"name": PropertyName.text.value, "description": "Text from which sentence transformer embedding is derived", "data_type": DataType.TEXT}
    has_word_schema: RefPropertySchema = {"name": RefName.has_word.value, "target_collection": CollectionName.sent.value}
    sent_collection: CollectionSchema = {"name": CollectionName.sent.value, "description": "Text which is encoded with its sentence transformer embedding", "properties": [
        sent_text_schema,
        ntype_schema
    ],
    "references": [
        has_word_schema
    ]
    }

    return [word_collection, sent_collection]

#
# Declaratives
#

def add_collections(collections_schema: List[CollectionSchema]) -> None:
    with instantiate_client() as client:
        if len(client.collections.list_all()) > 0:
            raise KeyError("Collections already exist in the Weaviate instance")

        for schema in collections_schema:
            client.collections.create(name=schema["name"], description=schema["description"], properties=[Property(property_schema) for property_schema in schema["properties"]],
                                      references=[ReferenceProperty(**ref_schema) for ref_schema in schema["references"]])

        logger.info(f"Added schema for {len(collections_schema)} collections")

# TODO: Add declaratives for batch imports
