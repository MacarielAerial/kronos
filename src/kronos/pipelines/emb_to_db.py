from pathlib import Path

from pydantic_core import Url

from kronos.data_interfaces.text_emb_local_data_interface import (
    TextEmbLocalDataInterface,
)
from kronos.nodes.operate_vector_db import (
    CollectionName,
    NameToSchema,
    VectorDBEndPoint,
    add_collections,
    batch_import,
    collection_exists,
    del_collection,
    instantiate_client,
)


def emb_to_db(
    path_text_emb: Path, collection_name: CollectionName, end_point: VectorDBEndPoint
) -> None:
    # Data Access - Input
    text_emb_local_data_interface = TextEmbLocalDataInterface(filepath=path_text_emb)
    text, emb = text_emb_local_data_interface.load()

    # Task Processing
    with instantiate_client(end_point=end_point) as client:
        if collection_exists(client=client, collection_name=collection_name):
            del_collection(client=client, collection_name=collection_name)

        add_collections(
            client=client, collections_schema=[NameToSchema[collection_name]]
        )

        # Task Processing
        batch_import(client=client, collection_name=collection_name, text=text, emb=emb)


if __name__ == "__main__":
    import argparse

    from kronos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Uploads text array and embedding array of a certain type "
        "into the vector database"
    )
    parser.add_argument(
        "-pte",
        "--path_text_emb",
        type=Path,
        required=True,
        help="Path from which text array and embedding array of a certain type "
        "(aggregation of nodes of certain types) are loaded",
    )
    parser.add_argument(
        "-cn",
        "--collection_name",
        type=CollectionName,
        required=True,
        help="Type of the text embedding to be uploaded",
    )
    parser.add_argument(
        "-ep",
        "--end_point",
        type=lambda x: VectorDBEndPoint(url=Url(x)),
        required=False,
        default=VectorDBEndPoint(),
        help="End point to connect to a running vector database instance",
    )

    args = parser.parse_args()

    emb_to_db(
        path_text_emb=args.path_text_emb,
        collection_name=args.collection_name,
        end_point=args.end_point,
    )
