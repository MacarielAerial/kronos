import logging
import random
from pathlib import Path
from typing import Any, List, TypedDict

import orjson

logger = logging.getLogger(__name__)


class NERAnnotations(TypedDict):
    classes: List[str]
    annotations: List[List[Any]]


def split_ner_annotations(
    path_ner_annotations: Path,
    path_train_annotations: Path,
    path_dev_annotations: Path,
    sample_proportion: float = 0.8,
) -> None:
    # Data Access - Input
    with open(path_ner_annotations, "r") as f:
        ner_annotations: NERAnnotations = orjson.loads(f.read())

        logger.info(
            "Loaded NER Annotation file whose keys are:\n"
            f"{ner_annotations.keys()} and has "
            f"{len(ner_annotations['annotations'])} annotations"
        )

    # Task Processing
    random.shuffle(ner_annotations["annotations"])
    split_index = int(len(ner_annotations["annotations"]) * 0.8)

    train = ner_annotations["annotations"][:split_index]
    dev = ner_annotations["annotations"][split_index:]

    logger.info(
        f"Split the annotations into {len(train)} "
        f"training annotations and {len(dev)} annotations"
    )

    train_annotations: NERAnnotations = {
        "classes": ner_annotations["classes"],
        "annotations": train,
    }
    dev_annotations: NERAnnotations = {
        "classes": ner_annotations["classes"],
        "annotations": dev,
    }

    # Data Access - Output
    with open(path_train_annotations, "wb+") as f:
        f.write(orjson.dumps(train_annotations))

        logger.info(f"Saved training annotations to {path_train_annotations}")

    with open(path_dev_annotations, "wb+") as f:
        f.write(orjson.dumps(dev_annotations))

        logger.info(f"Saved dev annotations to {path_dev_annotations}")


if __name__ == "__main__":
    import argparse

    from kronos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Split JSON formatted NER annotations into training and dev sets"
    )
    parser.add_argument(
        "-pna",
        "--path_ner_annotations",
        type=Path,
        required=True,
        help="Path from which full NER annotations are loaded",
    )
    parser.add_argument(
        "-pta",
        "--path_train_annotations",
        type=Path,
        required=True,
        help="Path to which training annotations are saved",
    )
    parser.add_argument(
        "-pda",
        "--path_dev_annotations",
        type=Path,
        required=True,
        help="Path to which dev annotations are saved",
    )

    args = parser.parse_args()

    split_ner_annotations(
        path_ner_annotations=args.path_ner_annotations,
        path_train_annotations=args.path_train_annotations,
        path_dev_annotations=args.path_dev_annotations,
    )
