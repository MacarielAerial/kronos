from pathlib import Path

import orjson
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

from kronos.pipelines.split_ner_annotations import NERAnnotations

nlp = spacy.blank("en")


def create_training(TRAIN_DATA: NERAnnotations) -> DocBin:
    db = DocBin()
    for text, annot in tqdm(TRAIN_DATA["annotations"]):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db


def convert_jsons_to_docbins(
    path_train_json: Path,
    path_dev_json: Path,
    path_train_docbin: Path,
    path_dev_docbin: Path,
) -> None:
    # Data Access - Input
    with open(path_train_json, "rb") as f:
        train_json = orjson.loads(f.read())
    with open(path_dev_json, "rb") as f:
        dev_json = orjson.loads(f.read())

    # Task Processing
    train_docbin = create_training(train_json)
    dev_docbin = create_training(dev_json)

    # Data Access - Output
    train_docbin.to_disk(path_train_docbin)
    dev_docbin.to_disk(path_dev_docbin)


if __name__ == "__main__":
    import argparse

    from kronos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Converts JSON formatted spacy training data "
        "into DocBin formatted ones"
    )
    parser.add_argument(
        "-ptj",
        "--path_train_json",
        type=Path,
        required=True,
        help="Path from which JSON formatted spacy training data is loaded",
    )
    parser.add_argument(
        "-pdj",
        "--path_dev_json",
        type=Path,
        required=True,
        help="Path from which JSON formatted spacy development data is loaded",
    )
    parser.add_argument(
        "-ptd",
        "--path_train_docbin",
        type=Path,
        required=True,
        help="Path to which DocBin formatted spacy training data is saved",
    )
    parser.add_argument(
        "-pdd",
        "--path_dev_docbin",
        type=Path,
        required=True,
        help="Path to which DocBin formatted spacy development data is saved",
    )

    args = parser.parse_args()

    convert_jsons_to_docbins(
        path_train_json=args.path_train_json,
        path_dev_json=args.path_dev_json,
        path_train_docbin=args.path_train_docbin,
        path_dev_docbin=args.path_dev_docbin,
    )
