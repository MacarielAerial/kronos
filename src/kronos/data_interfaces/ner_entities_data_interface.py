import logging
import random
from dataclasses import dataclass
from enum import Enum
from pprint import pformat
from typing import Any, Dict, List

from joblib import cpu_count
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


class NERLabel(str, Enum):
    date = "DATE"
    event = "EVENT"
    loc = "LOC"
    person = "PERSON"
    role = "ROLE"
    time = "TIME"

    # Subtypes of DATE which are produced by Matcher instead of NER
    year = "YEAR"
    month = "MONTH"
    day_of_week = "DAY_OF_WEEK"
    month_day = "MONTH_DAY"
    day_month = "DAY_MONTH"
    duration = "DURATION"
    day_of_month = "DAY_OF_MONTH"
    special_date = "SPECIAL_DATE"
    remove = "REMOVE"


@dataclass
class NEREntity:
    text: str
    label: NERLabel
    start_char: int
    end_char: int

    def validate(self, parent_text: str) -> bool:
        """
        Validates that the entity's text corresponds to the specified range in the parent text.
        If the validation fails, it raises a ValueError.
        """
        # Extract the corresponding substring from the parent text
        substring = parent_text[self.start_char : self.end_char]

        # Check if the substring matches the entity's text
        if substring != self.text:
            raise ValueError(
                f"Entity text does not match the specified range "
                f"in the parent text '{parent_text}'. "
                f"Expected: '{self.text}', Found: '{substring}'"
            )

        return True

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            text=self.text,
            label=self.label.value,
            start_char=self.start_char,
            end_char=self.end_char,
        )


def doc_to_ner_entities(doc: Doc) -> List[NEREntity]:
    return [
        NEREntity(
            text=ent.text,
            label=NERLabel(ent.label_),
            start_char=ent.start_char,
            end_char=ent.end_char,
        )
        for ent in doc.ents
    ]


def list_dict_to_ner_entities(list_dict: List[Dict[str, Any]]) -> List[NEREntity]:
    return [
        NEREntity(
            text=entity_dict["text"],
            label=NERLabel(entity_dict["label"]),
            start_char=entity_dict["start_char"],
            end_char=entity_dict["end_char"],
        )
        for entity_dict in list_dict
    ]


def list_text_to_list_ner_entities(
    list_text: List[str], spacy_pipeline: Language
) -> List[List[NEREntity]]:
    list_ner_entities: List[List[NEREntity]] = []
    list_doc = list(
        spacy_pipeline.pipe(list_text, batch_size=32, n_process=cpu_count())
    )
    list_ner_entities = [doc_to_ner_entities(doc) for doc in list_doc]

    i_log: int = random.choice(range(len(list_ner_entities)))
    logger.info(
        f"{len(list_ner_entities)} NER entities found in "
        f"{len(list_text)} texts. Here is an example text and its entities:\n"
        f"Text: {list_text[i_log]}\nEntities: "
        f"{pformat(list_ner_entities[i_log])}"
    )

    return list_ner_entities
