import logging
from typing import Dict, List, Optional, Tuple

from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span

from kronos.data_interfaces.ner_entities_data_interface import NERLabel
from kronos.nodes.spacy_patterns import (
    DAY_MONTH_PATTERN,
    DAY_OF_MONTH_PATTERN,
    DAY_OF_WEEK_PATTERN,
    DURATION_PATTERN,
    KNOWN_LOC_PATTERN,
    KNOWN_PERSON_PATTERN,
    MONTH_DAY_PATTERN,
    MONTH_PATTERN,
    REMOVE_PATTERN,
    SPECIAL_DATE_PATTERN,
    TIME_PATTERN,
    YEAR_PATTERN,
)

logger = logging.getLogger(__name__)


def configure_matcher(spacy_pipeline: Language) -> Matcher:
    matcher: Matcher = Matcher(spacy_pipeline.vocab, validate=True)

    matcher.add(NERLabel.year.value, [YEAR_PATTERN])
    matcher.add(NERLabel.month.value, [MONTH_PATTERN])
    matcher.add(NERLabel.day_of_week.value, [DAY_OF_WEEK_PATTERN])
    matcher.add(NERLabel.month_day.value, MONTH_DAY_PATTERN)  # type: ignore
    matcher.add(NERLabel.day_month.value, DAY_MONTH_PATTERN)  # type: ignore
    matcher.add(NERLabel.duration.value, DURATION_PATTERN)  # type: ignore
    matcher.add(NERLabel.day_of_month.value, [DAY_OF_MONTH_PATTERN])  # type: ignore
    matcher.add(NERLabel.person.value, [KNOWN_PERSON_PATTERN])
    matcher.add(NERLabel.loc.value, [KNOWN_LOC_PATTERN])
    matcher.add(NERLabel.time.value, [TIME_PATTERN])
    matcher.add(NERLabel.special_date.value, SPECIAL_DATE_PATTERN)
    matcher.add(NERLabel.remove.value, REMOVE_PATTERN)

    added_pattern_names: List[str] = [
        spacy_pipeline.vocab.strings[key] for key in dict(matcher._patterns).keys()  # type: ignore[attr-defined]
    ]
    logger.info(
        "Configured Matcher has the following patterns:\n"
        f"{dict(zip(matcher._patterns.keys(), added_pattern_names))}"  # type: ignore
    )

    return matcher


class PostProcessNER:
    def __init__(self, nlp: Language) -> None:
        self.nlp = nlp
        self.matcher = configure_matcher(nlp)

    def __call__(self, doc: Doc) -> Doc:
        ents = list(doc.ents)
        for ent in doc.ents:
            # Only postprocess DATE entities at the moment
            if ent.label_ == NERLabel.date.value:
                matches: List[Tuple[int, int, int]] = self.matcher(ent)

                text_to_match: Dict[str, Tuple[int, int, int]] = {}
                for match_id, start, end in matches:
                    # Raise error if there are multiple matches for the same text
                    if ent[start:end].text in text_to_match:
                        raise ValueError(
                            f"Found multiple matches for entity {ent} in doc {doc}. "
                            f"Here are the matches: {matches}.\n"
                            f"Here are the original entities and "
                            "their labels in the doc: "
                            f"{[(ent.text, ent.label_) for ent in doc.ents]}"
                        )
                    else:
                        text_to_match.update(
                            {ent[start:end].text: (match_id, start, end)}
                        )

                # The below line fails on purpose if no exact match is found
                exact_match: Optional[Tuple[int, int, int]] = text_to_match.get(
                    ent.text, None
                )

                if exact_match is None:
                    raise ValueError(
                        f"Failed to find exact match for entity {ent} in doc {doc}. "
                        f"Here are the matches: {matches}.\n"
                        f"Here are the original entities and "
                        "their labels in the doc: "
                        f"{[(ent.text, ent.label_) for ent in doc.ents]}"
                    )

                # Construct a new entity
                match_id, _, _ = exact_match
                new_label = self.nlp.vocab.strings[match_id]

                if new_label == "REMOVE":
                    ents.remove(ent)
                    continue
                else:
                    # Be careful to use Doc start and end char offsets instead of the Span's
                    new_ent = Span(
                        doc=doc, start=ent.start, end=ent.end, label=new_label
                    )

                    # Replace the old entity with the new one
                    ents[ents.index(ent)] = new_ent

        # Set modified entities for Doc
        doc.set_ents(ents)

        return doc
