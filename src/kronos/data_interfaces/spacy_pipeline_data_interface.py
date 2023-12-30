import logging
from pathlib import Path

import spacy
from spacy.language import Language

from kronos.data_interfaces.matcher_data_interface import PostProcessNER

logger = logging.getLogger(__name__)


@Language.factory("post_process_ner")
def post_process_ner(nlp: Language, name: str) -> PostProcessNER:
    return PostProcessNER(nlp)


class SpacyPipelineDataInterface:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def load(self) -> Language:
        nlp: Language = spacy.load(self.filepath)
        nlp.add_pipe("post_process_ner", after="ner")

        logger.info(
            "Loaded spacy pipeline has the following components: " f"{nlp.pipe_names}"
        )

        return nlp
