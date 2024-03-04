import logging
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)


class SpacyPipelineDataInterface:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def load(self, **kwargs: Any) -> Language:
        nlp: Language = spacy.load(self.filepath, **kwargs)

        logger.info(
            f"Loaded spacy pipeline has the following components: {nlp.pipe_names}"
        )

        return nlp
