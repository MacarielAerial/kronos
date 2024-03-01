import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NPArrayName(str, Enum):
    text = "text"
    emb = "emb"


class TextEmbLocalDataInterface:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def save(self, text: np.ndarray, emb: np.ndarray) -> None:
        kwargs: Dict[str, Any] = {
            NPArrayName.text.value: text,
            NPArrayName.emb.value: emb,
        }

        np.savez(self.filepath, **kwargs)

        logger.info(
            f"Saved text array and embedding arrays shaped {emb.shape} to "
            f"{self.filepath}"
        )

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        npzfile = np.load(self.filepath)

        text, emb = npzfile[NPArrayName.text.value], npzfile[NPArrayName.emb.value]

        logger.info(
            f"Loaded text array shaped {text.shape} and embedding array "
            f"shaped {emb.shape} from {self.filepath}"
        )

        return text, emb
