from pathlib import Path
from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest

from kronos.data_interfaces.text_emb_local_data_interface import (
    NPArrayName,
    TextEmbLocalDataInterface,
)


@pytest.fixture
def mock_text_emb_arrays() -> Tuple[np.ndarray, np.ndarray]:
    text_array = np.array(["random", "text"])
    emb_array = np.random.rand(
        2, 628
    )  # Assuming embeddings are 628-dimensional for 2 samples
    return text_array, emb_array


@pytest.fixture
def mock_filepath(tmp_path: Path) -> Path:
    return tmp_path / "data.npz"


def test_initialization(mock_filepath: Path) -> None:
    interface = TextEmbLocalDataInterface(mock_filepath)
    assert interface.filepath == mock_filepath


@patch("numpy.savez")
def test_save(
    mock_savez: Mock,
    mock_filepath: Path,
    mock_text_emb_arrays: Tuple[np.ndarray, np.ndarray],
) -> None:
    text, emb = mock_text_emb_arrays
    interface = TextEmbLocalDataInterface(mock_filepath)
    interface.save(text, emb)

    mock_savez.assert_called_once_with(
        mock_filepath, **{NPArrayName.text.value: text, NPArrayName.emb.value: emb}
    )


@patch("numpy.load")
def test_load(
    mock_load: Mock,
    mock_filepath: Path,
    mock_text_emb_arrays: Tuple[np.ndarray, np.ndarray],
) -> None:
    text, emb = mock_text_emb_arrays
    mock_load.return_value = {NPArrayName.text.value: text, NPArrayName.emb.value: emb}

    interface = TextEmbLocalDataInterface(mock_filepath)
    loaded_text, loaded_emb = interface.load()

    assert np.array_equal(loaded_text, text)
    assert np.array_equal(loaded_emb, emb)
    mock_load.assert_called_once_with(mock_filepath)
