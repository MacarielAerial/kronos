from spacy.language import Language

from kronos.data_interfaces.spacy_pipeline_data_interface import (
    SpacyPipelineDataInterface,
)
from tests.conftest import TestDataPaths


def test_load(test_data_paths: TestDataPaths) -> None:
    data_interface = SpacyPipelineDataInterface(
        filepath=test_data_paths.path_en_sm_spacy_pipeline
    )
    spacy_pipeline = data_interface.load()

    assert isinstance(spacy_pipeline, Language)
