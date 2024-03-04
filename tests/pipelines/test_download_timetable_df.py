from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pandas as pd

from kronos.pipelines.download_timetable_df import download_df_timetable
from tests.conftest import TestDataPaths


@patch(
    "kronos.data_interfaces.timetable_df_data_interface.TimeTableDFDataInterface.download"
)
def test_download_df_timetable(
    mock_download: Mock,
    test_data_paths: TestDataPaths,
    mock_sheet_values: List[List[str]],
) -> None:
    # Arrange
    mock_download.return_value = mock_sheet_values

    # Not invoked so can be anything
    dummy_service_account_json_path = Path("/path/to/dummy/service_account.json")

    # Act
    download_df_timetable(
        path_service_account_json=dummy_service_account_json_path,
        path_timetable_df=test_data_paths.path_integration_saved_timetable_df,
        version="test_version",
    )

    # Assert
    filepath = (
        test_data_paths.path_integration_saved_timetable_df
        / "test_version"
        / test_data_paths.path_integration_saved_timetable_df.name
    )
    assert filepath.exists(), "The output CSV file should exist after processing."
    # Load the output CSV to verify its contents
    df_output = pd.read_csv(filepath)
    # Not testing data transformation logic here. Leaving that to unit testing
    assert len(df_output) > 0, "The output DataFrame should contain rows."
