from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pandas as pd

from kronos.data_interfaces.timetable_df_data_interface import TimeTableDFDataInterface


def test_preprocess() -> None:
    input_data = [["1", "2", "3"], ["4", "5", "6"]]
    expected_df = pd.DataFrame({0: ["1", "2", "3"], 1: ["4", "5", "6"]})
    processed_df = TimeTableDFDataInterface.preprocess(input_data)

    pd.testing.assert_frame_equal(processed_df, expected_df)


@patch(
    "kronos.data_interfaces.timetable_df_data_interface.TimeTableDFDataInterface.download"
)
def test_download(mock_download: Mock, mock_sheet_values: List[List[str]]) -> None:
    mock_download.return_value = mock_sheet_values
    data_interface = TimeTableDFDataInterface(filepath=Path("dummy/path"))
    result = data_interface.download(Path("dummy/service_account.json"))
    assert (
        result == mock_sheet_values
    ), "Download method should return the mock data correctly"
