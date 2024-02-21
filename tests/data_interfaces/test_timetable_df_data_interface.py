from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pandas as pd

from kronos.data_interfaces.timetable_df_data_interface import TimeTableDFDataInterface
from tests.conftest import TestDataPaths


def test_preprocess() -> None:
    # Arrange
    input_data = [["1", "2", "3"], ["4", "5", "6"]]
    expected_df = pd.DataFrame({0: ["1", "2", "3"], 1: ["4", "5", "6"]})

    # Act
    processed_df = TimeTableDFDataInterface.preprocess(input_data)

    # Assert
    pd.testing.assert_frame_equal(processed_df, expected_df)


@patch(
    "kronos.data_interfaces.timetable_df_data_interface.TimeTableDFDataInterface.download"
)
def test_download(mock_download: Mock, mock_sheet_values: List[List[str]]) -> None:
    # Arrange
    mock_download.return_value = mock_sheet_values
    data_interface = TimeTableDFDataInterface(filepath=Path("dummy/path"))

    # Act
    result = data_interface.download(Path("dummy/service_account.json"))

    # Assert
    assert (
        result == mock_sheet_values
    ), "Download method should return the mock data correctly"


def test_save(test_data_paths: TestDataPaths) -> None:
    # Arrange
    data_interface = TimeTableDFDataInterface(
        filepath=test_data_paths.path_saved_timetable_df, version="test_version"
    )
    df = pd.DataFrame({"A": ["231th dsa - fsd qwe rfwa", "SDDADS 1"]})

    # Act
    data_interface.save(df)

    # Assert
    filepath = (
        test_data_paths.path_saved_timetable_df
        / "test_version"
        / test_data_paths.path_saved_timetable_df.name
    )
    assert filepath.is_file()


def test_load_most_recent_version(tmp_path: Path) -> None:
    # Arrange
    filepath = tmp_path / "timetable_df.csv"
    path_old_version = filepath / "2024-02-20T17.34.32.739Z" / "timetable_df.csv"
    path_old_version.parent.mkdir(parents=True)
    path_recent_version = filepath / "2024-02-20T17.34.55.964Z" / "timetable_df.csv"
    path_recent_version.parent.mkdir(parents=True)

    old_timetable_df = pd.DataFrame({"A": ["321th Fsda - dwa fes fse", "FEAFW 1"]})
    recent_timetable_df = pd.DataFrame({"A": ["21th Fsda - dwa fes fse", "FEAFW 1"]})
    old_timetable_df.to_csv(path_old_version, index=False, header=False)
    recent_timetable_df.to_csv(path_recent_version, index=False, header=False)

    # Act
    data_interface = TimeTableDFDataInterface(filepath=filepath)
    loaded_df = data_interface.load()

    # Assert
    # Given header is removed during save, loaded dataframe has default column names
    recent_timetable_df = recent_timetable_df.rename({"A": 0}, axis=1)
    pd.testing.assert_frame_equal(loaded_df, recent_timetable_df, check_like=True)
