import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from kronos.data_interfaces.timetable_df_data_interface import TimeTableDFDataInterface


def test_download() -> None:
    mock_gc = Mock()
    mock_sheet = Mock()
    mock_gc.open_by_key.return_value.get_worksheet.return_value = mock_sheet
    mock_sheet.get_values.return_value = [["value1", "value2"], ["value3", "value4"]]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "timetable.csv"

        with patch("gspread.service_account", return_value=mock_gc):
            interface = TimeTableDFDataInterface(temp_file)
            df = interface.download()

            assert not df.empty
            assert df.shape == (2, 2)  # Adjust based on expected shape


def test_save() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "timetable.csv"

        interface = TimeTableDFDataInterface(temp_file)

        # Timetable csv does not have meaningful index or header
        df = pd.DataFrame({0: ["row1", "row2"], 1: ["row3", "row4"]})

        interface.save(df)

        # Save method does not persist indices and headers
        saved_df = pd.read_csv(temp_file, index_col=False, header=None)

        pd.testing.assert_frame_equal(df, saved_df)


def test_load() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "timetable.csv"

        # Timetable csv does not have meaningful index or header
        df = pd.DataFrame({0: ["row1", "row2"], 1: ["row3", "row4"]})

        # Load method ignores indices and headers
        df.to_csv(temp_file, index=False, header=False)

        interface = TimeTableDFDataInterface(temp_file)
        loaded_df = interface.load()

        pd.testing.assert_frame_equal(df, loaded_df)


def test_save_ner_annotation_input() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "timetable.csv"

        df = pd.DataFrame(
            {"column1": ["row1", "row2", np.nan], "column2": ["row3", "row4", "row5"]}
        )

        TimeTableDFDataInterface.save_ner_annotation_input(df, temp_file)

        with open(temp_file, "r") as file:
            lines = file.readlines()

        # Check if the output file contains the expected lines
        # NER annotation input is saved row by row
        assert lines == ["row1\n", "row3\n", "row2\n", "row4\n", "row5\n"]
