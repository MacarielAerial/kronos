import logging
import os
from pathlib import Path
from typing import Any, List, Set

import gspread
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas import DataFrame

logger = logging.getLogger(__name__)


class TimeTableDFDataInterface:
    ENV_KEY: str = "NewTimeTable2024Key"

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def download(self) -> DataFrame:
        load_dotenv()

        gc = gspread.service_account()
        google_sheet = gc.open_by_key(os.getenv(self.ENV_KEY)).get_worksheet(0)
        list_of_lists: List[List[Any]] = google_sheet.get_values()

        #
        # Data Cleaning
        #

        # Transpose the table because homogenous data is oriented column wise
        timetable_df = pd.DataFrame(list_of_lists).transpose()
        # Empty strings should be interpreted as nulls
        timetable_df.replace("", np.nan, inplace=True)

        # Year numbers should be interpreted as integers
        def convert_to_str(val: Any) -> Any:
            try:
                # Attempt to convert to float first, then to int and to string
                return str(int(float(val)))
            except (ValueError, TypeError):
                # Return the value as-is if it can't be converted
                return val

        timetable_df = timetable_df.map(convert_to_str)

        logger.info(f"Loaded Google Sheet has shape {timetable_df.shape}")

        return timetable_df

    def save(self, timetable_df: DataFrame) -> None:
        with open(self.filepath, "w+") as f:
            # Timetable sheet does not have meaningful index or header
            timetable_df.to_csv(f, index=False, header=False)

            logger.info(f"Saved a {type(timetable_df)} object to {self.filepath}")

    def load(self) -> DataFrame:
        with open(self.filepath, "r") as f:
            # Empty strings should be interpreted as empty strings
            timetable_df = pd.read_csv(f, index_col=False, header=None, dtype=str)

            logger.info(f"Loaded DataFrame has shape {timetable_df.shape}")
            logger.info(f"Loaded a {type(timetable_df)} object from {self.filepath}")

            return timetable_df

    @staticmethod
    def save_ner_annotation_input(timetable_df: DataFrame, filepath: Path) -> None:
        set_cell_value: Set[str] = set()
        with open(filepath, "w+") as f:
            for i in range(timetable_df.shape[0]):
                for j in range(timetable_df.shape[1]):
                    # Ignore nulls
                    if (
                        pd.notna(timetable_df.iloc[i, j])
                        and timetable_df.iloc[i, j] not in set_cell_value
                    ):
                        f.write(f"{timetable_df.iloc[i, j]}\n")
                        set_cell_value.add(str(timetable_df.iloc[i, j]))

            logger.info(
                f"Saved a {type(timetable_df)} object cell by cell to " f"{filepath}"
            )
