import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import gspread
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas import DataFrame

from kronos.nodes.utils_data_interfaces import generate_timestamp

logger = logging.getLogger(__name__)


class TimeTableDFDataInterface:
    ENV_KEY: str = "NewTimeTable2024Key"

    def __init__(self, filepath: Path, version: Optional[str] = None) -> None:
        # The data interface is assumed to be used
        # either to save or to load but not both
        self.filepath = filepath
        self.version = version

    def download(self, path_service_account_json: Path) -> List[List[Any]]:
        load_dotenv()

        gc = gspread.service_account(filename=path_service_account_json)
        google_sheet = gc.open_by_key(os.getenv(self.ENV_KEY)).get_worksheet(0)
        sheet_values: List[List[Any]] = list(google_sheet.get_values())

        return sheet_values

    @staticmethod
    def preprocess(sheet_values: List[List[Any]]) -> DataFrame:
        #
        # Data Cleaning
        #

        # Transpose the table because homogenous data is oriented column wise
        timetable_df = pd.DataFrame(sheet_values).transpose()
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

        logger.info(
            "Dataframe parsed from google sheet values has shape "
            f"{timetable_df.shape}"
        )

        return timetable_df

    def save(self, timetable_df: DataFrame) -> None:
        # Resolve save versioned path
        if self.version is None:
            filepath = self.filepath / generate_timestamp() / self.filepath.name
        else:
            filepath = self.filepath / self.version / self.filepath.name

        # Create directory tree if not exist
        if not filepath.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating {filepath.parent} because it does not yet exist")

        with open(filepath, "w") as f:
            # Timetable sheet does not have meaningful index or header
            timetable_df.to_csv(f, index=False, header=False)

            logger.info(f"Saved a {type(timetable_df)} object to {self.filepath}")

    def load(self) -> DataFrame:
        # Resolve load version path
        if self.version is None:
            filepaths = sorted(list(self.filepath.iterdir()), reverse=True)
            filepath = filepaths[0]

        with open(filepath, "r") as f:
            # Empty strings should be interpreted as empty strings
            timetable_df = pd.read_csv(f, index_col=False, header=None, dtype=str)

            logger.info(f"Loaded timetable dataFrame has shape {timetable_df.shape}")
            logger.info(f"Loaded a {type(timetable_df)} object from {self.filepath}")

            return timetable_df
