from pathlib import Path
from typing import Optional

from kronos.data_interfaces.timetable_df_data_interface import TimeTableDFDataInterface


def download_df_timetable(
    path_service_account_json: Path,
    path_timetable_df: Path,
    version: Optional[str] = None,
) -> None:
    # Task Processing
    timetable_df_data_interface = TimeTableDFDataInterface(
        filepath=path_timetable_df, version=version
    )
    sheet_values = timetable_df_data_interface.download(
        path_service_account_json=path_service_account_json
    )
    timetable_df = timetable_df_data_interface.preprocess(sheet_values=sheet_values)

    # Data Access - Output
    timetable_df_data_interface.save(timetable_df=timetable_df)


if __name__ == "__main__":
    import argparse

    from kronos.nodes.project_logging import default_logging

    default_logging()

    parser = argparse.ArgumentParser(
        description="Downloads timetable google sheet before serialises it into a csv"
    )
    parser.add_argument(
        "-psj",
        "--path_service_account_json",
        type=Path,
        required=True,
        help="Path from which a google sheet service account credential file is loaded",
    )
    parser.add_argument(
        "-ptd",
        "--path_timetable_df",
        type=Path,
        required=True,
        help="Path to which timetable dataframe is serialised",
    )

    args = parser.parse_args()

    download_df_timetable(
        path_service_account_json=args.path_service_account_json,
        path_timetable_df=args.path_timetable_df,
    )
