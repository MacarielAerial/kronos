import logging
import shutil
from pathlib import Path
from typing import List

from pytest import ExitCode, Session, fixture

logger = logging.getLogger(__name__)


class TestDataPaths:
    @property
    def own_path(self) -> Path:
        return Path(__file__).parent

    @property
    def path_dir_data(self) -> Path:
        return self.own_path / "data"

    # Test input data paths

    # Test output data paths

    @property
    def path_dir_output(self) -> Path:
        return self.path_dir_data / "output"

    @property
    def path_saved_timetable_df(self) -> Path:
        return self.path_dir_output / "saved_timetable_df.json"

    @property
    def path_integration_saved_timetable_df(self) -> Path:
        return self.path_dir_output / "integration_saved_timetable_df.json"


@fixture
def test_data_paths() -> TestDataPaths:
    return TestDataPaths()


@fixture
def mock_sheet_values() -> List[List[str]]:
    return [
        [
            "dwhwd + ??",
            "",
            "",
            "",
            "DWADW",
            "OJWODJA",
            "",
            "",
            "312fd Pdsad - Fsj sdkwj wasda",
        ],
        ["", "", "", "", "1900", "", "DS/GWF", "", "WWQDWDDDWD 1"],
    ]


def pytest_sessionstart(session: Session) -> None:
    path_dir_output = TestDataPaths().path_dir_output

    logger.info(
        f"A test data output directory at {path_dir_output} "
        "will be created if not exist already"
    )

    path_dir_output.mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session: Session, exitstatus: ExitCode) -> None:
    path_dir_output = TestDataPaths().path_dir_output

    logger.info(f"Deleting Test output data directory at {path_dir_output}")

    shutil.rmtree(path=path_dir_output)