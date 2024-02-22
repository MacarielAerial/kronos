import logging
import shutil
from pathlib import Path
from typing import List

from pandas import DataFrame
from pytest import ExitCode, Session, fixture

from kronos.data_interfaces.edge_dfs_data_interface import (
    EdgeAttrKey,
    EdgeDF,
    EdgeDFs,
    EdgeType,
)
from kronos.data_interfaces.node_dfs_data_interface import (
    NodeAttrKey,
    NodeDF,
    NodeDFs,
    NodeType,
)

logger = logging.getLogger(__name__)


class TestDataPaths:
    @property
    def own_path(self) -> Path:
        return Path(__file__).parent

    @property
    def path_dir_data(self) -> Path:
        return self.own_path / "data"

    @property
    def path_mock_node_dfs(self) -> Path:
        return self.path_dir_data / "mock_node_dfs.json"

    @property
    def path_mock_edge_dfs(self) -> Path:
        return self.path_dir_data / "mock_edge_dfs.json"

    @property
    def path_mock_timetable_df(self) -> Path:
        return self.path_dir_data / "mock_timetable_df.csv"

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

    @property
    def path_saved_node_dfs(self) -> Path:
        return self.path_dir_output / "saved_node_dfs.json"

    @property
    def path_saved_edge_dfs(self) -> Path:
        return self.path_dir_output / "saved_edge_dfs.json"

    @property
    def path_integration_node_dfs(self) -> Path:
        return self.path_dir_output / "integration_node_dfs.json"

    @property
    def path_integration_edge_dfs(self) -> Path:
        return self.path_dir_output / "integration_edge_dfs.json"


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


@fixture
def mock_node_dfs() -> NodeDFs:
    node_dfs = NodeDFs(
        members=[
            NodeDF(
                ntype=NodeType.sheet_cell,
                df=DataFrame(
                    {
                        NodeAttrKey.nid.value: [0, 1],
                        NodeAttrKey.ntype.value: [NodeType.sheet_cell.value] * 2,
                    }
                ),
            ),
            NodeDF(
                ntype=NodeType.token,
                df=DataFrame(
                    {
                        NodeAttrKey.nid.value: [0, 1, 2],
                        NodeAttrKey.ntype.value: [NodeType.token.value] * 3,
                    }
                ),
            ),
        ]
    )

    return node_dfs


@fixture
def mock_edge_dfs() -> EdgeDFs:
    edge_dfs = EdgeDFs(
        members=[
            EdgeDF(
                etype=EdgeType.token_to_cell,
                df=DataFrame(
                    {
                        EdgeAttrKey.eid.value: [(0, 0), (0, 1)],
                        EdgeAttrKey.etype.value: [EdgeType.token_to_cell.value] * 2,
                    }
                ),
            ),
            EdgeDF(
                etype=EdgeType.up,
                df=DataFrame(
                    {
                        EdgeAttrKey.eid.value: [(0, 1)],
                        EdgeAttrKey.etype.value: [EdgeType.up.value],
                    }
                ),
            ),
        ]
    )

    return edge_dfs


@fixture
def mock_json_str_df() -> str:
    return '{"schema":{"fields":[{"name":"index","type":"integer"},{"name":"eid","type":"string"},{"name":"etype","type":"string"}],"primaryKey":["index"],"pandas_version":"1.4.0"},"data":[{"index":0,"eid":[0,0],"etype":"TokenToCell"},{"index":1,"eid":[0,1],"etype":"TokenToCell"}]}'


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
