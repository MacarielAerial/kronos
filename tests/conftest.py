import logging
import shutil
from pathlib import Path
from typing import List

import spacy
from networkx import Graph
from pandas import DataFrame
from pytest import ExitCode, Session, fixture
from spacy.language import Language

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
    def path_mock_timetable_df(self) -> Path:
        return self.path_dir_data / "mock_timetable_df.csv"

    @property
    def path_mock_nx_g(self) -> Path:
        return self.path_dir_data / "mock_nx_g.json"

    @property
    def path_en_sm_spacy_pipeline(self) -> Path:
        return (
            self.path_dir_data
            / "en_core_web_sm-3.7.1/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
        )

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

    @property
    def path_saved_nx_g(self) -> Path:
        return self.path_dir_output / "saved_nx_g.json"

    @property
    def path_integration_nx_g(self) -> Path:
        return self.path_dir_output / "integration_nx_g.json"


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
def mock_node_df() -> NodeDF:
    df_sheet_cell = DataFrame(
        {
            NodeAttrKey.nid.value: [0, 1],
            NodeAttrKey.ntype.value: [NodeType.sheet_cell.value] * 2,
        }
    )
    return NodeDF(ntype=NodeType.sheet_cell, df=df_sheet_cell)


@fixture
def mock_node_dfs(mock_node_df: NodeDF) -> NodeDFs:
    node_dfs = NodeDFs(
        members=[
            mock_node_df,
            NodeDF(
                ntype=NodeType.token,
                df=DataFrame(
                    {
                        NodeAttrKey.nid.value: [0],
                        NodeAttrKey.ntype.value: [NodeType.token.value],
                    }
                ),
            ),
        ]
    )

    return node_dfs


@fixture
def mock_edge_df() -> EdgeDF:
    return EdgeDF(
        etype=EdgeType.token_to_cell,
        df=DataFrame(
            {
                EdgeAttrKey.src_nid.value: [0],
                EdgeAttrKey.dst_nid.value: [1],
                EdgeAttrKey.src_ntype.value: [NodeType.token.value],
                EdgeAttrKey.dst_ntype.value: [NodeType.sheet_cell.value],
                EdgeAttrKey.etype.value: [EdgeType.token_to_cell.value],
            }
        ),
    )


@fixture
def mock_edge_dfs(mock_edge_df: EdgeDF) -> EdgeDFs:
    edge_dfs = EdgeDFs(
        members=[
            mock_edge_df,
            EdgeDF(
                etype=EdgeType.up,
                df=DataFrame(
                    {
                        EdgeAttrKey.src_nid.value: [0, 1],
                        EdgeAttrKey.dst_nid.value: [1, 0],
                        EdgeAttrKey.src_ntype.value: [NodeType.sheet_cell.value] * 2,
                        EdgeAttrKey.dst_ntype.value: [NodeType.sheet_cell.value] * 2,
                        EdgeAttrKey.etype.value: [
                            EdgeType.up.value,
                            EdgeType.down.value,
                        ],
                        EdgeAttrKey.distance.value: [4, 4],
                    }
                ),
            ),
        ]
    )

    return edge_dfs


@fixture
def mock_json_str_df() -> str:
    return '{"schema":{"fields":[{"name":"index","type":"integer"},{"name":"dummy","type":"string"},{"name":"etype","type":"string"}],"primaryKey":["index"],"pandas_version":"1.4.0"},"data":[{"index":0,"dummy":[0,0],"etype":"TokenToCell"},{"index":1,"dummy":[0,1],"etype":"TokenToCell"}]}'


@fixture
def mock_nx_g() -> Graph:
    nx_g = Graph()
    nx_g.add_node(0, ntype="haha")
    nx_g.add_node(1, ntype="ohno")
    nx_g.add_edge(0, 1, etype="lol")

    return nx_g


@fixture
def en_sm_spacy_pipeline(test_data_paths: TestDataPaths) -> Language:
    return spacy.load(test_data_paths.path_en_sm_spacy_pipeline)


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
