import networkx as nx

from kronos.data_interfaces.nx_g_data_interface import NXGDataInterface
from tests.conftest import TestDataPaths


def test_save(mock_nx_g: nx.Graph, test_data_paths: TestDataPaths) -> None:
    nx_g_data_interface = NXGDataInterface(filepath=test_data_paths.path_saved_nx_g)
    nx_g_data_interface.save(nx_g=mock_nx_g)

    assert test_data_paths.path_saved_nx_g.is_file()


def test_load(test_data_paths: TestDataPaths) -> None:
    nx_g_data_interface = NXGDataInterface(filepath=test_data_paths.path_mock_nx_g)
    nx_g = nx_g_data_interface.load()

    assert isinstance(nx_g, nx.Graph)
