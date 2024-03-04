import logging
from pathlib import Path

import networkx as nx
import orjson
from networkx import Graph

logger = logging.getLogger(__name__)


class NXGDataInterface:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def save(self, nx_g: Graph) -> None:
        if not self.filepath.parent.exists():
            logger.info(
                f"Creating {self.filepath.parent} because it does not yet exist"
            )
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(self.filepath, "wb") as f:
            json_data = nx.node_link_data(nx_g)

            f.write(orjson.dumps(json_data, option=orjson.OPT_INDENT_2))

            logger.info(f"Saved a {type(nx_g)} object to {self.filepath}")

    def load(self) -> Graph:
        with open(self.filepath, "rb") as f:
            json_data = orjson.loads(f.read())

            nx_g = nx.node_link_graph(json_data)

            if not isinstance(nx_g, Graph):
                raise TypeError(f"Invalid return object type {type(nx_g)}")

            logger.info(f"Loaded a {type(nx_g)} object from {self.filepath}")

            return nx_g
