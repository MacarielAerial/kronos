from __future__ import annotations

import logging
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Tuple

import dacite
import orjson
from pandas import DataFrame

from kronos.nodes.utils_df_serialisation import default, df_type_hook

logger = logging.getLogger(__name__)


class NodeAttrKey(str, Enum):
    # Default
    nid = "nid"
    ntype = "ntype"

    # Layout
    text = "text"
    coord = "coord"


class NodeType(str, Enum):
    # Layout
    sheet_cell = "SheetCell"

    # Semantics
    token = "Token"
    ent = "Ent"
    ent_label = "EntLabel"


# Layout
SheetCellTuple = namedtuple(
    "SheetCellTuple",
    [
        "nid",
        "ntype",
        "text",
        "coord",
    ],
)

# Semantics
TokenTuple = namedtuple(
    "TokenTuple",
    ["nid", "ntype", "text"],
)
EntTuple = namedtuple("EntTuple", ["nid", "ntype", "text"])
EntLabelTuple = namedtuple("EntLabelTuple", ["nid", "ntype", "text"])


@dataclass
class NodeDF:
    ntype: NodeType
    df: DataFrame


@dataclass
class NodeDFs:
    members: List[NodeDF]

    def validate(self) -> None:
        list_ntype: List[NodeType] = [node_df.ntype for node_df in self.members]
        set_ntype: Set[NodeType] = {node_df.ntype for node_df in self.members}

        if len(set_ntype) != len(list_ntype):
            raise ValueError(
                "Node type dataframes including the following "
                f"node types are not unique:\n{list_ntype}"
            )

        list_nodes: List[Tuple[str, int]] = []
        set_nodes: Set[Tuple[str, int]] = set()
        for node_df in self.members:
            df = node_df.df
            list_nodes.extend(df[NodeAttrKey.nid.value].tolist())
            set_nodes = set_nodes | set(
                zip(df[NodeAttrKey.ntype.value], df[NodeAttrKey.nid.value])
            )

        if len(set_nodes) != len(list_nodes):
            raise ValueError(
                "Node ids are not unique within dataframes in "
                f"{self.__class__.__name__} object"
            )

    def to_dict(self) -> Dict[NodeType, DataFrame]:
        ntype_to_df: Dict[NodeType, DataFrame] = {
            node_df.ntype: node_df.df for node_df in self.members
        }

        return ntype_to_df

    @property
    def ntypes(self) -> List[NodeType]:
        return [node_df.ntype for node_df in self.members]


class NodeDFsDataInterface:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def save(self, node_dfs: NodeDFs) -> None:
        if not self.filepath.parent.exists():
            logger.info(
                f"Creating {self.filepath.parent} because it does not yet exist"
            )
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(self.filepath, "wb") as f:
            json_str = orjson.dumps(
                node_dfs, default=default, option=orjson.OPT_INDENT_2
            )
            f.write(json_str)

            logger.info(f"Saved a {type(node_dfs)} type object to {self.filepath}")

    def load(self) -> NodeDFs:
        with open(self.filepath, "rb") as f:
            json_data = orjson.loads(f.read())
            node_dfs = dacite.from_dict(
                data_class=NodeDFs,
                data=json_data,
                config=dacite.Config(
                    type_hooks={DataFrame: df_type_hook}, cast=[NodeType]
                ),
            )

            logger.info(f"Loaded a {type(node_dfs)} object from {self.filepath}")

            return node_dfs
