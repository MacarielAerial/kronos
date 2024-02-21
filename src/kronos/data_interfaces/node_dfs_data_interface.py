from __future__ import annotations

import logging
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set

import dacite
import orjson
from pandas import DataFrame

from kronos.nodes.utils_df_serialisation import default, df_type_hook

logger = logging.getLogger(__name__)


class NodeAttrKey(str, Enum):
    nid = "nid"
    ntype = "ntype"
    text = "text"
    coord = "coord"


class NodeType(str, Enum):
    sheet_cell = "SheetCell"
    token = "Token"


SheetCellTuple = namedtuple(
    "SheetCellTuple",
    [
        "nid",
        "ntype",
        "text",
        "coord",
    ],
)
TokenTuple = namedtuple(
    "TokenTuple",
    ["nid", "ntype", "text"],
)


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

        list_nid: List[int] = []
        set_nid: Set[int] = set()
        for node_df in self.members:
            df = node_df.df
            list_nid.extend(df[NodeAttrKey.nid.value].tolist())
            set_nid = set_nid | set(df[NodeAttrKey.nid.value])

        if len(set_nid) != len(list_nid):
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
