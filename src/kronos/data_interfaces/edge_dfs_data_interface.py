from __future__ import annotations

import logging
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import dacite
import orjson
from pandas import DataFrame

from kronos.nodes.utils_df_serialisation import default, df_type_hook

logger = logging.getLogger(__name__)


class EdgeAttrKey(str, Enum):
    # Default
    # Default
    src_nid = "src_nid"
    dst_nid = "dst_nid"
    src_ntype = "src_ntype"
    etype = "etype"
    dst_ntype = "dst_ntype"

    # Layout

    # Layout
    distance = "distance"

    # Semantics
    i_token_in_doc = "i_token_in_doc"
    i_token_in_ent = "i_token_in_ent"
    i_ent_in_doc = "i_ent_in_doc"

    # Semantics
    i_token_in_doc = "i_token_in_doc"
    i_token_in_ent = "i_token_in_ent"
    i_ent_in_doc = "i_ent_in_doc"


class EdgeType(str, Enum):
    # Layout
    # Layout
    up = "Up"
    down = "Down"
    left = "Left"
    right = "Right"

    # Semantics
    # Semantics
    token_to_cell = "TokenToCell"
    token_to_ent = "TokenToEnt"
    ent_to_cell = "EntToCell"
    ent_to_label = "EntToLabel"


TraversalEdgeTypes: Tuple[EdgeType, ...] = (
    EdgeType.up,
    EdgeType.down,
    EdgeType.left,
    EdgeType.right,
)

# Layout
UpTuple = namedtuple(
    "UpTuple",
    ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype", "distance"],
)
DownTuple = namedtuple(
    "DownTuple",
    ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype", "distance"],
)
LeftTuple = namedtuple(
    "LeftTuple",
    ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype", "distance"],
)
RightTuple = namedtuple(
    "RightTuple",
    ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype", "distance"],
)

# Semantics

# Semantics
TokenToCellTuple = namedtuple(
    "TokenToCellTuple",
    ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype", "i_token_in_doc"],
)
TokenToEntTuple = namedtuple(
    "TokenToEntTuple",
    ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype", "i_token_in_ent"],
)
EntToCellTuple = namedtuple(
    "EntToCellTuple",
    ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype", "i_ent_in_doc"],
)
EntToLabelTuple = namedtuple(
    "EntToLabelTuple", ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype"]
    "TokenToCellTuple",
    ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype", "i_token_in_doc"],
)
TokenToEntTuple = namedtuple(
    "TokenToEntTuple",
    ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype", "i_token_in_ent"],
)
EntToCellTuple = namedtuple(
    "EntToCellTuple",
    ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype", "i_ent_in_doc"],
)
EntToLabelTuple = namedtuple(
    "EntToLabelTuple", ["src_nid", "dst_nid", "src_ntype", "etype", "dst_ntype"]
)

TraversalTuple = Union[UpTuple, DownTuple, LeftTuple, RightTuple]


@dataclass
class EdgeDF:
    etype: EdgeType
    df: DataFrame


@dataclass
class EdgeDFs:
    members: List[EdgeDF]

    def validate(self) -> None:
        list_etype: List[EdgeType] = [edge_df.etype for edge_df in self.members]
        set_etype: Set[EdgeType] = {edge_df.etype for edge_df in self.members}

        if len(set_etype) != len(list_etype):
            raise ValueError(
                "Edge type dataframes including the following "
                f"canonical edge types are not unique:\n{list_etype}"
            )

    def to_dict(self) -> Dict[EdgeType, DataFrame]:
        etype_to_df: Dict[EdgeType, DataFrame] = {
            edge_df.etype: edge_df.df for edge_df in self.members
        }

        return etype_to_df

    @property
    def etypes(self) -> List[EdgeType]:
        return [edge_df.etype for edge_df in self.members]


class EdgeDFsDataInterface:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def save(self, edge_dfs: EdgeDFs) -> None:
        if not self.filepath.parent.exists():
            logger.info(
                f"Creating {self.filepath.parent} because it does not yet exist"
            )
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(self.filepath, "wb") as f:
            json_str = orjson.dumps(
                edge_dfs, default=default, option=orjson.OPT_INDENT_2
            )
            f.write(json_str)

            logger.info(f"Saved a {type(edge_dfs)} type object to {self.filepath}")

    def load(self) -> EdgeDFs:
        with open(self.filepath, "rb") as f:
            json_data = orjson.loads(f.read())
            edge_dfs = dacite.from_dict(
                data_class=EdgeDFs,
                data=json_data,
                config=dacite.Config(
                    type_hooks={DataFrame: df_type_hook}, cast=[EdgeType]
                ),
            )

            logger.info(f"Loaded a {type(edge_dfs)} object from {self.filepath}")

            return edge_dfs
