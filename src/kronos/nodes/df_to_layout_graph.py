import logging
from collections import namedtuple
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd
from pandas import DataFrame

from kronos.data_interfaces.edge_dfs_data_interface import (
    DownTuple,
    EdgeAttrKey,
    EdgeDF,
    EdgeDFs,
    EdgeType,
    LeftTuple,
    RightTuple,
    TraversalTuple,
    UpTuple,
)
from kronos.data_interfaces.node_dfs_data_interface import (
    NodeAttrKey,
    NodeDF,
    NodeType,
    SheetCellTuple,
)
from kronos.nodes.utils_df_to_layout_graph import squeeze_tuple

logger = logging.getLogger(__name__)


TraDstTuple = namedtuple("TraDstTuple", ["i_row", "i_col", "distance"])


class Direction(str, Enum):
    up = "up"
    down = "down"
    left = "left"
    right = "right"


def find_first_non_null(  # noqa: C901
    df: DataFrame, start_row: int, start_col: int, direction: Direction
) -> Optional[TraDstTuple]:
    """The functions returns value of the first non-null cell to one of
    the four directions of a google sheet cell"""
    rows, cols = df.shape
    r, c = start_row, start_col

    if direction == Direction.up:
        for i in range(r - 1, -1, -1):
            if pd.notna(df.iloc[i, c]):
                return TraDstTuple(i, c, r - i)  # Distance is r - i
    elif direction == Direction.down:
        for i in range(r + 1, rows):
            if pd.notna(df.iloc[i, c]):
                return TraDstTuple(i, c, i - r)  # Distance is i - r
    elif direction == Direction.left:
        for j in range(c - 1, -1, -1):
            if pd.notna(df.iloc[r, j]):
                return TraDstTuple(r, j, c - j)  # Distance is c - j
    elif direction == Direction.right:
        for j in range(c + 1, cols):
            if pd.notna(df.iloc[r, j]):
                return TraDstTuple(r, j, j - c)  # Distance is j - c

    return None


def _df_to_layout_graph(df: DataFrame) -> Tuple[NodeDF, EdgeDFs]:
    sheet_cells: List[SheetCellTuple] = []
    traversals: List[TraversalTuple] = []
    direction_to_tetype: Dict[Direction, EdgeType] = {
        Direction.up: EdgeType.up,
        Direction.down: EdgeType.down,
        Direction.left: EdgeType.left,
        Direction.right: EdgeType.right,
    }
    direction_to_ttuple_type: Dict[Direction, Type[TraversalTuple]] = {
        Direction.up: UpTuple,
        Direction.down: DownTuple,
        Direction.left: LeftTuple,
        Direction.right: RightTuple,
    }

    rows, cols = df.shape
    curr_nid: int = 0

    # Assemble sheet cell node dataframe first
    for r in range(rows):
        for c in range(cols):
            # Ignore null cells
            if pd.notna(df.iloc[r, c]):
                # Collect non null cells as nodes
                sheet_cell = SheetCellTuple(
                    curr_nid, NodeType.sheet_cell.value, df.iloc[r, c], (r, c)
                )
                sheet_cells.append(sheet_cell)
                curr_nid += 1
    df_sheet_cell = DataFrame(sheet_cells)
    multi_index = pd.MultiIndex.from_tuples(
        df_sheet_cell[NodeAttrKey.coord.value], names=["x", "y"]
    )
    indexed_df_sheet_cell = df_sheet_cell.set_index(multi_index)

    logger.info(
        f"{NodeType.sheet_cell.value} node dataframe has shape "
        f"{df_sheet_cell.shape}"
    )

    # Assemble node dataframe object
    sheet_cell_node_df = NodeDF(ntype=NodeType.sheet_cell, df=df_sheet_cell)

    # Assemble traversal edge edgeframe second
    for r_src, c_src in indexed_df_sheet_cell.index:
        src_nid = indexed_df_sheet_cell.loc[(r_src, c_src), NodeAttrKey.nid.value]
        # Collect one edge per direction per node to the first non null cell
        for direction in Direction:
            traversal_tuple_type = direction_to_ttuple_type[direction]
            tra_dst_tuple = find_first_non_null(df, r_src, c_src, Direction.up)
            if tra_dst_tuple is not None:
                r_dst, c_dst, dis = tra_dst_tuple
                dst_nid = indexed_df_sheet_cell.loc[
                    (r_dst, c_dst), NodeAttrKey.nid.value
                ]
                tetype = direction_to_tetype[direction]
                traversal_tuple = traversal_tuple_type(
                    (src_nid, dst_nid), tetype.value, dis
                )
                traversals.append(traversal_tuple)
    df_traversal = DataFrame(traversals)

    logger.info(f"Traversal edge dataframe has shape {df_traversal.values}")

    # Initialise edge dataframes as a single object
    edge_dfs = EdgeDFs(members=[])

    # Populate edge dataframes with sub dataframes grouped by edge type
    for etype_val, df_by_etype in df_traversal.groupby([EdgeAttrKey.etype.value]):
        etype = EdgeType(squeeze_tuple(etype_val))
        edge_df = EdgeDF(etype=etype, df=df_by_etype)
        edge_dfs.members.append(edge_df)
        logger.info(
            f"Factored out {etype} edge dataframe has shape " f"{df_by_etype.shape}"
        )

    edge_dfs.validate()

    return sheet_cell_node_df, edge_dfs
