import logging
import random
from enum import Enum
from typing import Any, Dict, List, Tuple

import networkx as nx
import pandas as pd

from kronos.nodes.graph_schema import EdgeAttrKey, EdgeType

logger = logging.getLogger(__name__)


class Direction(str, Enum):
    up = "up"
    down = "down"
    left = "left"
    right = "right"


def find_first_non_null(  # type: ignore # noqa: C901
    df, start_row, start_col, direction
):
    rows, cols = df.shape
    r, c = start_row, start_col

    if direction == Direction.up:
        for i in range(r - 1, -1, -1):
            if pd.notna(df.iloc[i, c]):
                return i, c, r - i  # Distance is r - i
    elif direction == Direction.down:
        for i in range(r + 1, rows):
            if pd.notna(df.iloc[i, c]):
                return i, c, i - r  # Distance is i - r
    elif direction == Direction.left:
        for j in range(c - 1, -1, -1):
            if pd.notna(df.iloc[r, j]):
                return r, j, c - j  # Distance is c - j
    elif direction == Direction.right:
        for j in range(c + 1, cols):
            if pd.notna(df.iloc[r, j]):
                return r, j, j - c  # Distance is j - c
    return None


def df_to_layout_nx_g(df: pd.DataFrame) -> nx.DiGraph:
    nx_g: nx.DiGraph = nx.DiGraph()
    edge_tuples: List[Tuple[int, int, Dict[str, Any]]] = []

    rows, cols = df.shape
    for r in range(rows):
        for c in range(cols):
            if pd.notna(df.iloc[r, c]):
                # Add node with 'raw_text' attribute
                nx_g.add_node((r, c), raw_text=df.iloc[r, c])

                # Check each direction and connect to the first non-null cell
                for direction in Direction:
                    target = find_first_non_null(df, r, c, direction)
                    if target:
                        target_row, target_col, distance = target
                        edge_attrs = {
                            EdgeAttrKey.direction.value: direction.value,
                            EdgeAttrKey.etype.value: EdgeType.layout.value,
                            EdgeAttrKey.distance.value: distance,
                        }
                        edge_tuple = ((r, c), (target_row, target_col), edge_attrs)
                        edge_tuples.append(edge_tuple)  # type: ignore[arg-type]

    # Add all edges from the prepared list
    nx_g.add_edges_from(edge_tuples)

    logger.info(
        f"Added {len(edge_tuples)} layout edges. Here is an example:\n"
        f"{random.choice(edge_tuples)}"
    )

    return nx_g


def split_mixed_type_nodes(nx_g: nx.DiGraph) -> nx.DiGraph:  # type: ignore
    """Splits nodes which contain multiple NER entities into separate nodes
    while maintaining the visibility graph structure"""
    # pseudocode below
    """
    parts = USE NER TO OBTAIN PARTS
    # Remove the parent node
    graph.remove_node(node)
    last_child_node = None
    child_nodes = []

    for i, part in enumerate(parts):
        # Create a child node position (could be based on the parent node position)
        child_node = (node[0], node[1], i)
        graph.add_node(child_node, raw_text=part)
        child_nodes.append(child_node)

        # Connect child nodes sequentially
        if last_child_node is not None:
            graph.add_edge(last_child_node, child_node,
                           **{EdgeAttrKey.direction.value: Direction.right.value,
                              EdgeAttrKey.etype.value: EdgeType.layout.value,
                              EdgeAttrKey.distance.value: 1})
        last_child_node = child_node

    # Replicate parent's edges for each child
    for child_node in child_nodes:
        for edge in graph.edges(node):
            source, target = edge
            if source == node:
                graph.add_edge(child_node, target, **graph.get_edge_data(source, target))
            else:
                graph.add_edge(source, child_node, **graph.get_edge_data(source, target))

    """
    pass
