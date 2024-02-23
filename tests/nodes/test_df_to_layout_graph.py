from typing import Optional

import pandas as pd
import pytest

from kronos.data_interfaces.edge_dfs_data_interface import EdgeDF, EdgeDFs
from kronos.data_interfaces.node_dfs_data_interface import NodeDF, NodeType
from kronos.nodes.df_to_layout_graph import (
    Direction,
    TraDstTuple,
    _df_to_layout_graph,
    find_first_non_null,
)


@pytest.mark.parametrize(
    "direction,expected",
    [
        (Direction.up, TraDstTuple(0, 1, 1)),
        (Direction.down, None),  # Assuming there's no non-null cell below
        (Direction.left, TraDstTuple(1, 0, 1)),
        (Direction.right, None),  # Assuming there's no non-null cell to the right
    ],
)
def test_find_first_non_null(
    direction: Direction, expected: Optional[TraDstTuple]
) -> None:
    df = pd.DataFrame({0: [None, "start"], 1: ["target", None]})

    result = find_first_non_null(df, start_row=1, start_col=1, direction=direction)
    assert result == expected, f"Failed for direction: {direction}"


def test_df_to_layout_graph() -> None:
    df = pd.DataFrame({0: ["A", None, "B"], 1: [None, "C", None]})

    node_df, edge_dfs = _df_to_layout_graph(df)

    # Validate NodeDF creation
    assert isinstance(node_df, NodeDF)
    assert node_df.ntype == NodeType.sheet_cell
    assert len(node_df.df) == 3  # Expecting 3 non-null cells

    # Validate EdgeDFs creation
    assert isinstance(edge_dfs, EdgeDFs)
    # TODO: Validate the content of edge dataframes as well
    assert all(isinstance(edge_df, EdgeDF) for edge_df in edge_dfs.members)
