import numpy as np
import pandas as pd

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
from kronos.nodes.contract_sheet_cell_nodes import (
    _contract_sheet_cell_nodes,
    element_wise_sum_coords,
    merge_sheet_cell_nodes,
    update_nids_in_df_edge,
)


def test_agg_coords() -> None:
    # Arrange
    input_coords = [[1, 2], [3, 4], [5, 6]]
    expected_output = np.array([9, 12])

    # Act & Assert
    assert np.array_equal(
        element_wise_sum_coords(input_coords), expected_output
    ), "agg_coords should sum coordinates correctly."


def test_merge_sheet_cell_nodes() -> None:
    # Arrange
    df_nodes = pd.DataFrame(
        {
            NodeAttrKey.nid.value: [0, 1, 2, 3],
            NodeAttrKey.ntype.value: [NodeType.sheet_cell.value] * 4,
            NodeAttrKey.text.value: ["A"] * 2 + ["B"] * 2,
            NodeAttrKey.coord.value: [[0, 1], [1, 2], [2, 3], [3, 4]],
        }
    )

    expected_df = pd.DataFrame(
        {
            NodeAttrKey.nid.value: [0, 1],  # min is assumed to be the aggregation rule
            NodeAttrKey.ntype.value: [NodeType.sheet_cell.value] * 2,
            NodeAttrKey.text.value: ["A", "B"],
            NodeAttrKey.coord.value: [[1, 3], [5, 7]],  # Coordinates aggregated
        }
    )

    # Act
    merged_df, _ = merge_sheet_cell_nodes(df_nodes)

    # Assert
    pd.testing.assert_frame_equal(merged_df, expected_df)


def test_update_nids_in_df_edge() -> None:
    # Arrange
    df_edges = pd.DataFrame(
        {
            EdgeAttrKey.src_nid.value: [0, 1, 2, 3],
            EdgeAttrKey.dst_nid.value: [1, 2, 3, 0],
            EdgeAttrKey.distance.value: [1, 2, 3, 4],
        }
    )
    nid_mapping = {0: 10, 1: 11, 2: 12, 3: 13}
    expected_df_edges = pd.DataFrame(
        {
            EdgeAttrKey.src_nid.value: [10, 11, 12, 13],
            EdgeAttrKey.dst_nid.value: [11, 12, 13, 10],
            EdgeAttrKey.distance.value: [1, 2, 3, 4],
        }
    )

    # Act
    updated_df_edges = update_nids_in_df_edge(df_edges, nid_mapping)

    # Assert
    pd.testing.assert_frame_equal(updated_df_edges, expected_df_edges)


def test_contract_sheet_cell_nodes() -> None:
    # Arrange
    df_sheet_cell = pd.DataFrame(
        {
            NodeAttrKey.nid.value: [0, 1, 2, 3],
            NodeAttrKey.ntype.value: [NodeType.sheet_cell.value] * 4,
            NodeAttrKey.text.value: ["A"] * 2 + ["B"] * 2,
            NodeAttrKey.coord.value: [[0, 1], [1, 1], [0, 1], [1, 1]],
        }
    )
    df_traversal = pd.DataFrame(
        {
            EdgeAttrKey.src_nid.value: [0, 1, 2, 3],
            EdgeAttrKey.dst_nid.value: [1, 2, 3, 0],
            # Some edge attributes are omitted
        }
    )

    node_dfs = NodeDFs([NodeDF(df=df_sheet_cell, ntype=NodeType.sheet_cell)])
    edge_dfs = EdgeDFs([EdgeDF(df=df_traversal, etype=EdgeType.up)])

    # Act
    contracted_node_dfs, contracted_edge_dfs = _contract_sheet_cell_nodes(
        node_dfs, edge_dfs
    )

    # Assert
    # Check if node_dfs is updated correctly
    assert (
        len(contracted_node_dfs.members[0].df) == 2
    ), "Should merge nodes into two unique nodes based on text."
    # Verify the nid_mapping is applied correctly to edge_dfs
    assert (
        contracted_edge_dfs.members[0].df[EdgeAttrKey.src_nid.value].nunique() == 2
    ), "Edge source IDs should be updated to reflect merged nodes."
    assert (
        contracted_edge_dfs.members[0].df[EdgeAttrKey.dst_nid.value].nunique() == 2
    ), "Edge destination IDs should be updated to reflect merged nodes."
