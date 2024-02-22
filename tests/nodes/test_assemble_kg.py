import pytest
from networkx import DiGraph
from pandas import DataFrame

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
from kronos.nodes.assemble_kg import (
    _assemble_kg,
    edge_tuples_from_edge_df,
    edge_tuples_from_edge_dfs,
    node_tuples_from_node_df,
    node_tuples_from_node_dfs,
    validate_node_dfs_and_edge_dfs,
)


def test_validate_node_dfs_and_edge_dfs_valid(
    mock_node_dfs: NodeDFs, mock_edge_dfs: EdgeDFs
) -> None:
    try:
        validate_node_dfs_and_edge_dfs(node_dfs=mock_node_dfs, edge_dfs=mock_edge_dfs)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


def test_validate_node_dfs_and_edge_dfs_invalid(mock_node_dfs: NodeDFs) -> None:
    # Arrange
    # Mock an invalid edge dataframe
    df_edge_invalid = DataFrame(
        {
            EdgeAttrKey.src_nid: [3],
            EdgeAttrKey.dst_nid: [0],
            EdgeAttrKey.src_ntype: NodeType.sheet_cell.value,
            EdgeAttrKey.etype.value: EdgeType.down.value,
            EdgeAttrKey.dst_ntype: NodeType.sheet_cell.value,
        }
    )  # Non-matching IDs
    edge_df_invalid = EdgeDF(etype=EdgeType.down, df=df_edge_invalid)
    mock_edge_dfs_invalid = EdgeDFs(members=[edge_df_invalid])

    # Act and Assert
    with pytest.raises(ValueError):
        validate_node_dfs_and_edge_dfs(
            node_dfs=mock_node_dfs, edge_dfs=mock_edge_dfs_invalid
        )


def test_node_tuples_from_node_df(mock_node_df: NodeDF) -> None:
    # Arrange
    expected = [
        (
            (NodeType.sheet_cell.value, 0),
            {NodeAttrKey.ntype.value: NodeType.sheet_cell.value},
        ),
        (
            (NodeType.sheet_cell.value, 1),
            {NodeAttrKey.ntype.value: NodeType.sheet_cell.value},
        ),
    ]

    # Act
    result = node_tuples_from_node_df(node_df=mock_node_df)

    # Assert
    assert (
        result == expected
    ), "Node tuples transformation did not match expected output."


def test_node_tuples_from_node_dfs(mock_node_dfs: NodeDFs) -> None:
    result = node_tuples_from_node_dfs(node_dfs=mock_node_dfs)

    assert len(result) == 3, "The total number of node tuples is not as expected"
    assert isinstance(result[0], tuple), "The output should be a list of tuples."


def test_edge_tuples_from_edge_df(mock_edge_df: EdgeDF) -> None:
    # Arrange
    expected = [
        (
            (NodeType.token.value, 0),
            (NodeType.sheet_cell.value, 1),
            {EdgeAttrKey.etype.value: EdgeType.token_to_cell.value},
        )
    ]

    # Act
    result = edge_tuples_from_edge_df(edge_df=mock_edge_df)

    # Assert
    assert (
        result == expected
    ), "Edge tuples transformation did not match expected output."


def test_edge_tuples_from_edge_dfs(mock_edge_dfs: EdgeDFs) -> None:
    result = edge_tuples_from_edge_dfs(edge_dfs=mock_edge_dfs)

    assert len(result) == 3, "The total number of edge tuples is not as expected"
    assert isinstance(result[0], tuple), "The output should be a list of tuples."


def test_assemble_kg_valid(mock_node_dfs: NodeDFs, mock_edge_dfs: EdgeDFs) -> None:
    kg = _assemble_kg(node_dfs=mock_node_dfs, edge_dfs=mock_edge_dfs)

    assert isinstance(
        kg, DiGraph
    ), "The assembled knowledge graph is not a DiGraph instance."
    assert kg.number_of_nodes() == 3, "Unexpected number of nodes is present in kg."
    assert kg.number_of_edges() == 3, "Unexpected number of edges is present in kg."
