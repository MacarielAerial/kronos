import logging
from typing import Any, Dict, List, Set, Tuple

from networkx import DiGraph

from kronos.data_interfaces.edge_dfs_data_interface import EdgeAttrKey, EdgeDF, EdgeDFs
from kronos.data_interfaces.node_dfs_data_interface import NodeAttrKey, NodeDF, NodeDFs

logger = logging.getLogger(__name__)


# TODO: Strength data validation
def validate_node_dfs_and_edge_dfs(node_dfs: NodeDFs, edge_dfs: EdgeDFs) -> None:
    # Collect all unique (node type, node id) tuples
    # e.g. (SheetCell, 0)
    set_nodes: Set[Tuple[str, int]] = set()
    for node_df in node_dfs.members:
        df = node_df.df
        set_nodes = set_nodes | set(
            zip(df[NodeAttrKey.ntype.value], df[NodeAttrKey.nid.value])
        )

    # Collect all unique (node type, node id) tuples referenced in edge dataframes
    set_nodes_in_edges: Set[Tuple[str, int]] = set()
    for edge_df in edge_dfs.members:
        df = edge_df.df
        # Collect source nodes
        set_nodes_in_edges = set_nodes_in_edges | set(
            zip(df[EdgeAttrKey.src_ntype.value], df[EdgeAttrKey.src_nid.value])
        )
        # Collect destination nodes
        set_nodes_in_edges = set_nodes_in_edges | set(
            zip(df[EdgeAttrKey.dst_ntype.value], df[EdgeAttrKey.dst_nid.value])
        )

    if set_nodes != set_nodes_in_edges:  # Island nodes are not allowed
        raise ValueError(
            f"The set of {len(set_nodes)} nodes in node datagrames and "
            f"{len(set_nodes_in_edges)} nodes referenced in edge dataframes are "
            "not identical"
        )


def node_tuples_from_node_df(
    node_df: NodeDF,
) -> List[Tuple[Tuple[str, int], Dict[str, Any]]]:
    df = node_df.df

    node_tuples: List[Tuple[Tuple[str, int], Dict[str, Any]]] = []

    for record in df.to_dict(orient="records"):
        # e.g. [{"nid": ("SheetCell", 0), "ntype": ...}, ...]
        nid = int(record.pop(NodeAttrKey.nid.value))
        # Node type information is duplicated for convenience
        ntype = str(record[NodeAttrKey.ntype.value])
        # The rest is assumed all to be attributes
        record_str_key = {str(k): v for k, v in record.items()}
        # Use (NodeType, NID) to index nodes to facilitate DGL Graph conversion later
        # Nodes in networkx share node id space, requiring tuple indices
        node_tuple: Tuple[Tuple[str, int], Dict[str, Any]] = (
            (ntype, nid),
            record_str_key,
        )
        node_tuples.append(node_tuple)

    logger.info(
        f"Parsed {len(node_tuples)} node tuples from {node_df.ntype.value} "
        "node dataframe"
    )

    return node_tuples


def node_tuples_from_node_dfs(
    node_dfs: NodeDFs,
) -> List[Tuple[Tuple[str, int], Dict[str, Any]]]:
    node_tuples: List[Tuple[Tuple[str, int], Dict[str, Any]]] = []

    for node_df in node_dfs.members:
        node_tuples.extend(node_tuples_from_node_df(node_df=node_df))

    return node_tuples


def edge_tuples_from_edge_df(
    edge_df: EdgeDF,
) -> List[Tuple[Tuple[str, int], Tuple[str, int], Dict[str, Any]]]:
    df = edge_df.df

    edge_tuples: List[Tuple[Tuple[str, int], Tuple[str, int], Dict[str, Any]]] = []

    for record in df.to_dict(orient="records"):
        # e.g. [{"src_nid": 0, "dst_nid": 1, "src_ntype": "SheetCell", "etype": "Up",
        # "dst_ntype": "SheetCell", ...}, ...]
        u = int(record.pop(EdgeAttrKey.src_nid.value))
        ntype_u = str(record.pop(EdgeAttrKey.src_ntype.value))
        v = int(record.pop(EdgeAttrKey.dst_nid.value))
        ntype_v = str(record.pop(EdgeAttrKey.dst_ntype.value))
        # The rest is assumed all to be attributes
        record_str_key = {str(k): v for k, v in record.items()}
        edge_tuple: Tuple[Tuple[str, int], Tuple[str, int], Dict[str, Any]] = (
            (ntype_u, u),
            (ntype_v, v),
            record_str_key,
        )
        edge_tuples.append(edge_tuple)

    logger.info(
        f"Parsed {len(edge_tuples)} edge tuples from {edge_df.etype.value} "
        "edge dataframe"
    )

    return edge_tuples


def edge_tuples_from_edge_dfs(
    edge_dfs: EdgeDFs,
) -> List[Tuple[Tuple[str, int], Tuple[str, int], Dict[str, Any]]]:
    edge_tuples: List[Tuple[Tuple[str, int], Tuple[str, int], Dict[str, Any]]] = []

    for edge_df in edge_dfs.members:
        edge_tuples.extend(edge_tuples_from_edge_df(edge_df=edge_df))

    return edge_tuples


def _assemble_kg(node_dfs: NodeDFs, edge_dfs: EdgeDFs) -> DiGraph:
    # Validate input
    node_dfs.validate()
    edge_dfs.validate()

    # Sanity check coherence between input
    validate_node_dfs_and_edge_dfs(node_dfs=node_dfs, edge_dfs=edge_dfs)

    # Transform graph elements into networkx graph compatible form
    node_tuples = node_tuples_from_node_dfs(node_dfs=node_dfs)
    edge_tuples = edge_tuples_from_edge_dfs(edge_dfs=edge_dfs)

    # Initialise the knowledge graph
    nx_g = DiGraph()
    nx_g.add_nodes_from(node_tuples)
    nx_g.add_edges_from(edge_tuples)

    logger.info(
        f"Initialised knowledge graph has {nx_g.number_of_nodes()} nodes "
        f"and {nx_g.number_of_edges()} edges"
    )

    return nx_g
