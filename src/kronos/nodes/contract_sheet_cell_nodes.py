from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from kronos.data_interfaces.edge_dfs_data_interface import (
    EdgeAttrKey,
    EdgeDFs,
    TraversalEdgeTypes,
)
from kronos.data_interfaces.node_dfs_data_interface import (
    NodeAttrKey,
    NodeDFs,
    NodeType,
)


def element_wise_sum_coords(coord: List[List[int]]) -> List[int]:
    """e.g. pd.Series([[1, 0], [0, 1]]) row-wise aggregation to [[1, 1]]"""
    # Aggregate coord values by adding corresponding elements
    return np.sum(np.array(coord), axis=0).tolist()  # type: ignore[no-any-return]


def concat_nid(nid: pd.Series) -> list:
    return nid.tolist()


# TODO: Refactor the following function into a factory merge node function
# supporting custom attribute aggregation logic
def merge_sheet_cell_nodes(
    df_sheet_cell: DataFrame,
) -> Tuple[DataFrame, Dict[int, int]]:
    print(df_sheet_cell)
    # Nodes with identical node type and text values set to be merged
    grouped = df_sheet_cell.groupby([NodeAttrKey.ntype.value, NodeAttrKey.text.value])

    # Custom aggregation logic for each node attribute
    df_merged = grouped.apply(
        lambda x: pd.Series(
            {
                # Node ids are concatenated as graph element merging input
                NodeAttrKey.nid.value: concat_nid(x[NodeAttrKey.nid.value]),
                NodeAttrKey.coord.value: element_wise_sum_coords(
                    x[NodeAttrKey.coord.value].tolist()
                ),
            }
        )
    ).reset_index()[df_sheet_cell.columns]

    # Map old node ids to new node ids
    nid_mapping: Dict[int, int] = {
        i_old: i_new
        for i_new in df_merged.index
        for i_old in df_merged.loc[i_new, NodeAttrKey.nid.value]
    }

    # Set node ids with 0-indexed consecutive integer sequence
    df_merged[NodeAttrKey.nid.value] = df_merged.index

    return df_merged, nid_mapping


def update_nids_in_df_edge(
    df_edge: DataFrame, nid_mapping: Dict[int, int]
) -> DataFrame:
    # Update src_nid and dst_nid based on the nid_mapping
    df_edge[EdgeAttrKey.src_nid.value] = df_edge[EdgeAttrKey.src_nid.value].apply(
        lambda x: nid_mapping[x]
    )
    df_edge[EdgeAttrKey.dst_nid.value] = df_edge[EdgeAttrKey.dst_nid.value].apply(
        lambda x: nid_mapping[x]
    )

    return df_edge


def _contract_sheet_cell_nodes(
    node_dfs: NodeDFs, edge_dfs: EdgeDFs
) -> Tuple[NodeDFs, EdgeDFs]:
    # Identify index of sheelt cell node dataframe
    i = node_dfs.ntypes.index(NodeType.sheet_cell)

    # Merge nodes based on common text
    node_dfs.members[i].df, nid_mapping = merge_sheet_cell_nodes(
        df_sheet_cell=node_dfs.members[i].df
    )

    # Replace node references in edge dataframes
    for i in range(len(edge_dfs.members)):
        if edge_dfs.members[i].etype in TraversalEdgeTypes:
            edge_dfs.members[i].df = update_nids_in_df_edge(
                df_edge=edge_dfs.members[i].df, nid_mapping=nid_mapping
            )

    return node_dfs, edge_dfs
