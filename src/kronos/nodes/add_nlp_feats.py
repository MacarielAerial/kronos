import logging
from typing import List, Tuple

from pandas import DataFrame
from spacy.language import Language

from kronos.data_interfaces.edge_dfs_data_interface import (
    EdgeDF,
    EdgeDFs,
    EdgeType,
    EntToCellTuple,
    EntToLabelTuple,
    TokenToCellTuple,
    TokenToEntTuple,
)
from kronos.data_interfaces.node_dfs_data_interface import (
    EntLabelTuple,
    EntTuple,
    NodeAttrKey,
    NodeDF,
    NodeDFs,
    NodeType,
    TokenTuple,
)
from kronos.nodes.assemble_kg import validate_node_dfs_and_edge_dfs

logger = logging.getLogger(__name__)


def assemble_nlp_ntuples_etuples(
    token_text: List[str],
    ent_text: List[str],
    ent_label_text: List[str],
    i_token_to_cell: List[Tuple[int, int]],
    i_token_to_ent: List[Tuple[int, int]],
    i_ent_to_cell: List[Tuple[int, int]],
    i_ent_to_label: List[Tuple[int, int]],
    i_token_in_doc: List[Tuple[int, int]],
    i_token_in_ent: List[Tuple[int, int]],
    i_ent_in_doc: List[Tuple[int, int]],
) -> Tuple[
    List[TokenTuple],
    List[EntTuple],
    List[EntLabelTuple],
    List[TokenToCellTuple],
    List[TokenToEntTuple],
    List[EntToCellTuple],
    List[EntToLabelTuple],
]:
    logger.info("Assembling node tuples and edge tuples based on derived nlp features")

    # Assemble node tuples
    token = []
    for i, text in enumerate(token_text):
        t_tuple = TokenTuple(nid=i, ntype=NodeType.token.value, text=text)
        token.append(t_tuple)

    ent = []
    for i, text in enumerate(ent_text):
        e_tuple = EntTuple(nid=i, ntype=NodeType.ent.value, text=text)
        ent.append(e_tuple)

    ent_label = []
    for i, text in enumerate(ent_label_text):
        el_tuple = EntLabelTuple(nid=i, ntype=NodeType.ent_label.value, text=text)
        ent_label.append(el_tuple)

    # Assemble edge tuples
    token_to_cell = []
    for ttc, tid in zip(i_token_to_cell, i_token_in_doc):
        src_nid, dst_nid = ttc
        src_ntype, etype, dst_ntype = (
            NodeType.token.value,
            EdgeType.token_to_cell.value,
            NodeType.sheet_cell.value,
        )
        ttc_tuple = TokenToCellTuple(
            src_nid=src_nid,
            dst_nid=dst_nid,
            src_ntype=src_ntype,
            etype=etype,
            dst_ntype=dst_ntype,
            i_token_in_doc=tid,
        )
        token_to_cell.append(ttc_tuple)

    token_to_ent = []
    for tte, tie in zip(i_token_to_ent, i_token_in_ent):
        src_nid, dst_nid = tte
        src_ntype, etype, dst_ntype = (
            NodeType.token.value,
            EdgeType.token_to_ent.value,
            NodeType.ent.value,
        )
        tte_tuple = TokenToEntTuple(
            src_nid=src_nid,
            dst_nid=dst_nid,
            src_ntype=src_ntype,
            etype=etype,
            dst_ntype=dst_ntype,
            i_token_in_ent=tie,
        )
        token_to_ent.append(tte_tuple)

    ent_to_cell = []
    for etc, eid in zip(i_ent_to_cell, i_ent_in_doc):
        src_nid, dst_nid = etc
        src_ntype, etype, dst_ntype = (
            NodeType.ent.value,
            EdgeType.ent_to_cell.value,
            NodeType.sheet_cell.value,
        )
        etc_tuple = EntToCellTuple(
            src_nid=src_nid,
            dst_nid=dst_nid,
            src_ntype=src_ntype,
            etype=etype,
            dst_ntype=dst_ntype,
            i_ent_in_doc=eid,
        )
        ent_to_cell.append(etc_tuple)

    ent_to_label = []
    for etl in i_ent_to_label:
        src_nid, dst_nid = etl
        src_ntype, etype, dst_ntype = (
            NodeType.ent.value,
            EdgeType.ent_to_label.value,
            NodeType.ent_label.value,
        )
        etl_tuple = EntToLabelTuple(
            src_nid=src_nid,
            dst_nid=dst_nid,
            src_ntype=src_ntype,
            etype=etype,
            dst_ntype=dst_ntype,
        )
        ent_to_label.append(etl_tuple)

    return (
        token,
        ent,
        ent_label,
        token_to_cell,
        token_to_ent,
        ent_to_cell,
        ent_to_label,
    )


def prep_nlp_ntuples_etuples_input(
    df: DataFrame, spacy_pipeline: Language
) -> Tuple[
    List[str],
    List[str],
    List[str],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[int],
    List[int],
    List[int],
]:
    # Assume the input node dataframe has a text attribute
    cell = df[NodeAttrKey.text.value].tolist()

    logger.info(f"{len(cell)} layout graph nodes are to be processed for nlp features")

    # Node ids are assumed to be identical to 0-indexed integer index
    if df[NodeAttrKey.nid.value].tolist() != list(range(df.shape[0])):
        raise ValueError(
            f"{NodeAttrKey.nid.value} field is not a consecutive "
            "sequence of 0-indexed integers"
        )

    # Initialise iterables to keep track of entities
    token: List[str] = []
    ent: List[str] = []
    ent_label: List[str] = []

    # Initialise iterables to keep track of relations between entities
    i_token_to_cell: List[Tuple[int, int]] = []
    i_token_to_ent: List[Tuple[int, int]] = []
    i_ent_to_cell: List[Tuple[int, int]] = []
    i_ent_to_label: List[Tuple[int, int]] = []

    # Initialise iterables to keep track of relation attributes
    i_token_in_doc: List[int] = []
    i_token_in_ent: List[int] = []
    i_ent_in_doc: List[int] = []

    # Use a small number of processes and a large batch size
    # to reduce overhead and to improve platform compatibility
    for doc in spacy_pipeline.pipe(texts=cell, n_process=2, batch_size=2000):
        for t in doc:
            # Collect token as a class of entities
            if t.text not in token:
                token.append(t.text)

            # Collect token to cell as a class of relations
            i_token_to_cell.append((token.index(t.text), cell.index(doc.text)))

            # Collect token to cell attributes
            i_token_in_doc.append(t.i)

        for i_e_i_d, e in enumerate(doc.ents):
            # Collect ner entity as a class of entities
            if e.text not in ent:
                ent.append(e.text)

            # Collect ner label as a class of entities
            if e.label_ not in ent_label:
                ent_label.append(e.label_)

            for et in e:
                # Collect token to ner entity as a class of relations
                i_token_to_ent.append((token.index(et.text), ent.index(e.text)))

                # Collect token to ner entity attributes
                i_token_in_ent.append(et.i)

            # Collect ner entity to cell as a class of relations
            i_ent_to_cell.append((ent.index(e.text), cell.index(doc.text)))

            # Collect ner entity to cell attributes
            i_ent_in_doc.append(i_e_i_d)

            # Collect ner entity to ner label as a class of relations
            i_ent_to_label.append((ent.index(e.text), ent_label.index(e.label_)))

    return (
        token,
        ent,
        ent_label,
        i_token_to_cell,
        i_token_to_ent,
        i_ent_to_cell,
        i_ent_to_label,
        i_token_in_doc,
        i_token_in_ent,
        i_ent_in_doc,
    )


def assemble_nlp_ndfs_edfs(
    token: List[TokenTuple],
    ent: List[EntTuple],
    ent_label: List[EntLabelTuple],
    token_to_cell: List[TokenToCellTuple],
    token_to_ent: List[TokenToEntTuple],
    ent_to_cell: List[EntToCellTuple],
    ent_to_label: List[EntToLabelTuple],
) -> Tuple[List[NodeDF], List[EdgeDF]]:
    logger.info(
        "Assembling node and edge dataframes based on " "parsed node and edge tuples"
    )

    token_node_df = NodeDF(ntype=NodeType.token, df=DataFrame(token))
    ent_node_df = NodeDF(ntype=NodeType.ent, df=DataFrame(ent))
    ent_label_node_df = NodeDF(ntype=NodeType.ent_label, df=DataFrame(ent_label))

    token_to_cell_edge_df = EdgeDF(
        etype=EdgeType.token_to_cell, df=DataFrame(token_to_cell)
    )
    token_to_ent_edge_df = EdgeDF(
        etype=EdgeType.token_to_ent, df=DataFrame(token_to_ent)
    )
    ent_to_cell_edge_df = EdgeDF(etype=EdgeType.ent_to_cell, df=DataFrame(ent_to_cell))
    ent_to_label_edge_df = EdgeDF(
        etype=EdgeType.ent_to_label, df=DataFrame(ent_to_label)
    )

    return (
        [token_node_df, ent_node_df, ent_label_node_df],
        [
            token_to_cell_edge_df,
            token_to_ent_edge_df,
            ent_to_cell_edge_df,
            ent_to_label_edge_df,
        ],
    )


def _add_nlp_feats(
    node_dfs: NodeDFs, edge_dfs: EdgeDFs, spacy_pipeline: Language
) -> Tuple[NodeDFs, EdgeDFs]:
    tuple_assembly_input = prep_nlp_ntuples_etuples_input(
        df=node_dfs.to_dict()[NodeType.sheet_cell], spacy_pipeline=spacy_pipeline
    )
    # TODO: Investigates the cause of the following mypy error
    ntuples_etuples = assemble_nlp_ntuples_etuples(*tuple_assembly_input)  # type: ignore[arg-type]
    list_nlp_node_df, list_nlp_edge_df = assemble_nlp_ndfs_edfs(*ntuples_etuples)

    node_dfs.members.extend(list_nlp_node_df)
    node_dfs.validate()
    edge_dfs.members.extend(list_nlp_edge_df)
    edge_dfs.validate()

    validate_node_dfs_and_edge_dfs(node_dfs=node_dfs, edge_dfs=edge_dfs)

    return node_dfs, edge_dfs
