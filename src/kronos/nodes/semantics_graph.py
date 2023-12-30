import logging
from collections import Counter
from typing import Any, Dict, List

import networkx as nx
from spacy.language import Language

from kronos.data_interfaces.ner_entities_data_interface import (
    NEREntity,
    list_dict_to_ner_entities,
    list_text_to_list_ner_entities,
)
from kronos.nodes.graph_schema import NodeAttrKey, NodeType
from kronos.nodes.rule_based_linking.information_extraction import (
    is_day_of_week_node,
    is_event_node,
    is_month_node,
    is_year_node,
)

logger = logging.getLogger(__name__)


def augment_nx_g_with_semantics(
    nx_g: nx.DiGraph, spacy_pipeline: Language
) -> nx.DiGraph:
    # Obtain a list of raw texts
    list_raw_text: List[str] = list(
        nx.get_node_attributes(nx_g, NodeAttrKey.raw_text.value).values()
    )

    # Obtain list of lists of NEREntity
    list_ner_entities: List[List[NEREntity]] = list_text_to_list_ner_entities(
        list_raw_text, spacy_pipeline
    )
    list_serialised_ner_entities = [
        [entity.to_dict() for entity in ner_entities]
        for ner_entities in list_ner_entities
    ]

    # Add the new node attribute to the graph
    nx.set_node_attributes(
        G=nx_g,
        values=dict(zip(nx_g.nodes(), list_serialised_ner_entities)),
        name=NodeAttrKey.ner.value,
    )

    logger.info(
        f"Set {NodeAttrKey.ner.value} attribute for "
        f"{len(list_serialised_ner_entities)} nodes in the graph."
    )

    return nx_g


def collect_ntype_based_on_semantics(nx_g: nx.DiGraph) -> Dict[Any, str]:
    ntypes: List[str] = [
        NodeType.year.value,
        NodeType.month.value,
        NodeType.day_of_week.value,
        NodeType.event.value,
    ]

    nid_to_ntype: Dict[Any, str] = {}
    for nid in nx_g.nodes:
        bools_ntype: List[bool] = [False] * len(ntypes)

        # Determine node type based on the node attributes
        raw_text = nx_g.nodes[nid][NodeAttrKey.raw_text.value]
        ner_entity_dicts = nx_g.nodes[nid][NodeAttrKey.ner.value]
        ner_entities = list_dict_to_ner_entities(ner_entity_dicts)

        if is_year_node(raw_text):
            bools_ntype[ntypes.index(NodeType.year.value)] = True

        if is_month_node(raw_text, ner_entities):
            bools_ntype[ntypes.index(NodeType.month.value)] = True

        if is_day_of_week_node(raw_text):
            bools_ntype[ntypes.index(NodeType.day_of_week.value)] = True

        if is_event_node(raw_text, ner_entities):
            bools_ntype[ntypes.index(NodeType.event.value)] = True

        # Each node can only have one node type
        if sum(bools_ntype) != 1:
            raise ValueError(
                f"Node {nid} with text {raw_text} with the following "
                f"ner entities value: {ner_entity_dicts}\n"
                f"has {sum(bools_ntype)} ntypes. Here's its node type data "
                f"{dict(zip(ntypes, bools_ntype))}"
            )
        else:
            ntype = ntypes[bools_ntype.index(True)]
            nid_to_ntype[nid] = ntype

    logger.info(
        "Frequency distribution of the inferred node types: "
        f"{Counter(nid_to_ntype.values())}"
    )

    return nid_to_ntype
