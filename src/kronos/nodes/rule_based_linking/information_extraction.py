import logging
import random
import re
from typing import Any, Dict, List, Tuple

import networkx as nx

from kronos.data_interfaces.ner_entities_data_interface import NEREntity
from kronos.nodes.graph_schema import EdgeAttrKey, EdgeType, NodeAttrKey, NodeType
from kronos.nodes.layout_graph import Direction

logger = logging.getLogger(__name__)


def is_year_node(text: str) -> bool:
    """Check if the given node represents a year."""
    return bool(re.match(r"(19|20)\d{2}", text))


# TODO: Use spacy to improve the function's robustness
def is_month_node(text: str, ner_entities: List[NEREntity]) -> bool:
    """Check if the given node represents a month."""
    # Disqualify mixed type nodes
    if len(ner_entities) != 1:
        return False

    # Regular expressions for each month
    month_patterns = {
        "january": r"jan(uary)?",
        "february": r"feb(ruary)?",
        "march": r"mar(ch)?",
        "april": r"apr(il)?",
        "may": r"may",
        "june": r"jun(e)?",
        "july": r"jul(y)?",
        "august": r"aug(ust)?",
        "september": r"sep(t(ember)?)?",
        "october": r"oct(ober)?",
        "november": r"nov(ember)?",
        "december": r"dec(ember)?",
    }

    # Preprocess the input text
    clean_text = text.lower().strip()

    # Check against each pattern
    for pattern in month_patterns.values():
        if re.match(pattern, clean_text):
            return True

    return False


def is_day_of_week_node(text: str) -> bool:
    """Check if the given node represents a day of the week."""
    days_of_week = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    return text.lower().strip() in days_of_week


def is_event_node(text: str, ner_entities: List[NEREntity]) -> bool:
    result: bool = False

    other_conditions: List[bool] = [
        is_year_node(text),
        is_month_node(text, ner_entities),
        is_day_of_week_node(text),
    ]
    if not any(other_conditions):
        result = True

    return result


def find_year_nodes(nx_g: nx.DiGraph) -> List[Any]:
    """Find all year nodes in the graph."""
    return [
        nid
        for nid, ntype in nx_g.nodes(NodeAttrKey.ntype.value)
        if ntype == NodeType.year.value
    ]


def find_month_nodes(nx_g: nx.DiGraph) -> List[Any]:
    """Find all month nodes in the graph."""

    return [
        nid
        for nid, ntype in nx_g.nodes(NodeAttrKey.ntype.value)
        if ntype == NodeType.month.value
    ]


def find_day_of_week_nodes(nx_g: nx.DiGraph) -> List[Any]:
    """Find all day of week nodes in the graph."""
    return [
        nid
        for nid, ntype in nx_g.nodes(NodeAttrKey.ntype.value)
        if ntype == NodeType.day_of_week.value
    ]


def find_event_nodes(nx_g: nx.DiGraph) -> List[Any]:
    """Find all event nodes in the graph."""

    return [
        nid
        for nid, ntype in nx_g.nodes(NodeAttrKey.ntype.value)
        if ntype == NodeType.event.value
    ]


def gather_edges_year_to_month(
    nx_g: nx.DiGraph,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    """Connect each year node to all month nodes to its right until the boundary."""
    year_nodes = find_year_nodes(nx_g)
    month_nodes = find_month_nodes(nx_g)
    year_to_month_edges: List[Tuple[int, int, Dict[str, Any]]] = []

    for year_node in year_nodes:
        current_node = year_node
        while True:
            # Get the neighbors to the right
            right_neighbors = [
                neighbor
                for neighbor, edge_attr in nx_g[current_node].items()
                if edge_attr[EdgeAttrKey.direction.value] == Direction.right.value
            ]

            # Stop if we reached the boundary
            if not right_neighbors:
                break

            # Move to the right neighbor and if it's a month node, create an edge
            for neighbor in right_neighbors:
                current_node = neighbor
                if current_node in month_nodes:
                    edge_attrs: Dict[str, Any] = {
                        EdgeAttrKey.etype.value: EdgeType.year_to_month.value
                    }
                    year_to_month_edges.append((year_node, current_node, edge_attrs))

    # Logging
    i_logging: int = random.choice(range(len(year_to_month_edges)))
    i_year = year_to_month_edges[i_logging][0]
    i_month = year_to_month_edges[i_logging][1]
    text_year = nx_g.nodes[i_year][NodeAttrKey.raw_text.value]
    text_month = nx_g.nodes[i_month][NodeAttrKey.raw_text.value]
    logger.debug(f"Example year to month edge: {text_year} to {text_month}")

    return year_to_month_edges


def gather_edges_month_to_day_of_week(
    nx_g: nx.DiGraph,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    """
    Connect each month node to all day-of-week nodes beneath it until the boundary.
    """
    month_nodes = find_month_nodes(nx_g)
    day_of_week_nodes = find_day_of_week_nodes(nx_g)
    month_to_day_of_week_edges: List[Tuple[int, int, Dict[str, Any]]] = []

    for month_node in month_nodes:
        current_node = month_node
        while True:
            # Get the neighbors to the right
            right_neighbors = [
                neighbor
                for neighbor, edge_attr in nx_g[current_node].items()
                if edge_attr[EdgeAttrKey.direction.value] == Direction.right.value
            ]

            # Stop if we reached the boundary
            if not right_neighbors:
                break

            # Move to the right neighbor and if it's a day-of-week node, create an edge
            for neighbor in right_neighbors:
                current_node = neighbor
                if current_node in day_of_week_nodes:
                    edge_attrs: Dict[str, Any] = {
                        EdgeAttrKey.etype.value: EdgeType.month_to_day_of_week.value
                    }
                    month_to_day_of_week_edges.append(
                        (month_node, current_node, edge_attrs)
                    )

    # Logging
    i_logging: int = random.choice(range(len(month_to_day_of_week_edges)))
    i_month = month_to_day_of_week_edges[i_logging][0]
    i_day_of_week = month_to_day_of_week_edges[i_logging][1]
    text_month = nx_g.nodes[i_month][NodeAttrKey.raw_text.value]
    text_day_of_week = nx_g.nodes[i_day_of_week][NodeAttrKey.raw_text.value]
    logger.debug(
        f"Example month to day-of-week edge: {text_month} to {text_day_of_week}"
    )

    return month_to_day_of_week_edges


def gather_edges_day_of_week_to_event(
    nx_g: nx.DiGraph,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    day_of_week_nodes = find_day_of_week_nodes(nx_g)
    event_nodes = find_event_nodes(nx_g)
    day_of_week_to_event_edges: List[Tuple[int, int, Dict[str, Any]]] = []

    for day_of_week_node in day_of_week_nodes:
        current_node = day_of_week_node
        while True:
            # Get the neighbors to the right
            right_neighbors = [
                neighbor
                for neighbor, edge_attr in nx_g[current_node].items()
                if edge_attr[EdgeAttrKey.direction.value] == Direction.right.value
            ]

            # Stop if we reached the boundary
            if not right_neighbors:
                break

            # Move to the right neighbor and if it's an event node, create an edge
            for neighbor in right_neighbors:
                current_node = neighbor
                if current_node in event_nodes:
                    edge_attrs: Dict[str, Any] = {
                        EdgeAttrKey.etype.value: EdgeType.day_of_week_to_event.value
                    }
                    day_of_week_to_event_edges.append(
                        (day_of_week_node, current_node, edge_attrs)
                    )

    # Logging
    i_logging: int = random.choice(range(len(day_of_week_to_event_edges)))
    i_day_of_week = day_of_week_to_event_edges[i_logging][0]
    i_event = day_of_week_to_event_edges[i_logging][1]
    text_day_of_week = nx_g.nodes[i_day_of_week][NodeAttrKey.raw_text.value]
    text_event = nx_g.nodes[i_event][NodeAttrKey.raw_text.value]
    logger.debug(
        f"Example day-of-week to event edge: {text_day_of_week} to {text_event}"
    )

    return day_of_week_to_event_edges
