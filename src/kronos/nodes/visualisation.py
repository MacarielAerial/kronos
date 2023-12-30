from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import plotly.graph_objs as go


def vis_layout_nx_g(G: nx.DiGraph) -> go.Figure:  # type: ignore[no-any-unimported]
    """
    Visualizes the network using Plotly with improved node sizing, figure scaling,
    and reasonable figure dimensions.

    :param G: A NetworkX Graph object with 'raw_text' node attributes.
    :return: Plotly figure object.
    """
    # Determine the layout positions
    pos: Dict[Any, Tuple[float, float]] = {
        node: (node[1], -node[0]) for node in G.nodes()
    }  # Flip y-axis for alignment

    # Extract edge coordinates
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Determine the node size: Let's aim for smaller nodes
    node_count: int = len(list(G.nodes()))
    # Set a minimal node size and a factor for scaling
    min_node_size: float = 5
    node_size_factor: float = 100 / node_count
    node_size: float = max(min_node_size, node_size_factor)

    # Extract node coordinates and hover text
    node_x: List[float] = []
    node_y: List[float] = []
    hover_text: List[str] = []
    for node in G.nodes(data=True):
        node_x.append(pos[node[0]][0])
        node_y.append(pos[node[0]][1])
        hover_text.append(node[1]["raw_text"] if "raw_text" in node[1] else "")

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=hover_text,
        marker=dict(
            showscale=False, color="blue", size=node_size, line_width=0  # Remove border
        ),
    )

    # Define a reasonable figure size
    figure_width: int = 1000  # Fixed width for consistency
    figure_height: int = 800  # Fixed height for consistency

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Network graph of non-null cells in the table",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
            width=figure_width,
            height=figure_height,
        ),
    )

    return fig
