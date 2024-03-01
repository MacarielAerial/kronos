import pytest
from pandas import DataFrame

from kronos.data_interfaces.node_dfs_data_interface import NodeAttrKey, NodeDF, NodeType
from kronos.nodes.vectorise_text_feats import prep_emb_input


def test_prep_emb_input_unique_texts() -> None:
    mock_list_node_df = [
        NodeDF(
            ntype=NodeType.token.value,
            df=DataFrame(
                {NodeAttrKey.text.value: ["Non overlapping text", "Overlapping text"]}
            ),
        ),
        NodeDF(
            ntype=NodeType.sheet_cell.value,
            df=DataFrame({NodeAttrKey.text.value: ["Overlapping text"]}),
        ),
    ]

    unique_text = prep_emb_input(mock_list_node_df)

    assert isinstance(unique_text, set)
    assert len(unique_text) == 2  # Number of unique texts across both NodeDFs
    assert "Non overlapping text" in unique_text
    assert "Overlapping text" in unique_text


def test_prep_emb_input_empty_input() -> None:
    with pytest.raises(ValueError):
        _ = prep_emb_input([])
