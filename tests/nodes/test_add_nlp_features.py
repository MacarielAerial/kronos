from typing import Any, List, Tuple

import pytest
from pandas import DataFrame
from spacy.language import Language

from kronos.data_interfaces.edge_dfs_data_interface import (
    EntToCellTuple,
    EntToLabelTuple,
    TokenToCellTuple,
    TokenToEntTuple,
)
from kronos.data_interfaces.node_dfs_data_interface import (
    EntLabelTuple,
    EntTuple,
    TokenTuple,
)
from kronos.nodes.add_nlp_feats import (
    assemble_nlp_ntuples_etuples,
    prep_nlp_ntuples_etuples_input,
)


def test_typical_case() -> None:
    # Arrange
    # Function input
    token_text = ["Token1", "Token2"]
    ent_text = ["Ent1"]
    ent_label_text = ["Label1"]
    i_token_to_cell = [(0, 1)]
    i_token_to_ent = [(1, 0)]
    i_ent_to_cell = [(0, 1)]
    i_ent_to_label = [(0, 0)]
    i_token_in_doc = [(0, 0), (1, 1)]
    i_token_in_ent = [(1, 0)]
    i_ent_in_doc = [(0, 0)]

    # Expected function outputs
    expected_token_tuples = [
        TokenTuple(nid=0, ntype="Token", text="Token1"),
        TokenTuple(nid=1, ntype="Token", text="Token2"),
    ]
    expected_ent_tuples = [EntTuple(nid=0, ntype="Ent", text="Ent1")]
    expected_ent_label_tuples = [EntLabelTuple(nid=0, ntype="EntLabel", text="Label1")]
    expected_token_to_cell = [
        TokenToCellTuple(
            src_nid=0,
            dst_nid=1,
            src_ntype="Token",
            etype="TokenToCell",
            dst_ntype="SheetCell",
            i_token_in_doc=(0, 0),
        )
    ]
    expected_token_to_ent = [
        TokenToEntTuple(
            src_nid=1,
            dst_nid=0,
            src_ntype="Token",
            etype="TokenToEnt",
            dst_ntype="Ent",
            i_token_in_ent=(1, 0),
        )
    ]
    expected_ent_to_cell = [
        EntToCellTuple(
            src_nid=0,
            dst_nid=1,
            src_ntype="Ent",
            etype="EntToCell",
            dst_ntype="SheetCell",
            i_ent_in_doc=(0, 0),
        )
    ]
    expected_ent_to_label = [
        EntToLabelTuple(
            src_nid=0,
            dst_nid=0,
            src_ntype="Ent",
            etype="EntToLabel",
            dst_ntype="EntLabel",
        )
    ]

    # Act
    result = assemble_nlp_ntuples_etuples(
        token_text,
        ent_text,
        ent_label_text,
        i_token_to_cell,
        i_token_to_ent,
        i_ent_to_cell,
        i_ent_to_label,
        i_token_in_doc,
        i_token_in_ent,
        i_ent_in_doc,
    )

    # Assert
    assert result[0] == expected_token_tuples
    assert result[1] == expected_ent_tuples
    assert result[2] == expected_ent_label_tuples
    assert result[3] == expected_token_to_cell
    assert result[4] == expected_token_to_ent
    assert result[5] == expected_ent_to_cell
    assert result[6] == expected_ent_to_label


def test_empty_inputs() -> None:
    # Arrange & Act
    expected: Tuple[List[Any], ...] = ([], [], [], [], [], [], [])
    result = assemble_nlp_ntuples_etuples([], [], [], [], [], [], [], [], [], [])

    # Assert
    assert result == expected


def test_prep_nlp_ntuples_etuples_input_typical_case(
    sm_spacy_pipeline: Language,
) -> None:
    # Arrange
    data = {"nid": [0, 1], "text": ["Cape Town South Africa", "Shanghai repeat repeat"]}
    df = DataFrame(data)

    # Act
    (
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
    ) = prep_nlp_ntuples_etuples_input(df, sm_spacy_pipeline)

    # Assert
    assert len(token) > 0
    assert len(ent) == 2
    assert len(ent_label) == 1
    assert (len(i_token_to_cell) - 1) == len(token)  # "repeat" token repeats once
    assert len(i_token_to_ent) > 0
    assert len(i_ent_to_cell) == 2
    assert len(i_ent_to_label) == 2
    assert len(i_token_in_doc) > 0
    assert len(i_token_in_ent) > 0
    assert len(i_ent_in_doc) == 2
    # Continue with assertions for other outputs


def test_empty_dataframe(sm_spacy_pipeline: Language) -> None:
    empty_df = DataFrame()

    with pytest.raises(KeyError):
        prep_nlp_ntuples_etuples_input(empty_df, sm_spacy_pipeline)
