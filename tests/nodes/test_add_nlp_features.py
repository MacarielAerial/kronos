from typing import Any, List, Tuple

import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from spacy.language import Language

from kronos.data_interfaces.edge_dfs_data_interface import (
    EdgeDF,
    EntToCellTuple,
    EntToLabelTuple,
    TokenToCellTuple,
    TokenToEntTuple,
)
from kronos.data_interfaces.node_dfs_data_interface import (
    EntLabelTuple,
    EntTuple,
    NodeDF,
    TokenTuple,
)
from kronos.nodes.add_nlp_feats import (
    assemble_nlp_ndfs_edfs,
    assemble_nlp_ntuples_etuples,
    prep_nlp_ntuples_etuples_input,
)


def test_assemble_nlp_ntuples_etuples_typical() -> None:
    # Arrange
    # Function input
    token_text = ["Token1", "Token2"]
    ent_text = ["Ent1"]
    ent_label_text = ["Label1"]
    i_token_to_cell = [(0, 1)]
    i_token_to_ent = [(1, 0)]
    i_ent_to_cell = [(0, 1)]
    i_ent_to_label = [(0, 0)]
    i_token_in_doc = [0]
    i_token_in_ent = [0]
    i_ent_in_doc = [0]

    # Expected function outpu
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
            i_token_in_doc=0,
        )
    ]
    expected_token_to_ent = [
        TokenToEntTuple(
            src_nid=1,
            dst_nid=0,
            src_ntype="Token",
            etype="TokenToEnt",
            dst_ntype="Ent",
            i_token_in_ent=0,
        )
    ]
    expected_ent_to_cell = [
        EntToCellTuple(
            src_nid=0,
            dst_nid=1,
            src_ntype="Ent",
            etype="EntToCell",
            dst_ntype="SheetCell",
            i_ent_in_doc=0,
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


def test_assemble_nlp_ntuples_etuples_edge() -> None:
    # Arrange & Act
    expected: Tuple[List[Any], ...] = ([], [], [], [], [], [], [])
    result = assemble_nlp_ntuples_etuples([], [], [], [], [], [], [], [], [], [])

    # Assert
    assert result == expected


def test_prep_nlp_ntuples_etuples_input_typical(
    en_sm_spacy_pipeline: Language,
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
    ) = prep_nlp_ntuples_etuples_input(df, en_sm_spacy_pipeline)

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


def test_prep_nlp_ntuples_etuples_input_edge(en_sm_spacy_pipeline: Language) -> None:
    empty_df = DataFrame()

    with pytest.raises(KeyError):
        prep_nlp_ntuples_etuples_input(empty_df, en_sm_spacy_pipeline)


def test_assemble_nlp_ndfs_edfs_typical() -> None:
    # Arrange
    # Function input
    token = [TokenTuple(nid=0, ntype="Token", text="example")]
    ent = [EntTuple(nid=0, ntype="Ent", text="example")]
    ent_label = [EntLabelTuple(nid=0, ntype="EntLabel", text="example")]
    token_to_cell = [
        TokenToCellTuple(
            src_nid=0,
            dst_nid=1,
            src_ntype="Token",
            etype="TokenToCell",
            dst_ntype="SheetCell",
            i_token_in_doc=0,
        )
    ]
    token_to_ent = [
        TokenToEntTuple(
            src_nid=0,
            dst_nid=1,
            src_ntype="Token",
            etype="TokenToEnt",
            dst_ntype="Ent",
            i_token_in_ent=0,
        )
    ]
    ent_to_cell = [
        EntToCellTuple(
            src_nid=0,
            dst_nid=1,
            src_ntype="Ent",
            etype="EntToCell",
            dst_ntype="SheetCell",
            i_ent_in_doc=0,
        )
    ]
    ent_to_label = [
        EntToLabelTuple(
            src_nid=0,
            dst_nid=1,
            src_ntype="Ent",
            etype="EntToLabel",
            dst_ntype="EntLabel",
        )
    ]

    # Act
    node_dfs, edge_dfs = assemble_nlp_ndfs_edfs(
        token, ent, ent_label, token_to_cell, token_to_ent, ent_to_cell, ent_to_label
    )

    # Assert
    # Verify correct number of NodeDF and EdgeDF objects
    assert len(node_dfs) == 3
    assert len(edge_dfs) == 4

    # Verify data in returned DataFrames matches input tuples
    assert_frame_equal(node_dfs[0].df, DataFrame(token))
    assert_frame_equal(node_dfs[1].df, DataFrame(ent))
    assert_frame_equal(node_dfs[2].df, DataFrame(ent_label))
    assert_frame_equal(edge_dfs[0].df, DataFrame(token_to_cell))
    assert_frame_equal(edge_dfs[1].df, DataFrame(token_to_ent))
    assert_frame_equal(edge_dfs[2].df, DataFrame(ent_to_cell))
    assert_frame_equal(edge_dfs[3].df, DataFrame(ent_to_label))


def test_assemble_nlp_ndfs_edfs_edge() -> None:
    empty_lists: tuple = ([], [], [], [], [], [], [])
    node_dfs, edge_dfs = assemble_nlp_ndfs_edfs(*empty_lists)

    # Verify correct structure is returned with empty DataFrames
    assert len(node_dfs) == 3
    assert all(isinstance(ndf, NodeDF) for ndf in node_dfs)
    assert all(ndf.df.empty for ndf in node_dfs)
    assert len(edge_dfs) == 4
    assert all(isinstance(edf, EdgeDF) for edf in edge_dfs)
    assert all(edf.df.empty for edf in edge_dfs)
