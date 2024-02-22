import pandas as pd

from kronos.nodes.utils_df_serialisation import default, df_type_hook


def test_default() -> None:
    # Arrange
    df = pd.DataFrame({"A": [1, 2]})

    # Act
    result = default(df)

    # Assert
    assert "schema" in result


def test_df_type_hook(mock_json_str_df: str) -> None:
    # Assert
    result = df_type_hook(mock_json_str_df)

    # Act
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result.columns) == ["dummy", "etype"]
    assert result.iloc[0]["dummy"] == [0, 0]
