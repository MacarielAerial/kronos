from io import StringIO
from typing import Any

import pandas as pd
from pandas import DataFrame


def default(obj: Any) -> Any:
    if isinstance(obj, DataFrame):
        json_df = obj.to_json(orient="table")
        return json_df

    raise TypeError(
        f"Type {type(obj)} does not have corresponding serialisation logic defined"
    )


def df_type_hook(json_str_df: str) -> DataFrame:
    df = pd.read_json(StringIO(json_str_df), orient="table")

    return df
