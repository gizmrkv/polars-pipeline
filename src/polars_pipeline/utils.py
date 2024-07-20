from typing import Any, Dict, List

import polars as pl

from .typing import FrameType


def numerical_columns(frame: FrameType) -> List[str]:
    return frame.head(1).select(pl.col(pl.Float32, pl.Float64, pl.Decimal)).columns


def categorical_columns(frame: FrameType) -> List[str]:
    return frame.head(1).select(pl.col(pl.Categorical, pl.Enum, pl.Boolean)).columns


def list_of_dict_to_dict_of_list(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    if len(data) == 0:
        return {}

    keys = data[0].keys()
    result: Dict[str, List[Any]] = {key: [] for key in keys}
    for d in data:
        for key, value in d.items():
            result[key].append(value)
    return result
