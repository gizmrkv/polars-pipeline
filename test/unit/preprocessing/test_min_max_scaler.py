import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_pipeline.preprocessing import MinMaxScaler


def test_basic():
    input = pl.DataFrame(
        {
            "a": pl.Series([10.0, 0.0, 500.0, -500.0], dtype=pl.Float64),
            "b": pl.Series([10, 0, 500, -500], dtype=pl.Int64),
            "c": pl.Series(["foo", "bar", "foo", "bar"], dtype=pl.Utf8),
        }
    )
    expected = pl.DataFrame(
        {
            "a": pl.Series([0.51, 0.5, 1.0, 0.0], dtype=pl.Float64),
            "b": pl.Series([10, 0, 500, -500], dtype=pl.Int64),
            "c": pl.Series(["foo", "bar", "foo", "bar"], dtype=pl.Utf8),
        }
    )
    scaler = MinMaxScaler("a")
    output = scaler.fit_transform(input)
    assert_frame_equal(output, expected)


def test_constant():
    input = pl.DataFrame({"a": pl.Series([42] * 10, dtype=pl.Float64)})
    scaler = MinMaxScaler("a")
    with pytest.raises(ZeroDivisionError):
        scaler.fit(input)


def test_empty():
    input = pl.DataFrame({})
    scaler = MinMaxScaler("a")
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        scaler.fit(input)


def test_null():
    input = pl.DataFrame(
        {
            "a": pl.Series([10.0, 0.0, 500.0, -500.0, None], dtype=pl.Float64),
            "b": pl.Series([10, 0, None, 500, -500], dtype=pl.Int64),
            "c": pl.Series(["foo", None, "bar", "foo", "bar"], dtype=pl.Utf8),
        }
    )
    expected = pl.DataFrame(
        {
            "a": pl.Series([0.51, 0.5, 1.0, 0.0, None], dtype=pl.Float64),
            "b": pl.Series([10, 0, None, 500, -500], dtype=pl.Int64),
            "c": pl.Series(["foo", None, "bar", "foo", "bar"], dtype=pl.Utf8),
        }
    )
    scaler = MinMaxScaler("a")
    output = scaler.fit_transform(input)
    assert_frame_equal(output, expected)
