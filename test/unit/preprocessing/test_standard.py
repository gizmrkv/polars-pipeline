import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_pipeline.preprocessing import StandardScaler


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
            "a": pl.Series(
                [
                    0.018369795387880923,
                    -0.006123265129293641,
                    1.2185297607294345,
                    -1.2307762909880218,
                ],
                dtype=pl.Float64,
            ),
            "b": pl.Series([10, 0, 500, -500], dtype=pl.Int64),
            "c": pl.Series(["foo", "bar", "foo", "bar"], dtype=pl.Utf8),
        }
    )
    scaler = StandardScaler("a")
    output = scaler.fit_transform(input)
    assert_frame_equal(output, expected)


def test_constant():
    input = pl.DataFrame({"a": pl.Series([42] * 10, dtype=pl.Float64)})
    scaler = StandardScaler("a")
    with pytest.raises(ZeroDivisionError):
        scaler.fit(input)


def test_empty():
    input = pl.DataFrame({})
    scaler = StandardScaler("a")
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
            "a": pl.Series(
                [
                    0.018369795387880923,
                    -0.006123265129293641,
                    1.2185297607294345,
                    -1.2307762909880218,
                    None,
                ],
                dtype=pl.Float64,
            ),
            "b": pl.Series([10, 0, None, 500, -500], dtype=pl.Int64),
            "c": pl.Series(["foo", None, "bar", "foo", "bar"], dtype=pl.Utf8),
        }
    )
    scaler = StandardScaler("a")
    output = scaler.fit_transform(input)
    assert_frame_equal(output, expected)
