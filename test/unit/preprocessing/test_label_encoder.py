import polars as pl
from polars.testing import assert_frame_equal
from polars_pipeline.preprocessing import LabelEncoder


def test_basic():
    input = pl.DataFrame(
        {
            "a": pl.Series(["a", "b", "c", "a", "b", "c"], dtype=pl.Utf8),
            "b": pl.Series([1, 2, -3, 4, 5, 600], dtype=pl.Int64),
        }
    )
    expected = pl.DataFrame(
        {
            "b": pl.Series([1, 2, -3, 4, 5, 600], dtype=pl.Int64),
            "a": pl.Series([0, 1, 2, 0, 1, 2], dtype=pl.Int64),
        }
    )
    encoder = LabelEncoder("a", maintain_order=True)
    output = encoder.fit_transform(input)
    assert_frame_equal(output, expected)


def test_null():
    input = pl.DataFrame(
        {
            "a": pl.Series(["a", "b", None, None, "b", "c"], dtype=pl.Utf8),
            "b": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int64),
        }
    )
    expected = pl.DataFrame(
        {
            "b": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int64),
            "a": pl.Series([0, 1, None, None, 1, 3], dtype=pl.Int64),
        }
    )
    encoder = LabelEncoder("a", maintain_order=True)
    output = encoder.fit_transform(input)
    assert_frame_equal(output, expected)


def test_unknown():
    input = pl.DataFrame(
        {
            "a": pl.Series(["a", "b", "c", "a", "b", "c"], dtype=pl.Utf8),
        }
    )
    input_unknown = pl.DataFrame(
        {
            "a": pl.Series(["a", "b", "d", "e", "b", "c"], dtype=pl.Utf8),
        }
    )
    expected = pl.DataFrame(
        {
            "a": pl.Series([0, 1, None, None, 1, 2], dtype=pl.Int64),
        }
    )
    encoder = LabelEncoder("a", maintain_order=True)
    encoder.fit(input)
    output = encoder.transform(input_unknown)
    assert_frame_equal(output, expected)
