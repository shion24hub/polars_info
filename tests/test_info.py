"""Tests for polars_info output correctness after refactoring."""

from __future__ import annotations

import io
import re

import polars as pl
import pytest

from polars_info import DFInfoSummary, print_df_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture(df: pl.DataFrame, **kwargs) -> tuple[str, DFInfoSummary]:
    buf = io.StringIO()
    info = print_df_info(df, file=buf, **kwargs)
    return buf.getvalue(), info


# ---------------------------------------------------------------------------
# Header tests
# ---------------------------------------------------------------------------

class TestHeader:
    def test_class_line(self):
        df = pl.DataFrame({"a": [1]})
        text, _ = _capture(df)
        assert "<class 'polars.dataframe.frame.DataFrame'>" in text

    def test_name_shown(self):
        df = pl.DataFrame({"a": [1]})
        text, _ = _capture(df, name="my_df")
        assert "Name: my_df" in text

    def test_shape(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        text, _ = _capture(df)
        assert "Shape: (2, 2)" in text

    def test_estimated_size(self):
        df = pl.DataFrame({"a": [1]})
        text, _ = _capture(df)
        assert "Estimated size:" in text
        assert "Unknown" not in text


# ---------------------------------------------------------------------------
# Column table tests
# ---------------------------------------------------------------------------

class TestColumnTable:
    def test_full_display(self):
        df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.0]})
        text, _ = _capture(df, display="full")
        assert "Columns:" in text
        for col in ["a", "b", "c"]:
            assert col in text

    def test_head_tail_display(self):
        df = pl.DataFrame({f"c{i}": [i] for i in range(30)})
        text, _ = _capture(df, display="head_tail", head=3, tail=3)
        assert "c0" in text
        assert "c1" in text
        assert "c2" in text
        assert "c27" in text
        assert "c28" in text
        assert "c29" in text
        assert "..." in text
        # Middle columns should NOT appear
        assert "c15" not in text

    def test_auto_display_small(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        text, _ = _capture(df, display="auto")
        assert "..." not in text

    def test_auto_display_large(self):
        df = pl.DataFrame({f"c{i}": [i] for i in range(100)})
        text, _ = _capture(df, display="auto", max_cols=60, head=5, tail=5)
        assert "..." in text

    def test_empty_df(self):
        df = pl.DataFrame()
        text, _ = _capture(df)
        assert "(no columns)" in text

    def test_dtypes_shown(self):
        df = pl.DataFrame({"a": [1], "b": ["x"]})
        text, _ = _capture(df)
        assert "Int64" in text or "i64" in text.lower()
        assert "String" in text or "Utf8" in text or "str" in text.lower()


# ---------------------------------------------------------------------------
# Null statistics tests
# ---------------------------------------------------------------------------

class TestNullStats:
    def test_null_stats_shown(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": [None, None, "x"]})
        text, _ = _capture(df, show_null_stats=True)
        assert "Non-Null" in text
        assert "Null%" in text

    def test_null_stats_hidden(self):
        df = pl.DataFrame({"a": [1, None]})
        text, _ = _capture(df, show_null_stats=False)
        assert "Non-Null" not in text
        assert "Null%" not in text

    def test_null_counts_correct(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
        text, _ = _capture(df, show_null_stats=True)
        lines = text.strip().split("\n")
        # Find row for column 'b' - should show 3 nulls
        b_lines = [l for l in lines if re.search(r"\bb\b", l)]
        assert len(b_lines) == 1
        assert "3" in b_lines[0]


# ---------------------------------------------------------------------------
# Sample display tests
# ---------------------------------------------------------------------------

class TestSample:
    def test_sample_shown(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        text, _ = _capture(df, show_sample=2)
        assert "Sample (head 2):" in text

    def test_no_sample_by_default(self):
        df = pl.DataFrame({"a": [1]})
        text, _ = _capture(df)
        assert "Sample" not in text


# ---------------------------------------------------------------------------
# DFInfoSummary tests
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_fields(self):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        _, info = _capture(df)
        assert info.rows == 2
        assert info.cols == 2
        assert info.estimated_size_bytes is not None
        assert isinstance(info.dtypes, dict)
        assert "a" in info.dtypes
        assert "b" in info.dtypes

    def test_summary_empty_df(self):
        df = pl.DataFrame()
        _, info = _capture(df)
        assert info.rows == 0
        assert info.cols == 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_df_type(self):
        with pytest.raises(TypeError):
            print_df_info("not a dataframe")  # type: ignore[arg-type]

    def test_invalid_display(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError):
            print_df_info(df, display="invalid")

    def test_negative_head(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError):
            print_df_info(df, head=-1)


# ---------------------------------------------------------------------------
# Output destination tests
# ---------------------------------------------------------------------------

class TestOutput:
    def test_file_output(self):
        df = pl.DataFrame({"a": [1]})
        buf = io.StringIO()
        print_df_info(df, file=buf)
        assert len(buf.getvalue()) > 0

    def test_stdout_output(self, capsys):
        df = pl.DataFrame({"a": [1]})
        print_df_info(df)
        captured = capsys.readouterr()
        assert "<class" in captured.out
