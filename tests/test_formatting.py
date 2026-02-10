"""Tests for the _formatting module."""

from __future__ import annotations

import polars as pl

from polars_info._formatting import (
    _ColumnLayout,
    _build_column_table,
    _bytes_to_human,
    _compute_column_layout,
    _format_column_row,
    _format_ellipsis_row,
    _format_table_header,
)


class TestBytesToHuman:
    def test_none(self):
        assert _bytes_to_human(None) == "Unknown"

    def test_negative(self):
        assert _bytes_to_human(-1) == "-1"

    def test_zero(self):
        assert _bytes_to_human(0) == "0 B"

    def test_bytes(self):
        assert _bytes_to_human(512) == "512 B"

    def test_kib(self):
        result = _bytes_to_human(1024)
        assert "KiB" in result

    def test_mib(self):
        result = _bytes_to_human(1024 * 1024)
        assert "MiB" in result

    def test_gib(self):
        result = _bytes_to_human(1024**3)
        assert "GiB" in result


class TestColumnLayout:
    def test_frozen(self):
        layout = _ColumnLayout(max_name_len=10, max_dtype_len=8, has_null_stats=True)
        assert layout.max_name_len == 10
        assert layout.max_dtype_len == 8
        assert layout.has_null_stats is True


class TestComputeColumnLayout:
    def test_basic(self):
        layout = _compute_column_layout(
            col_names=["a", "bb", "ccc"],
            dtypes=[pl.Int64, pl.Utf8, pl.Float64],
            indices=[0, 1, 2],
            omitted=False,
            show_null_stats=False,
            null_by_col={},
        )
        assert layout.max_name_len >= len("Column")
        assert layout.max_dtype_len >= len("Dtype")
        assert layout.has_null_stats is False

    def test_with_null_stats(self):
        layout = _compute_column_layout(
            col_names=["a"],
            dtypes=[pl.Int64],
            indices=[0],
            omitted=False,
            show_null_stats=True,
            null_by_col={"a": 1},
        )
        assert layout.has_null_stats is True

    def test_empty_indices(self):
        layout = _compute_column_layout(
            col_names=[],
            dtypes=[],
            indices=[],
            omitted=False,
            show_null_stats=False,
            null_by_col={},
        )
        assert layout.max_name_len == len("Column")
        assert layout.max_dtype_len == len("Dtype")


class TestFormatTableHeader:
    def test_without_null_stats(self):
        layout = _ColumnLayout(max_name_len=10, max_dtype_len=8, has_null_stats=False)
        header = _format_table_header(layout)
        assert "#" in header
        assert "Column" in header
        assert "Dtype" in header
        assert "Null" not in header

    def test_with_null_stats(self):
        layout = _ColumnLayout(max_name_len=10, max_dtype_len=8, has_null_stats=True)
        header = _format_table_header(layout)
        assert "Non-Null" in header
        assert "Null%" in header


class TestFormatColumnRow:
    def test_basic_row(self):
        layout = _ColumnLayout(max_name_len=10, max_dtype_len=8, has_null_stats=False)
        row = _format_column_row(0, "a", pl.Int64, 100, layout, {})
        assert "0" in row
        assert "a" in row

    def test_row_with_nulls(self):
        layout = _ColumnLayout(max_name_len=10, max_dtype_len=8, has_null_stats=True)
        row = _format_column_row(0, "a", pl.Int64, 100, layout, {"a": 10})
        assert "90" in row  # non-null
        assert "10" in row  # null count
        assert "10.00%" in row  # null pct


class TestFormatEllipsisRow:
    def test_contains_dots(self):
        layout = _ColumnLayout(max_name_len=10, max_dtype_len=8, has_null_stats=False)
        row = _format_ellipsis_row(layout)
        assert "..." in row


class TestBuildColumnTable:
    def test_full_table(self):
        lines = _build_column_table(
            col_names=["a", "b"],
            dtypes=[pl.Int64, pl.Utf8],
            indices=[0, 1],
            omitted=False,
            rows=10,
            show_null_stats=False,
            null_by_col={},
        )
        assert lines[0] == "Columns:"
        assert "a" in lines[2]
        assert "b" in lines[3]

    def test_with_omission(self):
        lines = _build_column_table(
            col_names=["a", "b", "c", "d", "e"],
            dtypes=[pl.Int64] * 5,
            indices=[0, 1, 3, 4],
            omitted=True,
            rows=10,
            show_null_stats=False,
            null_by_col={},
        )
        text = "\n".join(lines)
        assert "..." in text
        assert "a" in text
        assert "e" in text

    def test_empty(self):
        lines = _build_column_table(
            col_names=[],
            dtypes=[],
            indices=[],
            omitted=False,
            rows=0,
            show_null_stats=False,
            null_by_col={},
        )
        assert any("(no columns)" in l for l in lines)
