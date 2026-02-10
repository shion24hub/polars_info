from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import polars as pl


def _bytes_to_human(n_bytes: Optional[int]) -> str:
    """Convert a byte count into a human-readable string.

    Args:
        n_bytes: Number of bytes. Treated as unknown when None.

    Returns:
        A human-readable size string, e.g. "12.34 MiB", or "Unknown".
    """
    if n_bytes is None:
        return "Unknown"
    if n_bytes < 0:
        return str(n_bytes)

    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    size = float(n_bytes)
    i = 0
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    if i == 0:
        return f"{int(size)} {units[i]}"
    return f"{size:.2f} {units[i]}"


@dataclass(frozen=True)
class _ColumnLayout:
    """Column width parameters for table formatting."""

    max_name_len: int
    max_dtype_len: int
    has_null_stats: bool


def _compute_column_layout(
    col_names: Sequence[str],
    dtypes: Sequence[pl.DataType],
    indices: list[int],
    omitted: bool,
    *,
    show_null_stats: bool,
    null_by_col: dict[str, int],
) -> _ColumnLayout:
    """Compute column widths based on displayed columns."""
    display_names = [col_names[i] for i in indices]
    if omitted:
        display_names.append("...")

    max_name_len = (
        max([len("Column")] + [len(n) for n in display_names])
        if indices
        else len("Column")
    )
    max_dtype_len = (
        max([len("Dtype")] + [len(str(dtypes[i])) for i in indices])
        if indices
        else len("Dtype")
    )
    return _ColumnLayout(
        max_name_len=max_name_len,
        max_dtype_len=max_dtype_len,
        has_null_stats=show_null_stats and bool(null_by_col),
    )


def _format_table_header(layout: _ColumnLayout) -> str:
    """Generate the table header line."""
    if layout.has_null_stats:
        return (
            f"{'#':>3}  "
            f"{'Column':<{layout.max_name_len}}  "
            f"{'Dtype':<{layout.max_dtype_len}}  "
            f"{'Non-Null':>8}  "
            f"{'Null':>6}  "
            f"{'Null%':>6}"
        )
    return (
        f"{'#':>3}  "
        f"{'Column':<{layout.max_name_len}}  "
        f"{'Dtype':<{layout.max_dtype_len}}"
    )


def _format_column_row(
    idx: int,
    col: str,
    dt: pl.DataType,
    rows: int,
    layout: _ColumnLayout,
    null_by_col: dict[str, int],
) -> str:
    """Format a single column row."""
    if layout.has_null_stats:
        n_null = null_by_col.get(col, 0)
        n_non_null = rows - n_null
        null_pct = (n_null / rows * 100.0) if rows else 0.0
        return (
            f"{idx:>3}  "
            f"{col:<{layout.max_name_len}}  "
            f"{str(dt):<{layout.max_dtype_len}}  "
            f"{n_non_null:>8,}  "
            f"{n_null:>6,}  "
            f"{null_pct:>5.2f}%"
        )
    return (
        f"{idx:>3}  "
        f"{col:<{layout.max_name_len}}  "
        f"{str(dt):<{layout.max_dtype_len}}"
    )


def _format_ellipsis_row(layout: _ColumnLayout) -> str:
    """Generate the ellipsis row for omitted columns."""
    return f"{'':>3}  {'...':<{layout.max_name_len}}  {'':<{layout.max_dtype_len}}"


def _build_column_table(
    col_names: Sequence[str],
    dtypes: Sequence[pl.DataType],
    indices: list[int],
    omitted: bool,
    rows: int,
    *,
    show_null_stats: bool,
    null_by_col: dict[str, int],
) -> list[str]:
    """Build the complete column table as a list of lines."""
    layout = _compute_column_layout(
        col_names, dtypes, indices, omitted,
        show_null_stats=show_null_stats,
        null_by_col=null_by_col,
    )

    lines: list[str] = []
    lines.append("Columns:")
    lines.append(_format_table_header(layout))

    if not indices:
        lines.append("(no columns)")
    else:
        prev = None
        for i in indices:
            if prev is not None and i != prev + 1:
                lines.append(_format_ellipsis_row(layout))
            lines.append(
                _format_column_row(i, col_names[i], dtypes[i], rows, layout, null_by_col)
            )
            prev = i

    return lines
