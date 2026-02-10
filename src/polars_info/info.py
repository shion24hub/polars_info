from __future__ import annotations

from dataclasses import dataclass
from typing import IO, Optional, Sequence

import polars as pl

from ._formatting import _build_column_table, _bytes_to_human


@dataclass(frozen=True)
class DFInfoSummary:
    """Summary information returned by print_df_info().

    Attributes:
        rows: Number of rows.
        cols: Number of columns.
        estimated_size_bytes: Estimated memory usage in bytes. None if unavailable.
        dtypes: Mapping of column name to Polars dtype.
    """

    rows: int
    cols: int
    estimated_size_bytes: Optional[int]
    dtypes: dict[str, pl.DataType]


def _safe_estimated_size_bytes(df: pl.DataFrame) -> Optional[int]:
    """Return the estimated size in bytes of a Polars DataFrame, if available.

    Args:
        df: A Polars DataFrame.

    Returns:
        Estimated size in bytes, or None if it cannot be determined.
    """
    try:
        est = getattr(df, "estimated_size", None)
        if est is None:
            return None
        return int(est())
    except Exception:
        return None


def _build_header_lines(
    df: pl.DataFrame,
    name: Optional[str],
    est_bytes: Optional[int],
) -> list[str]:
    """Build the header section of the info output."""
    lines: list[str] = []
    lines.append(f"<class '{df.__class__.__module__}.{df.__class__.__name__}'>")
    if name:
        lines.append(f"Name: {name}")
    rows, cols = df.shape
    lines.append(f"Shape: ({rows:,}, {cols:,})")
    lines.append(f"Estimated size: {_bytes_to_human(est_bytes)}")
    return lines


def _collect_null_counts(df: pl.DataFrame) -> dict[str, int]:
    """Collect per-column null counts, returning empty dict on failure."""
    try:
        raw = df.null_count().row(0, named=True)  # type: ignore[assignment]
        return {k: int(v) for k, v in raw.items()}
    except Exception:
        return {}


def _compute_display_indices(
    n_cols: int,
    *,
    display: str,
    head: int,
    tail: int,
    max_cols: int,
) -> tuple[list[int], bool]:
    """Return column indices to display and whether any columns were omitted.

    Args:
        n_cols: Total number of columns.
        display: Display mode.
            - "full": Show all columns if possible (capped by max_cols).
            - "head_tail": Show only the first ``head`` and last ``tail`` columns
              (columns in between are omitted).
            - "auto": Use "full" when the column count is within max_cols,
              otherwise fall back to "head_tail".
        head: Number of leading columns to show in head_tail mode.
        tail: Number of trailing columns to show in head_tail mode.
        max_cols: Upper limit for full/auto display. Exceeding this falls back
            to head_tail mode.

    Returns:
        (indices, omitted):
            indices: Column indices to display (in ascending order).
            omitted: Whether an ellipsis ("...") placeholder is needed.

    Raises:
        ValueError: If display is invalid or head/tail/max_cols are out of range.
    """
    if n_cols < 0:
        raise ValueError("n_cols must be >= 0")
    if head < 0 or tail < 0:
        raise ValueError("head and tail must be >= 0")
    if max_cols < 1:
        raise ValueError("max_cols must be >= 1")
    if display not in {"full", "head_tail", "auto"}:
        raise ValueError('display must be one of {"full", "head_tail", "auto"}')

    if n_cols == 0:
        return ([], False)

    if display == "auto":
        display = "head_tail" if n_cols > max_cols else "full"

    if display == "full":
        if n_cols <= max_cols:
            return (list(range(n_cols)), False)
        display = "head_tail"

    # head_tail
    show_head = min(head, n_cols)
    show_tail = min(tail, max(0, n_cols - show_head))
    indices = list(range(show_head))
    if show_tail > 0:
        indices.extend(range(n_cols - show_tail, n_cols))
    omitted = (show_head + show_tail) < n_cols
    return (indices, omitted)


def print_df_info(
    df: pl.DataFrame,
    *,
    name: Optional[str] = None,
    display: str = "auto",
    head: int = 5,
    tail: int = 5,
    max_cols: int = 60,
    show_null_stats: bool = True,
    show_sample: int = 0,
    file: Optional[IO[str]] = None,
) -> DFInfoSummary:
    """Print an info-style summary of a Polars DataFrame, with column truncation support.

    Not fully compatible with pandas.DataFrame.info(), but formats and prints:
    - DataFrame class name
    - Shape (rows and columns)
    - Estimated memory size (when available)
    - Dtype of each column
    - (Optional) Null / non-null statistics
    - (Optional) Sample of the first N rows

    For DataFrames with many columns, readability in notebooks can suffer.
    Use ``display`` to control how many columns are shown.  For example,
    showing only the first 5 and last 5 columns with "..." in between can be
    achieved with ``display="head_tail", head=5, tail=5`` (the default
    ``display="auto"`` does this automatically when the column count exceeds
    ``max_cols``).

    Args:
        df: The Polars DataFrame to summarize.
        name: An optional label for display purposes.
        display: Column display mode.
            - "auto": Use full display when columns <= ``max_cols``,
              otherwise fall back to head_tail.
            - "head_tail": Show only the first ``head`` and last ``tail``
              columns (with "..." in between).
            - "full": Show all columns if possible (falls back to head_tail
              when exceeding ``max_cols``).
        head: Number of leading columns to show in head_tail mode.
        tail: Number of trailing columns to show in head_tail mode.
        max_cols: Upper limit for full/auto display. Exceeds this triggers
            head_tail mode.
        show_null_stats: If True, display null count, non-null count and null%.
        show_sample: If greater than 0, append the first ``show_sample`` rows
            at the end of the output.
        file: Output destination (e.g. sys.stdout). Defaults to standard output
            via print when None.

    Returns:
        DFInfoSummary: Summary containing row/column counts, estimated size,
        and a dtype mapping.

    Raises:
        TypeError: If df is not a polars.DataFrame.
        ValueError: If any argument is invalid.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({f"c{i}": [i, None] for i in range(30)})
        >>> _ = print_df_info(df, display="head_tail", head=5, tail=5)
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"df must be polars.DataFrame, got {type(df)!r}")

    rows, cols = df.shape
    est_bytes = _safe_estimated_size_bytes(df)

    out_lines = _build_header_lines(df, name, est_bytes)

    indices, omitted = _compute_display_indices(
        cols, display=display, head=head, tail=tail, max_cols=max_cols
    )

    null_by_col = _collect_null_counts(df) if show_null_stats and cols > 0 else {}

    out_lines.extend(
        _build_column_table(
            df.columns, df.dtypes, indices, omitted, rows,
            show_null_stats=show_null_stats,
            null_by_col=null_by_col,
        )
    )

    if show_sample and show_sample > 0:
        out_lines.append(f"Sample (head {show_sample}):")
        out_lines.append(repr(df.head(show_sample)))

    text = "\n".join(out_lines)
    if file is None:
        print(text)
    else:
        file.write(text + "\n")

    return DFInfoSummary(
        rows=rows,
        cols=cols,
        estimated_size_bytes=est_bytes,
        dtypes={c: t for c, t in df.schema.items()},
    )
