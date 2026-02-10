from __future__ import annotations

from dataclasses import dataclass
from typing import IO, Optional, Sequence

import polars as pl


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
        # Fall back to head_tail when exceeding max_cols to avoid
        # cluttering notebooks with too many columns.
        display = "head_tail" if n_cols > max_cols else "full"

    if display == "full":
        if n_cols <= max_cols:
            return (list(range(n_cols)), False)
        # Even in full mode, respect the upper limit and fall back to head_tail
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

    out_lines: list[str] = []

    # Header section
    out_lines.append(f"<class '{df.__class__.__module__}.{df.__class__.__name__}'>")
    if name:
        out_lines.append(f"Name: {name}")

    rows, cols = df.shape
    est_bytes = _safe_estimated_size_bytes(df)
    out_lines.append(f"Shape: ({rows:,}, {cols:,})")
    out_lines.append(f"Estimated size: {_bytes_to_human(est_bytes)}")

    col_names: Sequence[str] = df.columns
    dtypes: Sequence[pl.DataType] = df.dtypes

    indices, omitted = _compute_display_indices(
        cols, display=display, head=head, tail=tail, max_cols=max_cols
    )

    # Null statistics (fetched in a single pass)
    null_by_col: dict[str, int] = {}
    if show_null_stats and cols > 0:
        try:
            null_by_col = df.null_count().row(0, named=True)  # type: ignore[assignment]
            null_by_col = {k: int(v) for k, v in null_by_col.items()}
        except Exception:
            null_by_col = {}

    # Compute column widths (based on displayed columns only)
    display_names = [col_names[i] for i in indices]
    if omitted:
        display_names.append("...")

    max_name_len = (
        max([len("Column")] + [len(n) for n in display_names])
        if cols
        else len("Column")
    )
    max_dtype_len = (
        max([len("Dtype")] + [len(str(dtypes[i])) for i in indices])
        if cols
        else len("Dtype")
    )

    # Table header
    out_lines.append("Columns:")
    if show_null_stats and null_by_col:
        header = (
            f"{'#':>3}  "
            f"{'Column':<{max_name_len}}  "
            f"{'Dtype':<{max_dtype_len}}  "
            f"{'Non-Null':>8}  "
            f"{'Null':>6}  "
            f"{'Null%':>6}"
        )
    else:
        header = f"{'#':>3}  {'Column':<{max_name_len}}  {'Dtype':<{max_dtype_len}}"
    out_lines.append(header)

    def fmt_row(idx: int, col: str, dt: pl.DataType) -> str:
        """Format a single table row."""
        if show_null_stats and null_by_col:
            n_null = null_by_col.get(col, 0)
            n_non_null = rows - n_null
            null_pct = (n_null / rows * 100.0) if rows else 0.0
            return (
                f"{idx:>3}  "
                f"{col:<{max_name_len}}  "
                f"{str(dt):<{max_dtype_len}}  "
                f"{n_non_null:>8,}  "
                f"{n_null:>6,}  "
                f"{null_pct:>5.2f}%"
            )
        return f"{idx:>3}  {col:<{max_name_len}}  {str(dt):<{max_dtype_len}}"

    # Output rows (insert a single "..." row where columns are omitted)
    if not indices:
        out_lines.append("(no columns)")
    else:
        # indices consists of a head block + an optional tail block.
        # Insert "..." where there is a gap between consecutive indices.
        prev = None
        for i in indices:
            if prev is not None and i != prev + 1:
                # Leave the index column blank for visual clarity
                out_lines.append(
                    f"{'':>3}  {'...':<{max_name_len}}  {'':<{max_dtype_len}}"
                )
            out_lines.append(fmt_row(i, col_names[i], dtypes[i]))
            prev = i

    # Sample display
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
