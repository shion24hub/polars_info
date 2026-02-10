from __future__ import annotations

from dataclasses import dataclass
from typing import IO, Optional, Sequence

import polars as pl


@dataclass(frozen=True)
class DFInfoSummary:
    """print_df_info() の要約情報。

    Attributes:
        rows: 行数。
        cols: 列数。
        estimated_size_bytes: 推定メモリ使用量（バイト）。取得できない場合は None。
        dtypes: 列名 -> Polars dtype の辞書。
    """

    rows: int
    cols: int
    estimated_size_bytes: Optional[int]
    dtypes: dict[str, pl.DataType]


def _bytes_to_human(n_bytes: Optional[int]) -> str:
    """バイト数を人間向け表記に変換する。

    Args:
        n_bytes: バイト数。None の場合は不明扱い。

    Returns:
        例: "12.34 MiB"、または "Unknown"。
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
    """Polars DataFrame の推定サイズ（bytes）を可能なら取得する。

    Args:
        df: Polars DataFrame。

    Returns:
        推定サイズ（bytes）。取得不可なら None。
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
    """表示対象の列インデックスと、省略が発生したかを返す。

    Args:
        n_cols: 全列数。
        display: 表示モード。
            - "full": 可能なら全列表示（ただし max_cols で上限）。
            - "head_tail": 先頭 head 列と末尾 tail 列のみ表示（間は省略）。
            - "auto": 列数が多い場合は head_tail、少なければ full。
        head: head_tail モードの先頭表示列数。
        tail: head_tail モードの末尾表示列数。
        max_cols: full/auto で full 表示するときの上限（超える場合は head_tail に落とす）。

    Returns:
        (indices, omitted):
            indices: 表示する列のインデックス（昇順）。
            omitted: 途中省略（...）が入るかどうか。

    Raises:
        ValueError: display が不正、または head/tail/max_cols が不正な場合。
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
        # 「全列を出すとNotebookが崩れる」問題を避けるため、
        # max_cols を超えたら head_tail にフォールバック。
        display = "head_tail" if n_cols > max_cols else "full"

    if display == "full":
        if n_cols <= max_cols:
            return (list(range(n_cols)), False)
        # full 指定でも上限は守り、head_tail に落とす
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
    # 追加: 表示短縮の制御
    display: str = "auto",
    head: int = 5,
    tail: int = 5,
    max_cols: int = 60,
    # 既存: 内容
    show_null_stats: bool = True,
    show_sample: int = 0,
    file: Optional[IO[str]] = None,
) -> DFInfoSummary:
    """polars.DataFrame の info 風サマリを見やすく出力する（列数が多い場合は省略対応）。

    pandas.DataFrame.info() と完全互換ではありませんが、以下を整形して出力します。
    - DataFrame クラス名
    - shape（行数・列数）
    - 推定メモリサイズ（可能な場合）
    - 各列の dtype
    - （任意）null / non-null の統計
    - （任意）先頭 N 行サンプル

    列数が多い DataFrame では Notebook の視認性が落ちやすいため、`display` で表示量を制御できます。
    典型例として「先頭5列＋末尾5列だけ出して真ん中を ...」は `display="head_tail", head=5, tail=5`
    で実現できます（`display="auto"` の既定でも、列数が `max_cols` を超えると自動でそうなります）。

    Args:
        df: 対象の Polars DataFrame。
        name: 表示用の任意ラベル。
        display: 列表示モード。
            - "auto": 列数が `max_cols` を超えると head_tail 表示、それ以下は full 表示。
            - "head_tail": 先頭 `head` 列＋末尾 `tail` 列のみ表示（間は ...）。
            - "full": 可能なら全列表示（ただし `max_cols` を超える場合は head_tail にフォールバック）。
        head: head_tail モードの先頭表示列数。
        tail: head_tail モードの末尾表示列数。
        max_cols: full/auto で full 表示するときの上限（超えたら head_tail）。
        show_null_stats: True の場合、null / non-null 数と null% を表示します。
        show_sample: 0 より大きい場合、先頭 show_sample 行を最後に表示します。
        file: 出力先（例: sys.stdout）。None の場合は標準出力に print します。

    Returns:
        DFInfoSummary: サマリ情報（行数・列数・推定サイズ・dtype辞書）。

    Raises:
        TypeError: df が polars.DataFrame ではない場合。
        ValueError: 引数が不正な場合。

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({f"c{i}": [i, None] for i in range(30)})
        >>> _ = print_df_info(df, display="head_tail", head=5, tail=5)
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"df must be polars.DataFrame, got {type(df)!r}")

    out_lines: list[str] = []

    # ヘッダ部
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

    # null統計（まとめて一発で取得）
    null_by_col: dict[str, int] = {}
    if show_null_stats and cols > 0:
        try:
            null_by_col = df.null_count().row(0, named=True)  # type: ignore[assignment]
            null_by_col = {k: int(v) for k, v in null_by_col.items()}
        except Exception:
            null_by_col = {}

    # 表の列幅（表示対象だけで計算）
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

    # ヘッダ
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
        """表の1行を整形する。"""
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

    # 行の出力（省略箇所は "..." 行を1つ挿入）
    if not indices:
        out_lines.append("(no columns)")
    else:
        # indices は先頭ブロック＋末尾ブロック（必要なら）という前提
        # 途中が飛ぶタイミングで ... を入れる
        prev = None
        for i in indices:
            if prev is not None and i != prev + 1:
                # 見た目重視で idx 列は空欄にする
                out_lines.append(
                    f"{'':>3}  {'...':<{max_name_len}}  {'':<{max_dtype_len}}"
                )
            out_lines.append(fmt_row(i, col_names[i], dtypes[i]))
            prev = i

    # サンプル表示
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
