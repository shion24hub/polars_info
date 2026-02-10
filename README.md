# polars-info

Your tiny sidekick for understanding `polars.DataFrame` fast.

With `print_df_info()`, you can print a clean, friendly summary of:

- shape (rows and columns)
- estimated size (bytes)
- per-column dtypes
- null / non-null stats (optional)
- sample rows from the head (optional)

Even when your table has tons of columns, `head + tail` display keeps it readable.

## 📦 Install

```bash
pip install polars-info
```

## 🚀 Quick Start

```python
import polars as pl
from polars_info import print_df_info

df = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["A", "B", None],
        "score": [10.2, None, 8.7],
    }
)

summary = print_df_info(df, name="demo_df")
print(summary.rows, summary.cols)
```

Output example:

```text
<class 'polars.dataframe.frame.DataFrame'>
Name: Example DataFrame
Shape: (3, 3)
Estimated size: 51 B
Columns:
  #  Column  Dtype    Non-Null    Null   Null%
  0  id      Int64           3       0   0.00%
  1  name    String          2       1  33.33%
  2  score   Float64         2       1  33.33%
Sample (head 2):
shape: (2, 3)
┌─────┬──────┬───────┐
│ id  ┆ name ┆ score │
│ --- ┆ ---  ┆ ---   │
│ i64 ┆ str  ┆ f64   │
╞═════╪══════╪═══════╡
│ 1   ┆ A    ┆ 10.2  │
│ 2   ┆ B    ┆ null  │
└─────┴──────┴───────┘
```

## 🌟 Great For

- fixing "too many columns, can't read anything" in notebooks
- quickly checking dtype / null health before preprocessing
- running lightweight, repeatable data sanity checks

## ⚙️ Main Options

```python
print_df_info(
    df,
    name="train_df",
    display="auto",      # "auto" | "head_tail" | "full"
    head=5,              # head columns shown in head_tail mode
    tail=5,              # tail columns shown in head_tail mode
    max_cols=60,         # full-display limit in auto/full mode
    show_null_stats=True,
    show_sample=3,       # print first 3 rows
)
```

## 📚 API

- `print_df_info(df, ..., show_sample=0) -> DFInfoSummary`
- `DFInfoSummary`
- `rows: int`
- `cols: int`
- `estimated_size_bytes: Optional[int]`
- `dtypes: dict[str, pl.DataType]`

## 📄 License

The license follows the configuration in `pyproject.toml`.
