# polars-info

**Pandas-like `df.info()` for Polars.**

Polars has no built-in `.info()`. This library brings the familiar summary view
to `polars.DataFrame` — column dtypes, null counts, memory usage — just like
`pandas.DataFrame.info()`, plus extras like head/tail column truncation.

With `print_df_info()`, you can print a clean, friendly summary of:

- shape (rows and columns)
- estimated size (bytes)
- per-column dtypes
- null / non-null stats (optional)
- sample rows from the head (optional)

Even when your table has tons of columns, `head + tail` display keeps it readable.

## ✏️ Pandas `df.info()` vs `polars_info`

<table>
<tr>
<th>Pandas <code>df.info()</code></th>
<th>polars-info <code>print_df_info(df)</code></th>
</tr>
<tr>
<td>

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   id      3 non-null      int64
 1   name    2 non-null      object
 2   score   2 non-null      float64
dtypes: float64(1), int64(1), object(1)
memory usage: 200.0+ bytes
```

</td>
<td>

```text
<class 'polars.dataframe.frame.DataFrame'>
Shape: (3, 3)
Estimated size: 51 B
Columns:
  #  Column  Dtype    Non-Null    Null   Null%
  0  id      Int64           3       0   0.00%
  1  name    String          2       1  33.33%
  2  score   Float64         2       1  33.33%
```

</td>
</tr>
</table>

Key differences from Pandas:

- **Null% column** — instantly spot columns with high missing rates
- **Head/tail truncation** — DataFrames with 100+ columns stay readable
- **Returns a summary object** — `DFInfoSummary` for programmatic use

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

summary = print_df_info(df)
```

```text
<class 'polars.dataframe.frame.DataFrame'>
Shape: (3, 3)
Estimated size: 51 B
Columns:
  #  Column  Dtype    Non-Null    Null   Null%
  0  id      Int64           3       0   0.00%
  1  name    String          2       1  33.33%
  2  score   Float64         2       1  33.33%
```

Use `name` and `show_sample` to add a label and preview rows.

```python
summary = print_df_info(df, name="demo_df", show_sample=2)
```

```text
<class 'polars.dataframe.frame.DataFrame'>
Name: demo_df
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

The returned `DFInfoSummary` gives you programmatic access to the metadata.

```python
print(summary.rows, summary.cols)  # 3 3
print(summary.dtypes)              # {'id': Int64, 'name': String, 'score': Float64}
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

## 📄 License

[MIT](LICENSE)
