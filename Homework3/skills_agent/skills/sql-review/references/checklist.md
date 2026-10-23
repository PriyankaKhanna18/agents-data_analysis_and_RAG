# SQL review checklist

## Security
- **SEC-1 (high)**: User input concatenated into the query string (SQL
  injection). Fix: use bound parameters (`?`, `%s` or `:name`).
- **SEC-2 (medium)**: `GRANT ALL` or queries run as a superuser account.

## Correctness
- **COR-1 (high)**: `UPDATE` or `DELETE` without a `WHERE` clause.
- **COR-2 (medium)**: Comparing with `= NULL` / `!= NULL` instead of
  `IS NULL` / `IS NOT NULL`.
- **COR-3 (medium)**: Non-aggregated columns in `SELECT` that are missing from
  `GROUP BY`.
- **COR-4 (low)**: `ORDER BY` a column position number (`ORDER BY 2`) instead
  of a name.

## Performance
- **PERF-1 (medium)**: `SELECT *` in application queries. Fix: list the
  columns you need.
- **PERF-2 (medium)**: A function wrapped around an indexed column in `WHERE`
  (e.g. `WHERE YEAR(created_at) = 2024`), which stops the index being used.
  Fix: use a range (`created_at >= '2024-01-01' AND created_at < '2025-01-01'`).
- **PERF-3 (low)**: Leading wildcard in `LIKE '%term'`, which forces a full
  scan.
