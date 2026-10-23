---
name: sql-review
description: Reviews SQL queries for bugs, security issues and performance problems. Use when the user shares a SQL query and asks to review, check, fix or optimize it.
license: MIT
---

# sql-review

1. Read the checklist at `/skills/sql-review/references/checklist.md` with
   `read_file` and go through every rule in it against the query.
2. Report each problem you find as a row in this table, most severe first:

| Rule | Severity | Problem | Fix |
|------|----------|---------|-----|

   Use the rule IDs from the checklist (e.g. `SEC-1`) and the severities
   `high`, `medium` or `low`.
3. After the table, give the full corrected query in a ```sql block.
4. If the query has no problems, say "No issues found" and don't make up any.
