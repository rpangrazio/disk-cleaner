# disk-cleaner

CLI tool that analyzes a folder, detects duplicate/superseded files, and produces a privacy-preserving JSON report. It can also optionally apply a reorganization plan.

## Quick start

```bash
python3 disk_cleaner.py --source /path/to/folder --report report.json
```

With optional embedding clustering:

```bash
python3 disk_cleaner.py \
  --source /path/to/folder \
  --report report.json \
  --api-base https://api.openai.com \
  --api-token "$TOKEN" \
  --model text-embedding-3-small
```

## SQLite backend

Use the SQLite backend to persist per-file attributes from each scan for later analysis.

### CLI flags

- `--sqlite-db <path>`
  - Enables SQLite persistence and writes scan attributes to the database file at `<path>`.
- `--sqlite-clear-existing`
  - Deletes existing rows for the current `--source` root before inserting the new scan.
  - Useful when you want each source root to reflect only the latest scan.

Example:

```bash
python3 disk_cleaner.py \
  --source /data/archive \
  --report report.json \
  --sqlite-db ./artifacts/file-attributes.db
```

Replace existing rows for the same source root:

```bash
python3 disk_cleaner.py \
  --source /data/archive \
  --sqlite-db ./artifacts/file-attributes.db \
  --sqlite-clear-existing
```

### Schema

Table: `file_attributes`

| Column | Type | Notes |
|---|---|---|
| `id` | `INTEGER` | Primary key (`AUTOINCREMENT`) |
| `source_root` | `TEXT` | Absolute source directory scanned |
| `path` | `TEXT` | Absolute file path |
| `relative_path` | `TEXT` | Path relative to `source_root` |
| `name` | `TEXT` | File name |
| `extension` | `TEXT` | Lowercased extension (without dot), or `no_ext` |
| `size` | `INTEGER` | File size in bytes |
| `mtime` | `REAL` | UNIX modified timestamp (seconds) |
| `mtime_iso` | `TEXT` | UTC modified timestamp in ISO-8601 |
| `scanned_at` | `TEXT` | UTC scan timestamp in ISO-8601 |

Constraints and indexes:

- `UNIQUE(source_root, path)`
- `INDEX idx_file_attributes_source_root (source_root)`
- `INDEX idx_file_attributes_extension (extension)`

Upsert behavior:

- Rows are inserted with `ON CONFLICT(source_root, path) DO UPDATE`.
- Re-scanning the same source updates metadata (`size`, `mtime`, etc.) and `scanned_at`.

## Example analysis queries

Use `sqlite3`:

```bash
sqlite3 ./artifacts/file-attributes.db
```

Largest files by extension:

```sql
SELECT extension, name, relative_path, size
FROM file_attributes
WHERE source_root = '/data/archive'
ORDER BY size DESC
LIMIT 50;
```

Total size by extension:

```sql
SELECT extension,
       COUNT(*) AS file_count,
       SUM(size) AS total_bytes
FROM file_attributes
WHERE source_root = '/data/archive'
GROUP BY extension
ORDER BY total_bytes DESC;
```

Files not modified in the last year:

```sql
SELECT relative_path, size, mtime_iso
FROM file_attributes
WHERE source_root = '/data/archive'
  AND mtime < strftime('%s','now','-365 day')
ORDER BY mtime ASC;
```

Most recently scanned source roots:

```sql
SELECT source_root,
       MAX(scanned_at) AS last_scan,
       COUNT(*) AS files_in_latest_rows
FROM file_attributes
GROUP BY source_root
ORDER BY last_scan DESC;
```

Potential extension outliers (largest `no_ext` files):

```sql
SELECT relative_path, size, mtime_iso
FROM file_attributes
WHERE source_root = '/data/archive'
  AND extension = 'no_ext'
ORDER BY size DESC
LIMIT 100;
```
