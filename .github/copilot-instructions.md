# Copilot instructions (12tone)

## Project shape
- This repo intentionally tracks almost nothing: see `.gitignore` (it ignores `*` and only whitelists a few files). If you add new files (including docs), ensure they are un-ignored.
- Main (and effectively only) code is `12tone.py`.

## What this script does
- CLI tool for analyzing pitch-class matrices (values 0..11) for:
  - adjacent major/minor triads (rows + columns), optionally wrap-around scanning
  - consonant dyads (perfect fifth, octave/unison)
  - symmetry / combinatoriality signals and a weighted `consonance_factor()` score
  - optional: seed-based 12×12 dodecaphonic matrix generation

## How to run (common workflows)
- Preferred interpreter in this repo (local venv): `/Users/arehart/Desktop/.venv/bin/python`
  - If the path doesn’t exist on a machine, use `python3` (or activate the venv first).
- Basic smoke checks:
  - `/Users/arehart/Desktop/.venv/bin/python 12tone.py --example`
  - `/Users/arehart/Desktop/.venv/bin/python 12tone.py --seed "0,6,9,4,5,11,7,1,8,10,3,2" --transpositions --note-names`
  - `/Users/arehart/Desktop/.venv/bin/python 12tone.py --file matrix.csv`
- Best-only search (multiprocess top-K random sampling):
  - `/Users/arehart/Desktop/.venv/bin/python 12tone.py --search-best 20000 --top-k 50 --jobs 4 --progress --quiet`
  - Use `--jobs 0` to auto-pick ~90% of CPU cores.
  - Use throttling to reduce sustained CPU/heat: `--throttle-ms 2 --throttle-every 200`.

## Optional exports / dependencies
- MySQL export:
  - Flag: `--export-sql-db "mysql://user:pass@host:3306/12Tone"` (table name via `--export-sql-table`)
  - Dependency: `pymysql` (the script raises a clear error if missing).
  - The script redacts credentials when reporting DB connection errors (`_redact_db_url`).
- Heatmap export (12×12 only):
  - Flags: `--export-heatmap out.png --heatmap-z 0 --heatmap-labels numbers|pitches`
  - Dependencies: `matplotlib` + `seaborn`.

## Conventions to preserve when editing
- Pitch classes are always treated mod 12; keep inputs/outputs consistent with `PITCH_NAMES`.
- Inversions are enabled by default; the opt-out switch is `--no-inversions`.
- Best-only search uses multiprocessing with the `spawn` start method and avoids sending unpicklable objects to workers.
  - Live DB snapshots use a `Manager().Queue()` proxy for cross-process updates.
- Table names are validated before SQL (`_mysql_ensure_table` checks alnum/underscore); keep that safety behavior.
- Scoring defaults live in `DEFAULT_WEIGHTS`; tune there rather than scattering constants.

## When changing behavior
- Keep CLI flags stable unless you also update `README.md`.
- Prefer extending existing helpers (e.g., `consonance_factor`, `_best_only_worker`, `_mysql_*`) over adding new parallel code paths.
