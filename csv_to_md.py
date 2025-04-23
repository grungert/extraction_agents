#!/usr/bin/env python3
"""
csv_head_to_rich.py
───────────────────
For each CSV file given (or every *.csv in a folder) print the first N raw
lines as a Markdown table *rendered in-terminal via Rich* and, unless
--stdout-only is given, also write <file>.md alongside the CSV.

• Encoding: BOM → charset-normalizer → latin-1
• Delimiter: csv.Sniffer()  (choices: , ; | \t)  or --sep
• Keeps banners / blank lines / headers exactly intact
"""

from __future__ import annotations
import argparse, csv, pathlib, sys
from typing import Iterable, Sequence

import pandas as pd
from rich.console import Console
from rich.markdown import Markdown

try:
    from charset_normalizer import from_bytes as _probe_enc
except ImportError:
    _probe_enc = None

console = Console()

# ────────────────────── encoding helpers ──────────────────────
def _bom_detect(buf: bytes) -> str | None:
    import codecs
    boms = [
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
    ]
    for bom, enc in boms:
        if buf.startswith(bom):
            return enc
    return None

def detect_encoding(path: pathlib.Path, sample: int = 8192) -> str:
    raw = path.read_bytes()[:sample]
    enc = _bom_detect(raw)
    if enc:
        return enc
    if _probe_enc:
        guess = _probe_enc(raw).best()
        if guess and guess.encoding and guess.encoding.lower() != "ascii":
            return guess.encoding
    return "latin-1"                                   # never fails

# ───────────────────── delimiter helper ───────────────────────
def sniff_delim(text: str,
                choices: Sequence[str] = (",", ";", "|", "\t")) -> str:
    try:
        return csv.Sniffer().sniff(text, delimiters=choices).delimiter
    except csv.Error:
        return ","

# ───────────────────── per-file conversion ────────────────────
def csv_to_markdown(path: pathlib.Path,
                    rows: int,
                    sep: str | None = None,
                    enc_override: str | None = None) -> str:
    enc_trials: list[str] = [enc_override] if enc_override else []
    enc_trials += [detect_encoding(path), "utf-8-sig", "latin-1"]

    seen: set[str] = set()
    enc_trials = [e for e in enc_trials if e not in seen and not seen.add(e)]

    for enc in enc_trials:
        try:
            with path.open("r", encoding=enc, newline="") as fh:
                sample = fh.read(4096)
                fh.seek(0)
                delimiter = sep or sniff_delim(sample)
                df = pd.read_csv(
                    fh,
                    sep=delimiter,
                    engine="python",
                    header=None,
                    nrows=rows,
                    skip_blank_lines=False,
                    keep_default_na=False,
                    skipinitialspace=True,
                    quoting=csv.QUOTE_MINIMAL,
                    on_bad_lines="warn",
                )
            return df.to_markdown(index=False)
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"All attempted encodings failed: {enc_trials}")

# ──────────────────── folder/file dispatcher ──────────────────
def emit(csv_path: pathlib.Path,
         rows: int,
         sep: str | None,
         enc: str | None,
         stdout_only: bool) -> None:
    try:
        md = csv_to_markdown(csv_path, rows, sep, enc)
    except Exception as exc:
        console.print(f"[red]ERROR[/] reading {csv_path}: {exc}")
        return

    console.rule(f"[bold cyan]{csv_path.name}")
    console.print(Markdown(md))

    if not stdout_only:
        out_path = csv_path.with_suffix(".md")
        out_path.write_text(md, encoding="utf-8")
        console.print(f"[green]✔  wrote[/] {out_path}")

def walk_targets(targets: Iterable[pathlib.Path],
                 rows: int,
                 sep: str | None,
                 enc: str | None,
                 stdout_only: bool) -> None:
    for p in targets:
        if p.is_dir():
            for csv_file in sorted(p.glob("*.csv")):
                emit(csv_file, rows, sep, enc, stdout_only)
        elif p.is_file():
            emit(p, rows, sep, enc, stdout_only)
        else:
            console.print(f"[yellow]WARNING:[/] {p} is not a file or directory")

# ────────────────────────── CLI glue ──────────────────────────
def cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render first N raw lines of CSVs as Markdown tables with Rich."
    )
    p.add_argument("paths", nargs="+", type=pathlib.Path,
                   help="CSV files or folders containing CSVs")
    p.add_argument("--rows", type=int, default=15,
                   help="Number of raw lines to include [default: 15]")
    p.add_argument("--sep",
                   help="Delimiter override (skip auto-detect)")
    p.add_argument("-e", "--encoding",
                   help="Force text encoding (e.g. cp1250)")
    p.add_argument("--stdout-only", action="store_true",
                   help="Skip writing <file>.md, just print to console")
    return p

def main(argv: list[str] | None = None) -> None:
    args = cli().parse_args(argv)
    walk_targets(args.paths, args.rows, args.sep, args.encoding, args.stdout_only)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(f"[bold red]FATAL:[/] {exc}")
        sys.exit(1)
