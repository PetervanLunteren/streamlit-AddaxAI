#!/usr/bin/env python3
"""
Utility script to normalize AddaxAI taxon-mapping CSV files into a simplified format.

Transforms rows shaped like:
    model_class, level_class, level_order, level_family, level_genus, level_species, ...
into:
    model_class,class,order,family,genus,species

The simplified output is written to `taxonomy.csv` alongside the original file
without modifying or deleting the source CSV.

Usage:
    python scripts/convert_taxon_mapping.py \
        --root models/cls \
        --pattern "*/taxon-mapping.csv"
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Iterable, List

PREFIXES = {
    "level_class": "class",
    "level_order": "order",
    "level_family": "family",
    "level_genus": "genus",
    "level_species": "species",
}


def simplify_row(row: dict[str, str]) -> List[str]:
    """Convert a raw CSV row to the simplified column order."""
    model_class = (row.get("model_class") or "").strip()

    parts: dict[str, str] = {}
    for source, prefix in PREFIXES.items():
        raw = (row.get(source) or "").strip()
        raw_lower = raw.lower()
        if not raw_lower or not raw_lower.startswith(f"{prefix.lower()} "):
            parts[prefix] = ""
            continue
        parts[prefix] = raw[len(prefix) + 1 :].strip().lower()

    if not any(parts.values()):
        return [model_class, "", "", "", "", ""]

    # Collapse downstream levels when only the class is truly known.
    if (
        parts["class"]
        and all(parts[level] in ("", parts["class"]) for level in ("order", "family", "genus", "species"))
    ):
        parts.update({"order": "", "family": "", "genus": "", "species": ""})

    genus = parts["genus"]
    species = parts["species"]
    if genus and species:
        if species == genus:
            parts["species"] = ""
        elif species.startswith(f"{genus} "):
            parts["species"] = species[len(genus) + 1 :]

    return [
        model_class,
        parts["class"],
        parts["order"],
        parts["family"],
        parts["genus"],
        parts["species"],
    ]


def convert_file(csv_path: pathlib.Path) -> None:
    """Write a simplified taxonomy CSV alongside the original file."""
    simplified_rows: List[List[str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in PREFIXES if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path}: missing expected columns {missing}. Cannot convert."
            )
        for row in reader:
            simplified_rows.append(simplify_row(row))

    headers = ["model_class", "class", "order", "family", "genus", "species"]
    out_path = csv_path.with_name("taxonomy.csv")
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(simplified_rows)


def find_targets(root: pathlib.Path, pattern: str) -> Iterable[pathlib.Path]:
    """Yield all CSV files matching the glob pattern relative to root."""
    return root.glob(pattern)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize taxon-mapping CSV files.")
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("models/cls"),
        help="Root directory to search (default: models/cls).",
    )
    parser.add_argument(
        "--pattern",
        default="*/taxon-mapping.csv",
        help='Glob pattern relative to --root (default: "*/taxon-mapping.csv").',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root: pathlib.Path = args.root
    pattern: str = args.pattern

    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    targets = list(find_targets(root, pattern))
    if not targets:
        raise SystemExit(f"No files matched pattern {pattern!r} under {root}")

    for csv_path in targets:
        convert_file(csv_path)
        print(f"Wrote {csv_path.with_name('taxonomy.csv').relative_to(root.parent)}")


if __name__ == "__main__":
    main()
