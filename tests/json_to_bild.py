#!/usr/bin/env python3
"""
Convert multiple coordinate arrays from a JSON into separate Chimera/ChimeraX .bild files.

Writes up to 4 files (if data exists):
  - <input_stem>_dna_base_scaf_pos.bild
  - <input_stem>_dna_base_scaf_pos_ref.bild
  - <input_stem>_node_pos.bild
  - <input_stem>_node_pos_ref.bild
  
Usage:
  python json_to_bild.py input.json bild_out --radius 2.0 --color 0.2 0.6 0.9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Tuple, Optional


Coord = Tuple[float, float, float]


def _as_coord(v: Any) -> Optional[Coord]:
    if not (isinstance(v, (list, tuple)) and len(v) == 3):
        return None
    try:
        x, y, z = float(v[0]), float(v[1]), float(v[2])
    except (TypeError, ValueError):
        return None
    return (x, y, z)


def iter_positions(records: Any, key: str) -> Iterable[Coord]:
    if not isinstance(records, list):
        return
    for item in records:
        if not isinstance(item, dict):
            continue
        coord = _as_coord(item.get(key))
        if coord is None:
            continue
        yield coord


def write_bild(
    positions: Iterable[Coord],
    out_path: Path,
    radius: float,
    color: Optional[Tuple[float, float, float]],
) -> int:
    if radius <= 0:
        raise ValueError("--radius must be > 0.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Generated from JSON coordinates\n")
        if color is not None:
            r, g, b = color
            for c in (r, g, b):
                if not (0.0 <= c <= 1.0):
                    raise ValueError("--color values must be in [0, 1].")
            f.write(f".color {r:.6g} {g:.6g} {b:.6g}\n")

        for (x, y, z) in positions:
            f.write(f".sphere {x:.6g} {y:.6g} {z:.6g} {radius:.6g}\n")
            count += 1

    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump dna/base_scaf, dna/base_stap and node coordinates into separate .bild files.")
    ap.add_argument("input_json", type=Path, help="Input JSON file")
    ap.add_argument("out_dir", type=Path, help="Output directory for .bild files")
    ap.add_argument("--radius", type=float, default=1.0, help="Sphere radius (default: 1.0)")
    ap.add_argument(
        "--color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=None,
        help="Optional RGB color in [0,1] (e.g. --color 1 0 0)",
    )
    args = ap.parse_args()

    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    out_dir: Path = args.out_dir
    color = tuple(args.color) if args.color else None

    input_stem = args.input_json.stem

    dna_base_scaf = None
    dna_base_stap = None
    if isinstance(data, dict):
        dna = data.get("dna")
        if isinstance(dna, dict):
            dna_base_scaf = dna.get("base_scaf")
            dna_base_stap = dna.get("base_stap")

    node_records = data.get("node") if isinstance(data, dict) else None

    targets = [
        (dna_base_scaf, "pos",     f"{input_stem}_dna_base_scaf_pos.bild"),
        (dna_base_scaf, "pos_ref", f"{input_stem}_dna_base_scaf_pos_ref.bild"),
        (dna_base_stap, "pos",     f"{input_stem}_dna_base_stap_pos.bild"),
        (dna_base_stap, "pos_ref", f"{input_stem}_dna_base_stap_pos_ref.bild"),
        (node_records,  "pos",     f"{input_stem}_node_pos.bild"),
        (node_records,  "pos_ref", f"{input_stem}_node_pos_ref.bild"),
    ]

    wrote_any = False
    for records, key, fname in targets:
        positions = list(iter_positions(records, key))
        if not positions:
            print(f"Skip: no valid coordinates for {key} -> {fname}")
            continue

        out_path = out_dir / fname
        n = write_bild(positions, out_path, radius=args.radius, color=color)
        print(f"Wrote {n} spheres -> {out_path}")
        wrote_any = True

    if not wrote_any:
        raise SystemExit("No output written: none of the requested coordinate sets were found/valid.")


if __name__ == "__main__":
    main()