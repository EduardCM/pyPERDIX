from __future__ import annotations

import argparse

from .data_prob import ProbType
from .data_geom import GeomType
from .input import input_initialize


def run_mvp(svg_path: str) -> None:
    prob = ProbType()
    geom = GeomType()
    input_initialize(prob, geom, svg_path=svg_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="PERDIX MVP: SVG to BILD")
    parser.add_argument("svg", help="Input SVG file")
    parser.add_argument("--config", help="Path to JSON config file", default=None)
    args = parser.parse_args()
    prob = ProbType()
    geom = GeomType()
    input_initialize(prob, geom, svg_path=args.svg, config_path=args.config)


if __name__ == "__main__":
    main()
