from __future__ import annotations

from .data_prob import ProbType
from .data_geom import GeomType
from .data_mesh import MeshType
from .data_dna import DNAType

import argparse
from pathlib import Path

from .input import (
    input_initialize,
    compute_svg_shared_frame,
    compute_svg_shared_route_start,
    SvgSharedFrame,
    SvgSharedRouteStart,
)
from .modgeo import modgeo_modification
from .section import section_generation
from .basepair import basepair_discretize
from .route import route_generation
from .seqdesign import seqdesign_design
from .output import output_generation, print_summary
from .debug_dump import dump_mesh_snapshot
from .svg_import import list_svg_layers
from .validation import enforce_pipeline_validation
from . import para


def _resolve_cli_input_path(svg_path: str) -> str:
    path = Path(svg_path)
    if path.exists():
        return str(path)
    candidate = Path.cwd() / "input" / svg_path
    if candidate.exists():
        return str(candidate)
    return svg_path


def _run_pipeline(
    svg_path: str,
    config_path: str | None,
    edge_len: int | None,
    frame_mode: str,
    summary: bool,
    debug_mesh: bool,
    svg_layer: int | None = None,
    svg_import_layer: int | None = None,
    svg_output_subdir: str = "",
    svg_shared_frame: SvgSharedFrame | None = None,
    svg_shared_route_start: SvgSharedRouteStart | None = None,
) -> None:
    prob = ProbType()
    geom = GeomType()
    mesh = MeshType()
    dna = DNAType()

    input_initialize(
        prob,
        geom,
        svg_path=svg_path,
        config_path=config_path,
        edge_len_override=edge_len,
        frame_mode=frame_mode,
        svg_layer=svg_import_layer if svg_import_layer is not None else svg_layer,
        svg_output_subdir=svg_output_subdir,
        svg_shared_frame=svg_shared_frame,
        svg_shared_route_start=svg_shared_route_start,
    )
    enforce_pipeline_validation("input_initialize", prob, geom, mesh, dna)
    modgeo_modification(prob, geom)
    enforce_pipeline_validation("modgeo_modification", prob, geom, mesh, dna)
    section_generation(prob, geom)
    enforce_pipeline_validation("section_generation", prob, geom, mesh, dna)
    basepair_discretize(prob, geom, mesh)
    enforce_pipeline_validation("basepair_discretize", prob, geom, mesh, dna)
    if debug_mesh or para.para_debug_mesh:
        dump_mesh_snapshot(prob, geom, mesh, "pre_route", dna)
    route_generation(prob, geom, mesh, dna)
    enforce_pipeline_validation("route_generation", prob, geom, mesh, dna)
    if debug_mesh or para.para_debug_mesh:
        dump_mesh_snapshot(prob, geom, mesh, "post_route", dna)
    seqdesign_design(prob, geom, mesh, dna)
    enforce_pipeline_validation("seqdesign_design", prob, geom, mesh, dna)
    output_generation(prob, geom, mesh, dna)
    if summary or para.para_print_summary:
        print_summary(prob, geom, mesh, dna)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PERDIX Python port")
    parser.add_argument("svg", help="Input SVG file")
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    parser.add_argument(
        "--edge-len",
        type=int,
        default=None,
        help="Override minimum edge length (bp), bypassing config",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary to stdout (off by default)",
    )
    parser.add_argument(
        "--debug-mesh",
        action="store_true",
        help="Dump mesh JSON before and after routing",
    )
    parser.add_argument(
        "--frame-mode",
        choices=["legacy", "svg-rect"],
        default="legacy",
        help="Coordinate frame mode for SVG import",
    )
    parser.add_argument(
        "--svg-layer",
        type=int,
        default=None,
        help="When reading a multi-layer SVG, import only <g id=\"Layer_N\">",
    )
    parser.add_argument(
        "--route-start",
        choices=["default", "shared-boundary"],
        default="default",
        help="Control how the routed start edge is chosen for multi-layer SVGs",
    )
    return parser


def _validate_route_start_args(args, svg_layers: list[int]) -> None:
    if args.route_start != "shared-boundary":
        return
    if not svg_layers:
        raise ValueError("--route-start shared-boundary requires a multi-layer SVG with Layer_N groups")
    if args.svg_layer is not None:
        raise ValueError("--route-start shared-boundary cannot be combined with --svg-layer; it requires all layers")


def _compute_shared_route_start(args, svg_path: str, svg_layers: list[int]) -> SvgSharedRouteStart | None:
    if not svg_layers or args.route_start != "shared-boundary":
        return None
    return compute_svg_shared_route_start(
        svg_path,
        config_path=args.config,
        frame_mode=args.frame_mode,
        svg_layers=svg_layers,
    )


def _run_layered_pipelines(args, svg_path: str, svg_layers: list[int], shared_route_start: SvgSharedRouteStart | None) -> bool:
    if args.svg_layer is not None or not svg_layers:
        return False

    shared_frame = compute_svg_shared_frame(
        svg_path,
        config_path=args.config,
        frame_mode=args.frame_mode,
        svg_layers=svg_layers,
    )
    for layer in svg_layers:
        _run_pipeline(
            svg_path=svg_path,
            config_path=args.config,
            edge_len=args.edge_len,
            frame_mode=args.frame_mode,
            summary=args.summary,
            debug_mesh=args.debug_mesh,
            svg_layer=layer,
            svg_import_layer=layer,
            svg_output_subdir=f"Layer_{layer}",
            svg_shared_frame=shared_frame,
            svg_shared_route_start=shared_route_start,
        )
    return True


def main() -> None:
    """Entry point mirroring PERDIX.f90::Main (SVG-driven)."""
    args = _build_parser().parse_args()
    svg_path = _resolve_cli_input_path(args.svg)
    svg_layers = list_svg_layers(svg_path) if svg_path.lower().endswith(".svg") else []
    _validate_route_start_args(args, svg_layers)
    shared_route_start = _compute_shared_route_start(args, svg_path, svg_layers)
    if _run_layered_pipelines(args, svg_path, svg_layers, shared_route_start):
        return

    svg_output_subdir = f"Layer_{args.svg_layer}" if args.svg_layer is not None and svg_layers else ""
    _run_pipeline(
        svg_path=svg_path,
        config_path=args.config,
        edge_len=args.edge_len,
        frame_mode=args.frame_mode,
        summary=args.summary,
        debug_mesh=args.debug_mesh,
        svg_layer=args.svg_layer,
        svg_import_layer=args.svg_layer,
        svg_output_subdir=svg_output_subdir,
        svg_shared_frame=None,
        svg_shared_route_start=shared_route_start,
    )


if __name__ == "__main__":
    main()
