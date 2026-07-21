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
from .route import (
    SharedScaffoldStartReport,
    SharedScaffoldStartSpec,
    SharedScaffoldXoverReport,
    SharedScaffoldXoverSpec,
    apply_shared_scaffold_start,
    collect_exterior_shared_scaffold_edge_keys,
    collect_safe_shared_scaffold_start_node_keys,
    collect_shared_scaffold_node_keys,
    export_shared_scaffold_start,
    export_shared_scaffold_xovers,
    route_generation,
)
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
    shared_scaffold_xovers: list[SharedScaffoldXoverSpec] | None = None,
    shared_scaffold_report: SharedScaffoldXoverReport | None = None,
    shared_scaffold_start: SharedScaffoldStartSpec | None = None,
    shared_scaffold_start_report: SharedScaffoldStartReport | None = None,
) -> tuple[ProbType, GeomType, MeshType, DNAType]:
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
    route_generation(
        prob,
        geom,
        mesh,
        dna,
        shared_scaffold_xovers=shared_scaffold_xovers,
        shared_scaffold_report=shared_scaffold_report,
    )
    enforce_pipeline_validation("route_generation", prob, geom, mesh, dna)
    if debug_mesh or para.para_debug_mesh:
        dump_mesh_snapshot(prob, geom, mesh, "post_route", dna)
    seqdesign_design(prob, geom, mesh, dna)
    enforce_pipeline_validation("seqdesign_design", prob, geom, mesh, dna)
    if shared_scaffold_start is not None:
        apply_shared_scaffold_start(
            geom,
            mesh,
            dna,
            shared_scaffold_start,
            shared_scaffold_start_report,
            prob=prob,
        )
    output_generation(prob, geom, mesh, dna)
    if summary or para.para_print_summary:
        print_summary(prob, geom, mesh, dna)
    return prob, geom, mesh, dna


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
    parser.add_argument(
        "--shared-crossover-indices",
        type=int,
        default=None,
        metavar="LAYER",
        help="Route LAYER normally, then force same scaffold crossover indices on other SVG layers",
    )
    parser.add_argument(
        "--shared-start",
        choices=["smallest", "exterior"],
        default=None,
        metavar="MODE",
        help="Align scaffold starts using the smallest layer's native start or a safe exterior shared edge",
    )
    return parser


def _validate_route_start_args(args, svg_layers: list[int]) -> None:
    if (
        args.route_start != "shared-boundary"
        and args.shared_crossover_indices is None
        and args.shared_start is None
    ):
        return
    if not svg_layers:
        raise ValueError("shared route options require a multi-layer SVG with Layer_N groups")
    if args.svg_layer is not None:
        raise ValueError("shared route options cannot be combined with --svg-layer; they require all layers")
    if args.shared_crossover_indices is not None and args.shared_crossover_indices not in svg_layers:
        raise ValueError("--shared-crossover-indices must name an existing Layer_N group")


def _compute_shared_route_start(args, svg_path: str, svg_layers: list[int]) -> SvgSharedRouteStart | None:
    if not svg_layers or (
        args.route_start != "shared-boundary"
        and args.shared_crossover_indices is None
        and args.shared_start is None
    ):
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
    ordered_layers = list(svg_layers)
    shared_reference_layer = args.shared_crossover_indices
    if shared_reference_layer is not None:
        ordered_layers = [shared_reference_layer] + [layer for layer in svg_layers if layer != shared_reference_layer]

    shared_specs: list[SharedScaffoldXoverSpec] | None = None
    shared_start: SharedScaffoldStartSpec | None = None
    reports: dict[int, SharedScaffoldXoverReport] = {}
    start_reports: dict[int, SharedScaffoldStartReport] = {}
    layer_outputs: dict[int, tuple[ProbType, GeomType, MeshType, DNAType]] = {}
    for layer in ordered_layers:
        report = None
        layer_specs = None
        if args.shared_crossover_indices is not None and layer != args.shared_crossover_indices:
            layer_specs = shared_specs if shared_specs is not None else []
            report = SharedScaffoldXoverReport()
            reports[layer] = report
        _prob, geom, mesh, dna = _run_pipeline(
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
            shared_scaffold_xovers=layer_specs,
            shared_scaffold_report=report,
        )
        layer_outputs[layer] = (_prob, geom, mesh, dna)
        if args.shared_crossover_indices is not None and layer == args.shared_crossover_indices:
            shared_specs = export_shared_scaffold_xovers(geom, mesh, dna)

    shared_start_source_layer = None
    shared_start_requested = args.shared_start is not None
    if shared_start_requested:
        shared_start_source_layer = _select_shared_start_source_layer(layer_outputs)
        if args.shared_start == "exterior":
            shared_start = _select_exterior_shared_start(layer_outputs)
        else:
            shared_node_keys = None
            for _layer, (_prob, geom, mesh, _dna) in layer_outputs.items():
                node_keys = collect_shared_scaffold_node_keys(geom, mesh)
                shared_node_keys = node_keys if shared_node_keys is None else shared_node_keys & node_keys
            _prob, geom, mesh, dna = layer_outputs[shared_start_source_layer]
            shared_start = export_shared_scaffold_start(
                geom,
                mesh,
                dna,
                allowed_node_keys=shared_node_keys,
            )
            if shared_start is None:
                raise ValueError(
                    f"Layer_{shared_start_source_layer} scaffold start is not an exact node shared by every layer"
                )
        for layer, (_prob, geom, mesh, dna) in layer_outputs.items():
            report = SharedScaffoldStartReport()
            apply_shared_scaffold_start(geom, mesh, dna, shared_start, report, prob=_prob)
            start_reports[layer] = report
            if report.failed:
                raise ValueError(
                    f"Layer_{layer} cannot safely use the shared scaffold start: "
                    f"missing_node={report.missing_node} missing_scaffold={report.missing_scaffold} "
                    f"unsafe_target={report.unsafe_target} invalid_topology={report.invalid_topology}"
                )
            output_generation(_prob, geom, mesh, dna)

    if args.shared_crossover_indices is not None:
        _print_shared_crossover_report(args.shared_crossover_indices, shared_specs or [], reports)
    if shared_start_requested:
        _print_shared_start_report(shared_start_source_layer, shared_start, start_reports)
    return True


def _select_shared_start_source_layer(
    layer_outputs: dict[int, tuple[ProbType, GeomType, MeshType, DNAType]],
) -> int:
    candidates = list(layer_outputs)
    best_layer = candidates[0]
    best_count = None
    for layer in candidates:
        _prob, _geom, _mesh, dna = layer_outputs[layer]
        count = dna.n_base_scaf
        if best_count is None or count < best_count or (count == best_count and layer < best_layer):
            best_count = count
            best_layer = layer
    return best_layer


def _select_exterior_shared_start(
    layer_outputs: dict[int, tuple[ProbType, GeomType, MeshType, DNAType]],
) -> SharedScaffoldStartSpec:
    exterior_edges = None
    safe_nodes = None
    for _layer, (_prob, geom, mesh, dna) in layer_outputs.items():
        layer_exterior = collect_exterior_shared_scaffold_edge_keys(geom)
        layer_safe = collect_safe_shared_scaffold_start_node_keys(geom, mesh, dna)
        exterior_edges = layer_exterior if exterior_edges is None else exterior_edges & layer_exterior
        safe_nodes = layer_safe if safe_nodes is None else safe_nodes & layer_safe
    spec = _select_longest_exterior_start_run(exterior_edges or set(), safe_nodes or set())
    if spec is None:
        raise ValueError("No common safe scaffold start exists on an exterior shared edge")
    return spec


def _select_longest_exterior_start_run(
    exterior_edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]],
    safe_nodes: set[tuple[tuple[tuple[int, int, int], tuple[int, int, int]], int, int]],
) -> SharedScaffoldStartSpec | None:
    grouped: dict[tuple[tuple[tuple[int, int, int], tuple[int, int, int]], int], list[int]] = {}
    for edge_key, bp, sec in safe_nodes:
        if edge_key in exterior_edges:
            grouped.setdefault((edge_key, sec), []).append(bp)

    runs = []
    for (edge_key, sec), values in grouped.items():
        ordered = sorted(set(values))
        start = 0
        for stop in range(1, len(ordered) + 1):
            if stop < len(ordered) and ordered[stop] == ordered[stop - 1] + 1:
                continue
            run = ordered[start:stop]
            runs.append((len(run), edge_key, sec, run))
            start = stop
    if not runs:
        return None
    _length, edge_key, sec, run = min(runs, key=lambda item: (-item[0], item[1], item[2], item[3][0]))
    return SharedScaffoldStartSpec(edge_key=edge_key, bp=run[(len(run) - 1) // 2], sec=sec)


def _print_shared_crossover_report(
    reference_layer: int,
    specs: list[SharedScaffoldXoverSpec],
    reports: dict[int, SharedScaffoldXoverReport],
) -> None:
    print(f"shared scaffold xovers: reference Layer_{reference_layer}, exported {len(specs)}")
    for layer, report in sorted(reports.items()):
        print(
            "shared scaffold xovers: "
            f"Layer_{layer} requested={report.requested} applied={report.applied} "
            f"already_present={report.already_present} missing_node={report.missing_node} "
            f"missing_neighbor={report.missing_neighbor} blocked={report.blocked}"
        )


def _print_shared_start_report(
    source_layer: int,
    spec: SharedScaffoldStartSpec | None,
    reports: dict[int, SharedScaffoldStartReport],
) -> None:
    exported = 0 if spec is None else 1
    print(f"shared scaffold start: source Layer_{source_layer}, exported {exported}")
    for layer, report in sorted(reports.items()):
        print(
            "shared scaffold start: "
            f"Layer_{layer} requested={report.requested} applied={report.applied} "
            f"missing_node={report.missing_node} missing_scaffold={report.missing_scaffold} "
            f"unsafe_target={report.unsafe_target} invalid_topology={report.invalid_topology}"
        )


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
