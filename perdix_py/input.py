from __future__ import annotations

from pathlib import Path

from . import para
from .chimera_bild import write_target_geometry_bild
from .config import load_config
from .data_geom import GeomType
from .data_prob import ProbType
from .input_geom import (
    _convert_faces_to_lines,
    _polygonize_lines,
    _read_seq_txt,
    _round_geom_points,
    _scale_init_geom,
    _set_section_connectivity,
)
from .input_svg import (
    SvgSharedFrame,
    SvgSharedRouteStart,
    _apply_split_triangle_svg_compatibility,
    _apply_svg_shared_frame,
    _apply_svg_shared_route_start,
    compute_svg_shared_frame,
    compute_svg_shared_route_start,
    _normalize_svg_to_canvas_rect,
    _scale_init_geom_canvas,
)
from .naming import bild_path
from .svg_import import import_svg_to_geom, _canonicalize_svg_geom


_GEOM_COPY_FIELDS = (
    "n_sec",
    "n_iniP",
    "n_modP",
    "n_croP",
    "n_iniL",
    "n_croL",
    "n_face",
    "min_edge_length",
    "max_edge_length",
    "n_junc",
    "canvas_rect",
    "sec",
    "iniP",
    "modP",
    "croP",
    "iniL",
    "croL",
    "face",
    "junc",
)


def input_initialize(
    prob: ProbType,
    geom: GeomType,
    svg_path: str | None = None,
    config_path: str | None = None,
    edge_len_override: int | None = None,
    frame_mode: str = "legacy",
    svg_layer: int | None = None,
    svg_output_subdir: str = "",
    svg_shared_frame: SvgSharedFrame | None = None,
    svg_shared_route_start: SvgSharedRouteStart | None = None,
) -> None:
    if svg_path is None:
        raise ValueError("input path is required for MVP")

    in_path = Path(svg_path)
    if not in_path.exists():
        candidate = Path.cwd() / "input" / svg_path
        if candidate.exists():
            in_path = candidate

    prob.name_file = in_path.stem
    prob.type_file = in_path.suffix.lstrip(".").lower()
    prob.path_input = str(Path.cwd() / "input") + "/"

    cfg = load_config(config_path)
    if prob.type_file != "svg":
        raise ValueError(f"Unsupported input type: {prob.type_file}; only SVG input is supported")

    prob.name_prob = cfg.name_prob or prob.name_file
    geom_tmp = import_svg_to_geom(str(in_path), scale=cfg.svg_scale, layer_index=svg_layer)
    _replace_geom_contents(geom, geom_tmp)
    _prepare_geometry(geom, cfg, frame_mode, svg_layer, svg_shared_route_start)
    _apply_problem_defaults(prob, geom, cfg, edge_len_override)
    _prepare_output_dir(prob, cfg, svg_output_subdir)
    _ensure_section_grid(geom, cfg)
    _set_section_connectivity(prob, geom)
    _apply_final_scaling(prob, geom, frame_mode, svg_shared_frame)
    _round_geom_points(geom, digits=4)

    if para.para_write_102:
        out_path = bild_path(prob, "01_target_geometry")
        write_target_geometry_bild(geom, str(out_path), target_len=20.0)


def _replace_geom_contents(target: GeomType, source: GeomType) -> None:
    for field_name in _GEOM_COPY_FIELDS:
        setattr(target, field_name, getattr(source, field_name))


def _prepare_geometry(
    geom: GeomType,
    cfg,
    frame_mode: str,
    svg_layer: int | None,
    svg_shared_route_start: SvgSharedRouteStart | None,
) -> None:
    if frame_mode == "svg-rect":
        _normalize_svg_to_canvas_rect(geom)
    if geom.n_face == 0:
        _polygonize_lines(geom)
    _convert_faces_to_lines(geom)
    if svg_layer is not None:
        _canonicalize_svg_geom(geom)
        if cfg.svg_layer_snap_compat:
            _apply_split_triangle_svg_compatibility(geom)
    if svg_shared_route_start is not None:
        _apply_svg_shared_route_start(geom, svg_shared_route_start)


def _apply_problem_defaults(prob: ProbType, geom: GeomType, cfg, edge_len_override: int | None) -> None:
    geom.sec.types = cfg.sec_type
    geom.sec.n_row = cfg.sec_n_row
    geom.sec.n_col = cfg.sec_n_col
    geom.sec.minR = 1
    geom.sec.maxR = cfg.sec_n_row
    geom.sec.minC = 1
    geom.sec.maxC = cfg.sec_n_col
    geom.sec.ref_row = cfg.sec_ref_row
    geom.sec.ref_minC = cfg.sec_ref_minC
    geom.sec.ref_maxC = cfg.sec_ref_maxC
    prob.n_edge_len = edge_len_override if edge_len_override is not None else cfg.edge_len
    prob.sel_edge_ref = cfg.edge_ref
    prob.sel_edge_sec = 1
    prob.sel_vertex = 2
    if para.para_start_bp_ID == -1:
        para.para_start_bp_ID = 4
    if not cfg.para_scaf_seq_explicit:
        _read_seq_txt(prob)


def _prepare_output_dir(prob: ProbType, cfg, svg_output_subdir: str) -> None:
    out_base = Path(cfg.output_dir) if cfg.output_dir else (Path.cwd() / "output")
    out_dir = out_base / f"{prob.name_file}_{prob.n_edge_len}bp"
    if svg_output_subdir:
        out_dir = out_dir / svg_output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    prob.path_work = str(out_dir)


def _ensure_section_grid(geom: GeomType, cfg) -> None:
    if geom.sec.id:
        if geom.sec.types == "honeycomb":
            geom.sec.dir = 90
        return

    geom.n_sec = cfg.sec_n_row * cfg.sec_n_col
    geom.sec.id = list(range(geom.n_sec))
    geom.sec.posR = []
    geom.sec.posC = []
    for r in range(1, cfg.sec_n_row + 1):
        for c in range(1, cfg.sec_n_col + 1):
            geom.sec.posR.append(r)
            geom.sec.posC.append(c)
    geom.sec.conn = [-1 for _ in range(geom.n_sec)]
    if geom.sec.types == "honeycomb":
        geom.sec.dir = 90


def _apply_final_scaling(
    prob: ProbType,
    geom: GeomType,
    frame_mode: str,
    svg_shared_frame: SvgSharedFrame | None,
) -> None:
    if svg_shared_frame is not None:
        _apply_svg_shared_frame(prob, geom, svg_shared_frame)
    elif frame_mode == "svg-rect":
        _scale_init_geom_canvas(prob, geom, 100.0)
    else:
        _scale_init_geom(prob, geom, 100.0)
