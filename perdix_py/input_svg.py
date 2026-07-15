from __future__ import annotations

import math
from dataclasses import dataclass

from .config import load_config
from .data_geom import GeomType, LineType
from .svg_import import import_svg_to_geom, _canonicalize_svg_geom
from .input_geom import _convert_faces_to_lines, _polygonize_lines


@dataclass
class SvgSharedFrame:
    center: tuple[float, float, float]
    scale: float


@dataclass
class SvgSharedRouteStart:
    p1: tuple[float, float, float]
    p2: tuple[float, float, float]


def _normalize_svg_to_canvas_rect(geom: GeomType) -> None:
    if geom.canvas_rect is None:
        raise ValueError(
            "frame_mode=svg-rect requires an SVG viewBox=\"min-x min-y width height\""
        )
    x0, y0, width, height = geom.canvas_rect
    if float(width) <= 1e-12 or float(height) <= 1e-12:
        raise ValueError("Invalid perdix_canvas size")
    for p in geom.iniP:
        x, y, z = p.pos
        nx = x - x0
        ny = y - y0
        p.pos = (nx, ny, z)
        p.ori_pos = (nx, ny, z)


def _scale_init_geom_canvas(prob, geom: GeomType, scale: float) -> None:
    if not geom.iniP:
        return
    prob.input_center = (0.0, 0.0, 0.0)
    prob.input_init_scale = float(scale)
    for p in geom.iniP:
        p.pos = (p.pos[0] * scale, p.pos[1] * scale, p.pos[2] * scale)


def _apply_split_triangle_svg_compatibility(geom: GeomType) -> None:
    if geom.n_iniP == 3 and geom.n_iniL == 3:
        _normalize_triangle_split(geom)
        return

    if geom.n_iniP != 4 or geom.n_iniL != 5:
        return
    _normalize_diamond_split(geom)


def _normalize_triangle_split(geom: GeomType) -> None:
    pts = [p.pos for p in geom.iniP]
    top = max(range(3), key=lambda idx: (float(pts[idx][1]), -float(pts[idx][0])))
    bottom = [idx for idx in range(3) if idx != top]
    if len(bottom) != 2:
        return
    left, right = sorted(bottom, key=lambda idx: float(pts[idx][0]))
    mapping = {top: 0, right: 1, left: 2}
    if [mapping[i] for i in range(3)] != [0, 1, 2]:
        geom.iniP = [geom.iniP[top], geom.iniP[right], geom.iniP[left]]
        for line in geom.iniL:
            line.poi = [mapping[line.poi[0]], mapping[line.poi[1]]]
            line.iniP = [mapping[line.iniP[0]], mapping[line.iniP[1]]]
    _reorder_edges(geom, [[2, 1], [1, 0], [0, 2]])


def _normalize_diamond_split(geom: GeomType) -> None:
    degree = [0 for _ in range(geom.n_iniP)]
    for line in geom.iniL:
        a, b = line.poi
        if a == b:
            return
        degree[a] += 1
        degree[b] += 1
    if sorted(degree) != [2, 2, 3, 3]:
        return

    ys = [float(p.pos[1]) for p in geom.iniP]
    base_y = max(set(ys), key=ys.count)
    base_pts = [idx for idx, y in enumerate(ys) if abs(y - base_y) < 1e-6]
    if len(base_pts) != 3:
        return
    apex_pts = [idx for idx in range(geom.n_iniP) if idx not in base_pts]
    center_candidates = [idx for idx in base_pts if degree[idx] == 3]
    if len(apex_pts) != 1 or len(center_candidates) != 1:
        return
    apex = apex_pts[0]
    center = center_candidates[0]
    side_pts = [idx for idx in base_pts if idx != center]
    if len(side_pts) != 2:
        return
    left, right = sorted(side_pts, key=lambda idx: float(geom.iniP[idx].pos[0]))
    _reposition_diamond_points(geom, apex, center, left, right, base_y)
    _reorder_edges(geom, [[center, right], [right, apex], [apex, center], [apex, left], [left, center]])


def _reposition_diamond_points(geom: GeomType, apex: int, center: int, left: int, right: int, base_y: float) -> None:
    left_x = float(geom.iniP[left].pos[0])
    right_x = float(geom.iniP[right].pos[0])
    mid_x = (left_x + right_x) / 2.0
    half_span = ((mid_x - left_x) + (right_x - mid_x)) / 2.0
    if half_span <= 1e-12:
        return
    z = float(geom.iniP[0].pos[2])
    apex_y = base_y - half_span if float(geom.iniP[apex].pos[1]) < base_y else base_y + half_span
    for idx, pos in (
        (apex, (mid_x, apex_y, z)),
        (center, (mid_x, base_y, z)),
        (left, (mid_x - half_span, base_y, z)),
        (right, (mid_x + half_span, base_y, z)),
    ):
        geom.iniP[idx].pos = pos
        geom.iniP[idx].ori_pos = pos


def _reorder_edges(geom: GeomType, desired: list[list[int]]) -> None:
    edge_map: dict[tuple[int, int], LineType] = {}
    for line in geom.iniL:
        a, b = line.poi
        edge_map[(a, b) if a < b else (b, a)] = line
    ordered: list[LineType] = []
    for idx, (a, b) in enumerate(desired):
        key = (a, b) if a < b else (b, a)
        if key not in edge_map:
            return
        line = edge_map[key]
        line.poi = [a, b]
        line.iniP = [a, b]
        line.iniL = idx
        ordered.append(line)
    geom.iniL = ordered


def _apply_svg_shared_frame(prob, geom: GeomType, frame: SvgSharedFrame) -> None:
    prob.input_center = frame.center
    prob.input_init_scale = float(frame.scale)
    cx, cy, cz = frame.center
    for p in geom.iniP:
        p.pos = (
            (p.pos[0] - cx) * frame.scale,
            (p.pos[1] - cy) * frame.scale,
            (p.pos[2] - cz) * frame.scale,
        )


def _edge_points(geom: GeomType, line: LineType) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return geom.iniP[line.poi[0]].pos, geom.iniP[line.poi[1]].pos


def _edge_key_from_points(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    tol: float = 1e-6,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    def q(p: tuple[float, float, float]) -> tuple[int, int, int]:
        return (
            int(round(float(p[0]) / tol)),
            int(round(float(p[1]) / tol)),
            int(round(float(p[2]) / tol)),
        )

    a = q(p1)
    b = q(p2)
    return (a, b) if a <= b else (b, a)


def _canonical_edge_orientation(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (p1, p2) if p1 <= p2 else (p2, p1)


def _apply_svg_shared_route_start(geom: GeomType, route_start: SvgSharedRouteStart) -> None:
    target_key = _edge_key_from_points(route_start.p1, route_start.p2, tol=_ROUTE_START_TOL)
    oriented_start = _canonical_edge_orientation(route_start.p1, route_start.p2)
    reordered: list[LineType] = []
    start_line: LineType | None = None

    for line in geom.iniL:
        p1, p2 = _edge_points(geom, line)
        if _edge_key_from_points(p1, p2, tol=_ROUTE_START_TOL) != target_key:
            reordered.append(line)
            continue
        line_p1, line_p2 = _canonical_edge_orientation(p1, p2)
        if line_p1 != oriented_start[0] or line_p2 != oriented_start[1]:
            line.poi = [line.poi[1], line.poi[0]]
            line.iniP = [line.iniP[1], line.iniP[0]]
        start_line = line

    if start_line is None:
        raise ValueError("Shared boundary route start was not found in layer geometry")

    geom.iniL = [start_line] + reordered
    geom.n_iniL = len(geom.iniL)
    for idx, line in enumerate(geom.iniL):
        line.iniL = idx


def compute_svg_shared_frame(
    svg_path: str,
    config_path: str | None = None,
    frame_mode: str = "legacy",
    svg_layers: list[int] | None = None,
) -> SvgSharedFrame:
    cfg = load_config(config_path)
    layers = svg_layers or [None]
    geoms: list[GeomType] = []
    for layer in layers:
        geom = import_svg_to_geom(str(svg_path), scale=cfg.svg_scale, layer_index=layer)
        if frame_mode == "svg-rect":
            _normalize_svg_to_canvas_rect(geom)
        geoms.append(geom)

    if frame_mode == "svg-rect":
        return SvgSharedFrame(center=(0.0, 0.0, 0.0), scale=100.0)

    all_points = [p.pos for geom in geoms for p in geom.iniP]
    if not all_points:
        return SvgSharedFrame(center=(0.0, 0.0, 0.0), scale=1.0)

    cx = sum(p[0] for p in all_points) / float(len(all_points))
    cy = sum(p[1] for p in all_points) / float(len(all_points))
    cz = sum(p[2] for p in all_points) / float(len(all_points))

    min_len: float | None = None
    for geom in geoms:
        for line in geom.iniL:
            p1 = geom.iniP[line.poi[0]].pos
            p2 = geom.iniP[line.poi[1]].pos
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dz = p1[2] - p2[2]
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            if length <= 1e-12:
                continue
            if min_len is None or length < min_len:
                min_len = length
    scale = 1.0 if min_len is None else 100.0 / min_len
    return SvgSharedFrame(center=(float(cx), float(cy), float(cz)), scale=float(scale))


def _prepare_svg_geom_for_layer_analysis(svg_path: str, cfg, frame_mode: str, layer: int | None) -> GeomType:
    geom = import_svg_to_geom(str(svg_path), scale=cfg.svg_scale, layer_index=layer)
    if frame_mode == "svg-rect":
        _normalize_svg_to_canvas_rect(geom)
    if geom.n_face == 0:
        _polygonize_lines(geom)
    _convert_faces_to_lines(geom)
    if layer is not None:
        _canonicalize_svg_geom(geom)
        if cfg.svg_layer_snap_compat:
            _apply_split_triangle_svg_compatibility(geom)
    return geom


def compute_svg_shared_route_start(
    svg_path: str,
    config_path: str | None = None,
    frame_mode: str = "legacy",
    svg_layers: list[int] | None = None,
) -> SvgSharedRouteStart | None:
    cfg = load_config(config_path)
    layers = svg_layers or [None]
    geoms = [_prepare_svg_geom_for_layer_analysis(svg_path, cfg, frame_mode, layer) for layer in layers]
    if not geoms:
        return None

    shared_keys: set[tuple[tuple[int, int, int], tuple[int, int, int]]] | None = None
    edge_points: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        tuple[tuple[float, float, float], tuple[float, float, float]],
    ] = {}
    for geom in geoms:
        layer_keys: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
        for line in geom.iniL:
            p1, p2 = _edge_points(geom, line)
            key = _edge_key_from_points(p1, p2)
            layer_keys.add(key)
            edge_points.setdefault(key, _canonical_edge_orientation(p1, p2))
        shared_keys = layer_keys if shared_keys is None else (shared_keys & layer_keys)

    if not shared_keys:
        return None

    all_points: list[tuple[float, float, float]] = [p.pos for geom in geoms for p in geom.iniP]
    min_x = min(float(p[0]) for p in all_points)
    max_x = max(float(p[0]) for p in all_points)
    min_y = min(float(p[1]) for p in all_points)
    max_y = max(float(p[1]) for p in all_points)
    tol = 1e-6

    def on_bbox(mid: tuple[float, float, float]) -> bool:
        return (
            abs(float(mid[0]) - min_x) <= tol
            or abs(float(mid[0]) - max_x) <= tol
            or abs(float(mid[1]) - min_y) <= tol
            or abs(float(mid[1]) - max_y) <= tol
        )

    boundary_keys = []
    for key in shared_keys:
        p1, p2 = edge_points[key]
        mid = (
            (float(p1[0]) + float(p2[0])) / 2.0,
            (float(p1[1]) + float(p2[1])) / 2.0,
            (float(p1[2]) + float(p2[2])) / 2.0,
        )
        if on_bbox(mid):
            boundary_keys.append(key)

    candidates = boundary_keys if boundary_keys else list(shared_keys)

    def candidate_key(key):
        p1, p2 = edge_points[key]
        p1, p2 = _canonical_edge_orientation(p1, p2)
        mid_x = (float(p1[0]) + float(p2[0])) / 2.0
        mid_y = (float(p1[1]) + float(p2[1])) / 2.0
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        dz = float(p2[2]) - float(p1[2])
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        return (round(mid_x, 12), round(mid_y, 12), round(length, 12), p1, p2)

    chosen = min(candidates, key=candidate_key)
    p1, p2 = _canonical_edge_orientation(*edge_points[chosen])
    return SvgSharedRouteStart(p1=p1, p2=p2)


_ROUTE_START_TOL = 1e-6
