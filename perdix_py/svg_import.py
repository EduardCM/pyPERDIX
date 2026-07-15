from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .data_geom import FaceType, GeomType, LineType, PointType


_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _parse_points_attr(points: str) -> List[Tuple[float, float]]:
    nums = [float(x) for x in _FLOAT_RE.findall(points)]
    if len(nums) % 2 != 0:
        raise ValueError("Invalid points attribute; expected even number of values")
    return list(zip(nums[0::2], nums[1::2]))


def _tokenize_path(d: str) -> List[str]:
    # Split into commands and numbers
    tokens: List[str] = []
    i = 0
    while i < len(d):
        c = d[i]
        if c.isalpha():
            tokens.append(c)
            i += 1
            continue
        if c.isspace() or c == ",":
            i += 1
            continue
        m = _FLOAT_RE.match(d, i)
        if not m:
            raise ValueError(f"Unexpected path data at position {i}")
        tokens.append(m.group(0))
        i = m.end()
    return tokens


def _parse_path(d: str) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    # Minimal path parser: M/m, L/l, H/h, V/v, Z/z
    tokens = _tokenize_path(d)
    segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    i = 0
    cmd = None
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in ("Z", "z"):
                if cur != start:
                    segs.append((cur, start))
                cur = start
                cmd = None
            continue
        if cmd is None:
            raise ValueError("Path data missing command")
        if cmd in ("M", "m"):
            x = float(tokens[i]); y = float(tokens[i + 1]); i += 2
            if cmd == "m":
                cur = (cur[0] + x, cur[1] + y)
            else:
                cur = (x, y)
            start = cur
            cmd = "L" if cmd == "M" else "l"
        elif cmd in ("L", "l"):
            x = float(tokens[i]); y = float(tokens[i + 1]); i += 2
            nxt = (x + cur[0], y + cur[1]) if cmd == "l" else (x, y)
            segs.append((cur, nxt))
            cur = nxt
        elif cmd in ("H", "h"):
            x = float(tokens[i]); i += 1
            nxt = (x + cur[0], cur[1]) if cmd == "h" else (x, cur[1])
            segs.append((cur, nxt))
            cur = nxt
        elif cmd in ("V", "v"):
            y = float(tokens[i]); i += 1
            nxt = (cur[0], y + cur[1]) if cmd == "v" else (cur[0], y)
            segs.append((cur, nxt))
            cur = nxt
        else:
            raise ValueError(f"Unsupported path command: {cmd}")
    return segs


def _dedup_point(points: Dict[Tuple[int, int], int], p: Tuple[float, float], tol: float) -> int:
    key = (int(round(p[0] / tol)), int(round(p[1] / tol)))
    if key in points:
        return points[key]
    idx = len(points)
    points[key] = idx
    return idx


def _parse_viewbox(root: ET.Element, scale: float) -> tuple[float, float, float, float] | None:
    vb = root.get("viewBox")
    if not vb:
        return None
    nums = [float(x) for x in _FLOAT_RE.findall(vb)]
    if len(nums) != 4:
        raise ValueError("Invalid SVG viewBox; expected 4 numbers: min-x min-y width height")
    x, y, w, h = nums
    if w <= 0.0 or h <= 0.0:
        raise ValueError("SVG viewBox width/height must be positive")
    return (x * scale, y * scale, w * scale, h * scale)


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1]


def _find_layer_group(root: ET.Element, layer_index: int) -> ET.Element:
    target_id = f"Layer_{layer_index}"
    for el in root.iter():
        if _strip_ns(el.tag) != "g":
            continue
        if el.get("id") == target_id:
            return el
    raise ValueError(f"SVG layer group not found: {target_id}")


def list_svg_layers(svg_path: str) -> list[int]:
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    layers: list[int] = []
    for el in root.iter():
        if _strip_ns(el.tag) != "g":
            continue
        layer_id = el.get("id", "")
        if not layer_id.startswith("Layer_"):
            continue
        try:
            layers.append(int(layer_id.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(set(layers))


def _canonicalize_svg_geom(geom: GeomType) -> None:
    if not geom.iniP:
        return

    order = sorted(
        range(len(geom.iniP)),
        key=lambda idx: (
            -float(geom.iniP[idx].pos[1]),
            -float(geom.iniP[idx].pos[0]),
            float(geom.iniP[idx].pos[2]),
            idx,
        ),
    )
    if order != list(range(len(geom.iniP))):
        mapping = {old: new for new, old in enumerate(order)}
        geom.iniP = [geom.iniP[i] for i in order]
        for line in geom.iniL:
            line.poi = [mapping[line.poi[0]], mapping[line.poi[1]]]
            line.iniP = [mapping[line.iniP[0]], mapping[line.iniP[1]]]
        for face in geom.face:
            face.poi = [mapping[idx] for idx in face.poi]

    cx = sum(float(p.pos[0]) for p in geom.iniP) / float(len(geom.iniP))
    cy = sum(float(p.pos[1]) for p in geom.iniP) / float(len(geom.iniP))
    start_angle = -math.pi / 4.0

    def line_key(line: LineType) -> tuple[float, float, float, float]:
        p1 = geom.iniP[line.poi[0]].pos
        p2 = geom.iniP[line.poi[1]].pos
        mx = (float(p1[0]) + float(p2[0])) / 2.0
        my = (float(p1[1]) + float(p2[1])) / 2.0
        angle = (math.atan2(my - cy, mx - cx) - start_angle) % (2.0 * math.pi)
        return (
            round(angle, 12),
            -round(mx, 12),
            round(my, 12),
            min(line.poi),
            max(line.poi),
        )

    geom.iniL.sort(key=line_key)
    for idx, line in enumerate(geom.iniL):
        line.iniL = idx


def _segments_from_line(
    el: ET.Element,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[list[tuple[float, float]]]]:
    x1 = float(el.get("x1", "0"))
    y1 = float(el.get("y1", "0"))
    x2 = float(el.get("x2", "0"))
    y2 = float(el.get("y2", "0"))
    return [((x1, y1), (x2, y2))], []


def _segments_from_polyline(
    el: ET.Element,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[list[tuple[float, float]]]]:
    pts = _parse_points_attr(el.get("points", ""))
    return list(zip(pts, pts[1:])), []


def _segments_from_polygon(
    el: ET.Element,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[list[tuple[float, float]]]]:
    pts = _parse_points_attr(el.get("points", ""))
    if len(pts) < 2:
        return [], []
    return list(zip(pts, pts[1:] + pts[:1])), [pts]


def _segments_from_path_element(
    el: ET.Element,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[list[tuple[float, float]]]]:
    d = el.get("d")
    if not d:
        return [], []
    return _parse_path(d), []


_SVG_ELEMENT_HANDLERS = {
    "line": _segments_from_line,
    "path": _segments_from_path_element,
    "polygon": _segments_from_polygon,
    "polyline": _segments_from_polyline,
}


def _collect_svg_geometry(
    iter_root: ET.Element,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[list[tuple[float, float]]]]:
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    faces: List[List[Tuple[float, float]]] = []
    for el in iter_root.iter():
        handler = _SVG_ELEMENT_HANDLERS.get(_strip_ns(el.tag))
        if handler is None:
            continue
        new_segments, new_faces = handler(el)
        segments.extend(new_segments)
        faces.extend(new_faces)
    return segments, faces


def _point_on_segment(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    tol: float,
) -> bool:
    ax, ay = a
    bx, by = b
    px, py = p
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    cross = abs(abx * apy - aby * apx)
    if cross > tol:
        return False
    dot = apx * abx + apy * aby
    if dot < -tol:
        return False
    ab_len2 = abx * abx + aby * aby
    return dot - ab_len2 <= tol


def _split_lines_at_points(
    lines: List[Tuple[int, int]],
    point_coords: Dict[int, Tuple[float, float]],
    tol: float,
) -> List[Tuple[int, int]]:
    new_lines: List[Tuple[int, int]] = []
    for ia, ib in lines:
        a = point_coords[ia]
        b = point_coords[ib]
        abx = b[0] - a[0]
        aby = b[1] - a[1]
        denom = abx * abx + aby * aby
        pts = []
        for idx, p in point_coords.items():
            if _point_on_segment(p, a, b, tol):
                t = 0.0 if denom == 0.0 else ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / denom
                pts.append((t, idx))
        pts.sort(key=lambda item: item[0])
        for (_, p1), (_, p2) in zip(pts, pts[1:]):
            if p1 != p2:
                new_lines.append((p1, p2))
    return new_lines


@dataclass(frozen=True)
class _GeomBuildInputs:
    point_map: Dict[Tuple[int, int], int]
    point_coords: Dict[int, Tuple[float, float]]
    lines: List[Tuple[int, int]]
    faces: List[List[Tuple[float, float]]]
    scale: float
    tol: float
    canvas_rect: tuple[float, float, float, float] | None


def _build_geom(inputs: _GeomBuildInputs) -> GeomType:
    geom = GeomType()
    geom.n_iniP = len(inputs.point_coords)
    geom.n_iniL = len(inputs.lines)
    geom.n_face = len(inputs.faces)

    geom.iniP = [
        PointType(pos=(x * inputs.scale, y * inputs.scale, 0.0), ori_pos=(x * inputs.scale, y * inputs.scale, 0.0))
        for x, y in (inputs.point_coords[idx] for idx in range(geom.n_iniP))
    ]
    geom.iniL = [LineType(iniL=i, poi=[a, b], iniP=[a, b]) for i, (a, b) in enumerate(inputs.lines)]
    if inputs.faces:
        geom.face = [
            FaceType(n_poi=len(ids), poi=ids)
            for ids in ([_dedup_point(inputs.point_map, p, inputs.tol) for p in face_pts] for face_pts in inputs.faces)
        ]
    else:
        geom.face = []
    geom.canvas_rect = inputs.canvas_rect
    return geom


def import_svg_to_geom(
    svg_path: str,
    scale: float = 100.0,
    tol: float = 1e-6,
    layer_index: int | None = None,
) -> GeomType:
    """Parse SVG and build GeomType with points, lines, and faces.

    Supported elements: line, polyline, polygon, path (M/L/H/V/Z only).
    Units are taken as-is; a uniform scale is applied after parsing.
    """
    svg_path = str(svg_path)
    tree = ET.parse(svg_path)
    root = tree.getroot()

    canvas_rect: tuple[float, float, float, float] | None = _parse_viewbox(root, scale)
    iter_root = _find_layer_group(root, layer_index) if layer_index is not None else root
    segments, faces = _collect_svg_geometry(iter_root)

    if not segments:
        raise ValueError("No supported geometry found in SVG")

    # Deduplicate points with tolerance
    point_map: Dict[Tuple[int, int], int] = {}
    point_coords: Dict[int, Tuple[float, float]] = {}
    lines: List[Tuple[int, int]] = []

    for a, b in segments:
        ia = _dedup_point(point_map, a, tol)
        ib = _dedup_point(point_map, b, tol)
        point_coords[ia] = a
        point_coords[ib] = b
        if ia != ib:
            lines.append((ia, ib))

    if lines:
        lines = _split_lines_at_points(lines, point_coords, tol)

    return _build_geom(_GeomBuildInputs(point_map, point_coords, lines, faces, scale, tol, canvas_rect))
