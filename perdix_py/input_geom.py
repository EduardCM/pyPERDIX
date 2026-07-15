from __future__ import annotations

from pathlib import Path

from . import para
from .data_geom import GeomType
from .data_prob import ProbType
from .resource_utils import read_packaged_text


def _scale_init_geom(prob: ProbType, geom: GeomType, scale: float) -> None:
    if not geom.iniP or not geom.iniL:
        return
    cx = sum(p.pos[0] for p in geom.iniP) / float(geom.n_iniP)
    cy = sum(p.pos[1] for p in geom.iniP) / float(geom.n_iniP)
    cz = sum(p.pos[2] for p in geom.iniP) / float(geom.n_iniP)
    prob.input_center = (float(cx), float(cy), float(cz))
    for p in geom.iniP:
        p.pos = (p.pos[0] - cx, p.pos[1] - cy, p.pos[2] - cz)

    def edge_len(line) -> float:
        p1 = geom.iniP[line.poi[0]].pos
        p2 = geom.iniP[line.poi[1]].pos
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    min_len = edge_len(geom.iniL[0])
    for line in geom.iniL[1:]:
        length = edge_len(line)
        if length < min_len:
            min_len = length

    if min_len == 0:
        return

    factor = scale / min_len
    prob.input_init_scale = float(factor)
    for p in geom.iniP:
        p.pos = (p.pos[0] * factor, p.pos[1] * factor, p.pos[2] * factor)


def _round_geom_points(geom: GeomType, digits: int = 4) -> None:
    for p in geom.iniP:
        p.pos = tuple(round(float(v), digits) for v in p.pos)
        p.ori_pos = tuple(round(float(v), digits) for v in p.ori_pos)


def _convert_faces_to_lines(geom: GeomType) -> None:
    edge_set: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []

    for face in geom.face:
        n = face.n_poi
        for j in range(n):
            point_1 = face.poi[j]
            point_2 = face.poi[0] if j == n - 1 else face.poi[j + 1]
            key = (point_1, point_2) if point_1 < point_2 else (point_2, point_1)
            if key in edge_set:
                continue
            edge_set.add(key)
            edges.insert(0, (point_1, point_2))

    from .data_geom import LineType

    geom.n_iniL = len(edges)
    geom.iniL = []
    for a, b in reversed(edges):
        line = LineType()
        line.poi = [a, b]
        line.iniP = [a, b]
        geom.iniL.append(line)


def _polygonize_lines(geom: GeomType) -> None:
    try:
        from shapely.geometry import MultiLineString, MultiPolygon
        from shapely.ops import polygonize_full, unary_union
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Shapely is required to polygonize SVG line input") from exc

    if not geom.iniL:
        raise ValueError("No lines available to polygonize")

    linepoints = []
    for line in geom.iniL:
        p1 = geom.iniP[line.poi[0]].pos
        p2 = geom.iniP[line.poi[1]].pos
        linepoints.append(((p1[0], -p1[1]), (p2[0], -p2[1])))

    multilines = MultiLineString(linepoints)
    inter = multilines.intersection(multilines)
    result, _dangles, _cuts, _invalids = polygonize_full(inter)
    result = MultiPolygon(result)
    polygon = unary_union(result)

    multilines = polygon.boundary.union(result.boundary)
    result, _dangles, _cuts, _invalids = polygonize_full(multilines)
    polygon = MultiPolygon(result)
    polygons = list(polygon.geoms)
    if not polygons:
        raise ValueError("Failed to polygonize SVG line input")

    points: list[tuple[float, float]] = []
    for poly in polygons:
        for coord in list(poly.exterior.coords):
            pt = (coord[0], coord[1])
            if pt not in points:
                points.append(pt)

    point_index = {pt: idx for idx, pt in enumerate(points)}
    conns: list[list[int]] = []
    for poly in polygons:
        coords = list(poly.exterior.coords)
        face = [point_index[(coords[j][0], coords[j][1])] for j in range(len(coords) - 1)]
        conns.append(list(reversed(face)))

    from .data_geom import FaceType, PointType

    geom.n_iniP = len(points)
    geom.iniP = [PointType(pos=(x, y, 0.0), ori_pos=(x, y, 0.0)) for x, y in points]
    geom.n_face = len(conns)
    geom.face = [FaceType(n_poi=len(ids), poi=ids) for ids in conns]
    geom.n_iniL = 0
    geom.iniL = []


def _read_seq_txt(prob: ProbType) -> None:
    seq_path = Path("seq.txt")
    if seq_path.exists():
        lines = seq_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = read_packaged_text("seq.txt").splitlines()
    if not lines:
        para.para_scaf_seq = "m13"
        return

    para.para_scaf_seq = lines[0].strip()
    if para.para_scaf_seq not in ("m13", "user", "rand"):
        raise ValueError("Please check the field in the seq.txt file.")

    if para.para_scaf_seq == "user":
        if len(lines) < 2:
            raise ValueError("Please check file format in seq.txt.")
        prob.scaf_seq = lines[1].strip().upper()


def _merge_colinear_edges(edges: list[tuple[int, int]], points: list) -> list[tuple[int, int]]:
    import numpy as np

    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def is_colinear(p, a, b) -> bool:
        v1 = np.array(points[a].pos) - np.array(points[p].pos)
        v2 = np.array(points[b].pos) - np.array(points[p].pos)
        if np.linalg.norm(v1) < 1e-8 or np.linalg.norm(v2) < 1e-8:
            return False
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return abs(abs(np.dot(v1, v2)) - 1.0) < 1e-6

    changed = True
    while changed:
        changed = False
        for p in list(adj.keys()):
            if p not in adj or len(adj[p]) != 2:
                continue
            a, b = list(adj[p])
            if not is_colinear(p, a, b):
                continue
            adj[a].discard(p)
            adj[b].discard(p)
            adj[p].clear()
            adj[a].add(b)
            adj[b].add(a)
            changed = True
        for p in [k for k, v in adj.items() if len(v) == 0]:
            adj.pop(p, None)

    new_edges = []
    seen = set()
    for a, nbrs in adj.items():
        for b in nbrs:
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            new_edges.append((a, b))
    return new_edges


def _set_section_connectivity(prob: ProbType, geom: GeomType) -> None:
    from .section import section_connection_scaf

    geom.sec.conn = [-1 for _ in range(geom.n_sec)]
    for i in range(geom.n_sec):
        sec_cur = geom.sec.id[i]
        row_cur = geom.sec.posR[i]
        for j in range(geom.n_sec):
            sec_com = geom.sec.id[j]
            row_com = geom.sec.posR[j]
            if sec_cur == sec_com:
                continue
            b_connect = section_connection_scaf(geom, sec_cur, sec_com, 1)
            if (
                (para.para_vertex_design == "flat" and b_connect)
                or (
                    para.para_vertex_design == "mitered"
                    and para.para_vertex_crash == "mod1"
                    and row_cur < geom.sec.ref_row
                    and row_cur == row_com
                )
            ):
                geom.sec.conn[i] = sec_com
                break

    count = sum(1 for sec in geom.sec.conn if sec == -1)
    if count == 0 or count % 2 != 0:
        raise ValueError("The section connect was wrong.")
