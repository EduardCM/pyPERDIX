from __future__ import annotations

from math import cos, pi, sin
from typing import List, Tuple

from .data_geom import FaceType, GeomType, LineType, PointType
from .data_prob import ProbType
from .mani import Mani_Set_Prob


def _set_geom_from_points_faces(geom: GeomType, points: List[Tuple[float, float, float]], faces: List[List[int]]) -> None:
    geom.n_iniP = len(points)
    geom.iniP = [PointType(pos=p) for p in points]
    geom.n_face = len(faces)
    geom.face = [FaceType(n_poi=len(f), poi=list(f)) for f in faces]
    _convert_faces_to_lines(geom)


def _convert_faces_to_lines(geom: GeomType) -> None:
    edges = []
    edge_set = set()
    for face in geom.face:
        for i in range(face.n_poi):
            a = face.poi[i]
            b = face.poi[(i + 1) % face.n_poi]
            key = (min(a, b), max(a, b))
            if key in edge_set:
                continue
            edge_set.add(key)
            edges.append((a, b))

    geom.n_iniL = len(edges)
    geom.iniL = []
    for a, b in edges:
        line = LineType()
        line.poi = [a, b]
        line.iniP = [a, b]
        geom.iniL.append(line)


def _polygon_points(n: int, radius: float = 1.0, center: Tuple[float, float] = (0.0, 0.0), start_angle: float = 0.0) -> List[Tuple[float, float, float]]:
    cx, cy = center
    return [
        (cx + radius * cos(start_angle + 2.0 * pi * i / n), cy + radius * sin(start_angle + 2.0 * pi * i / n), 0.0)
        for i in range(n)
    ]


def _polygon_face(n: int) -> List[int]:
    return list(range(n))


def _grid_points(nx: int, ny: int, width: float = 1.0, height: float = 1.0) -> List[Tuple[float, float, float]]:
    return [(width * i / nx, height * j / ny, 0.0) for j in range(ny + 1) for i in range(nx + 1)]


def _grid_faces_quad(nx: int, ny: int) -> List[List[int]]:
    faces = []
    n_i = nx + 1
    for j in range(ny):
        for i in range(nx):
            a = n_i * j + i
            faces.append([a, a + 1, a + n_i + 1, a + n_i])
    return faces


def _grid_faces_tri(nx: int, ny: int) -> List[List[int]]:
    faces = []
    n_i = nx + 1
    for j in range(ny):
        for i in range(nx):
            a = n_i * j + i
            faces.append([a, a + 1, a + n_i + 1])
            faces.append([a, a + n_i + 1, a + n_i])
    return faces


def _circle_polygon(n: int, radius: float = 1.0) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    return _polygon_points(n, radius), [_polygon_face(n)]


def _quarter_circle(n: int, radius: float = 1.0) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    pts = [(0.0, 0.0, 0.0)]
    for i in range(n + 1):
        ang = 0.5 * pi * i / n
        pts.append((radius * cos(ang), radius * sin(ang), 0.0))
    return pts, [list(range(len(pts)))]


def _cross_shape(size: float = 1.0, thickness: float = 0.3) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    s = size
    t = thickness
    pts = [(-t, -s, 0.0), (t, -s, 0.0), (t, -t, 0.0), (s, -t, 0.0), (s, t, 0.0), (t, t, 0.0), (t, s, 0.0), (-t, s, 0.0), (-t, t, 0.0), (-s, t, 0.0), (-s, -t, 0.0), (-t, -t, 0.0)]
    return pts, [list(range(len(pts)))]


def _arrowhead(size: float = 1.0) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    s = size
    pts = [(-s, -s, 0.0), (0.0, -s, 0.0), (0.0, -2 * s, 0.0), (2 * s, 0.0, 0.0), (0.0, 2 * s, 0.0), (0.0, s, 0.0), (-s, s, 0.0)]
    return pts, [list(range(len(pts)))]


def _annulus(n: int, r_out: float = 1.0, r_in: float = 0.5) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    return _polygon_points(n, r_out), [_polygon_face(n)]


def _l_shape(size: float = 2.0, width: float = 1.0) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    s = size
    w = width
    return [(0.0, 0.0, 0.0), (s, 0.0, 0.0), (s, w, 0.0), (w, w, 0.0), (w, s, 0.0), (0.0, s, 0.0)], [list(range(6))]


def _star(n: int = 5, r1: float = 1.0, r2: float = 0.5) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    pts = []
    for i in range(n * 2):
        r = r1 if i % 2 == 0 else r2
        ang = pi * i / n
        pts.append((r * cos(ang), r * sin(ang), 0.0))
    return pts, [list(range(len(pts)))]


def _set_prob_name(prob: ProbType, name: str) -> None:
    prob.name_prob = name
    prob.name_file = name
    prob.type_file = "primitive"
    Mani_Set_Prob(prob, (52, 152, 219))
