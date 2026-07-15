from __future__ import annotations

from pathlib import Path

import numpy as np

from .data_geom import GeomType
from . import para


def _pos(geom: GeomType, idx: int) -> tuple[float, float, float]:
    point = geom.iniP[idx]
    return point.ori_pos if point.ori_pos != (0.0, 0.0, 0.0) else point.pos


def _min_edge_len(geom: GeomType) -> float:
    if not geom.iniL:
        return 1.0

    min_len = None
    for line in geom.iniL:
        p1 = _pos(geom, line.poi[0])
        p2 = _pos(geom, line.poi[1])
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        length = (dx * dx + dy * dy + dz * dz) ** 0.5
        min_len = length if min_len is None else min(min_len, length)

    if not min_len:
        return 1.0
    return min_len


def _get_geometry_shift(geom: GeomType, scale: float) -> tuple[float, float]:
    xs = [p[0] for p in (_pos(geom, i) for i in range(len(geom.iniP)))]
    ys = [p[1] for p in (_pos(geom, i) for i in range(len(geom.iniP)))]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_shift = (min_x + max_x) / 2.0
    if abs(max_y + min_y) < 1e-6:
        return x_shift, 0.0
    return x_shift, min_y + 5.0 / scale


def _fmt_92(value: float) -> str:
    if abs(value) < 5e-4:
        value = 0.0
    return f"{value:9.2f}"


def _scaled_pos(
    geom: GeomType,
    idx: int,
    scale: float,
    x_shift: float,
    y_shift: float,
) -> tuple[float, float, float]:
    p = _pos(geom, idx)
    return ((p[0] - x_shift) * scale, (p[1] - y_shift) * scale, p[2] * scale)


def _write_geometry_points(f, geom: GeomType, scale: float, x_shift: float, y_shift: float) -> None:
    f.write(".color red\n")
    for idx, _point in enumerate(geom.iniP):
        px, py, pz = _scaled_pos(geom, idx, scale, x_shift, y_shift)
        f.write(".sphere " f"{_fmt_92(px)}" f"{_fmt_92(py)}" f"{_fmt_92(pz)}{_fmt_92(0.75)}\n")


def _write_geometry_edges(f, geom: GeomType, scale: float, x_shift: float, y_shift: float) -> None:
    f.write(".color dark green\n")
    for line in geom.iniL:
        p1x, p1y, p1z = _scaled_pos(geom, line.poi[0], scale, x_shift, y_shift)
        p2x, p2y, p2z = _scaled_pos(geom, line.poi[1], scale, x_shift, y_shift)
        f.write(
            ".cylinder "
            f"{_fmt_92(p1x)}"
            f"{_fmt_92(p1y)}"
            f"{_fmt_92(p1z)}"
            f"{_fmt_92(p2x)}"
            f"{_fmt_92(p2y)}"
            f"{_fmt_92(p2z)}"
            f"{_fmt_92(0.3)}\n"
        )


def _write_point_labels(f, geom: GeomType, scale: float, x_shift: float, y_shift: float) -> None:
    for i, _point in enumerate(geom.iniP, start=1):
        px, py, pz = _scaled_pos(geom, i - 1, scale, x_shift, y_shift)
        f.write(".cmov " f"{_fmt_92(px + 1.0)}" f"{_fmt_92(py + 1.0)}" f"{_fmt_92(pz + 1.0)}\n")
        f.write(".color red\n")
        f.write(".font Helvetica 12 bold\n")
        f.write(f"{i:12d}\n")


def _write_edge_labels(f, geom: GeomType, scale: float, x_shift: float, y_shift: float) -> None:
    for i, line in enumerate(geom.iniL, start=1):
        p1 = np.array(_scaled_pos(geom, line.poi[0], scale, x_shift, y_shift), dtype=float)
        p2 = np.array(_scaled_pos(geom, line.poi[1], scale, x_shift, y_shift), dtype=float)
        pc = (p1 + p2) / 2.0
        length = float(np.linalg.norm(p2 - p1))
        f.write(".cmov " f"{_fmt_92(pc[0] + 0.5)}{_fmt_92(pc[1] + 0.5)}{_fmt_92(pc[2] + 0.5)}\n")
        f.write(".color dark green\n")
        f.write(".font Helvetica 12 bold\n")
        f.write(f"{i:12d}({length:5.1f})\n")


def _write_face_labels(f, geom: GeomType, scale: float, x_shift: float, y_shift: float) -> None:
    for i, face in enumerate(geom.face, start=1):
        if face.n_poi <= 0:
            continue
        pts = np.array([_scaled_pos(geom, idx, scale, x_shift, y_shift) for idx in face.poi], dtype=float)
        pc = pts.mean(axis=0)
        f.write(".cmov " f"{_fmt_92(pc[0] + 1.0)}{_fmt_92(pc[1] + 1.0)}{_fmt_92(pc[2] + 1.0)}\n")
        f.write(".color black\n")
        f.write(".font Helvetica 12 bold\n")
        f.write(f"{i:7d}\n")


def _write_geometry_info_labels(f, geom: GeomType, scale: float, x_shift: float, y_shift: float) -> None:
    _write_point_labels(f, geom, scale, x_shift, y_shift)
    _write_edge_labels(f, geom, scale, x_shift, y_shift)
    _write_face_labels(f, geom, scale, x_shift, y_shift)


def write_target_geometry_bild(geom: GeomType, path: str, target_len: float | None = None) -> None:
    """Write _01_target_geometry.bild (Input_Chimera_Init_Geom equivalent)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scale = (target_len / _min_edge_len(geom)) if target_len else 1.0
    x_shift, y_shift = _get_geometry_shift(geom, scale)

    with out_path.open("w", encoding="utf-8") as f:
        _write_geometry_points(f, geom, scale, x_shift, y_shift)
        _write_geometry_edges(f, geom, scale, x_shift, y_shift)
        if para.para_chimera_102_info:
            _write_geometry_info_labels(f, geom, scale, x_shift, y_shift)

        if para.para_chimera_axis:
            _write_axis(f)


def _write_axis(f) -> None:
    # Simple axis helper; matches Mani_Set_Chimera_Axis intent.
    f.write(".color red\n")
    f.write(".cylinder 0 0 0 10 0 0 0.1\n")
    f.write(".color green\n")
    f.write(".cylinder 0 0 0 0 10 0 0.1\n")
    f.write(".color blue\n")
    f.write(".cylinder 0 0 0 0 0 10 0.1\n")
