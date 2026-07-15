from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .data_prob import ProbType
from .data_geom import GeomType, LineType, PointType
from .math_utils import normalize
from . import para


@dataclass(frozen=True)
class _SectionPositionArgs:
    r: float
    s: float
    t: float
    num: int
    n_row: int
    n_col: int


def section_generation(prob: ProbType, geom: GeomType) -> None:
    """Generate cross-sectional geometry (square lattice only for now)."""
    _set_sectional_data(geom)
    if geom.sec.types == "square":
        _generate_square(geom)
    elif geom.sec.types == "honeycomb":
        _generate_honeycomb(geom)
    else:
        raise NotImplementedError("Honeycomb lattice not yet ported")
    _reset_local_coordinate(geom)
    _round_section_positions(geom, digits=4)


def _round_section_positions(geom: GeomType, digits: int = 4) -> None:
    for p in geom.croP:
        p.pos = tuple(round(float(v), digits) for v in p.pos)
        p.ori_pos = tuple(round(float(v), digits) for v in p.ori_pos)


def section_connection_scaf(geom: GeomType, sec_cur: int, sec_com: int, bp_id: int) -> bool:
    """Determine section connection for scaffold (square lattice only)."""
    if geom.sec.types == "square":
        bp = bp_id + para.para_start_bp_ID - 1
        if bp < 0:
            bp = 32 + bp
        bp = bp % 32
    elif geom.sec.types == "honeycomb":
        bp = bp_id + para.para_start_bp_ID - 1
        if bp < 0:
            bp = 21 + bp
        bp = bp % 21
    else:
        raise NotImplementedError("Unsupported section type")

    row_cur = geom.sec.posR[sec_cur]
    col_cur = geom.sec.posC[sec_cur]
    row_com = geom.sec.posR[sec_com]
    col_com = geom.sec.posC[sec_com]

    if geom.sec.types == "square":
        if row_cur == row_com and col_cur == col_com - 1:
            if sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (4, 5, 15, 16, 26, 27)
            if sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (0, 10, 11, 20, 21, 31)
        elif row_cur == row_com and col_cur == col_com + 1:
            if sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (0, 10, 11, 20, 21, 31)
            if sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (4, 5, 15, 16, 26, 27)
        elif row_cur == row_com + 1 and col_cur == col_com:
            if sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (7, 8, 18, 19, 28, 29)
            if sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (2, 3, 12, 13, 23, 24)
        elif row_cur == row_com - 1 and col_cur == col_com:
            if sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (2, 3, 12, 13, 23, 24)
            if sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (7, 8, 18, 19, 28, 29)
    else:
        # honeycomb
        if row_cur == row_com and col_cur == col_com - 1:
            if geom.sec.dir == -90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (8, 9, 18, 19)
            if geom.sec.dir == 90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (8, 9, 18, 19)
            if geom.sec.dir == 150 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (1, 2, 11, 12)
        elif row_cur == row_com and col_cur == col_com + 1:
            if geom.sec.dir == -90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (8, 9, 18, 19)
            if geom.sec.dir == 90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (8, 9, 18, 19)
            if geom.sec.dir == 150 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (1, 2, 11, 12)
        elif row_cur == row_com + 1 and col_cur == col_com:
            if geom.sec.dir == -90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (4, 5, 15, 16)
            if geom.sec.dir == 90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (4, 5, 15, 16)
            if geom.sec.dir == 150 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (4, 5, 15, 16)
            if geom.sec.dir == -90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (1, 2, 11, 12)
            if geom.sec.dir == 90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (1, 2, 11, 12)
            if geom.sec.dir == 150 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (1, 2, 11, 12)
        elif row_cur == row_com - 1 and col_cur == col_com:
            if geom.sec.dir == -90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (1, 2, 11, 12)
            if geom.sec.dir == 90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (1, 2, 11, 12)
            if geom.sec.dir == 150 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (4, 5, 15, 16)
            if geom.sec.dir == -90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (4, 5, 15, 16)
            if geom.sec.dir == 90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (4, 5, 15, 16)
            if geom.sec.dir == 150 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (4, 5, 15, 16)
    return False


def section_connection_stap(geom: GeomType, sec_cur: int, sec_com: int, bp_id: int) -> bool:
    """Determine section connection for staple (square lattice only)."""
    if geom.sec.types == "square":
        bp = bp_id + para.para_start_bp_ID - 1
        if bp < 0:
            bp = 32 + bp
        bp = bp % 32
    elif geom.sec.types == "honeycomb":
        bp = bp_id + para.para_start_bp_ID - 1
        if bp < 0:
            bp = 21 + bp
        bp = bp % 21
    else:
        raise NotImplementedError("Unsupported section type")

    row_cur = geom.sec.posR[sec_cur]
    col_cur = geom.sec.posC[sec_cur]
    row_com = geom.sec.posR[sec_com]
    col_com = geom.sec.posC[sec_com]

    if geom.sec.types == "square":
        if row_cur == row_com and col_cur == col_com - 1:
            if sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (0, 31)
            if sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (15, 16)
        elif row_cur == row_com and col_cur == col_com + 1:
            if sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (15, 16)
            if sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (0, 31)
        elif row_cur == row_com + 1 and col_cur == col_com:
            if sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (23, 24)
            if sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (7, 8)
        elif row_cur == row_com - 1 and col_cur == col_com:
            if sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (7, 8)
            if sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (23, 24)
    else:
        # honeycomb
        if row_cur == row_com and col_cur == col_com - 1:
            if geom.sec.dir == -90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (13, 14)
            if geom.sec.dir == 90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (13, 14)
            if geom.sec.dir == 150 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (6, 7)
        elif row_cur == row_com and col_cur == col_com + 1:
            if geom.sec.dir == -90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (13, 14)
            if geom.sec.dir == 90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (13, 14)
            if geom.sec.dir == 150 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (6, 7)
        elif row_cur == row_com + 1 and col_cur == col_com:
            if geom.sec.dir == -90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (0, 20)
            if geom.sec.dir == 90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (0, 20)
            if geom.sec.dir == 150 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (0, 20)
            if geom.sec.dir == -90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (6, 7)
            if geom.sec.dir == 90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (6, 7)
            if geom.sec.dir == 150 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (13, 14)
        elif row_cur == row_com - 1 and col_cur == col_com:
            if geom.sec.dir == -90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (6, 7)
            if geom.sec.dir == 90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (6, 7)
            if geom.sec.dir == 150 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (13, 14)
            if geom.sec.dir == -90 and sec_cur % 2 == 1 and sec_com % 2 == 0:
                return bp in (0, 20)
            if geom.sec.dir == 90 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (0, 20)
            if geom.sec.dir == 150 and sec_cur % 2 == 0 and sec_com % 2 == 1:
                return bp in (0, 20)
    return False


def _set_sectional_data(geom: GeomType) -> None:
    geom.n_croP = geom.n_sec * geom.n_modP
    geom.n_croL = geom.n_sec * geom.n_iniL
    geom.croP = [PointType() for _ in range(geom.n_croP)]
    geom.croL = [LineType() for _ in range(geom.n_croL)]

    for i in range(geom.n_sec):
        offset = i * geom.n_modP
        for j in range(geom.n_modP):
            geom.croP[offset + j].pos = geom.modP[j].pos
            for m in range(geom.n_junc):
                for n in range(geom.junc[m].n_arm):
                    if geom.junc[m].modP[n] == j:
                        geom.junc[m].croP[n][i] = offset + j
        for j in range(geom.n_iniL):
            line = geom.croL[i * geom.n_iniL + j]
            line.poi = [offset + geom.iniL[j].poi[0], offset + geom.iniL[j].poi[1]]
            line.sec = geom.sec.id[i]
            line.iniL = j
            line.t = list(geom.iniL[j].t)


def _generate_square(geom: GeomType) -> None:
    for i in range(geom.n_sec):
        for j in range(geom.n_iniL):
            pos_row = geom.sec.posR[i] - geom.sec.minR + 1
            pos_col = geom.sec.posC[i] - geom.sec.minC + 1
            r = 1.0
            s = 2.0 * float(pos_col - 1) / float(max(geom.sec.n_col - 1, 1)) - 1.0
            t = 2.0 * float(pos_row - 1) / float(max(geom.sec.n_row - 1, 1)) - 1.0
            t_mod = 2.0 * float(geom.sec.ref_row - 1) / float(max(geom.sec.n_row - 1, 1)) - 1.0
            t = t - t_mod
            if geom.sec.n_row == 1:
                t = 0.0

            pos = _get_position(
                geom,
                _SectionPositionArgs(-r, s, t, j, geom.sec.n_row, geom.sec.n_col),
            )
            node = geom.iniL[j].poi[0] + i * geom.n_modP
            geom.croP[node].pos = pos

            pos = _get_position(
                geom,
                _SectionPositionArgs(r, s, t, j, geom.sec.n_row, geom.sec.n_col),
            )
            node = geom.iniL[j].poi[1] + i * geom.n_modP
            geom.croP[node].pos = pos


def _generate_honeycomb(geom: GeomType) -> None:
    for i in range(geom.n_sec):
        for j in range(geom.n_iniL):
            pos_row = geom.sec.posR[i] - geom.sec.minR + 1
            pos_col = geom.sec.posC[i] - geom.sec.minC + 1
            r = 1.0

            if (
                (geom.sec.posC[i] % 2 == 0 and geom.sec.posR[i] % 2 == 0)
                or (geom.sec.posC[i] % 2 != 0 and geom.sec.posR[i] % 2 != 0)
            ):
                s = 2.0 * float(3 * pos_col - 1) / float(3 * geom.sec.n_col) - 1.0
            else:
                s = 2.0 * float(3 * pos_col - 2) / float(3 * geom.sec.n_col) - 1.0

            t = 2.0 * float(pos_row - 1) / float(max(geom.sec.n_row - 1, 1)) - 1.0
            t_mod = 2.0 * float(geom.sec.ref_row - 1) / float(max(geom.sec.n_row - 1, 1)) - 1.0
            t = t - t_mod
            if geom.sec.n_row == 1:
                t = 0.0

            pos = _get_position(
                geom,
                _SectionPositionArgs(-r, s, t, j, geom.sec.n_row, geom.sec.n_col),
            )
            node = geom.iniL[j].poi[0] + i * geom.n_modP
            geom.croP[node].pos = pos

            pos = _get_position(
                geom,
                _SectionPositionArgs(r, s, t, j, geom.sec.n_row, geom.sec.n_col),
            )
            node = geom.iniL[j].poi[1] + i * geom.n_modP
            geom.croP[node].pos = pos


def _get_position(geom: GeomType, args: _SectionPositionArgs) -> Tuple[float, float, float]:
    hr, hst = _get_shape_function(args.r, args.s, args.t)
    Vs, Vt = _get_director(geom, args.num)
    yz_bar = _get_parameter(geom.sec.types, hst, args.n_row, args.n_col)

    p1 = np.array(geom.modP[geom.iniL[args.num].poi[0]].pos)
    p2 = np.array(geom.modP[geom.iniL[args.num].poi[1]].pos)
    pos = (
        hr[0] * p1
        + hr[1] * p2
        + hr[0] * yz_bar[0] * Vt
        + hr[1] * yz_bar[0] * Vt
        + hr[0] * yz_bar[1] * Vs
        + hr[1] * yz_bar[1] * Vs
    )
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _get_shape_function(r: float, s: float, t: float) -> Tuple[List[float], List[float]]:
    hr = [0.5 * (1.0 - r), 0.5 * (1.0 + r)]
    hst = [
        0.25 * (1.0 - s) * (1.0 + t),
        0.25 * (1.0 + s) * (1.0 + t),
        0.25 * (1.0 - s) * (1.0 - t),
        0.25 * (1.0 + s) * (1.0 - t),
    ]
    return hr, hst


def _get_director(geom: GeomType, line: int) -> Tuple[np.ndarray, np.ndarray]:
    Vs = np.array(geom.croL[line].t[1], dtype=float)
    Vt = np.array(geom.croL[line].t[2], dtype=float)
    return Vs, Vt


def _get_parameter(type_sec: str, hst: List[float], n_row: int, n_col: int) -> Tuple[float, float]:
    rad_cylinder = para.para_rad_helix + para.para_gap_helix / 2.0
    if type_sec == "square":
        y_scale = rad_cylinder * float(n_col - 1)
        z_scale = rad_cylinder * float(n_row - 1)
    else:
        y_scale = 0.5 * rad_cylinder * 3.0 * float(n_col)
        z_scale = 0.5 * rad_cylinder * math.sqrt(3.0) * float(n_row - 1)

    y = [-y_scale, y_scale, -y_scale, y_scale]
    z = [z_scale, z_scale, -z_scale, -z_scale]
    yz_bar_0 = sum(hst[i] * y[i] for i in range(4))
    yz_bar_1 = sum(hst[i] * z[i] for i in range(4))
    return (yz_bar_0, yz_bar_1)


def _reset_local_coordinate(geom: GeomType) -> None:
    for i in range(geom.n_croL):
        p1 = np.array(geom.croP[geom.croL[i].poi[0]].pos)
        p2 = np.array(geom.croP[geom.croL[i].poi[1]].pos)
        if geom.croL[i].sec % 2 == 0:
            t1 = np.array(normalize(p2 - p1))
        else:
            t1 = np.array(normalize(p1 - p2))
        t2 = np.array(geom.croL[i].t[1])
        t3 = np.array(normalize(np.cross(t1, t2)))
        geom.croL[i].t[0] = (float(t1[0]), float(t1[1]), float(t1[2]))
        geom.croL[i].t[1] = (float(t2[0]), float(t2[1]), float(t2[2]))
        geom.croL[i].t[2] = (float(t3[0]), float(t3[1]), float(t3[2]))
