from __future__ import annotations

import numpy as np

from . import para
from .data_geom import GeomType
from .math_utils import deg2rad, pi
from .modgeo_neighbors import _set_angle_junction


def _separate_line(geom: GeomType) -> None:
    geom.n_modP = 2 * geom.n_iniL
    geom.modP = [type(geom.iniP[0])() for _ in range(geom.n_modP)]
    count_arm = [0 for _ in range(geom.n_junc)]
    for i in range(geom.n_iniL):
        poi_1 = geom.iniL[i].poi[0]
        poi_2 = geom.iniL[i].poi[1]
        geom.modP[2 * i].pos = geom.iniP[poi_1].pos
        geom.modP[2 * i + 1].pos = geom.iniP[poi_2].pos
        geom.iniL[i].poi[0] = 2 * i
        geom.iniL[i].poi[1] = 2 * i + 1
        for j in range(geom.n_junc):
            if poi_1 == geom.junc[j].poi_c:
                geom.junc[j].modP[count_arm[j]] = geom.iniL[i].poi[0]
                count_arm[j] += 1
            elif poi_2 == geom.junc[j].poi_c:
                geom.junc[j].modP[count_arm[j]] = geom.iniL[i].poi[1]
                count_arm[j] += 1


def _set_width_section(geom: GeomType) -> float:
    if geom.sec.types == "square":
        n_column = geom.sec.ref_maxC - geom.sec.ref_minC + 1
    elif geom.sec.ref_row == 1:
        n_column = 2
    else:
        n_column = 3 * (geom.sec.ref_maxC - geom.sec.ref_minC + 1) / 2
    return (2.0 * para.para_rad_helix + para.para_gap_helix) * float(n_column)


def _find_scale_factor(prob, geom: GeomType) -> float:
    _set_angle_junction(geom)
    width = _set_width_section(geom)

    pos_modP = np.zeros((geom.n_modP, 3), dtype=float)
    for junc in geom.junc:
        ref_ang = junc.ref_ang
        tot_ang = junc.tot_ang if prob.type_geo != "open" else 2.0 * pi
        ang = ref_ang * (2.0 * pi / tot_ang)
        if junc.n_arm <= 1:
            junc.gap = 0.0
            for k in range(len(junc.iniL)):
                pos_modP[junc.modP[k], :] = np.array(geom.modP[junc.modP[k]].pos)
            continue
        factor = (0.20 - 0.0) * (ang - ref_ang) / deg2rad(60.0) if geom.sec.n_col == 2 else 0.0
        if tot_ang <= 2.0 * pi:
            junc.gap = (width / 2.0 / np.tan(ang / 2.0) + factor) * (2.0 * pi / tot_ang)
        else:
            junc.gap = width / 2.0 / np.tan(ang / 2.0)

        for k, line_idx in enumerate(junc.iniL):
            poi_cur = junc.modP[k]
            pos_cur = np.array(geom.modP[poi_cur].pos)
            poi_1 = geom.iniL[line_idx].poi[0]
            poi_2 = geom.iniL[line_idx].poi[1]
            pos_opp = np.array(geom.modP[poi_2].pos) if poi_1 == poi_cur else np.array(geom.modP[poi_1].pos)
            length = float(np.linalg.norm(pos_opp - pos_cur))
            pos_modP[poi_cur, :] = (junc.gap * pos_opp + (length - junc.gap) * pos_cur) / length

    if prob.sel_edge_ref == 0:
        min_length = None
        for i in range(geom.n_iniL):
            poi_1 = geom.iniL[i].poi[0]
            poi_2 = geom.iniL[i].poi[1]
            length = float(np.linalg.norm(pos_modP[poi_1] - pos_modP[poi_2]))
            if min_length is None or length < min_length:
                min_length = length
                cur_length = float(np.linalg.norm(np.array(geom.modP[poi_1].pos) - np.array(geom.modP[poi_2].pos)))
                diff = cur_length - min_length
    else:
        poi_1 = geom.iniL[prob.sel_edge_ref].poi[0]
        poi_2 = geom.iniL[prob.sel_edge_ref].poi[1]
        min_length = float(np.linalg.norm(pos_modP[poi_1] - pos_modP[poi_2]))
        cur_length = float(np.linalg.norm(np.array(geom.modP[poi_1].pos) - np.array(geom.modP[poi_2].pos)))
        diff = cur_length - min_length

    length = diff + para.para_dist_bp * float(prob.n_edge_len - 2)
    return length / cur_length


def _scale_geometry(geom: GeomType, scale: float) -> None:
    for i in range(geom.n_modP):
        p = np.array(geom.modP[i].pos, dtype=float) * scale
        geom.modP[i].pos = (float(p[0]), float(p[1]), float(p[2]))
    for i in range(geom.n_iniP):
        p = np.array(geom.iniP[i].pos, dtype=float) * scale
        geom.iniP[i].pos = (float(p[0]), float(p[1]), float(p[2]))


def _set_gap_junction(geom: GeomType) -> None:
    for junc in geom.junc:
        for k, line_idx in enumerate(junc.iniL):
            poi_cur = junc.modP[k]
            pos_cur = np.array(geom.modP[poi_cur].pos)
            poi_1 = geom.iniL[line_idx].poi[0]
            poi_2 = geom.iniL[line_idx].poi[1]
            pos_opp = np.array(geom.modP[poi_2].pos) if poi_1 == poi_cur else np.array(geom.modP[poi_1].pos)
            length = float(np.linalg.norm(pos_opp - pos_cur))
            pos = (junc.gap * pos_opp + (length - junc.gap) * pos_cur) / length
            geom.modP[poi_cur].pos = (float(pos[0]), float(pos[1]), float(pos[2]))
