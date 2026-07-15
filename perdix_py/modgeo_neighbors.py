from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from . import para
from .data_geom import GeomType
from .math_utils import normalize


def _round_geom_positions(geom: GeomType, digits: int = 4) -> None:
    for collection_name in ("iniP", "modP", "croP"):
        for p in getattr(geom, collection_name, []):
            p.pos = tuple(round(float(v), digits) for v in p.pos)
            p.ori_pos = tuple(round(float(v), digits) for v in p.ori_pos)


def _find_neighbor_face(geom: GeomType, line_idx: int) -> Tuple[int, int]:
    nei_face = [-1, -1]
    if not geom.face:
        return (-1, -1)
    poi_1 = geom.iniL[line_idx].poi[0]
    poi_2 = geom.iniL[line_idx].poi[1]
    for i, face in enumerate(geom.face):
        for j in range(face.n_poi):
            poi_a = face.poi[j]
            poi_b = face.poi[(j + 1) % face.n_poi]
            if poi_a == poi_1 and poi_b == poi_2:
                nei_face[0] = i
            if poi_b == poi_1 and poi_a == poi_2:
                nei_face[1] = i
    return (nei_face[0], nei_face[1])


def _find_neighbor_points(geom: GeomType, line_idx: int, faces: Tuple[int, int]) -> List[List[int]]:
    poi_1 = geom.iniL[line_idx].poi[0]
    poi_2 = geom.iniL[line_idx].poi[1]
    nei_poi = [[-1, -1], [-1, -1]]
    if faces[0] == -1 and faces[1] == -1:
        def order_neighbors(p_center: int, p_other: int) -> tuple[int, int]:
            center = np.array(geom.iniP[p_center].pos, dtype=float)
            other = np.array(geom.iniP[p_other].pos, dtype=float)
            line_dir = normalize(other - center)
            angles: list[tuple[float, int]] = []
            for i, line in enumerate(geom.iniL):
                if i == line_idx:
                    continue
                if line.poi[0] == p_center:
                    neighbor = line.poi[1]
                elif line.poi[1] == p_center:
                    neighbor = line.poi[0]
                else:
                    continue
                vec = normalize(np.array(geom.iniP[neighbor].pos, dtype=float) - center)
                ang = math.atan2(np.cross(line_dir, vec)[2], float(np.dot(line_dir, vec)))
                angles.append((ang, neighbor))
            if not angles:
                return -1, -1
            angles.sort(key=lambda item: item[0])
            if len(angles) == 1:
                return angles[0][1], angles[0][1]
            return angles[0][1], angles[-1][1]

        nei_poi[0][0], nei_poi[0][1] = order_neighbors(poi_1, poi_2)
        nei_poi[1][0], nei_poi[1][1] = order_neighbors(poi_2, poi_1)
        return nei_poi

    for face_idx in faces:
        if face_idx == -1:
            continue
        face = geom.face[face_idx]
        poi_first = face.poi[0]
        poi_last = face.poi[face.n_poi - 1]
        for j in range(face.n_poi - 1):
            poi_a = face.poi[j]
            poi_b = face.poi[j + 1]
            if poi_1 == poi_a and poi_2 == poi_b:
                nei_poi[1][0] = poi_first if j + 1 == face.n_poi - 1 else face.poi[j + 2]
                nei_poi[0][0] = poi_last if j == 0 else face.poi[j - 1]
            elif poi_2 == poi_first and poi_1 == poi_last:
                nei_poi[1][0] = face.poi[1]
                nei_poi[0][0] = face.poi[face.n_poi - 2]

            if poi_2 == poi_a and poi_1 == poi_b:
                nei_poi[1][1] = poi_last if j == 0 else face.poi[j - 1]
                nei_poi[0][1] = poi_first if j + 1 == face.n_poi - 1 else face.poi[j + 2]
            elif poi_1 == poi_first and poi_2 == poi_last:
                nei_poi[1][1] = face.poi[face.n_poi - 2]
                nei_poi[0][1] = face.poi[1]
    return nei_poi


def _set_neighbor_point(prob, geom: GeomType) -> None:
    for i in range(geom.n_iniL):
        faces = _find_neighbor_face(geom, i)
        if faces[0] == -1 or faces[1] == -1:
            prob.type_geo = "open"
        geom.iniL[i].neiF[0] = faces[0]
        geom.iniL[i].neiF[1] = faces[1]
        nei_poi = _find_neighbor_points(geom, i, faces)
        geom.iniL[i].neiP[0][0] = nei_poi[0][0]
        geom.iniL[i].neiP[0][1] = nei_poi[0][1]
        geom.iniL[i].neiP[1][0] = nei_poi[1][0]
        geom.iniL[i].neiP[1][1] = nei_poi[1][1]


def _find_neighbor_line(geom: GeomType, line_idx: int) -> List[List[int]]:
    nei_line = [[-1, -1], [-1, -1]]
    poi_1 = geom.iniL[line_idx].poi[0]
    poi_2 = geom.iniL[line_idx].poi[1]
    for i in range(geom.n_iniL):
        line = geom.iniL[i]
        if (
            (geom.iniL[line_idx].neiP[0][0] == line.poi[0] and poi_1 == line.poi[1])
            or (geom.iniL[line_idx].neiP[0][0] == line.poi[1] and poi_1 == line.poi[0])
        ):
            nei_line[0][0] = i
        if (
            (geom.iniL[line_idx].neiP[1][0] == line.poi[0] and poi_2 == line.poi[1])
            or (geom.iniL[line_idx].neiP[1][0] == line.poi[1] and poi_2 == line.poi[0])
        ):
            nei_line[1][0] = i
        if (
            (geom.iniL[line_idx].neiP[0][1] == line.poi[0] and poi_1 == line.poi[1])
            or (geom.iniL[line_idx].neiP[0][1] == line.poi[1] and poi_1 == line.poi[0])
        ):
            nei_line[0][1] = i
        if (
            (geom.iniL[line_idx].neiP[1][1] == line.poi[0] and poi_2 == line.poi[1])
            or (geom.iniL[line_idx].neiP[1][1] == line.poi[1] and poi_2 == line.poi[0])
        ):
            nei_line[1][1] = i
    return nei_line


def _set_neighbor_line(geom: GeomType) -> None:
    for i in range(geom.n_iniL):
        nei_line = _find_neighbor_line(geom, i)
        geom.iniL[i].neiL[0][0] = nei_line[0][0]
        geom.iniL[i].neiL[0][1] = nei_line[0][1]
        geom.iniL[i].neiL[1][0] = nei_line[1][0]
        geom.iniL[i].neiL[1][1] = nei_line[1][1]


def _set_junction_data(geom: GeomType) -> None:
    juncs = []
    for poi in range(geom.n_iniP):
        incident = [li for li, line in enumerate(geom.iniL) if poi == line.poi[0] or poi == line.poi[1]]
        if len(incident) < 2:
            raise ValueError("The geometry is not closed (junction with <2 arms).")
        juncs.append((poi, incident))

    from .data_geom import JuncType

    geom.n_junc = len(juncs)
    geom.junc = []
    for poi_c, incident in juncs:
        junc = JuncType()
        junc.n_arm = len(incident)
        junc.poi_c = poi_c
        junc.iniL = incident[:]
        junc.modP = [-1 for _ in incident]
        junc.croP = [[-1 for _ in range(max(geom.n_sec, 1))] for _ in incident]
        junc.node = [[-1 for _ in range(max(geom.n_sec, 1))] for _ in incident]
        junc.conn = [[-1, -1] for _ in range(len(incident) * max(geom.n_sec, 1))]
        junc.type_conn = [-1 for _ in range(len(incident) * max(geom.n_sec, 1))]
        geom.junc.append(junc)


def _set_local_coordinate(geom: GeomType) -> None:
    for i in range(geom.n_iniL):
        t = _set_local_vectors(geom, i)
        geom.iniL[i].t[0] = t[0]
        geom.iniL[i].t[1] = t[1]
        geom.iniL[i].t[2] = t[2]


def _set_local_vectors(geom: GeomType, line_idx: int) -> List[Tuple[float, float, float]]:
    p1 = np.array(geom.iniP[geom.iniL[line_idx].poi[0]].pos, dtype=float)
    p2 = np.array(geom.iniP[geom.iniL[line_idx].poi[1]].pos, dtype=float)
    t1 = np.array(normalize(p2 - p1))

    face1 = geom.iniL[line_idx].neiF[0]
    face2 = geom.iniL[line_idx].neiF[1]
    if face1 == -1 and face2 == -1:
        t2 = np.array([1.0, 0.0, 0.0]) if abs(t1[0]) < 1e-8 and abs(t1[1]) < 1e-8 else np.array([0.0, 0.0, 1.0])
        t3 = np.array(normalize(np.cross(t1, t2)))
        return [tuple(t1), tuple(t2), tuple(t3)]

    def face_normal(face_idx: int) -> np.ndarray:
        face = geom.face[face_idx]
        pts = np.array([geom.iniP[i].pos for i in face.poi], dtype=float)
        pc = pts.mean(axis=0)
        vec_c = np.cross(pts[0] - pc, pts[1] - pc)
        if np.linalg.norm(vec_c) < 1e-8 and len(pts) > 2:
            vec_c = np.cross(pts[1] - pc, pts[2] - pc)
        return np.array(normalize(vec_c))

    vec_face1 = face_normal(face1) if face1 != -1 else np.zeros(3)
    vec_face2 = face_normal(face2) if face2 != -1 else np.zeros(3)
    t2 = vec_face1 + vec_face2 if face1 == -1 or face2 == -1 else 0.5 * (vec_face1 + vec_face2)
    t2 = np.array(normalize(t2))
    t3 = np.array(normalize(np.cross(t1, t2)))
    return [tuple(t1), tuple(t2), tuple(t3)]


def _set_angle_junction(geom: GeomType) -> None:
    if not geom.face:
        for junc in geom.junc:
            if junc.n_arm <= 0:
                junc.tot_ang = 2.0 * math.pi
                junc.ref_ang = 2.0 * math.pi
                continue
            center = np.array(geom.iniP[junc.poi_c].pos, dtype=float)
            angles: list[float] = []
            for line_idx in junc.iniL:
                p1 = geom.iniL[line_idx].iniP[0]
                p2 = geom.iniL[line_idx].iniP[1]
                other = np.array(geom.iniP[p2].pos if p1 == junc.poi_c else geom.iniP[p1].pos, dtype=float)
                vec = other - center
                if np.linalg.norm(vec) < 1e-8:
                    continue
                angles.append(math.atan2(vec[1], vec[0]))
            if len(angles) <= 1:
                junc.tot_ang = 2.0 * math.pi
                junc.ref_ang = 2.0 * math.pi
                continue
            angles.sort()
            diffs = []
            for idx, ang_cur in enumerate(angles):
                diff = angles[(idx + 1) % len(angles)] - ang_cur
                if diff < 0:
                    diff += 2.0 * math.pi
                diffs.append(diff)
            junc.tot_ang = float(sum(diffs))
            if para.para_vertex_angle in ("min", "opt"):
                junc.ref_ang = min(diffs)
            elif para.para_vertex_angle == "max":
                junc.ref_ang = max(diffs)
            else:
                junc.ref_ang = junc.tot_ang / float(len(diffs))
        return

    junc_angles = [[] for _ in range(geom.n_iniP)]
    for face in geom.face:
        for idx in range(face.n_poi):
            poi_c = face.poi[idx]
            pos_cur = np.array(geom.iniP[poi_c].pos)
            pos_pre = np.array(geom.iniP[face.poi[idx - 1]].pos)
            pos_next = np.array(geom.iniP[face.poi[(idx + 1) % face.n_poi]].pos)
            ang = math.atan2(np.linalg.norm(np.cross(pos_pre - pos_cur, pos_next - pos_cur)), float(np.dot(pos_pre - pos_cur, pos_next - pos_cur)))
            junc_angles[poi_c].append(ang)

    for junc in geom.junc:
        angs = junc_angles[junc.poi_c]
        if not angs:
            junc.tot_ang = 2.0 * math.pi
            junc.ref_ang = 2.0 * math.pi / max(junc.n_arm, 1)
            continue
        junc.tot_ang = float(sum(angs))
        if para.para_vertex_angle in ("min", "opt"):
            junc.ref_ang = min(angs)
        elif para.para_vertex_angle == "max":
            junc.ref_ang = max(angs)
        else:
            junc.ref_ang = junc.tot_ang / len(angs)
