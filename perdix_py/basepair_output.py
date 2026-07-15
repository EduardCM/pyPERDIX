from __future__ import annotations

import numpy as np

from . import para
from .data_geom import GeomType
from .data_mesh import EleType, MeshType, NodeType
from .math_utils import is_same_vector, nint, norm
from .naming import bild_path


def _make_sticky_end(geom: GeomType, mesh: MeshType) -> None:
    if geom.sec.types == "square" or not geom.face:
        return
    for i in range(geom.n_junc):
        for j in range(geom.junc[i].n_arm):
            for k in range(geom.n_sec):
                node = geom.junc[i].node[j][k]
                sec = mesh.node[node].sec
                if mesh.node[node].conn == 4 and para.para_sticky_self == "off":
                    continue
                if mesh.node[node].conn == 3:
                    continue
                result = _append_sticky_end_node(geom, mesh, node, sec)
                if result is None:
                    continue
                new_id, pre = result
                _copy_sticky_end_node_fields(mesh, node, new_id)
                _place_sticky_end_node(mesh, node, new_id, pre)
                mesh.ele[-1].cn = (node, new_id)
                _replace_sticky_end_junction_node(geom, node, new_id, i, j, k)
                geom.croP[geom.junc[i].croP[j][k]].pos = mesh.node[new_id].pos


def _copy_sticky_end_node_fields(mesh: MeshType, node: int, new_id: int) -> None:
    mesh.node[new_id].id = new_id
    mesh.node[new_id].sec = mesh.node[node].sec
    mesh.node[new_id].iniL = mesh.node[node].iniL
    mesh.node[new_id].croL = mesh.node[node].croL
    mesh.node[new_id].conn = mesh.node[node].conn
    mesh.node[new_id].ghost = mesh.node[node].ghost


def _place_sticky_end_node(mesh: MeshType, node: int, new_id: int, pre: int) -> None:
    pos = np.array(mesh.node[node].pos)
    vec = pos - np.array(mesh.node[pre].pos)
    pos = pos + vec / norm(vec) * para.para_dist_bp
    mesh.node[new_id].pos = (float(pos[0]), float(pos[1]), float(pos[2]))


def _replace_sticky_end_junction_node(geom: GeomType, old_node: int, new_id: int, junc_index: int, arm_index: int, sec_index: int) -> None:
    geom.junc[junc_index].node[arm_index][sec_index] = new_id
    for junc in geom.junc:
        for idx in range(junc.n_arm * geom.n_sec):
            if junc.conn[idx][0] == old_node:
                junc.conn[idx][0] = new_id
            if junc.conn[idx][1] == old_node:
                junc.conn[idx][1] = new_id


def _append_sticky_end_node(geom: GeomType, mesh: MeshType, node: int, sec: int) -> tuple[int, int] | None:
    if mesh.node[node].dn == -1:
        if para.para_sticky_self == "off" and geom.sec.conn[sec] != -1 and mesh.node[node].sec % 2 == 0:
            return None
        new_id = _reserve_mesh_node_and_element(mesh)
        mesh.node[new_id].bp = mesh.node[node].bp - 1 if mesh.node[node].sec % 2 == 0 else mesh.node[node].bp + 1
        mesh.node[new_id].up = mesh.node[node].id
        mesh.node[new_id].dn = -1
        mesh.node[node].dn = new_id
        return new_id, mesh.node[node].up
    if mesh.node[node].up == -1 and geom.sec.conn[sec] != -1 and mesh.node[node].sec % 2 == 0:
        if para.para_sticky_self != "off":
            return None
        new_id = _reserve_mesh_node_and_element(mesh)
        mesh.node[new_id].bp = mesh.node[node].bp + 1
        mesh.node[new_id].up = -1
        mesh.node[new_id].dn = mesh.node[node].id
        mesh.node[node].up = new_id
        return new_id, mesh.node[node].dn
    return None


def _reserve_mesh_node_and_element(mesh: MeshType) -> int:
    new_id = mesh.n_node
    mesh.n_node += 1
    mesh.n_ele += 1
    mesh.node.append(NodeType())
    mesh.ele.append(EleType())
    return new_id


def _write_edge_length(prob, geom: GeomType) -> None:
    if not para.para_write_505:
        return
    from pathlib import Path

    path = Path(prob.path_work) / "TXT_Edge_Length.txt"
    num_bp: list[int] = []
    num_count: list[int] = []
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{'Multi line':>15}{'Init line':>15}{'Section ID':>15}{'Edge length':>15}\n")
        for i in range(geom.n_croL):
            poi_1 = geom.croL[i].poi[0]
            poi_2 = geom.croL[i].poi[1]
            length = float(np.linalg.norm(np.array(geom.croP[poi_1].pos) - np.array(geom.croP[poi_2].pos)))
            n_bp = nint(length / para.para_dist_bp) + 1
            f.write(f"{i+1:15d}{geom.croL[i].iniL+1:15d}{geom.croL[i].sec+1:15d}{n_bp:15d}\n")
            if n_bp in num_bp:
                num_count[num_bp.index(n_bp)] += 1
            else:
                num_bp.append(n_bp)
                num_count.append(1)
        pairs = sorted(zip(num_bp, num_count), key=lambda x: x[0])
        f.write("\n")
        f.write(f"{'Edge length':>15}{'# of multi lines':>15}\n")
        for n_bp, n_ct in pairs:
            f.write(f"{n_bp:15d}{n_ct:15d}\n")
    if num_bp:
        geom.min_edge_length = min(num_bp)
        geom.max_edge_length = max(num_bp)


def _write_cylinder_bild(prob, geom: GeomType, mesh: MeshType, mode: str) -> None:
    if not para.para_write_502:
        return
    path = bild_path(prob, "05_cylindrical_model_1" if mode == "cylinder_prior" else "06_cylindrical_model_2")
    radius = para.para_rad_helix + para.para_gap_helix / 2.0
    with path.open("w", encoding="utf-8") as f:
        def _fmt_92(value: float) -> str:
            return f"{0.0 if abs(value) < 5e-4 else value:9.2f}"

        r = prob.color[0] / 255.0
        g = prob.color[1] / 255.0
        b = prob.color[2] / 255.0
        if mode == "cylinder_prior":
            f.write(f".color {r:9.4f}{g:9.4f}{b:9.4f}\n")
            min_length = 1.0e9
            min_croL = 0
            for i in range(geom.n_croL):
                p1 = geom.croP[geom.croL[i].poi[0]].pos
                p2 = geom.croP[geom.croL[i].poi[1]].pos
                length = float(np.linalg.norm(np.array(p1) - np.array(p2)))
                if min_length >= length:
                    min_length = length
                    min_croL = i
                f.write(f".cylinder {p1[0]:10.3f}{p1[1]:10.3f}{p1[2]:10.3f}{p2[0]:10.3f}{p2[1]:10.3f}{p2[2]:10.3f}{radius:10.3f}\n")
            p1 = geom.croP[geom.croL[min_croL].poi[0]].pos
            p2 = geom.croP[geom.croL[min_croL].poi[1]].pos
            f.write(".color red\n")
            f.write(f".cylinder {p1[0]:10.3f}{p1[1]:10.3f}{p1[2]:10.3f}{p2[0]:10.3f}{p2[1]:10.3f}{p2[2]:10.3f}{radius*4.0:10.3f}\n")
        else:
            for i in range(geom.n_croL):
                p1 = geom.croP[geom.croL[i].poi[0]].ori_pos
                p2 = geom.croP[geom.croL[i].poi[1]].ori_pos
                f.write(f".color {r:9.4f}{g:9.4f}{b:9.4f}\n")
                f.write(f".cylinder {p1[0]:10.3f}{p1[1]:10.3f}{p1[2]:10.3f}{p2[0]:10.3f}{p2[1]:10.3f}{p2[2]:10.3f}{radius:10.3f}\n")
                for poi in (geom.croL[i].poi[0], geom.croL[i].poi[1]):
                    pos_1 = geom.croP[poi].pos
                    pos_2 = geom.croP[poi].ori_pos
                    if is_same_vector(pos_1, pos_2):
                        continue
                    f.write(".color dark gray\n" if float(np.linalg.norm(np.array(pos_1) - np.array(pos_2))) > 0.4 else f".color {r:9.4f}{g:9.4f}{b:9.4f}\n")
                    f.write(f".cylinder {pos_1[0]:12.5f}{pos_1[1]:12.5f}{pos_1[2]:12.5f}{pos_2[0]:12.5f}{pos_2[1]:12.5f}{pos_2[2]:12.5f}{radius:9.3f}\n")
        for i in range(geom.n_junc):
            for j in range(geom.n_sec * geom.junc[i].n_arm):
                node_1 = geom.junc[i].conn[j][0]
                node_2 = geom.junc[i].conn[j][1]
                if node_1 == -1 or node_2 == -1:
                    continue
                p1 = mesh.node[node_1].pos
                p2 = mesh.node[node_2].pos
                f.write(".color red\n" if geom.junc[i].type_conn[j] == 1 else ".color steel blue\n")
                f.write(f".cylinder {p1[0]:10.3f}{p1[1]:10.3f}{p1[2]:10.3f}{p2[0]:10.3f}{p2[1]:10.3f}{p2[2]:10.3f}{radius*0.1:10.3f}\n")
                f.write(".color steel blue\n")
                f.write(f".sphere {p1[0]:10.3f}{p1[1]:10.3f}{p1[2]:10.3f}{0.2:10.3f}\n")
                f.write(f".sphere {p2[0]:10.3f}{p2[1]:10.3f}{p2[2]:10.3f}{0.2:10.3f}\n")
        if mode == "cylinder_prior":
            f.write(".color red\n")
            for p in geom.iniP:
                f.write(f".sphere {_fmt_92(p.pos[0])}{_fmt_92(p.pos[1])}{_fmt_92(p.pos[2])}{_fmt_92(0.30)}\n")
            f.write(".color dark green\n")
            for junc in geom.junc:
                for iniL in junc.iniL:
                    p1 = geom.iniP[geom.iniL[iniL].iniP[0]].pos
                    p2 = geom.iniP[geom.iniL[iniL].iniP[1]].pos
                    f.write(f".cylinder {_fmt_92(p1[0])}{_fmt_92(p1[1])}{_fmt_92(p1[2])}{_fmt_92(p2[0])}{_fmt_92(p2[1])}{_fmt_92(p2[2])}{_fmt_92(0.12)}\n")


def _write_mesh_bild(prob, geom: GeomType, mesh: MeshType) -> None:
    if not para.para_write_503:
        return
    path = bild_path(prob, "mesh")
    with path.open("w", encoding="utf-8") as f:
        for i in range(mesh.n_node):
            if mesh.node[i].ghost == 1:
                continue
            p = mesh.node[i].pos
            f.write(".color orange\n")
            f.write(f".sphere {p[0]:9.3f}{p[1]:9.3f}{p[2]:9.3f}{0.15:9.3f}\n")
        for ele in mesh.ele:
            p1 = mesh.node[ele.cn[0]].pos
            p2 = mesh.node[ele.cn[1]].pos
            f.write(".color blue\n")
            f.write(f".cylinder {p1[0]:9.3f}{p1[1]:9.3f}{p1[2]:9.3f}{p2[0]:9.3f}{p2[1]:9.3f}{p2[2]:9.3f}{0.08:9.3f}\n")
