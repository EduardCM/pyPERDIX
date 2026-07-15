from __future__ import annotations

import math

import numpy as np

from . import para
from .data_geom import GeomType
from .data_mesh import EleType, MeshType, NodeType
from .math_utils import cross, find_closest_point, nint, norm


def _increase_edge(prob, geom: GeomType, mesh: MeshType, node_cur: int, node_com: int) -> None:
    endpoint_pair = _edge_increase_endpoint_pair(mesh, node_cur, node_com)
    if endpoint_pair is None:
        return
    node_in, node_out = endpoint_pair
    if mesh.node[node_in].up != -1 and mesh.node[node_in].dn != -1:
        return
    if mesh.node[node_out].up != -1 and mesh.node[node_out].dn != -1:
        return
    node_in_dn = mesh.node[node_in].dn
    node_out_up = mesh.node[node_out].up
    vec_in = np.array(mesh.node[node_in].pos) - np.array(mesh.node[node_in_dn].pos)
    vec_out = np.array(mesh.node[node_out].pos) - np.array(mesh.node[node_out_up].pos)
    angle = math.atan2(norm(cross(vec_in, vec_out)), float(np.dot(vec_in, vec_out)))
    len_side = para.para_rad_helix + para.para_gap_helix / 2.0
    a = math.sqrt(2.0 * len_side**2.0 - 2.0 * len_side * len_side * math.cos(math.pi - angle))
    b = norm(np.array(mesh.node[node_in].pos) - np.array(mesh.node[node_out].pos))
    c = norm(np.array(mesh.node[node_in_dn].pos) - np.array(mesh.node[node_out_up].pos))
    if abs(a - b) < 0.001:
        _mark_edge_endpoint_connection(prob, geom, mesh, node_in)
        _mark_edge_endpoint_connection(prob, geom, mesh, node_out)
        return
    count = _edge_extension_count_from_lengths(a, b, c, para.para_dist_bp)
    if count is None:
        return
    line_a = mesh.node[node_in].croL
    line_b = mesh.node[node_out].croL
    pos_a1 = np.array(geom.croP[geom.croL[line_a].poi[0]].pos)
    pos_a2 = np.array(geom.croP[geom.croL[line_a].poi[1]].pos)
    pos_b1 = np.array(geom.croP[geom.croL[line_b].poi[0]].pos)
    pos_b2 = np.array(geom.croP[geom.croL[line_b].poi[1]].pos)
    pos_inter, ok = find_closest_point(pos_a1, pos_a2, pos_b1, pos_b2)
    if ok:
        rad = len_side / math.tan(angle / 2.0)
        len_a = norm(np.array(mesh.node[node_in].pos) - np.array(pos_inter)) - rad
        len_b = norm(np.array(mesh.node[node_out].pos) - np.array(pos_inter)) - rad
        count_in = nint(len_a / para.para_dist_bp)
        count_out = nint(len_b / para.para_dist_bp)
        if not geom.face and count_in == count_out and 0 < count_in < 10:
            count_in -= 1
            count_out -= 1
    else:
        count_in = count
        count_out = count
    for i in range(count_in):
        node_in = _add_edge_extension_basepair(geom, mesh, node_in, vec_in, is_terminal=i == count_in - 1)
    for i in range(count_out):
        node_out = _add_edge_extension_basepair(geom, mesh, node_out, vec_out, is_terminal=i == count_out - 1)
    if count_in == 0:
        _mark_edge_endpoint_connection(prob, geom, mesh, node_in)
    if count_out == 0:
        _mark_edge_endpoint_connection(prob, geom, mesh, node_out)


def _edge_extension_count_from_lengths(a: float, b: float, c: float, y: float) -> int | None:
    denom = 1.0 - ((y**2.0 - ((c - b) / 2.0) ** 2.0) / y**2.0)
    if denom <= 0.0:
        return None
    length = math.sqrt(((b - a) / 2.0) ** 2.0 / denom)
    return int(math.floor(length / y))


def _add_edge_extension_basepair(geom: GeomType, mesh: MeshType, node: int, vec: np.ndarray, is_terminal: bool) -> int:
    node = _add_basepair(geom, mesh, node, vec)
    if is_terminal:
        mesh.node[node].conn = 3
    return node


def _edge_increase_endpoint_pair(mesh: MeshType, node_cur: int, node_com: int) -> tuple[int, int] | None:
    if mesh.node[node_cur].up == -1:
        return node_cur, node_com
    if mesh.node[node_cur].dn == -1:
        return node_com, node_cur
    if mesh.node[node_com].up == -1:
        return node_com, node_cur
    if mesh.node[node_com].dn == -1:
        return node_cur, node_com
    return None


def _mark_edge_endpoint_connection(prob, geom: GeomType, mesh: MeshType, node: int) -> None:
    mesh.node[node].conn = 1
    poi_1 = geom.croL[mesh.node[node].croL].poi[0]
    poi_2 = geom.croL[mesh.node[node].croL].poi[1]
    length = norm(np.array(geom.croP[poi_1].pos) - np.array(geom.croP[poi_2].pos))
    if prob.n_edge_len - 2 != nint(length / para.para_dist_bp):
        mesh.node[node].conn = 3


def _add_basepair(geom: GeomType, mesh: MeshType, node: int, vec: np.ndarray) -> int:
    prev = mesh.node[node]
    new_id = mesh.n_node
    new_node = NodeType()
    new_node.id = new_id
    if prev.up == -1:
        new_node.bp = prev.bp + 1 if prev.sec % 2 == 0 else prev.bp - 1
        new_node.up = -1
        new_node.dn = node
        prev.up = new_id
    elif prev.dn == -1:
        new_node.bp = prev.bp - 1 if prev.sec % 2 == 0 else prev.bp + 1
        new_node.up = node
        new_node.dn = -1
        prev.dn = new_id
    else:
        raise ValueError("Check the junction connectivity")
    new_node.sec = prev.sec
    new_node.iniL = prev.iniL
    new_node.croL = prev.croL
    new_node.mitered = 1
    new_node.conn = -1
    new_node.ghost = -1
    new_node.pos = tuple(np.array(prev.pos) + vec)
    new_node.ori = prev.ori
    mesh.node.append(new_node)
    mesh.n_node += 1
    mesh.ele.append(EleType(cn=(node, new_id)))
    mesh.n_ele += 1
    for i in range(geom.n_junc):
        for j in range(geom.n_sec):
            for k in range(geom.junc[i].n_arm):
                if geom.junc[i].node[k][j] == node:
                    geom.junc[i].node[k][j] = new_id
                    geom.croP[geom.junc[i].croP[k][j]].pos = new_node.pos
                idx = j * geom.junc[i].n_arm + k
                if geom.junc[i].conn[idx][0] == node:
                    geom.junc[i].conn[idx][0] = new_id
                if geom.junc[i].conn[idx][1] == node:
                    geom.junc[i].conn[idx][1] = new_id
    mesh.n_mitered += 1
    return new_id
