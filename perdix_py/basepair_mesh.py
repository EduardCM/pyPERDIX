from __future__ import annotations

import numpy as np

from .data_geom import GeomType
from .data_mesh import EleType, MeshType, NodeType
from . import para
from .math_utils import nint


def _round_mesh_positions(mesh: MeshType, digits: int = 4) -> None:
    for node in mesh.node:
        node.pos = tuple(round(float(v), digits) for v in node.pos)
        node.ori = tuple(tuple(round(float(c), digits) for c in axis) for axis in node.ori)


def _count_basepair(prob, geom: GeomType, mesh: MeshType) -> None:
    mesh.n_node = 0
    mesh.n_ele = 0
    for i in range(geom.n_croL):
        p1 = np.array(geom.croP[geom.croL[i].poi[0]].pos)
        p2 = np.array(geom.croP[geom.croL[i].poi[1]].pos)
        count = nint(float(np.linalg.norm(p2 - p1)) / para.para_dist_bp)
        mesh.n_node += count + 1
        mesh.n_ele += count


def _generate_basepair(geom: GeomType, mesh: MeshType) -> None:
    mesh.node = [NodeType() for _ in range(mesh.n_node)]
    mesh.ele = [EleType() for _ in range(mesh.n_ele)]
    n_node = 0
    n_conn = 0
    for i in range(geom.n_croL):
        p1 = np.array(geom.croP[geom.croL[i].poi[0]].pos)
        p2 = np.array(geom.croP[geom.croL[i].poi[1]].pos)
        count = nint(float(np.linalg.norm(p2 - p1)) / para.para_dist_bp)
        node_start = n_node
        mesh.node[node_start].id = node_start
        mesh.node[node_start].bp = 1
        mesh.node[node_start].sec = geom.croL[i].sec
        mesh.node[node_start].iniL = geom.croL[i].iniL
        mesh.node[node_start].croL = i
        mesh.node[node_start].mitered = -1
        mesh.node[node_start].conn = -1
        mesh.node[node_start].ghost = -1
        mesh.node[node_start].pos = (float(p1[0]), float(p1[1]), float(p1[2]))
        if geom.croL[i].sec % 2 == 0:
            mesh.node[node_start].up = node_start + 1
            mesh.node[node_start].dn = -1
        else:
            mesh.node[node_start].up = -1
            mesh.node[node_start].dn = node_start + 1

        for j in range(1, count + 1):
            idx = node_start + j
            mesh.node[idx].id = idx
            mesh.node[idx].bp = j + 1
            mesh.node[idx].sec = geom.croL[i].sec
            mesh.node[idx].iniL = geom.croL[i].iniL
            mesh.node[idx].croL = i
            mesh.node[idx].mitered = -1
            mesh.node[idx].conn = -1
            mesh.node[idx].ghost = -1
            pos = (j * p2 + (count - j) * p1) / (j + (count - j))
            mesh.node[idx].pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            if geom.croL[i].sec % 2 == 0:
                mesh.node[idx].dn = idx - 1
                mesh.node[idx].up = -1 if j == count else idx + 1
            else:
                mesh.node[idx].up = idx - 1
                mesh.node[idx].dn = -1 if j == count else idx + 1
            mesh.ele[n_conn].cn = (idx - 1, idx)
            n_conn += 1

        poi_1 = geom.croL[i].poi[0]
        poi_2 = geom.croL[i].poi[1]
        node_end = node_start + count
        for j in range(geom.n_junc):
            for k in range(geom.junc[j].n_arm):
                for m in range(geom.n_sec):
                    poi_arm = geom.junc[j].croP[k][m]
                    if poi_1 == poi_arm:
                        geom.junc[j].node[k][m] = node_start
                    if poi_2 == poi_arm:
                        geom.junc[j].node[k][m] = node_end
        n_node = node_start + count + 1


def _get_direction_inil(geom: GeomType, mesh: MeshType, node: int) -> str:
    sec = geom.croL[mesh.node[node].croL].sec
    if mesh.node[node].dn == -1 and sec % 2 == 0:
        return "outward"
    if mesh.node[node].dn == -1 and sec % 2 != 0:
        return "inward"
    if mesh.node[node].up == -1 and sec % 2 == 0:
        return "inward"
    if mesh.node[node].up == -1 and sec % 2 != 0:
        return "outward"
    raise ValueError("Node does not lie on junction")


def _get_section_grid_position(geom: GeomType, sec_idx: int) -> tuple[int, int]:
    return geom.sec.posR[sec_idx], geom.sec.posC[sec_idx]


def _get_neighbor_line_pair(geom: GeomType, mesh: MeshType, node: int) -> list[int]:
    iniL_cur = mesh.node[node].iniL
    n_column = geom.sec.maxC - geom.sec.minC + 1
    if geom.sec.posC[mesh.node[node].sec] < n_column // 2 + 1:
        return [geom.iniL[iniL_cur].neiL[0][0], geom.iniL[iniL_cur].neiL[1][0]]
    return [geom.iniL[iniL_cur].neiL[0][1], geom.iniL[iniL_cur].neiL[1][1]]


def _find_neighbor_nodes_for_junction(geom: GeomType, mesh: MeshType, junc_idx: int, nei_line: list[int]) -> list[int]:
    node_con = [-1 for _ in range(geom.n_sec)]
    for sec_idx in range(geom.n_sec):
        for arm_idx in range(geom.junc[junc_idx].n_arm):
            node_com = geom.junc[junc_idx].node[arm_idx][sec_idx]
            if nei_line[0] == mesh.node[node_com].iniL or nei_line[1] == mesh.node[node_com].iniL:
                node_con[sec_idx] = node_com
    return node_con


def _nodes_form_junction_connection(geom: GeomType, mesh: MeshType, node_cur: int, node_com: int) -> bool:
    sec_cur = geom.croL[mesh.node[node_cur].croL].sec
    sec_com = geom.croL[mesh.node[node_com].croL].sec
    row_cur, col_cur = _get_section_grid_position(geom, sec_cur)
    row_com, col_com = _get_section_grid_position(geom, sec_com)
    if row_cur != row_com:
        return False
    dir_cur = _get_direction_inil(geom, mesh, node_cur)
    dir_com = _get_direction_inil(geom, mesh, node_com)
    if dir_cur == dir_com:
        return col_cur == (geom.sec.maxC - col_com + 1)
    return col_cur == col_com


def _populate_single_junction_connection(geom: GeomType, mesh: MeshType, junc_idx: int, sec_idx: int, arm_idx: int, connect: list[list[int]]) -> None:
    node_cur = geom.junc[junc_idx].node[arm_idx][sec_idx]
    conn_idx = sec_idx * geom.junc[junc_idx].n_arm + arm_idx
    nei_line = _get_neighbor_line_pair(geom, mesh, node_cur)
    if nei_line[0] == -1 and nei_line[1] == -1:
        geom.junc[junc_idx].type_conn[sec_idx] = 1
        connect[conn_idx] = [node_cur, -1]
        return
    for node_com in _find_neighbor_nodes_for_junction(geom, mesh, junc_idx, nei_line):
        if node_com != -1 and _nodes_form_junction_connection(geom, mesh, node_cur, node_com):
            connect[conn_idx] = [node_cur, node_com]
            return


def _set_conn_junction(geom: GeomType, mesh: MeshType) -> None:
    for i in range(geom.n_junc):
        connect = [[-1, -1] for _ in range(geom.junc[i].n_arm * geom.n_sec)]
        for j in range(geom.n_sec):
            for k in range(geom.junc[i].n_arm):
                _populate_single_junction_connection(geom, mesh, i, j, k, connect)
        for j in range(geom.n_sec * geom.junc[i].n_arm):
            geom.junc[i].conn[j][0] = connect[j][0]
            geom.junc[i].conn[j][1] = connect[j][1]
            geom.junc[i].type_conn[j] = 1
    _apply_open_geometry_junction_fallback(geom, mesh)
    _apply_junction_self_connections(geom, mesh)


def _apply_open_geometry_junction_fallback(geom: GeomType, mesh: MeshType) -> None:
    for i in range(geom.n_junc):
        for j in range(geom.n_sec * geom.junc[i].n_arm):
            if geom.junc[i].conn[j][1] != -1:
                continue
            for k in range(j + 1, geom.n_sec * geom.junc[i].n_arm):
                if geom.junc[i].conn[k][1] != -1:
                    continue
                sec_cur = mesh.node[geom.junc[i].conn[j][0]].sec
                sec_com = mesh.node[geom.junc[i].conn[k][0]].sec
                row_cur, col_cur = _get_section_grid_position(geom, sec_cur)
                row_com, col_com = _get_section_grid_position(geom, sec_com)
                if row_cur == row_com and col_cur == col_com:
                    geom.junc[i].conn[j][1] = geom.junc[i].conn[k][0]
                    geom.junc[i].conn[k][1] = geom.junc[i].conn[j][0]
                    break


def _apply_junction_self_connections(geom: GeomType, mesh: MeshType) -> None:
    for i in range(geom.n_junc):
        for j in range(geom.junc[i].n_arm):
            for m in range(geom.n_sec):
                node_cur = geom.junc[i].node[j][m]
                sec_cur = mesh.node[node_cur].sec
                for n in range(geom.n_sec):
                    node_com = geom.junc[i].node[j][n]
                    sec_com = mesh.node[node_com].sec
                    if sec_cur == sec_com:
                        continue
                    if geom.sec.conn and geom.sec.conn[sec_cur] != sec_com:
                        continue
                    idx = m * geom.junc[i].n_arm + j
                    geom.junc[i].conn[idx][0] = node_cur
                    geom.junc[i].conn[idx][1] = node_com
                    geom.junc[i].type_conn[idx] = 2
