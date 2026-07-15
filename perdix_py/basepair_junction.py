from __future__ import annotations

import numpy as np

from . import para
from .data_geom import GeomType, JuncType
from .data_mesh import MeshType
from .basepair_edge_ops import _add_basepair, _increase_edge


def _modify_junction(prob, geom: GeomType, mesh: MeshType) -> None:
    for i in range(geom.n_croP):
        geom.croP[i].ori_pos = geom.croP[i].pos
    mesh.n_mitered = 0
    internal_lines = _find_internal_junction_lines(geom)
    for i in range(geom.n_junc):
        conn, type_conn = _unique_junction_connections(geom.junc[i])
        for j in range(len(conn)):
            node_cur = conn[j][0]
            node_com = conn[j][1]
            if type_conn[j] == 2:
                _apply_vertex_crash_strategy(geom, mesh, node_cur, node_com)
            elif type_conn[j] == 1 and _should_increase_mitered_edge(geom, mesh, i, node_cur, internal_lines):
                _increase_edge(prob, geom, mesh, node_cur, node_com)


def _apply_vertex_crash_strategy(geom: GeomType, mesh: MeshType, node_cur: int, node_com: int) -> None:
    if para.para_vertex_crash == "const":
        mesh.node[node_cur].conn = 2
        mesh.node[node_com].conn = 2
        n_move = _find_xover_nearby(geom, mesh, node_cur, node_com)
        if n_move > 0:
            _increase_basepair(geom, mesh, node_cur, node_com, n_move)
        elif n_move < 0:
            _decrease_basepair(geom, mesh, node_cur, node_com, abs(n_move))
        return
    if para.para_vertex_crash == "mod1":
        if geom.sec.posR[mesh.node[node_cur].sec] >= geom.sec.ref_row:
            mesh.node[node_cur].conn = 2
            mesh.node[node_com].conn = 2
            return
        _make_ghost_node(geom, mesh, node_cur, node_com)
        return
    if para.para_vertex_crash == "mod2":
        n_move = _find_xover_nearby(geom, mesh, node_cur, node_com)
        if n_move == 0:
            mesh.node[node_cur].conn = 4
            mesh.node[node_com].conn = 4
        elif n_move > 0:
            _increase_basepair(geom, mesh, node_cur, node_com, n_move)
        else:
            _decrease_basepair(geom, mesh, node_cur, node_com, abs(n_move))


def _should_increase_mitered_edge(geom: GeomType, mesh: MeshType, junc_idx: int, node_cur: int, internal_lines: set[int]) -> bool:
    if para.para_vertex_design != "mitered":
        return False
    if geom.face or geom.junc[junc_idx].n_arm <= 2:
        return True
    return mesh.node[node_cur].sec == 0 and mesh.node[node_cur].iniL not in internal_lines and geom.junc[junc_idx].ref_ang < 1.2


def _unique_junction_connections(junc: JuncType) -> tuple[list[list[int]], list[int]]:
    conn: list[list[int]] = []
    type_conn: list[int] = []
    for pair, tconn in zip(junc.conn, junc.type_conn):
        if pair[0] == -1 or pair[1] == -1:
            continue
        if any((existing[0], existing[1]) == (pair[0], pair[1]) or (existing[0], existing[1]) == (pair[1], pair[0]) for existing in conn):
            continue
        conn.append([pair[0], pair[1]])
        type_conn.append(tconn)
    return conn, type_conn


def _find_internal_junction_lines(geom: GeomType) -> set[int]:
    if geom.face:
        return set()
    line_arm_counts = [0 for _ in range(geom.n_iniL)]
    for junc in geom.junc:
        if junc.n_arm > 2:
            for line_idx in junc.iniL:
                line_arm_counts[line_idx] += 1
    return {idx for idx, count in enumerate(line_arm_counts) if count >= 2}


def _find_xover_nearby(geom: GeomType, mesh: MeshType, node_cur: int, node_com: int) -> int:
    from .section import section_connection_scaf

    sec_cur = mesh.node[node_cur].sec
    sec_com = mesh.node[node_com].sec
    bp_cur = mesh.node[node_cur].bp
    bp_inside = 0
    bp_outside = 0
    n_inside = 0
    n_outside = 0
    while True:
        b_inside = section_connection_scaf(geom, sec_cur, sec_com, bp_cur + bp_inside)
        b_outside = section_connection_scaf(geom, sec_cur, sec_com, bp_cur + bp_outside)
        if n_inside == 0 and n_outside == 0 and b_inside and b_outside:
            early = _initial_xover_move(geom, mesh, node_cur, sec_cur, sec_com, bp_cur)
            if early is not None:
                return early
        if b_inside:
            n_inside += 1
            if n_inside == 2:
                break
        elif b_outside:
            n_outside += 1
            if n_outside == 1:
                break
        step_inside, step_outside = _xover_search_steps(mesh, node_cur)
        bp_inside += step_inside
        bp_outside += step_outside
    if b_inside:
        return -abs(bp_inside)
    if b_outside:
        return abs(bp_outside)
    return 0


def _initial_xover_move(geom: GeomType, mesh: MeshType, node_cur: int, sec_cur: int, sec_com: int, bp_cur: int) -> int | None:
    from .section import section_connection_scaf

    if _xover_search_starts_inside(mesh, node_cur):
        return -1 if section_connection_scaf(geom, sec_cur, sec_com, bp_cur - 1) else 0
    if _xover_search_starts_outside(mesh, node_cur):
        return 0 if section_connection_scaf(geom, sec_cur, sec_com, bp_cur - 1) else -1
    return None


def _xover_search_starts_inside(mesh: MeshType, node_cur: int) -> bool:
    return (mesh.node[node_cur].up == -1 and mesh.node[node_cur].sec % 2 == 0) or (mesh.node[node_cur].dn == -1 and mesh.node[node_cur].sec % 2 == 1)


def _xover_search_starts_outside(mesh: MeshType, node_cur: int) -> bool:
    return (mesh.node[node_cur].dn == -1 and mesh.node[node_cur].sec % 2 == 0) or (mesh.node[node_cur].up == -1 and mesh.node[node_cur].sec % 2 == 1)


def _xover_search_steps(mesh: MeshType, node_cur: int) -> tuple[int, int]:
    if _xover_search_starts_outside(mesh, node_cur):
        return 1, -1
    if _xover_search_starts_inside(mesh, node_cur):
        return -1, 1
    return 0, 0


def _increase_basepair(geom: GeomType, mesh: MeshType, node_cur: int, node_com: int, n_move: int) -> None:
    endpoints = _basepair_adjustment_endpoints(mesh, node_cur, node_com)
    if endpoints is None:
        return
    node_dn, node_up = endpoints
    vec_dn = np.array(mesh.node[node_dn].pos) - np.array(mesh.node[mesh.node[node_dn].dn].pos)
    vec_up = np.array(mesh.node[node_up].pos) - np.array(mesh.node[mesh.node[node_up].up].pos)
    for _ in range(n_move):
        node_dn = _add_basepair(geom, mesh, node_dn, vec_dn)
        node_up = _add_basepair(geom, mesh, node_up, vec_up)
    mesh.node[node_dn].conn = 4
    mesh.node[node_up].conn = 4


def _decrease_basepair(geom: GeomType, mesh: MeshType, node_cur: int, node_com: int, n_move: int) -> None:
    endpoints = _basepair_adjustment_endpoints(mesh, node_cur, node_com)
    if endpoints is None:
        return
    node_dn, node_up = endpoints
    for _ in range(n_move):
        mesh.node[node_dn].ghost = 1
        mesh.node[node_up].ghost = 1
        node_dn = mesh.node[node_dn].dn
        node_up = mesh.node[node_up].up
    mesh.node[node_dn].conn = 4
    mesh.node[node_up].conn = 4


def _basepair_adjustment_endpoints(mesh: MeshType, node_cur: int, node_com: int) -> tuple[int, int] | None:
    if mesh.node[node_cur].up == -1:
        return node_cur, node_com
    if mesh.node[node_cur].dn == -1:
        return node_com, node_cur
    return None


def _make_ghost_node(geom: GeomType, mesh: MeshType, node_cur: int, node_com: int) -> None:
    walk = _ghost_node_walk_start(mesh, node_cur, node_com)
    if walk is None:
        return
    node_up, node_dn = walk
    mesh.node[node_cur].ghost = 1
    mesh.node[node_com].ghost = 1
    _mark_ghost_chain_until_second_xover(geom, mesh, node_up, node_dn)


def _ghost_node_walk_start(mesh: MeshType, node_cur: int, node_com: int) -> tuple[int, int] | None:
    if mesh.node[node_cur].up == -1:
        node_dn = mesh.node[node_cur].dn
        node_up = mesh.node[node_com].up
    else:
        node_up = mesh.node[node_cur].up
        node_dn = mesh.node[node_com].dn
    if node_up == -1 or node_dn == -1:
        return None
    return node_up, node_dn


def _mark_ghost_chain_until_second_xover(geom: GeomType, mesh: MeshType, node_up: int, node_dn: int) -> None:
    from .section import section_connection_scaf

    n_xover = 0
    while True:
        if node_up == -1 or node_dn == -1:
            return
        if section_connection_scaf(geom, mesh.node[node_up].sec, mesh.node[node_dn].sec, mesh.node[node_up].bp):
            n_xover += 1
            if n_xover == 2:
                break
        mesh.node[node_up].ghost = 1
        mesh.node[node_dn].ghost = 1
        node_up = mesh.node[node_up].up
        node_dn = mesh.node[node_dn].dn
