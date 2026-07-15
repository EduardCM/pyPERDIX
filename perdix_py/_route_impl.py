from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TextIO

import numpy as np

from .data_prob import ProbType
from .data_geom import GeomType
from .data_mesh import MeshType
from .data_dna import BaseType, DNAType
from .math_utils import nint
from . import para


def route_generation(prob: ProbType, geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    """Scaffold routing and crossover placement (partial port)."""
    _init_base_connectivity(mesh, dna)
    _set_base_position(geom, mesh, dna)
    _write_route_bild(prob, geom, mesh, dna, "route1")
    _reconnect_junction(geom, mesh, dna)
    _set_strand_id_scaf(dna)
    _write_route_bild(prob, geom, mesh, dna, "route2")
    _find_centered_scaf_xover(prob, geom, mesh, dna)
    _modify_scaf_xover(geom, mesh, dna)
    _graph_build_origami(mesh, dna)
    if para.para_method_MST == "greedy":
        _make_scaf_origami(dna)
    _set_strand_id_scaf(dna)
    _write_route_bild(prob, geom, mesh, dna, "route3")
    _write_route_bild(prob, geom, mesh, dna, "route4")
    _find_possible_stap_xover(geom, mesh, dna)
    if para.para_set_stap_sxover == "on":
        _set_stap_crossover(dna)
    _write_route_bild(prob, geom, mesh, dna, "route5")
    _set_orientation(mesh, dna)
    _round_route_coordinates(mesh, dna, digits=4)
    _write_crossovers_bild(prob, geom, mesh, dna)
    _write_orientation_bild(prob, mesh, dna)
    # Atomic model output is generated after sequence design (see output.py).


def _round_route_coordinates(mesh: MeshType, dna: DNAType, digits: int = 4) -> None:
    for node in mesh.node:
        node.pos = tuple(round(float(v), digits) for v in node.pos)
        node.ori = tuple(
            tuple(round(float(c), digits) for c in axis)
            for axis in node.ori
        )
    for base_list in (dna.base_scaf, dna.base_stap):
        for base in base_list:
            base.pos = tuple(round(float(v), digits) for v in base.pos)


def _init_base_connectivity(mesh: MeshType, dna: DNAType) -> None:
    dna.n_base_scaf = mesh.n_node
    dna.n_base_stap = mesh.n_node
    from .data_dna import BaseType

    dna.base_scaf = [BaseType() for _ in range(dna.n_base_scaf)]
    dna.base_stap = [BaseType() for _ in range(dna.n_base_stap)]

    for i in range(mesh.n_node):
        base_id = mesh.node[i].id
        dna.base_scaf[i].id = base_id
        dna.base_stap[i].id = base_id
        dna.base_scaf[i].node = base_id
        dna.base_stap[i].node = base_id
        dna.base_scaf[i].up = mesh.node[i].up
        dna.base_scaf[i].dn = mesh.node[i].dn

        if dna.base_scaf[i].up == -1:
            dna.base_stap[i].up = dna.base_scaf[i].dn
            dna.base_stap[i].dn = -1
        elif dna.base_scaf[i].dn == -1:
            dna.base_stap[i].up = -1
            dna.base_stap[i].dn = dna.base_scaf[i].up
        else:
            dna.base_stap[i].up = dna.base_scaf[i].dn
            dna.base_stap[i].dn = dna.base_scaf[i].up

        dna.base_scaf[i].xover = -1
        dna.base_stap[i].xover = -1
        dna.base_scaf[i].across = dna.base_stap[i].id
        dna.base_stap[i].across = dna.base_scaf[i].id
        dna.base_scaf[i].strand = -1
        dna.base_stap[i].strand = -1
        dna.base_scaf[i].pos = (0.0, 0.0, 0.0)
        dna.base_stap[i].pos = (0.0, 0.0, 0.0)


def _set_base_position(geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    if geom.sec.types == "square":
        ang_bp = 360.0 * 3.0 / 32.0
    else:
        ang_bp = 360.0 * 2.0 / 21.0

    def _dmod(a: float, b: float) -> float:
        return math.fmod(a, b)

    for i in range(mesh.n_node):
        if mesh.node[i].sec % 2 == 0:
            ang_init = 180.0 + ang_bp / 2.0
            if geom.sec.dir == 90:
                ang_init = 90.0
            if geom.sec.dir == 150:
                ang_init = 150.0
            if geom.sec.dir == -90:
                ang_init = 270.0
            ang_start = _dmod(ang_init + ang_bp * float(para.para_start_bp_ID), 360.0)
            ang_start = _dmod(ang_start + ang_bp * float(mesh.node[i].bp - 1), 360.0)
            ang_scaf = _dmod(ang_start + para.para_ang_correct, 360.0)
            ang_stap = _dmod(ang_scaf + para.para_ang_minor, 360.0)
        else:
            ang_init = 0.0 + ang_bp / 2.0
            if geom.sec.dir == 90:
                ang_init = 270.0
            if geom.sec.dir == 150:
                ang_init = 330.0
            if geom.sec.dir == -90:
                ang_init = 90.0
            ang_start = _dmod(ang_init + ang_bp * float(para.para_start_bp_ID), 360.0)
            ang_start = _dmod(ang_start + ang_bp * float(mesh.node[i].bp - 1), 360.0)
            ang_scaf = _dmod(ang_start - para.para_ang_correct, 360.0)
            ang_stap = _dmod(ang_scaf - para.para_ang_minor, 360.0)

        if ang_scaf < 0.0 or ang_stap < 0.0:
            ang_scaf = _dmod(360.0 + ang_scaf, 360.0)
            ang_stap = _dmod(360.0 + ang_stap, 360.0)

        t1 = geom.iniL[mesh.node[i].iniL].t[0]
        t2 = geom.iniL[mesh.node[i].iniL].t[1]
        t3 = geom.iniL[mesh.node[i].iniL].t[2]
        rot = (
            (t1[0], t2[0], t3[0]),
            (t1[1], t2[1], t3[1]),
            (t1[2], t2[2], t3[2]),
        )

        sin_scaf = math.sin(-ang_scaf * math.pi / 180.0)
        cos_scaf = math.cos(-ang_scaf * math.pi / 180.0)
        sin_stap = math.sin(-ang_stap * math.pi / 180.0)
        cos_stap = math.cos(-ang_stap * math.pi / 180.0)

        pos_scaf = (0.0, para.para_rad_helix * sin_scaf, para.para_rad_helix * cos_scaf)
        pos_stap = (0.0, para.para_rad_helix * sin_stap, para.para_rad_helix * cos_stap)
        base_pos = mesh.node[i].pos

        def _matmul(rot_mat: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], vec: tuple[float, float, float]) -> tuple[float, float, float]:
            return (
                rot_mat[0][0] * vec[0] + rot_mat[0][1] * vec[1] + rot_mat[0][2] * vec[2],
                rot_mat[1][0] * vec[0] + rot_mat[1][1] * vec[1] + rot_mat[1][2] * vec[2],
                rot_mat[2][0] * vec[0] + rot_mat[2][1] * vec[1] + rot_mat[2][2] * vec[2],
            )

        v_scaf = _matmul(rot, pos_scaf)
        v_stap = _matmul(rot, pos_stap)
        dna.base_scaf[i].pos = (base_pos[0] + v_scaf[0], base_pos[1] + v_scaf[1], base_pos[2] + v_scaf[2])
        dna.base_stap[i].pos = (base_pos[0] + v_stap[0], base_pos[1] + v_stap[1], base_pos[2] + v_stap[2])


def _reconnect_junction(geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    for i in range(geom.n_junc):
        join: list[tuple[int, int]] = []
        for j in range(geom.n_sec * geom.junc[i].n_arm):
            a = geom.junc[i].conn[j][0]
            b = geom.junc[i].conn[j][1]
            if a == -1 or b == -1:
                continue
            if (a, b) in join or (b, a) in join:
                continue
            join.append((a, b))

        for node_cur, node_com in join:
            # neighbor connection only when no section connection
            sec_cur = mesh.node[node_cur].sec
            if geom.sec.conn[sec_cur] == -1:
                n_un_scaf = _connect_scaf(mesh, dna, node_cur, node_com)
                n_un_stap = _connect_stap(mesh, dna, node_cur, node_com)
                geom.junc[i].n_un_scaf += n_un_scaf
                geom.junc[i].n_un_stap += n_un_stap

    # self connection in same section
    for i in range(geom.n_junc):
        for j in range(geom.junc[i].n_arm):
            for m in range(geom.n_sec):
                node_cur = geom.junc[i].node[j][m]
                sec_cur = mesh.node[node_cur].sec
                for n in range(m + 1, geom.n_sec):
                    node_com = geom.junc[i].node[j][n]
                    sec_com = mesh.node[node_com].sec
                    if sec_cur == sec_com:
                        continue
                    if geom.sec.conn and geom.sec.conn[sec_cur] != sec_com:
                        continue
                    n_un_scaf = _connect_scaf(mesh, dna, node_cur, node_com)
                    geom.junc[i].n_un_scaf += n_un_scaf


def _connect_scaf(mesh: MeshType, dna: DNAType, node_cur: int, node_com: int) -> int:
    cur = node_cur
    com = node_com
    pos_cur = np.array(dna.base_scaf[cur].pos)
    pos_com = np.array(dna.base_scaf[com].pos)

    count = 0
    if para.para_unpaired_scaf == "on":
        length = float(np.linalg.norm(pos_cur - pos_com))
        count = nint(length / para.para_dist_pp) - 1
        if count < 0:
            count = 0

    if dna.base_scaf[cur].up == -1 and dna.base_scaf[com].dn == -1:
        dna.base_scaf[cur].up = dna.base_scaf[com].id
        dna.base_scaf[com].dn = dna.base_scaf[cur].id
        for i in range(1, count + 1):
            pos = (i * pos_com + (count + 1 - i) * pos_cur) / (count + 1)
            ttt = _add_nucleotide(dna, dna.base_scaf, pos, is_scaf=True)
            dna.base_scaf[ttt].up = dna.base_scaf[com].id
            dna.base_scaf[ttt].dn = dna.base_scaf[cur].id
            dna.base_scaf[cur].up = dna.base_scaf[ttt].id
            dna.base_scaf[com].dn = dna.base_scaf[ttt].id
            cur = ttt
    elif dna.base_scaf[cur].dn == -1 and dna.base_scaf[com].up == -1:
        dna.base_scaf[cur].dn = dna.base_scaf[com].id
        dna.base_scaf[com].up = dna.base_scaf[cur].id
        for i in range(1, count + 1):
            pos = (i * pos_com + (count + 1 - i) * pos_cur) / (count + 1)
            ttt = _add_nucleotide(dna, dna.base_scaf, pos, is_scaf=True)
            dna.base_scaf[ttt].up = dna.base_scaf[cur].id
            dna.base_scaf[ttt].dn = dna.base_scaf[com].id
            dna.base_scaf[cur].dn = dna.base_scaf[ttt].id
            dna.base_scaf[com].up = dna.base_scaf[ttt].id
            cur = ttt
    else:
        return 0
    return count


def _connect_stap(mesh: MeshType, dna: DNAType, node_cur: int, node_com: int) -> int:
    cur = node_cur
    com = node_com
    pos_cur = np.array(dna.base_stap[cur].pos)
    pos_com = np.array(dna.base_stap[com].pos)

    if para.para_n_base_tn == -1:
        length = float(np.linalg.norm(pos_cur - pos_com))
        n_poly_tn = nint(length / para.para_dist_pp) - 1
    else:
        n_poly_tn = para.para_n_base_tn
    if n_poly_tn < 0:
        n_poly_tn = 0

    if dna.base_stap[cur].up == -1 and dna.base_stap[com].dn == -1:
        dna.base_stap[cur].up = dna.base_stap[com].id
        dna.base_stap[com].dn = dna.base_stap[cur].id
        for i in range(1, n_poly_tn + 1):
            pos = (i * pos_com + (n_poly_tn + 1 - i) * pos_cur) / (n_poly_tn + 1)
            ttt = _add_nucleotide(dna, dna.base_stap, pos, is_scaf=False)
            dna.base_stap[ttt].up = dna.base_stap[com].id
            dna.base_stap[ttt].dn = dna.base_stap[cur].id
            dna.base_stap[cur].up = dna.base_stap[ttt].id
            dna.base_stap[com].dn = dna.base_stap[ttt].id
            cur = ttt
    elif dna.base_stap[cur].dn == -1 and dna.base_stap[com].up == -1:
        dna.base_stap[cur].dn = dna.base_stap[com].id
        dna.base_stap[com].up = dna.base_stap[cur].id
        for i in range(1, n_poly_tn + 1):
            pos = (i * pos_com + (n_poly_tn + 1 - i) * pos_cur) / (n_poly_tn + 1)
            ttt = _add_nucleotide(dna, dna.base_stap, pos, is_scaf=False)
            dna.base_stap[ttt].up = dna.base_stap[cur].id
            dna.base_stap[ttt].dn = dna.base_stap[com].id
            dna.base_stap[cur].dn = dna.base_stap[ttt].id
            dna.base_stap[com].up = dna.base_stap[ttt].id
            cur = ttt
    else:
        return 0
    return n_poly_tn


def _add_nucleotide(dna: DNAType, base_list, pos: np.ndarray, is_scaf: bool) -> int:
    from .data_dna import BaseType

    new_id = len(base_list)
    b = BaseType()
    b.id = new_id
    b.node = -1
    b.up = -1
    b.dn = -1
    b.xover = -1
    b.across = -1
    b.strand = -1
    b.pos = (float(pos[0]), float(pos[1]), float(pos[2]))
    base_list.append(b)
    if is_scaf:
        dna.n_base_scaf += 1
    else:
        dna.n_base_stap += 1
    return new_id


def _set_strand_id_scaf(dna: DNAType) -> None:
    visit = [False for _ in range(dna.n_base_scaf)]
    n_strand = 0
    for i in range(dna.n_base_scaf):
        if visit[i]:
            continue
        n_strand += 1
        start = dna.base_scaf[i].id
        search = start
        visit[i] = True
        dna.base_scaf[i].strand = n_strand
        count = 0
        while True:
            search = dna.base_scaf[search].up
            if search == -1:
                break
            visit[search] = True
            if start == search:
                break
            dna.base_scaf[search].strand = n_strand
            count += 1
            if count > dna.n_base_scaf:
                raise ValueError("dna.n_base_scaf exceeds count")
    dna.n_scaf = n_strand


def _find_centered_scaf_xover(prob: ProbType, geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    """Centered scaffold crossover search (ported core logic)."""
    from .section import section_connection_scaf

    min_bp, max_bp = _scaffold_crossover_bp_bounds(geom, mesh)
    b_xover = [[False for _ in range(geom.n_sec)] for _ in range(geom.n_croL)]

    dna.n_xover_scaf = 0
    pre_iniL = -1
    n_gap = 0
    for i, j in _iter_centered_scaffold_xover_candidates(mesh):
        node_i = mesh.node[i]
        node_j = mesh.node[j]
        sec_cur = node_i.sec
        sec_com = node_j.sec
        croL_cur = node_i.croL
        croL_com = node_j.croL
        id_bp = node_i.bp
        min_bp1 = min_bp[croL_cur]
        max_bp1 = max_bp[croL_cur]
        min_bp2 = min_bp[croL_com]
        max_bp2 = max_bp[croL_com]

        if _skip_centered_scaffold_candidate(min_bp1, max_bp1, id_bp):
            continue
        if b_xover[croL_cur][sec_com]:
            continue
        if b_xover[croL_com][sec_cur]:
            continue

        if not section_connection_scaf(geom, sec_cur, sec_com, id_bp):
            continue

        node_cur = node_i.id
        node_com = node_j.id
        choice = _CenteredSplitChoice(
            node_cur=node_cur,
            node_com=node_com,
            max_gap=n_gap,
            max_cur=node_cur,
            max_com=node_com,
        )
        bounds = _ScaffoldSplitBounds(
            min_bp1=min_bp1,
            max_bp1=max_bp1,
            min_bp2=min_bp2,
            max_bp2=max_bp2,
        )
        iniL = node_i.iniL

        if (pre_iniL == -1 or iniL != pre_iniL) and sec_cur == 0:
            if prob.sel_edge_sec != 1 and (sec_cur != 0 or sec_com != 5):
                continue
            pre_iniL = iniL
            node_cur, node_com = _select_initial_centered_scaffold_split(
                geom, mesh, choice, bounds
            )
        else:
            node_cur, node_com, n_gap = _select_best_centered_scaffold_split(
                geom, mesh, dna, choice, bounds
            )

        b_xover[croL_cur][sec_com] = True
        neighbor_pair = _find_scaffold_neighbor_pair(geom, mesh, node_cur, node_com)
        if neighbor_pair is None:
            continue
        _apply_scaffold_xover_pair(dna, node_cur, node_com, neighbor_pair)


def _scaffold_crossover_bp_bounds(geom: GeomType, mesh: MeshType) -> tuple[list[int], list[int]]:
    min_bp = [10**9 for _ in range(geom.n_croL)]
    max_bp = [-10**9 for _ in range(geom.n_croL)]
    for node in mesh.node:
        croL_cur = node.croL
        id_bp = node.bp
        if id_bp > max_bp[croL_cur]:
            max_bp[croL_cur] = id_bp
        if id_bp < min_bp[croL_cur]:
            min_bp[croL_cur] = id_bp
    return min_bp, max_bp


def _iter_centered_scaffold_xover_candidates(mesh: MeshType):
    for i in range(mesh.n_node):
        for j in range(i + 1, mesh.n_node):
            if mesh.node[i].bp != mesh.node[j].bp:
                continue
            if mesh.node[i].iniL != mesh.node[j].iniL:
                continue
            if mesh.node[i].sec == mesh.node[j].sec:
                continue
            yield i, j


def _skip_centered_scaffold_candidate(min_bp: int, max_bp: int, id_bp: int) -> bool:
    ave_bp = (min_bp + max_bp) // 2 - 2
    return (
        id_bp < ave_bp
        or id_bp > max_bp - para.para_gap_xover_bound_scaf
    )


@dataclass(frozen=True)
class _ScaffoldSplitBounds:
    min_bp1: int
    max_bp1: int
    min_bp2: int
    max_bp2: int


@dataclass(frozen=True)
class _ScaffoldNeighborPair:
    b_nei_up: bool
    b_nei_dn: bool
    up_cur: int
    up_com: int
    dn_cur: int
    dn_com: int


@dataclass
class _CenteredSplitChoice:
    node_cur: int
    node_com: int
    max_gap: int
    max_cur: int
    max_com: int


@dataclass(frozen=True)
class _ScaffoldSplitRequest:
    cur: int
    com: int
    bounds: _ScaffoldSplitBounds
    step: int


def _select_initial_centered_scaffold_split(
    geom: GeomType,
    mesh: MeshType,
    choice: _CenteredSplitChoice,
    bounds: _ScaffoldSplitBounds,
) -> tuple[int, int]:
    if geom.sec.maxR == 1 and geom.sec.maxC == 2:
        step = 1 if para.para_set_xover_scaf == "center" else 5
    else:
        step = 1 if para.para_set_xover_scaf == "center" else 3
    b_fail, split_cur, split_com = _split_centered_scaf_xover(
        geom,
        mesh,
        _ScaffoldSplitRequest(
            cur=choice.node_cur,
            com=choice.node_com,
            bounds=bounds,
            step=step,
        ),
    )
    if step == 3 and b_fail:
        _b_fail, split_cur, split_com = _split_centered_scaf_xover(
            geom,
            mesh,
            _ScaffoldSplitRequest(
                cur=split_cur,
                com=split_com,
                bounds=bounds,
                step=step + 1,
            ),
        )
    return split_cur, split_com


def _select_best_centered_scaffold_split(
    geom: GeomType,
    mesh: MeshType,
    dna: DNAType,
    choice: _CenteredSplitChoice,
    bounds: _ScaffoldSplitBounds,
) -> tuple[int, int, int]:
    n_gap = choice.max_gap
    split_cur = choice.node_cur
    split_com = choice.node_com
    for k in range(1, 6):
        split_cur = choice.node_cur
        split_com = choice.node_com
        b_fail, split_cur, split_com = _split_centered_scaf_xover(
            geom,
            mesh,
            _ScaffoldSplitRequest(
                cur=split_cur,
                com=split_com,
                bounds=bounds,
                step=k,
            ),
        )
        n_gap = _check_nei_xover(geom, mesh, dna, split_cur, split_com)
        if not b_fail and n_gap == para.para_gap_xover_two_scaf:
            break
        if not b_fail and n_gap != para.para_gap_xover_two_scaf:
            if n_gap >= choice.max_gap:
                choice.max_gap = n_gap
                choice.max_cur = split_cur
                choice.max_com = split_com
        if k == 5:
            split_cur = choice.max_cur
            split_com = choice.max_com
    return split_cur, split_com, n_gap


def _find_scaffold_neighbor_pair(
    geom: GeomType,
    mesh: MeshType,
    node_cur: int,
    node_com: int,
) -> _ScaffoldNeighborPair | None:
    from .section import section_connection_scaf

    if node_cur == -1 or node_com == -1:
        return None
    up_cur = mesh.node[node_cur].up
    up_com = mesh.node[node_com].dn
    if up_cur == -1 or up_com == -1:
        return None
    sec_cur = mesh.node[up_cur].sec
    sec_com = mesh.node[up_com].sec
    id_bp = mesh.node[up_cur].bp
    b_nei_up = section_connection_scaf(geom, sec_cur, sec_com, id_bp)

    dn_cur = mesh.node[node_cur].dn
    dn_com = mesh.node[node_com].up
    if dn_cur == -1 or dn_com == -1:
        return None
    sec_cur = mesh.node[dn_cur].sec
    sec_com = mesh.node[dn_com].sec
    id_bp = mesh.node[dn_cur].bp
    b_nei_dn = section_connection_scaf(geom, sec_cur, sec_com, id_bp)

    return _ScaffoldNeighborPair(
        b_nei_up=b_nei_up,
        b_nei_dn=b_nei_dn,
        up_cur=up_cur,
        up_com=up_com,
        dn_cur=dn_cur,
        dn_com=dn_com,
    )


def _apply_scaffold_xover_pair(
    dna: DNAType,
    node_cur: int,
    node_com: int,
    neighbors: _ScaffoldNeighborPair,
) -> None:
    dna.n_xover_scaf += 2
    dna.base_scaf[node_cur].xover = dna.base_scaf[node_com].id
    dna.base_scaf[node_com].xover = dna.base_scaf[node_cur].id

    if neighbors.b_nei_up:
        dna.base_scaf[neighbors.up_cur].xover = dna.base_scaf[neighbors.up_com].id
        dna.base_scaf[neighbors.up_com].xover = dna.base_scaf[neighbors.up_cur].id
    elif neighbors.b_nei_dn:
        dna.base_scaf[neighbors.dn_cur].xover = dna.base_scaf[neighbors.dn_com].id
        dna.base_scaf[neighbors.dn_com].xover = dna.base_scaf[neighbors.dn_cur].id


def _modify_scaf_xover(geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    n_xover1 = dna.n_xover_scaf
    for i in range(dna.n_base_scaf):
        xover = dna.base_scaf[i].xover
        if xover != -1:
            if dna.base_scaf[i].strand == dna.base_scaf[xover].strand:
                dna.n_xover_scaf -= 1
                _clear_scaffold_xover_pair(dna, i, xover)

    n_xover2 = dna.n_xover_scaf
    check: list[dict[str, tuple[int, int]]] = []

    for i in range(dna.n_base_scaf):
        base = dna.base_scaf[i].id
        xover = dna.base_scaf[base].xover
        if xover == -1:
            continue
        up_base = dna.base_scaf[base].up
        dn_base = dna.base_scaf[base].dn
        up_xover = dna.base_scaf[up_base].xover if up_base != -1 else -1
        dn_xover = dna.base_scaf[dn_base].xover if dn_base != -1 else -1

        if xover != -1 and (up_xover != -1 or dn_xover != -1):
            strd_cur = dna.base_scaf[base].strand
            strd_com = dna.base_scaf[xover].strand

            if not _has_scaffold_strand_pair_record(check, strd_cur, strd_com):
                entry = {
                    "base1": (base, xover),
                    "strd": (dna.base_scaf[base].strand, dna.base_scaf[xover].strand),
                    "base2": (up_base, up_xover) if up_xover != -1 else (dn_base, dn_xover),
                }
                check.append(entry)
            else:
                if not _has_scaffold_xover_record(check, base, xover):
                    dna.n_xover_scaf -= 2
                    _clear_scaffold_xover_pair(dna, base, xover)
                    if up_xover != -1:
                        _clear_scaffold_xover_pair(dna, up_base, up_xover)
                    elif dn_xover != -1:
                        _clear_scaffold_xover_pair(dna, dn_base, dn_xover)


def _clear_scaffold_xover_pair(dna: DNAType, base_a: int, base_b: int) -> None:
    dna.base_scaf[base_a].xover = -1
    dna.base_scaf[base_b].xover = -1


def _has_scaffold_strand_pair_record(
    records: list[dict[str, tuple[int, int]]],
    strand_a: int,
    strand_b: int,
) -> bool:
    for item in records:
        if (item["strd"][0] == strand_a and item["strd"][1] == strand_b) or (
            item["strd"][1] == strand_a and item["strd"][0] == strand_b
        ):
            return True
    return False


def _has_scaffold_xover_record(
    records: list[dict[str, tuple[int, int]]],
    base: int,
    xover: int,
) -> bool:
    for item in records:
        if (base, xover) == item["base1"] or (xover, base) == item["base1"]:
            return True
        if (base, xover) == item["base2"] or (xover, base) == item["base2"]:
            return True
    return False


def _make_scaf_origami(dna: DNAType) -> None:
    if dna.n_scaf <= 0:
        return
    b_visit = [False for _ in range(dna.n_scaf)]
    id_entry = [-1 for _ in range(dna.n_scaf)]

    base = dna.base_scaf[0].id
    strand = dna.base_scaf[base].strand
    b_visit[strand - 1] = True
    id_entry[strand - 1] = base

    n_xover = 0
    for _ in range(1, dna.n_base_scaf):
        xover = dna.base_scaf[base].xover
        start = id_entry[strand - 1]
        up = dna.base_scaf[base].up

        if xover == -1:
            base = dna.base_scaf[base].up
            strand = dna.base_scaf[base].strand
        elif xover != -1 and not b_visit[dna.base_scaf[xover].strand - 1]:
            dna.base_scaf[base].up = xover
            dna.base_scaf[xover].dn = base
            base = dna.base_scaf[base].xover
            strand = dna.base_scaf[base].strand
            id_entry[strand - 1] = base
        elif xover != -1 and start == up:
            dna.base_scaf[base].up = xover
            dna.base_scaf[xover].dn = base
            base = dna.base_scaf[base].xover
            strand = dna.base_scaf[base].strand
        else:
            if xover != -1 and xover != dna.base_scaf[base].up and xover != dna.base_scaf[base].dn:
                _clear_scaffold_xover_pair(dna, base, xover)
                n_xover += 1
            base = dna.base_scaf[base].up
            strand = dna.base_scaf[base].strand

        b_visit[strand - 1] = True

    dna.n_xover_scaf -= n_xover


def _check_nei_xover(geom: GeomType, mesh: MeshType, dna: DNAType, cur: int, com: int) -> int:
    sec1 = mesh.node[cur].sec
    sec2 = mesh.node[com].sec
    if mesh.node[cur].up == -1 or mesh.node[cur].dn == -1:
        return 0
    from .section import section_connection_scaf

    b_nei_up = section_connection_scaf(geom, sec1, sec2, mesh.node[mesh.node[cur].up].bp)
    b_nei_dn = section_connection_scaf(geom, sec1, sec2, mesh.node[mesh.node[cur].dn].bp)
    if b_nei_up:
        up_node1 = mesh.node[cur].up
        up_node2 = mesh.node[com].dn
        dn_node1 = cur
        dn_node2 = com
    elif b_nei_dn:
        up_node1 = cur
        up_node2 = com
        dn_node1 = mesh.node[cur].dn
        dn_node2 = mesh.node[com].up
    else:
        return 0

    n_gap = 0
    for _ in range(para.para_gap_xover_two_scaf):
        if up_node1 == -1 or up_node2 == -1 or dn_node1 == -1 or dn_node2 == -1:
            break
        up_node1 = mesh.node[up_node1].up
        up_node2 = mesh.node[up_node2].dn
        dn_node1 = mesh.node[dn_node1].dn
        dn_node2 = mesh.node[dn_node2].up
        if up_node1 == -1 or up_node2 == -1 or dn_node1 == -1 or dn_node2 == -1:
            break
        if (
            dna.base_scaf[up_node1].xover != -1
            or dna.base_scaf[up_node2].xover != -1
            or dna.base_scaf[dn_node1].xover != -1
            or dna.base_scaf[dn_node2].xover != -1
        ):
            break
        n_gap += 1
    return n_gap


def _split_centered_scaf_xover(
    geom: GeomType,
    mesh: MeshType,
    request: _ScaffoldSplitRequest,
) -> tuple[bool, int, int]:
    node1 = request.cur
    node2 = request.com
    sec1 = mesh.node[node1].sec
    sec2 = mesh.node[node2].sec
    b_fail = False

    n_cross, direction = _scaffold_split_strategy(request.step)

    if direction == "down":
        if mesh.node[node1].dn == -1 or mesh.node[node2].up == -1:
            return True, request.cur, request.com
        node1, node2 = _advance_scaffold_split_pair(mesh, node1, node2, direction, steps=2)
        if node1 == -1 or node2 == -1:
            return True, request.cur, request.com
        b_fail, node1, node2 = _scan_scaffold_split_direction(
            geom,
            mesh,
            node1,
            node2,
            sec1,
            sec2,
            n_cross,
            direction,
            request.bounds.min_bp1,
            request.bounds.max_bp1,
            request.bounds.min_bp2,
            request.bounds.max_bp2,
        )
    elif direction == "up":
        if mesh.node[node1].up == -1 or mesh.node[node2].dn == -1:
            return True, request.cur, request.com
        node1, node2 = _advance_scaffold_split_pair(mesh, node1, node2, direction, steps=2)
        if node1 == -1 or node2 == -1:
            return True, request.cur, request.com
        b_fail, node1, node2 = _scan_scaffold_split_direction(
            geom,
            mesh,
            node1,
            node2,
            sec1,
            sec2,
            n_cross,
            direction,
            request.bounds.min_bp1,
            request.bounds.max_bp1,
            request.bounds.min_bp2,
            request.bounds.max_bp2,
        )
    else:
        b_fail = False

    if not b_fail:
        return False, node1, node2
    return True, request.cur, request.com


def _scaffold_split_strategy(step: int) -> tuple[int, str]:
    if para.para_set_xover_scaf == "split":
        if step == 1:
            return 3, "down"
        if step == 2:
            return 1, "down"
        if step == 3:
            return 3, "up"
        if step == 4:
            return 1, "up"
        return 0, "center"
    if step == 1:
        return 0, "center"
    if step == 2:
        return 1, "down"
    if step == 3:
        return 1, "up"
    if step == 4:
        return 3, "down"
    return 3, "up"


def _scaffold_split_hits_boundary_gap(
    id_bp1: int,
    id_bp2: int,
    min_bp1: int,
    max_bp1: int,
    min_bp2: int,
    max_bp2: int,
) -> bool:
    return (
        id_bp1 < min_bp1 + para.para_gap_xover_bound_scaf
        or id_bp1 > max_bp1 - para.para_gap_xover_bound_scaf
        or id_bp2 < min_bp2 + para.para_gap_xover_bound_scaf
        or id_bp2 > max_bp2 - para.para_gap_xover_bound_scaf
    )


def _scan_scaffold_split_direction(
    geom: GeomType,
    mesh: MeshType,
    node1: int,
    node2: int,
    sec1: int,
    sec2: int,
    n_cross: int,
    direction: str,
    min_bp1: int,
    max_bp1: int,
    min_bp2: int,
    max_bp2: int,
) -> tuple[bool, int, int]:
    from .section import section_connection_scaf

    id_bp1 = mesh.node[node1].bp
    id_bp2 = mesh.node[node2].bp
    n_skip = 0
    while True:
        if _scaffold_split_hits_boundary_gap(id_bp1, id_bp2, min_bp1, max_bp1, min_bp2, max_bp2):
            return True, node1, node2
        if section_connection_scaf(geom, sec1, sec2, id_bp1):
            if n_skip == n_cross:
                return False, node1, node2
            n_skip += 1
        node1, node2 = _advance_scaffold_split_pair(mesh, node1, node2, direction, steps=1)
        if node1 == -1 or node2 == -1:
            return True, node1, node2
        id_bp1 = mesh.node[node1].bp
        id_bp2 = mesh.node[node2].bp


def _advance_scaffold_split_pair(
    mesh: MeshType,
    node1: int,
    node2: int,
    direction: str,
    steps: int,
) -> tuple[int, int]:
    for _ in range(steps):
        if node1 == -1 or node2 == -1:
            return -1, -1
        if direction == "down":
            node1 = mesh.node[node1].dn
            node2 = mesh.node[node2].up
        else:
            node1 = mesh.node[node1].up
            node2 = mesh.node[node2].dn
    return node1, node2


def _graph_build_origami(mesh: MeshType, dna: DNAType) -> None:
    """Build dual graph and compute MST (Fortran port); remove non-spanning crossovers."""
    n_node = dna.n_scaf
    if n_node <= 1:
        return

    # Initialize node centers and edge lists (1-based indexing)
    pos_node: list[np.ndarray] = [np.zeros(3, dtype=float) for _ in range(n_node + 1)]
    n_base = [0] * (n_node + 1)
    tail: list[int] = [0]
    head: list[int] = [0]
    cost: list[int] = [0]

    for i in range(dna.n_base_scaf):
        strand = dna.base_scaf[i].strand
        xover = dna.base_scaf[i].xover
        if strand <= 0 or strand > n_node:
            continue
        n_base[strand] += 1
        pos_node[strand] += np.array(dna.base_scaf[i].pos, dtype=float)

        if xover == -1:
            continue
        strand_xover = dna.base_scaf[xover].strand
        if strand_xover <= 0 or strand_xover == strand or strand_xover > n_node:
            continue

        # Check duplicate edges (undirected)
        b_add = True
        for j in range(1, len(tail)):
            if (tail[j] == strand and head[j] == strand_xover) or (tail[j] == strand_xover and head[j] == strand):
                b_add = False
                break
        if not b_add:
            continue

        tail.append(strand)
        head.append(strand_xover)
        edge_cost = 1
        if para.para_weight_edge == "on":
            node1 = dna.base_scaf[i].id
            node2 = dna.base_scaf[i].xover
            sec1 = mesh.node[node1].sec
            sec2 = mesh.node[node2].sec
            con_pri1 = (0, 1)
            con_pri2 = (0, 1)
            con_span = (0, 1)

            def _match(pair: tuple[int, int]) -> bool:
                return (sec1 == pair[0] and sec2 == pair[1]) or (sec1 == pair[1] and sec2 == pair[0])

            if _match(con_pri1):
                edge_cost = 1
            elif _match(con_pri2):
                edge_cost = 1
            elif _match(con_span):
                edge_cost = 2
            else:
                edge_cost = 3
        cost.append(edge_cost)

    n_edge = len(tail) - 1
    if n_edge == 0:
        return

    for i in range(1, n_node + 1):
        if n_base[i] > 0:
            pos_node[i] = pos_node[i] / float(n_base[i])

    if para.para_method_sort == "quick":
        _spantree_quick_sort(tail, head, cost)

    tree = _spantree_prim_algorithm_1(tail, head, cost, n_node, n_edge)
    dna.graph_pos_node = [tuple(p.tolist()) for p in pos_node]
    dna.graph_tail = tail
    dna.graph_head = head
    dna.graph_tree = tree
    _graph_delete_scaf_xover(dna, tail, head, tree)


def _graph_delete_scaf_xover(dna: DNAType, tail: list[int], head: list[int], tree: list[int]) -> None:
    keep = set()
    for eid in tree:
        u = tail[eid]
        v = head[eid]
        keep.add((u, v))
        keep.add((v, u))

    for i in range(dna.n_base_scaf):
        xover = dna.base_scaf[i].xover
        if xover == -1:
            continue
        strand = dna.base_scaf[i].strand
        strand_xover = dna.base_scaf[xover].strand
        if (strand, strand_xover) not in keep:
            dna.base_scaf[i].xover = -1
            dna.base_scaf[xover].xover = -1
            dna.n_xover_scaf -= 1
        else:
            up = dna.base_scaf[i].up
            if up == -1:
                continue
            up_xover = dna.base_scaf[up].xover
            if up_xover != -1 and dna.base_scaf[xover].strand == dna.base_scaf[up_xover].strand:
                dna.base_scaf[i].up = xover
                dna.base_scaf[up].dn = up_xover


def _spantree_quick_sort(tail: list[int], head: list[int], cost: list[int]) -> None:
    n_edge = len(tail) - 1
    sl = [0] * (2 * n_edge + 1)
    sr = [0] * (2 * n_edge + 1)
    s = 1
    sl[s] = 1
    sr[s] = n_edge
    while s != 0:
        l = sl[s]
        r = sr[s]
        s -= 1
        while True:
            i = l
            j = r
            ha = (l + r) // 2
            me = cost[ha]
            while True:
                if cost[i] < me:
                    i += 1
                    continue
                while cost[j] > me:
                    j -= 1
                if i <= j:
                    head[i], head[j] = head[j], head[i]
                    tail[i], tail[j] = tail[j], tail[i]
                    cost[i], cost[j] = cost[j], cost[i]
                    i += 1
                    j -= 1
                if i <= j:
                    continue
                break
            if i < r:
                s += 1
                sl[s] = i
                sr[s] = r
            r = j
            if l < r:
                continue
            break


def _spantree_prim_algorithm_1(
    tail: list[int],
    head: list[int],
    cost: list[int],
    n_node: int,
    n_edge: int,
) -> list[int]:
    maxint = 2**31 - 1
    up = [0] * (n_node + 1)
    down = [0] * (n_node + 1)
    lsup = [0] * (2 * n_edge + 1)
    lsdn = [0] * (2 * n_edge + 1)
    node = [0] * (n_node + 1)
    posi = [0] * (n_node + 1)

    def idx(arc: int) -> int:
        return arc + n_edge

    for k in range(1, n_edge + 1):
        i = tail[k]
        if up[i] != 0:
            x = down[i]
            lsup[idx(x)] = k
            lsdn[idx(k)] = x
            down[i] = k
        else:
            up[i] = k
            down[i] = k

        kk = -k
        i = head[k]
        if up[i] != 0:
            x = down[i]
            lsup[idx(x)] = kk
            lsdn[idx(kk)] = x
            down[i] = kk
        else:
            up[i] = kk
            down[i] = kk

    tree: list[int] = []
    fi = 1
    node[fi] = 1
    posi[1] = 1
    es = 0

    while es < n_node - 1:
        me = maxint
        am = 0
        nm = 0
        for j in range(1, fi + 1):
            i = node[j]
            x = up[i]
            if x == 0:
                continue
            wo = cost[abs(x)]
            if wo < me:
                me = wo
                am = x
                h = abs(am)
                if head[h] != i:
                    nm = head[h]
                else:
                    nm = tail[h]
        if am == 0:
            break
        es += 1
        k = abs(am)
        tree.append(k)
        fi += 1
        node[fi] = nm
        posi[nm] = fi
        x = up[nm]
        while x != 0:
            if x > 0:
                su = head[x]
                if posi[su] > 0:
                    fi = _spantree_remove(nm, x, up, down, lsup, lsdn, node, posi, fi, n_edge)
                    fi = _spantree_remove(su, -x, up, down, lsup, lsdn, node, posi, fi, n_edge)
            else:
                su = tail[-x]
                if posi[su] > 0:
                    fi = _spantree_remove(su, -x, up, down, lsup, lsdn, node, posi, fi, n_edge)
                    fi = _spantree_remove(nm, x, up, down, lsup, lsdn, node, posi, fi, n_edge)
            x = lsup[idx(x)]
    return tree


def _spantree_remove(
    ant: int,
    arc: int,
    up: list[int],
    down: list[int],
    listup: list[int],
    listdn: list[int],
    node: list[int],
    posicao: list[int],
    fim: int,
    n_edge: int,
) -> int:
    def idx(arc_val: int) -> int:
        return arc_val + n_edge

    if up[ant] != arc:
        y = listdn[idx(arc)]
        if listup[idx(arc)] != 0:
            z = listup[idx(arc)]
            listup[idx(y)] = z
            listdn[idx(z)] = y
        else:
            listup[idx(y)] = 0
            down[ant] = y
    else:
        if down[ant] != arc:
            x = listup[idx(arc)]
            up[ant] = x
        else:
            up[ant] = 0
            down[ant] = 0
            x = posicao[ant]
            y = node[fim]
            node[x] = y
            posicao[y] = x
            fim -= 1
    return fim


def _find_possible_stap_xover(geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    min_bp, max_bp = _staple_crossover_bp_bounds(geom, mesh)

    dna.n_xover_stap = 0
    dna.n_sxover_stap = 0
    for i, j in _iter_staple_xover_candidates(mesh, dna):
        neighbor_pair = _accepted_staple_xover_neighbor_pair(geom, mesh, dna, min_bp, max_bp, i, j)
        if neighbor_pair is None:
            continue

        b_nei_up, up_cur, up_com, dn_cur, dn_com = neighbor_pair
        _apply_staple_xover_pair(dna, i, j, b_nei_up, up_cur, up_com, dn_cur, dn_com)


def _accepted_staple_xover_neighbor_pair(
    geom: GeomType,
    mesh: MeshType,
    dna: DNAType,
    min_bp: list[int],
    max_bp: list[int],
    node_cur: int,
    node_com: int,
) -> tuple[bool, int, int, int, int] | None:
    from .section import section_connection_stap

    sec_cur = mesh.node[node_cur].sec
    sec_com = mesh.node[node_com].sec
    croL_cur = mesh.node[node_cur].croL
    croL_com = mesh.node[node_com].croL
    id_bp = mesh.node[node_cur].bp

    if _staple_xover_hits_boundary_gap(min_bp, max_bp, croL_cur, croL_com, id_bp):
        return None
    if not section_connection_stap(geom, sec_cur, sec_com, id_bp):
        return None
    if _has_nearby_scaffold_xover(mesh, dna, node_cur, node_com):
        return None

    b_nei_up, b_nei_dn, up_cur, up_com, dn_cur, dn_com = _find_staple_neighbor_pair(
        geom,
        mesh,
        sec_cur,
        sec_com,
        node_cur,
        node_com,
    )
    if not b_nei_up and not b_nei_dn:
        return None
    return b_nei_up, up_cur, up_com, dn_cur, dn_com


def _staple_crossover_bp_bounds(geom: GeomType, mesh: MeshType) -> tuple[list[int], list[int]]:
    min_bp = [10**9 for _ in range(geom.n_croL)]
    max_bp = [-10**9 for _ in range(geom.n_croL)]
    for node in mesh.node:
        croL_cur = node.croL
        id_bp = node.bp
        if id_bp > max_bp[croL_cur]:
            max_bp[croL_cur] = id_bp
        if id_bp < min_bp[croL_cur]:
            min_bp[croL_cur] = id_bp
    return min_bp, max_bp


def _iter_staple_xover_candidates(mesh: MeshType, dna: DNAType):
    for i in range(mesh.n_node):
        for j in range(i + 1, mesh.n_node):
            if dna.base_stap[i].xover != -1 and dna.base_stap[j].xover != -1:
                continue
            if mesh.node[i].bp != mesh.node[j].bp:
                continue
            if mesh.node[i].iniL != mesh.node[j].iniL:
                continue
            if mesh.node[i].croL == mesh.node[j].croL:
                continue
            if mesh.node[i].sec == mesh.node[j].sec:
                continue
            yield i, j


def _staple_xover_hits_boundary_gap(
    min_bp: list[int],
    max_bp: list[int],
    croL_cur: int,
    croL_com: int,
    id_bp: int,
) -> bool:
    return (
        min_bp[croL_cur] + para.para_gap_xover_bound_stap > id_bp
        or max_bp[croL_cur] - para.para_gap_xover_bound_stap < id_bp
        or min_bp[croL_com] + para.para_gap_xover_bound_stap > id_bp
        or max_bp[croL_com] - para.para_gap_xover_bound_stap < id_bp
    )


def _has_nearby_scaffold_xover(mesh: MeshType, dna: DNAType, node_a: int, node_b: int) -> bool:
    up_scaf1 = mesh.node[node_a].id
    dn_scaf1 = mesh.node[node_a].id
    up_scaf2 = mesh.node[node_b].id
    dn_scaf2 = mesh.node[node_b].id
    for _ in range(para.para_gap_xover_two):
        if min(up_scaf1, dn_scaf1, up_scaf2, dn_scaf2) < 0:
            return False
        if (
            dna.base_scaf[mesh.node[up_scaf1].id].xover
            == dna.base_scaf[mesh.node[dn_scaf2].id].id
            and dna.base_scaf[mesh.node[dn_scaf2].id].xover
            == dna.base_scaf[mesh.node[up_scaf1].id].id
        ) or (
            dna.base_scaf[mesh.node[dn_scaf1].id].xover
            == dna.base_scaf[mesh.node[up_scaf2].id].id
            and dna.base_scaf[mesh.node[up_scaf2].id].xover
            == dna.base_scaf[mesh.node[dn_scaf1].id].id
        ):
            return True
        up_scaf1 = mesh.node[up_scaf1].up
        dn_scaf1 = mesh.node[dn_scaf1].dn
        up_scaf2 = mesh.node[up_scaf2].up
        dn_scaf2 = mesh.node[dn_scaf2].dn
    return False


def _find_staple_neighbor_pair(
    geom: GeomType,
    mesh: MeshType,
    sec_cur: int,
    sec_com: int,
    node_cur: int,
    node_com: int,
) -> tuple[bool, bool, int, int, int, int]:
    from .section import section_connection_stap

    up_cur = mesh.node[node_cur].dn
    up_com = mesh.node[node_com].up
    if up_cur == -1 or up_com == -1:
        b_nei_up = False
    else:
        id_bp_up = mesh.node[up_cur].bp
        b_nei_up = section_connection_stap(geom, sec_cur, sec_com, id_bp_up)

    dn_cur = mesh.node[node_cur].up
    dn_com = mesh.node[node_com].dn
    if dn_cur == -1 or dn_com == -1:
        b_nei_dn = False
    else:
        id_bp_dn = mesh.node[dn_cur].bp
        b_nei_dn = section_connection_stap(geom, sec_cur, sec_com, id_bp_dn)

    return b_nei_up, b_nei_dn, up_cur, up_com, dn_cur, dn_com


def _apply_staple_xover_pair(
    dna: DNAType,
    cur: int,
    com: int,
    b_nei_up: bool,
    up_cur: int,
    up_com: int,
    dn_cur: int,
    dn_com: int,
) -> None:
    dna.n_xover_stap += 2
    dna.base_stap[cur].xover = dna.base_stap[com].id
    dna.base_stap[com].xover = dna.base_stap[cur].id

    if b_nei_up:
        dna.base_stap[up_cur].xover = dna.base_stap[up_com].id
        dna.base_stap[up_com].xover = dna.base_stap[up_cur].id
        dna.base_stap[cur].up = dna.base_stap[com].id
        dna.base_stap[com].dn = dna.base_stap[cur].id
        dna.base_stap[up_cur].dn = dna.base_stap[up_com].id
        dna.base_stap[up_com].up = dna.base_stap[up_cur].id
    else:
        dna.base_stap[dn_cur].xover = dna.base_stap[dn_com].id
        dna.base_stap[dn_com].xover = dna.base_stap[dn_cur].id
        dna.base_stap[cur].dn = dna.base_stap[com].id
        dna.base_stap[com].up = dna.base_stap[cur].id
        dna.base_stap[dn_cur].up = dna.base_stap[dn_com].id
        dna.base_stap[dn_com].dn = dna.base_stap[dn_cur].id


def _set_stap_crossover(dna: DNAType) -> None:
    for i in range(dna.n_base_stap):
        cur = dna.base_stap[i].id
        up = dna.base_stap[cur].up
        down = dna.base_stap[cur].dn
        if up == -1 or down == -1:
            continue
        cur_xover = dna.base_stap[cur].xover
        up_xover = dna.base_stap[up].xover
        down_xover = dna.base_stap[down].xover
        if cur_xover != -1 and (up != cur_xover or down != cur_xover):
            if up_xover != -1:
                dna.base_stap[cur].up = cur_xover
                dna.base_stap[cur_xover].dn = cur
                dna.base_stap[up].dn = up_xover
                dna.base_stap[up_xover].up = up
            elif down_xover != -1:
                dna.base_stap[cur].dn = cur_xover
                dna.base_stap[cur_xover].up = cur
                dna.base_stap[down].up = down_xover
                dna.base_stap[down_xover].dn = down
            else:
                raise ValueError("The number of crossover should be even.")


def _set_orientation(mesh: MeshType, dna: DNAType) -> None:
    for i in range(mesh.n_node):
        if mesh.node[i].up != -1:
            pos_1 = np.array(mesh.node[i].pos)
            pos_2 = np.array(mesh.node[mesh.node[i].up].pos)
        elif mesh.node[i].dn != -1:
            pos_1 = np.array(mesh.node[mesh.node[i].dn].pos)
            pos_2 = np.array(mesh.node[i].pos)
        else:
            continue

        e3 = pos_2 - pos_1
        e3_norm = np.linalg.norm(e3)
        if e3_norm <= 1e-12:
            continue
        e3 = e3 / e3_norm
        e2 = np.array(dna.base_scaf[i].pos) - np.array(mesh.node[i].pos)
        e2_norm = np.linalg.norm(e2)
        if e2_norm <= 1e-12:
            continue
        e2 = e2 / e2_norm
        e1 = np.cross(e2, e3)
        e1_norm = np.linalg.norm(e1)
        if e1_norm <= 1e-12:
            continue
        e1 = e1 / e1_norm
        mesh.node[i].ori = (
            (float(e1[0]), float(e1[1]), float(e1[2])),
            (float(e2[0]), float(e2[1]), float(e2[2])),
            (float(e3[0]), float(e3[1]), float(e3[2])),
        )


def _write_crossovers_bild(prob: ProbType, geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    if not para.para_write_607:
        return
    from .naming import bild_path

    path = bild_path(prob, "08_crossovers")
    with path.open("w", encoding="utf-8") as f:
        # junction connections
        for i in range(geom.n_junc):
            for j in range(geom.n_sec * geom.junc[i].n_arm):
                a = geom.junc[i].conn[j][0]
                b = geom.junc[i].conn[j][1]
                if a == -1 or b == -1:
                    continue
                if mesh.node[a].conn in (2, 4):
                    f.write(".color red\n")
                else:
                    f.write(".color blue\n")
                p1 = mesh.node[a].pos
                p2 = mesh.node[b].pos
                f.write(f".cylinder {p1[0]:9.3f}{p1[1]:9.3f}{p1[2]:9.3f}{p2[0]:9.3f}{p2[1]:9.3f}{p2[2]:9.3f}{0.06:9.3f}\n")

        f.write(".color tan\n")
        for i in range(mesh.n_node):
            p = mesh.node[i].pos
            f.write(f".sphere {p[0]:9.3f}{p[1]:9.3f}{p[2]:9.3f}{0.08:9.3f}\n")

        f.write(".color steel blue\n")
        for i in range(mesh.n_node):
            p = mesh.node[i].pos
            v = mesh.node[i].ori[2]
            f.write(f".arrow {p[0]:8.2f}{p[1]:8.2f}{p[2]:8.2f}{p[0]+v[0]*0.25:8.2f}{p[1]+v[1]*0.25:8.2f}{p[2]+v[2]*0.25:8.2f}{0.04:8.2f}{0.12:8.2f}{0.60:8.2f}\n")

        f.write(".color steel blue\n")
        for i in range(mesh.n_node):
            xover = dna.base_scaf[i].xover
            if xover != -1 and i < xover:
                p1 = mesh.node[i].pos
                p2 = mesh.node[xover].pos
                f.write(f".cylinder {p1[0]:9.3f}{p1[1]:9.3f}{p1[2]:9.3f}{p2[0]:9.3f}{p2[1]:9.3f}{p2[2]:9.3f}{0.06:9.3f}\n")

        f.write(".color orange\n")
        for i in range(mesh.n_node):
            xover = dna.base_stap[i].xover
            if xover != -1 and i < xover:
                p1 = mesh.node[i].pos
                p2 = mesh.node[xover].pos
                f.write(f".cylinder {p1[0]:9.3f}{p1[1]:9.3f}{p1[2]:9.3f}{p2[0]:9.3f}{p2[1]:9.3f}{p2[2]:9.3f}{0.06:9.3f}\n")


def _write_orientation_bild(prob: ProbType, mesh: MeshType, dna: DNAType) -> None:
    if not para.para_write_608:
        return
    from .naming import bild_path

    path = bild_path(prob, "orientation")
    with path.open("w", encoding="utf-8") as f:
        f.write(".color tan\n")
        for i in range(mesh.n_node):
            p = mesh.node[i].pos
            f.write(f".sphere {p[0]:9.3f}{p[1]:9.3f}{p[2]:9.3f}{0.08:9.3f}\n")

        f.write(".color salmon\n")
        for i in range(mesh.n_node):
            p = mesh.node[i].pos
            v = mesh.node[i].ori[0]
            f.write(f".arrow {p[0]:8.2f}{p[1]:8.2f}{p[2]:8.2f}{p[0]+v[0]*0.85:8.2f}{p[1]+v[1]*0.85:8.2f}{p[2]+v[2]*0.85:8.2f}{0.03:8.2f}{0.09:8.2f}\n")

        f.write(".color sea green\n")
        for i in range(mesh.n_node):
            p = mesh.node[i].pos
            v = mesh.node[i].ori[1]
            f.write(f".arrow {p[0]:8.2f}{p[1]:8.2f}{p[2]:8.2f}{p[0]+v[0]*0.85:8.2f}{p[1]+v[1]*0.85:8.2f}{p[2]+v[2]*0.85:8.2f}{0.03:8.2f}{0.09:8.2f}\n")

        f.write(".color steel blue\n")
        for i in range(mesh.n_node):
            p = mesh.node[i].pos
            v = mesh.node[i].ori[2]
            f.write(f".arrow {p[0]:8.2f}{p[1]:8.2f}{p[2]:8.2f}{p[0]+v[0]*0.25:8.2f}{p[1]+v[1]*0.25:8.2f}{p[2]+v[2]*0.25:8.2f}{0.04:8.2f}{0.10:8.2f}{0.60:8.2f}\n")


def _write_route_bild(prob: ProbType, geom: GeomType, mesh: MeshType, dna: DNAType, step_route: str) -> None:
    if not _route_bild_enabled(step_route):
        return

    path_scaf, path_stap = _route_bild_paths(prob, step_route)
    with path_scaf.open("w", encoding="utf-8") as f_scaf, path_stap.open("w", encoding="utf-8") as f_stap:
        _write_scaffold_route_bild(f_scaf, mesh, dna)
        _write_staple_route_bild(f_stap, mesh, dna)


def _route_bild_enabled(step_route: str) -> bool:
    return {
        "route1": para.para_write_601_1,
        "route2": para.para_write_601_2,
        "route3": para.para_write_601_3,
        "route4": para.para_write_601_4,
        "route5": para.para_write_601_5,
    }.get(step_route, False)


def _route_bild_paths(prob: ProbType, step_route: str):
    from .naming import bild_path

    return bild_path(prob, f"{step_route}_scaf"), bild_path(prob, f"{step_route}_stap")


def _write_scaffold_route_bild(handle: TextIO, mesh: MeshType, dna: DNAType) -> None:
    _write_route_base_segments(handle, mesh, dna.base_scaf, dna.n_base_scaf, "steel blue")


def _write_staple_route_bild(handle: TextIO, mesh: MeshType, dna: DNAType) -> None:
    _write_route_base_segments(handle, mesh, dna.base_stap, dna.n_base_stap, "orange")


def _write_route_base_segments(
    handle: TextIO,
    mesh: MeshType,
    bases: list[BaseType],
    n_base: int,
    normal_color: str,
) -> None:
    for i in range(n_base):
        if bases[i].up == -1:
            continue
        base_1 = bases[i].id
        base_2 = bases[i].up
        resolved = _resolve_route_node_pair(bases, base_1, base_2)
        if resolved is None:
            continue
        node_1, node_2, bridges_unpaired_run = resolved
        p1 = mesh.node[node_1].pos
        p2 = mesh.node[node_2].pos

        if bridges_unpaired_run:
            _write_route_cylinder(handle, "red", p1, p2)
            continue

        if _write_route_xover_segment(handle, mesh, bases, base_1, p1):
            continue
        _write_route_cylinder(handle, normal_color, p1, p2)


def _resolve_route_node_pair(
    bases: list[BaseType],
    base_1: int,
    base_2: int,
) -> tuple[int, int, bool] | None:
    node_1 = bases[base_1].node
    node_2 = bases[base_2].node
    if node_1 == -1 or node_2 == -1:
        if node_1 != -1 and node_2 == -1:
            while base_2 != -1 and bases[base_2].node == -1:
                base_2 = bases[base_2].up
            if base_2 == -1:
                return None
            node_2 = bases[base_2].node
            return node_1, node_2, True
        return None
    return node_1, node_2, False


def _write_route_xover_segment(
    handle: TextIO,
    mesh: MeshType,
    bases: list[BaseType],
    base_1: int,
    p1: tuple[float, float, float],
) -> bool:
    if bases[base_1].xover == -1:
        return False
    base_x = bases[base_1].xover
    node_x = bases[base_x].node
    if node_x == -1:
        return False
    px = mesh.node[node_x].pos
    _write_route_cylinder(handle, "tan", p1, px)
    return bases[base_1].xover == bases[base_1].up


def _write_route_cylinder(
    handle: TextIO,
    color: str,
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> None:
    handle.write(f".color {color}\n")
    handle.write(f".cylinder {p1[0]:9.3f}{p1[1]:9.3f}{p1[2]:9.3f}{p2[0]:9.3f}{p2[1]:9.3f}{p2[2]:9.3f}{0.10:9.3f}\n")
