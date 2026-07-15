from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_prob import ProbType
from .data_geom import GeomType
from .data_mesh import MeshType
from .data_dna import DNAType
from .mani import Mani_Go_Start_Base
from . import para


def output_generation(prob: ProbType, geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    """Generate output files (partial port)."""
    _write_target_geometry_local(prob, geom)
    _write_sep_line(prob, geom)
    _write_doubled_lines(prob, geom, mesh)
    _write_spantree_bild(prob, mesh, dna)
    _write_cylinder_xover(prob, geom, mesh, dna)
    _write_routing_scaf(prob, dna)
    _write_routing_stap(prob, dna)
    _write_routing_all(prob, dna)
    _write_line_scaf(prob, mesh, dna)
    _write_line_stap(prob, mesh, dna)
    _write_atomic_model(prob, dna)
    if para.para_write_801:
        _write_basepair(prob, mesh, dna)
    if para.para_write_802:
        _write_base(prob, dna)
    if para.para_write_803:
        _write_cndo(prob, mesh, dna)
    if para.para_write_804:
        _write_tecplot(prob, mesh)
    if para.para_write_805:
        _write_adina(prob, mesh)
    if para.para_write_702:
        _write_guide_bild(prob, geom, mesh)
    # caDNAno JSON is not fully supported; write minimal placeholder
    if para.para_write_701:
        _write_cadnano_json(prob, geom, mesh, dna)
        _write_sequences_txt(prob, dna)
        _write_sequences_csv(prob, dna)
    if para.para_write_706:
        _write_strand_base_txt(prob, dna)


def _write_doubled_lines(prob: ProbType, geom: GeomType, mesh: MeshType) -> None:
    if not para.para_write_504:
        return
    from .naming import bild_path

    path = bild_path(prob, "04_doubled_lines")
    with path.open("w", encoding="utf-8") as f:
        def _fmt_93(value: float) -> str:
            if abs(value) < 5e-4:
                value = 0.0
            return f"{value:9.3f}"

        def _fmt_82(value: float) -> str:
            if abs(value) < 5e-4:
                value = 0.0
            if value < 0.0:
                value -= 1.0e-6
            return f"{value:8.2f}"

        f.write(".color red\n")
        for p in geom.croP:
            f.write(f".sphere {_fmt_93(p.pos[0])}{_fmt_93(p.pos[1])}{_fmt_93(p.pos[2])}{_fmt_93(0.35)}\n")

        f.write(".color dark green\n")
        for line in geom.croL:
            p1 = geom.croP[line.poi[0]].pos
            p2 = geom.croP[line.poi[1]].pos
            f.write(
                f".cylinder {_fmt_93(p1[0])}{_fmt_93(p1[1])}{_fmt_93(p1[2])}"
                f"{_fmt_93(p2[0])}{_fmt_93(p2[1])}{_fmt_93(p2[2])}{_fmt_93(0.15)}\n"
            )

        if para.para_chimera_504_info:
            f.write(".color dark green\n")
            for p in geom.modP:
                f.write(f".sphere {_fmt_93(p.pos[0])}{_fmt_93(p.pos[1])}{_fmt_93(p.pos[2])}{_fmt_93(0.30)}\n")

            # Write modified edges (every other unit length)
            for line in geom.iniL:
                p1 = np.array(geom.modP[line.poi[0]].pos, dtype=float)
                p2 = np.array(geom.modP[line.poi[1]].pos, dtype=float)
                vec = p2 - p1
                length = float(np.linalg.norm(vec))
                if length <= 1e-8:
                    continue
                direction = vec / length
                iter_count = int(length / 2.0)
                for j in range(1, iter_count * 2 + 1):
                    if j % 2 == 1:
                        pos_a = p1 + (j - 1) * direction
                        pos_b = p1 + j * direction
                        f.write(
                            f".cylinder {_fmt_93(pos_a[0])}{_fmt_93(pos_a[1])}{_fmt_93(pos_a[2])}"
                            f"{_fmt_93(pos_b[0])}{_fmt_93(pos_b[1])}{_fmt_93(pos_b[2])}{_fmt_93(0.10)}\n"
                        )

            # Information on edge number
            for i, line in enumerate(geom.croL, start=1):
                p1 = np.array(geom.croP[line.poi[0]].pos, dtype=float)
                p2 = np.array(geom.croP[line.poi[1]].pos, dtype=float)
                pc = (p1 + p2) / 2.0
                f.write(f".cmov {_fmt_93(pc[0] + 0.4)}{_fmt_93(pc[1] + 0.4)}{_fmt_93(pc[2] + 0.4)}\n")
                f.write(".color dark green\n")
                f.write(".font Helvetica 12 bold\n")
                f.write(f"{i:7d}({line.sec:3d})\n")

            # Local coordinate system on cross-sectional edges
            for line in geom.croL:
                p1 = np.array(geom.croP[line.poi[0]].pos, dtype=float)
                p2 = np.array(geom.croP[line.poi[1]].pos, dtype=float)
                pc = (p1 + p2) / 2.0
                t1 = line.t[0]
                t2 = line.t[1]
                t3 = line.t[2]

                f.write(".color red\n")
                f.write(
                    f".arrow {_fmt_82(pc[0])}{_fmt_82(pc[1])}{_fmt_82(pc[2])}"
                    f"{_fmt_82(pc[0] + t1[0] * 1.5)}{_fmt_82(pc[1] + t1[1] * 1.5)}{_fmt_82(pc[2] + t1[2] * 1.5)}"
                    f"{_fmt_82(0.18)}{_fmt_82(0.36)}{_fmt_82(0.6)}\n"
                )

                f.write(".color blue\n")
                f.write(
                    f".arrow {_fmt_82(pc[0])}{_fmt_82(pc[1])}{_fmt_82(pc[2])}"
                    f"{_fmt_82(pc[0] + t2[0] * 1.2)}{_fmt_82(pc[1] + t2[1] * 1.2)}{_fmt_82(pc[2] + t2[2] * 1.2)}"
                    f"{_fmt_82(0.15)}{_fmt_82(0.30)}{_fmt_82(0.6)}\n"
                )

                f.write(".color yellow\n")
                f.write(
                    f".arrow {_fmt_82(pc[0])}{_fmt_82(pc[1])}{_fmt_82(pc[2])}"
                    f"{_fmt_82(pc[0] + t3[0] * 1.2)}{_fmt_82(pc[1] + t3[1] * 1.2)}{_fmt_82(pc[2] + t3[2] * 1.2)}"
                    f"{_fmt_82(0.15)}{_fmt_82(0.30)}{_fmt_82(0.6)}\n"
                )


def _write_target_geometry_local(prob: ProbType, geom: GeomType) -> None:
    if not para.para_write_302:
        return
    from .naming import bild_path
    from .mani import Mani_Set_Chimera_Axis

    path = bild_path(prob, "02_target_geometry_local")
    with path.open("w", encoding="utf-8") as f:
        def _fmt_93(value: float) -> str:
            if abs(value) < 5e-4:
                value = 0.0
            return f"{value:9.3f}"

        def _fmt_82(value: float) -> str:
            if abs(value) < 5e-4:
                value = 0.0
            return f"{value:8.2f}"

        f.write(".color dark green\n")
        scale_local = 0.2
        for line in geom.iniL:
            p1 = tuple(v * scale_local for v in geom.iniP[line.iniP[0]].ori_pos)
            p2 = tuple(v * scale_local for v in geom.iniP[line.iniP[1]].ori_pos)
            f.write(
                f".cylinder {_fmt_93(p1[0])}{_fmt_93(p1[1])}{_fmt_93(p1[2])}"
                f"{_fmt_93(p2[0])}{_fmt_93(p2[1])}{_fmt_93(p2[2])}{_fmt_93(0.20)}\n"
            )

        f.write(".color red\n")
        for p in geom.iniP:
            pos = tuple(v * scale_local for v in p.ori_pos)
            f.write(
                f".sphere {_fmt_93(pos[0])}{_fmt_93(pos[1])}{_fmt_93(pos[2])}{_fmt_93(0.50)}\n"
            )

        for line in geom.iniL:
            p1 = np.array(geom.iniP[line.iniP[0]].ori_pos, dtype=float) * scale_local
            p2 = np.array(geom.iniP[line.iniP[1]].ori_pos, dtype=float) * scale_local
            pc = (p1 + p2) / 2.0

            f.write(".color red\n")
            t1 = line.t[0]
            f.write(
                f".arrow {_fmt_82(pc[0])}{_fmt_82(pc[1])}{_fmt_82(pc[2])}"
                f"{_fmt_82(pc[0] + t1[0]*1.8)}{_fmt_82(pc[1] + t1[1]*1.8)}{_fmt_82(pc[2] + t1[2]*1.8)}"
                f"{_fmt_82(0.22)}{_fmt_82(0.44)}{_fmt_82(0.6)}\n"
            )

            f.write(".color blue\n")
            t2 = line.t[1]
            f.write(
                f".arrow {_fmt_82(pc[0])}{_fmt_82(pc[1])}{_fmt_82(pc[2])}"
                f"{_fmt_82(pc[0] + t2[0]*1.8)}{_fmt_82(pc[1] + t2[1]*1.8)}{_fmt_82(pc[2] + t2[2]*1.8)}"
                f"{_fmt_82(0.20)}{_fmt_82(0.40)}{_fmt_82(0.6)}\n"
            )

            f.write(".color yellow\n")
            t3 = line.t[2]
            f.write(
                f".arrow {_fmt_82(pc[0])}{_fmt_82(pc[1])}{_fmt_82(pc[2])}"
                f"{_fmt_82(pc[0] + t3[0]*1.8)}{_fmt_82(pc[1] + t3[1]*1.8)}{_fmt_82(pc[2] + t3[2]*1.8)}"
                f"{_fmt_82(0.20)}{_fmt_82(0.40)}{_fmt_82(0.6)}\n"
            )

        if para.para_chimera_302_info:
            for idx, line in enumerate(geom.iniL, start=1):
                p1 = np.array(geom.iniP[line.iniP[0]].ori_pos, dtype=float) * scale_local
                p2 = np.array(geom.iniP[line.iniP[1]].ori_pos, dtype=float) * scale_local
                pc = (p1 + p2) / 2.0
                f.write(f".cmov {_fmt_93(pc[0] + 0.4)}{_fmt_93(pc[1] + 0.4)}{_fmt_93(pc[2] + 0.4)}\n")
                f.write(".color black\n")
                f.write(".font Helvetica 12 bold\n")
                f.write(f"{idx:7d}\n")

            for idx, p in enumerate(geom.iniP, start=1):
                pos = tuple(v * scale_local for v in p.ori_pos)
                f.write(f".cmov {_fmt_93(pos[0] + 0.4)}{_fmt_93(pos[1] + 0.4)}{_fmt_93(pos[2] + 0.4)}\n")
                f.write(".color red\n")
                f.write(".font Helvetica 12 bold\n")
                f.write(f"{idx:7d}\n")

        if para.para_chimera_axis:
            Mani_Set_Chimera_Axis(f)


def _write_sep_line(prob: ProbType, geom: GeomType) -> None:
    if not para.para_write_303:
        return
    from .naming import bild_path
    from .mani import Mani_Set_Chimera_Axis
    from .math_utils import norm, dble2str1, int2str, nint

    path = bild_path(prob, "03_sep_line")
    with path.open("w", encoding="utf-8") as f:
        def _fmt_93(value: float) -> str:
            if abs(value) < 5e-4:
                value = 0.0
            return f"{value:9.3f}"

        def _fmt_82(value: float) -> str:
            if abs(value) < 5e-4:
                value = 0.0
            return f"{value:8.2f}"

        f.write(".color red\n")
        for p in geom.modP:
            f.write(f".sphere {_fmt_93(p.pos[0])}{_fmt_93(p.pos[1])}{_fmt_93(p.pos[2])}{_fmt_93(0.35)}\n")

        f.write(".color dark green\n")
        for line in geom.iniL:
            p1 = geom.modP[line.poi[0]].pos
            p2 = geom.modP[line.poi[1]].pos
            f.write(
                f".cylinder {_fmt_93(p1[0])}{_fmt_93(p1[1])}{_fmt_93(p1[2])}"
                f"{_fmt_93(p2[0])}{_fmt_93(p2[1])}{_fmt_93(p2[2])}{_fmt_93(0.15)}\n"
            )

        if para.para_chimera_303_info:
            for idx, p in enumerate(geom.modP, start=1):
                f.write(f".cmov {_fmt_93(p.pos[0] + 0.4)}{_fmt_93(p.pos[1] + 0.4)}{_fmt_93(p.pos[2] + 0.4)}\n")
                f.write(".color red\n")
                f.write(".font Helvetica 12 bold\n")
                f.write(f"{idx:7d}\n")

            for idx, line in enumerate(geom.iniL, start=1):
                p1 = np.array(geom.modP[line.poi[0]].pos, dtype=float)
                p2 = np.array(geom.modP[line.poi[1]].pos, dtype=float)
                pc = (p1 + p2) / 2.0
                length = norm(p2 - p1)
                f.write(f".cmov {_fmt_93(pc[0] + 0.4)}{_fmt_93(pc[1] + 0.4)}{_fmt_93(pc[2] + 0.4)}\n")
                f.write(".color dark green\n")
                f.write(".font Helvetica 12 bold\n")
                bp_count = nint(length / para.para_dist_bp) + 1
                f.write(f"{idx:7d}({dble2str1(length)}, {int2str(bp_count)})\n")

            for line in geom.iniL:
                p1 = np.array(geom.modP[line.poi[0]].pos, dtype=float)
                p2 = np.array(geom.modP[line.poi[1]].pos, dtype=float)
                pc = (p1 + p2) / 2.0

                f.write(".color red\n")
                t1 = line.t[0]
                f.write(
                    f".arrow {_fmt_82(pc[0])}{_fmt_82(pc[1])}{_fmt_82(pc[2])}"
                    f"{_fmt_82(pc[0] + t1[0]*1.5)}{_fmt_82(pc[1] + t1[1]*1.5)}{_fmt_82(pc[2] + t1[2]*1.5)}"
                    f"{_fmt_82(0.18)}{_fmt_82(0.36)}{_fmt_82(0.6)}\n"
                )

                f.write(".color blue\n")
                t2 = line.t[1]
                f.write(
                    f".arrow {_fmt_82(pc[0])}{_fmt_82(pc[1])}{_fmt_82(pc[2])}"
                    f"{_fmt_82(pc[0] + t2[0]*1.2)}{_fmt_82(pc[1] + t2[1]*1.2)}{_fmt_82(pc[2] + t2[2]*1.2)}"
                    f"{_fmt_82(0.18)}{_fmt_82(0.36)}{_fmt_82(0.6)}\n"
                )

                f.write(".color yellow\n")
                t3 = line.t[2]
                f.write(
                    f".arrow {_fmt_82(pc[0])}{_fmt_82(pc[1])}{_fmt_82(pc[2])}"
                    f"{_fmt_82(pc[0] + t3[0]*1.2)}{_fmt_82(pc[1] + t3[1]*1.2)}{_fmt_82(pc[2] + t3[2]*1.2)}"
                    f"{_fmt_82(0.18)}{_fmt_82(0.36)}{_fmt_82(0.6)}\n"
                )

        if para.para_chimera_axis:
            Mani_Set_Chimera_Axis(f)


def _write_spantree_bild(prob: ProbType, mesh: MeshType, dna: DNAType) -> None:
    if not para.para_write_606:
        return
    if dna.n_base_scaf == 0:
        return
    if not dna.graph_pos_node or not dna.graph_tail or not dna.graph_head:
        return

    pos_node = dna.graph_pos_node
    tail = dna.graph_tail
    head = dna.graph_head
    tree = dna.graph_tree
    n_node = len(pos_node) - 1
    n_edge = len(tail) - 1
    if n_node <= 0 or n_edge <= 0:
        return
    tree_ids = set(tree)

    from .naming import bild_path

    path = bild_path(prob, "07_spantree")
    with path.open("w", encoding="utf-8") as f:
        def _fmt_93(value: float) -> str:
            if abs(value) < 5e-4:
                value = -0.0
            return f"{value:9.3f}"

        f.write(".color steel blue\n")
        for i in range(1, n_node + 1):
            p = pos_node[i]
            f.write(f".sphere {_fmt_93(p[0])}{_fmt_93(p[1])}{_fmt_93(p[2])}{_fmt_93(0.35)}\n")

        edge_indices = list(range(1, n_edge + 1))
        if n_edge == n_node and len(tree) == n_node - 1 and tree == list(range(1, len(tree) + 1)):
            edge_indices = list(reversed(edge_indices))

        for i in edge_indices:
            p1 = pos_node[tail[i]]
            p2 = pos_node[head[i]]
            if i in tree_ids:
                f.write(".color red\n")
                radius = 0.15
            else:
                f.write(".color tan\n")
                radius = 0.10
            f.write(
                f".cylinder {_fmt_93(p1[0])}{_fmt_93(p1[1])}{_fmt_93(p1[2])}"
                f"{_fmt_93(p2[0])}{_fmt_93(p2[1])}{_fmt_93(p2[2])}{_fmt_93(radius)}\n"
            )

        if para.para_chimera_axis:
            from .mani import Mani_Set_Chimera_Axis

            Mani_Set_Chimera_Axis(f)


def _write_routing_scaf(prob: ProbType, dna: DNAType) -> None:
    if not para.para_write_703:
        return
    from .naming import bild_path

    path = bild_path(prob, "09_atomic_model_scaf")
    with path.open("w", encoding="utf-8") as f:
        f.write(".color steel blue\n")
        for base in dna.base_scaf:
            x, y, z = base.pos
            f.write(f".sphere {x:9.3f}{y:9.3f}{z:9.3f}{0.15:9.3f}\n")
            if base.up != -1:
                u = base.up
                x2, y2, z2 = dna.base_scaf[u].pos
                f.write(
                    f".cylinder {x:9.3f}{y:9.3f}{z:9.3f}"
                    f"{x2:9.3f}{y2:9.3f}{z2:9.3f}{0.05:9.3f}\n"
                )


def _write_line_scaf(prob: ProbType, mesh: MeshType, dna: DNAType) -> None:
    if not para.para_write_703:
        return
    from .naming import bild_path

    path = bild_path(prob, "10_routing_scaf")
    _write_line_route(path, mesh, dna, strand_type="scaf")


def _write_routing_stap(prob: ProbType, dna: DNAType) -> None:
    if not para.para_write_703:
        return
    from .naming import bild_path

    path = bild_path(prob, "09_atomic_model_stap")
    with path.open("w", encoding="utf-8") as f:
        f.write(".color orange\n")
        for base in dna.base_stap:
            x, y, z = base.pos
            f.write(f".sphere {x:9.3f}{y:9.3f}{z:9.3f}{0.15:9.3f}\n")
            if base.up != -1:
                u = base.up
                x2, y2, z2 = dna.base_stap[u].pos
                f.write(
                    f".cylinder {x:9.3f}{y:9.3f}{z:9.3f}"
                    f"{x2:9.3f}{y2:9.3f}{z2:9.3f}{0.05:9.3f}\n"
                )


def _write_line_stap(prob: ProbType, mesh: MeshType, dna: DNAType) -> None:
    if not para.para_write_703:
        return
    from .naming import bild_path

    path = bild_path(prob, "11_routing_stap")
    _write_line_route(path, mesh, dna, strand_type="stap")


def _write_route_start_marker(f, mesh: MeshType, cur_node: int) -> None:
    f.write(".color dark green\n")
    pos = mesh.node[cur_node].pos
    f.write(f".sphere {pos[0]:9.3f}{pos[1]:9.3f}{pos[2]:9.3f}{0.200:9.3f}\n")


def _write_route_terminal_arrow(f, mesh: MeshType, cur_node: int, up_node: int) -> None:
    pos1 = np.array(mesh.node[cur_node].pos, dtype=float)
    pos2 = np.array(mesh.node[up_node].pos, dtype=float)
    vec = pos2 - pos1
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return
    vec = vec / norm
    f.write(".color dark green\n")
    p2 = pos1 + vec * 1.3
    f.write(
        f".arrow {pos1[0]:8.2f}{pos1[1]:8.2f}{pos1[2]:8.2f}"
        f"{p2[0]:8.2f}{p2[1]:8.2f}{p2[2]:8.2f}{0.11:8.2f}{0.35:8.2f}\n"
    )


def _advance_route_up_base(dna: DNAType, up_base: int) -> int:
    while up_base != -1 and dna.top[up_base].node == -1:
        up_base = dna.top[up_base].up
    return up_base


@dataclass(frozen=True)
class _RouteSegmentState:
    strand_type: str
    cur_base: int
    cur_node: int
    up_base: int
    up_node: int


def _resolve_route_segment(
    f,
    mesh: MeshType,
    dna: DNAType,
    state: _RouteSegmentState,
) -> int:
    if state.strand_type == "scaf":
        is_terminal = mesh.node[state.cur_node].up == -1
        default_color = "steel blue"
    else:
        is_terminal = mesh.node[state.cur_node].dn == -1
        default_color = "orange"

    if is_terminal:
        up_base = _advance_route_up_base(dna, state.up_base)
        up_node = dna.top[up_base].node if up_base != -1 else -1
        f.write(".color tan\n")
        return up_node

    f.write(".color red\n" if dna.top[state.cur_base].xover == state.up_base else f".color {default_color}\n")
    return state.up_node


def _write_route_segment(f, mesh: MeshType, cur_node: int, up_node: int) -> None:
    if up_node == -1:
        return
    pos1 = mesh.node[cur_node].pos
    pos2 = mesh.node[up_node].pos
    if pos1 == pos2:
        return
    f.write(
        f".cylinder {pos1[0]:9.3f}{pos1[1]:9.3f}{pos1[2]:9.3f}"
        f"{pos2[0]:9.3f}{pos2[1]:9.3f}{pos2[2]:9.3f}{0.100:9.3f}\n"
    )


def _write_line_route(path: Path, mesh: MeshType, dna: DNAType, strand_type: str) -> None:
    from .mani import Mani_Set_Chimera_Axis

    with path.open("w", encoding="utf-8") as f:
        for strand in dna.strand:
            if strand.type1 != strand_type:
                continue

            for cur_base in strand.base:
                up_base = dna.top[cur_base].up
                down_base = dna.top[cur_base].dn
                cur_node = dna.top[cur_base].node

                if cur_node == -1:
                    continue

                if up_base != -1:
                    up_node = dna.top[up_base].node
                else:
                    _write_route_start_marker(f, mesh, cur_node)
                    continue

                if down_base == -1 and up_node != -1:
                    _write_route_terminal_arrow(f, mesh, cur_node, up_node)

                up_node = _resolve_route_segment(
                    f,
                    mesh,
                    dna,
                    _RouteSegmentState(
                        strand_type=strand_type,
                        cur_base=cur_base,
                        cur_node=cur_node,
                        up_base=up_base,
                        up_node=up_node,
                    ),
                )
                _write_route_segment(f, mesh, cur_node, up_node)

        if para.para_chimera_axis:
            Mani_Set_Chimera_Axis(f)


def _write_routing_all(prob: ProbType, dna: DNAType) -> None:
    if not para.para_write_705:
        return
    from .naming import bild_path

    path = bild_path(prob, "09_atomic_model_all")
    with path.open("w", encoding="utf-8") as f:
        f.write(".color steel blue\n")
        for base in dna.base_scaf:
            x, y, z = base.pos
            f.write(f".sphere {x:9.3f}{y:9.3f}{z:9.3f}{0.12:9.3f}\n")
            if base.up != -1:
                u = base.up
                x2, y2, z2 = dna.base_scaf[u].pos
                f.write(
                    f".cylinder {x:9.3f}{y:9.3f}{z:9.3f}"
                    f"{x2:9.3f}{y2:9.3f}{z2:9.3f}{0.04:9.3f}\n"
                )
        f.write(".color orange\n")
        for base in dna.base_stap:
            x, y, z = base.pos
            f.write(f".sphere {x:9.3f}{y:9.3f}{z:9.3f}{0.12:9.3f}\n")
            if base.up != -1:
                u = base.up
                x2, y2, z2 = dna.base_stap[u].pos
                f.write(
                    f".cylinder {x:9.3f}{y:9.3f}{z:9.3f}"
                    f"{x2:9.3f}{y2:9.3f}{z2:9.3f}{0.04:9.3f}\n"
                )


def _write_cylinder_xover(prob: ProbType, geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    if not para.para_write_609:
        return
    from .naming import bild_path

    path = bild_path(prob, "13_cylindrical_model_xover")
    import numpy as np

    def same_vector(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
        return float(np.linalg.norm(np.array(a) - np.array(b))) < 1e-9

    radius = para.para_rad_helix + para.para_gap_helix / 2.0
    color = tuple(c / 255.0 for c in prob.color)
    with path.open("w", encoding="utf-8") as f:
        for line in geom.croL:
            pos_1 = geom.croP[line.poi[0]].ori_pos
            pos_2 = geom.croP[line.poi[1]].ori_pos
            f.write(f".color {color[0]:9.4f}{color[1]:9.4f}{color[2]:9.4f}\n")
            f.write(
                f".cylinder {pos_1[0]:9.3f}{pos_1[1]:9.3f}{pos_1[2]:9.3f}"
                f"{pos_2[0]:9.3f}{pos_2[1]:9.3f}{pos_2[2]:9.3f}{radius:9.3f}\n"
            )

            pos_1 = geom.croP[line.poi[0]].pos
            pos_2 = geom.croP[line.poi[0]].ori_pos
            if not same_vector(pos_1, pos_2):
                if float(np.linalg.norm(np.array(pos_1) - np.array(pos_2))) > 0.4:
                    f.write(".color dark gray\n")
                    rad_mod = radius * 1.02
                else:
                    f.write(f".color {color[0]:9.4f}{color[1]:9.4f}{color[2]:9.4f}\n")
                    rad_mod = radius
                f.write(
                    f".cylinder {pos_1[0]:12.5f}{pos_1[1]:12.5f}{pos_1[2]:12.5f}"
                    f"{pos_2[0]:12.5f}{pos_2[1]:12.5f}{pos_2[2]:12.5f}{rad_mod:9.3f}\n"
                )

            pos_1 = geom.croP[line.poi[1]].pos
            pos_2 = geom.croP[line.poi[1]].ori_pos
            if not same_vector(pos_1, pos_2):
                if float(np.linalg.norm(np.array(pos_1) - np.array(pos_2))) > 0.4:
                    f.write(".color dark gray\n")
                    rad_mod = radius * 1.02
                else:
                    f.write(f".color {color[0]:9.4f}{color[1]:9.4f}{color[2]:9.4f}\n")
                    rad_mod = radius
                f.write(
                    f".cylinder {pos_1[0]:12.5f}{pos_1[1]:12.5f}{pos_1[2]:12.5f}"
                    f"{pos_2[0]:12.5f}{pos_2[1]:12.5f}{pos_2[2]:12.5f}{rad_mod:9.3f}\n"
                )

        for strand in dna.strand:
            for base_id in strand.base:
                if dna.top[base_id].xover == -1:
                    continue
                node = dna.top[base_id].node
                if node == -1:
                    continue
                if mesh.node[node].up == -1:
                    pos_1 = mesh.node[node].pos
                else:
                    pos_1 = mesh.node[mesh.node[node].up].pos
                if mesh.node[node].dn == -1:
                    pos_2 = mesh.node[node].pos
                else:
                    pos_2 = mesh.node[mesh.node[node].dn].pos
                if strand.type1 == "scaf":
                    f.write(".color medium blue\n")
                else:
                    f.write(".color orange red\n")
                f.write(
                    f".cylinder {pos_1[0]:9.3f}{pos_1[1]:9.3f}{pos_1[2]:9.3f}"
                    f"{pos_2[0]:9.3f}{pos_2[1]:9.3f}{pos_2[2]:9.3f}{radius*1.05:9.3f}\n"
                )


def _write_atomic_model(prob: ProbType, dna: DNAType) -> None:
    if not para.para_write_702:
        return
    from .naming import bild_path
    from .mani import Mani_Set_Chimera_Axis

    def _fmt(value: float, is_y: bool = False) -> float:
        # Reference atomic models never emit -0.000 for Y components.
        if is_y and abs(value) < 5e-4:
            return 0.0
        return value

    path = bild_path(prob, "09_atomic_model")
    with path.open("w", encoding="utf-8") as f:
        for strand in dna.strand:
            for base in strand.base:
                if strand.type1 == "scaf":
                    f.write(".color steel blue\n")
                else:
                    f.write(".color orange\n")

                pos = dna.top[base].pos
                f.write(
                    f".sphere {_fmt(pos[0]):9.3f}{_fmt(pos[1], True):9.3f}{_fmt(pos[2]):9.3f}{0.15:9.3f}\n"
                )

                up = dna.top[base].up
                if up != -1:
                    pos_up = dna.top[up].pos
                    f.write(
                        f".cylinder {_fmt(pos[0]):9.3f}{_fmt(pos[1], True):9.3f}{_fmt(pos[2]):9.3f}"
                        f"{_fmt(pos_up[0]):9.3f}{_fmt(pos_up[1], True):9.3f}{_fmt(pos_up[2]):9.3f}{0.05:9.3f}\n"
                    )

                xover = dna.top[base].xover
                if xover != -1 and base < xover:
                    if strand.type1 == "scaf":
                        f.write(".color blue\n")
                    else:
                        f.write(".color red\n")
                    pos_x = dna.top[xover].pos
                    f.write(
                        f".cylinder {_fmt(pos[0]):9.3f}{_fmt(pos[1], True):9.3f}{_fmt(pos[2]):9.3f}"
                        f"{_fmt(pos_x[0]):9.3f}{_fmt(pos_x[1], True):9.3f}{_fmt(pos_x[2]):9.3f}{0.08:9.3f}\n"
                    )

                across = dna.top[base].across
                if across != -1:
                    f.write(".color light gray\n")
                    pos_a = dna.top[across].pos
                    f.write(
                        f".cylinder {_fmt(pos[0]):9.3f}{_fmt(pos[1], True):9.3f}{_fmt(pos[2]):9.3f}"
                        f"{_fmt(pos_a[0]):9.3f}{_fmt(pos_a[1], True):9.3f}{_fmt(pos_a[2]):9.3f}{0.025:9.3f}\n"
                    )

        if para.para_chimera_axis:
            Mani_Set_Chimera_Axis(f)


def _summary_pct(num: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return num / denom * 100.0


def _summary_lines(prob: ProbType, geom: GeomType, mesh: MeshType, dna: DNAType) -> list[str]:
    return [
        "",
        " +==================================================+",
        " | 8. Summary                                       |",
        " +==================================================+",
        "",
        " 8.1. Design parameters",
        f"   * Scaffold sequence              : {para.para_scaf_seq}",
        f"   * Vertex design                  : {para.para_vertex_design}",
        f"   * Vertex angle                   : {para.para_vertex_angle}",
        f"   * Vertex clash                   : {para.para_vertex_crash}",
        f"   * Constant edge length from mesh : {para.para_const_edge_mesh}",
        f"   * Gap b/w two scaf xovers        : {para.para_gap_xover_two_scaf}",
        f"   * Gap b/w xover(stap) and bound  : {para.para_gap_xover_bound_stap}",
        f"   * Gap b/w stap and scaf xovers   : {para.para_gap_xover_two}",
        f"   * Gap b/w xover and first nick   : {para.para_gap_xover_nick1}",
        f"   * Gap b/w xover and nick         : {para.para_gap_xover_nick}",
        f"   * Break rule for staples         : {para.para_cut_stap_method}",
        f"   * Unpaired scaffold nucleotides  : {para.para_unpaired_scaf}",
        f"   * # of changing for min staple   : {prob.n_cng_min_stap}",
        f"   * # of changing for max staple   : {prob.n_cng_max_stap}",
        "",
        " 8.2. Target geometry",
        f"   * File name            : {prob.name_file}",
        f"   * The number of faces  : {geom.n_face}",
        f"   * The number of points : {geom.n_iniP}",
        f"   * The number of edges  : {geom.n_iniL}",
        f"   * Minimum edge-length  : {prob.n_edge_len}",
        "",
        " 8.3. Basepair",
        f"   * # of basepairs            : {mesh.n_node}",
        "   * # of mitered nts          : "
        f"{mesh.n_mitered} [{_summary_pct(mesh.n_mitered, mesh.n_node):.3f} %]",
        "   * Edge length [ min - max ] : "
        f"[{geom.min_edge_length} - {geom.max_edge_length}]",
        "   * Min # xovers [scaf, stap] : "
        f"{dna.min_xover_scaf + dna.min_xover_stap} [{dna.min_xover_scaf}, {dna.min_xover_stap}]",
        "",
        " 8.4. Scaffold",
        f"   * # of the scaffold  : {dna.n_scaf}",
        f"   * # of total nts     : {dna.n_base_scaf}",
        f"   * # of unpaired nts  : {dna.n_nt_unpaired_scaf}",
        f"   * # of double Xovers : {dna.n_xover_scaf // 2}",
        "",
        " 8.5. Staple",
        f"   * # of staples         : {dna.n_stap}",
        "    @ with the 4nt dsDNA  : "
        f"{dna.n_4nt} [{_summary_pct(dna.n_4nt, dna.n_stap):.3f} %]",
        "    @ with the 14nt dsDNA : "
        f"{dna.n_14nt} [{_summary_pct(dna.n_14nt, dna.n_stap):.3f} %]",
        f"   * # of nts             : {dna.n_base_stap}",
        "    @ in 4nt dsDNA        : "
        f"{dna.n_nt_4nt} [{_summary_pct(dna.n_nt_4nt, dna.n_base_stap):.3f} %]",
        "    @ in 14nt dsDNA       : "
        f"{dna.n_nt_14nt} [{_summary_pct(dna.n_nt_14nt, dna.n_base_stap):.3f} %]",
        f"   * # of unpaired nts    : {dna.n_nt_unpaired_stap}",
        f"   * # of total Xover     : {dna.n_xover_stap}",
        f"   * # of single-Xovers   : {dna.n_sxover_stap}",
        "   * # of double-Xovers   : "
        f"{(dna.n_xover_stap - dna.n_sxover_stap) // 2}",
        "   * Length [min-max-ave] : "
        f"[{dna.len_min_stap} - {dna.len_max_stap} - {dna.len_ave_stap:.3f}]",
        "",
        " +=== completed =====================================+",
        " | PERDIX generated output files.                    |",
        " +===================================================+",
    ]


def print_summary(prob: ProbType, geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    """Print summary of design.

    Fortran: PERDIX.f90::Print_Summary
    """
    sys.stdout.write("\n".join(_summary_lines(prob, geom, mesh, dna)) + "\n")


def _write_basepair(prob: ProbType, mesh: MeshType, dna: DNAType) -> None:
    path = Path(prob.path_work) / f"{prob.name_file}_basepair.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{mesh.n_node:7d}\n")
        for i in range(mesh.n_node):
            node = mesh.node[i]
            ori = node.ori
            f.write(f"{i:7d}")
            f.write(f"{node.pos[0]:8.2f}{node.pos[1]:8.2f}{node.pos[2]:8.2f}")
            f.write(f"{ori[0][0]:8.2f}{ori[0][1]:8.2f}{ori[0][2]:8.2f}")
            f.write(f"{ori[1][0]:8.2f}{ori[1][1]:8.2f}{ori[1][2]:8.2f}")
            f.write(f"{ori[2][0]:8.2f}{ori[2][1]:8.2f}{ori[2][2]:8.2f}")
            f.write(f"{dna.base_scaf[i].xover:7d}{dna.base_stap[i].xover:7d}\n")
        f.write("\n")
        f.write(f"{mesh.n_ele:7d}\n")
        for i in range(mesh.n_ele):
            c1, c2 = mesh.ele[i].cn
            f.write(f"{i:7d}{c1:8d}{c2:8d}\n")


def _write_base(prob: ProbType, dna: DNAType) -> None:
    path = Path(prob.path_work) / f"{prob.name_file}_base.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{dna.n_base_scaf + dna.n_base_stap:7d}\n")
        for i in range(dna.n_top):
            t = dna.top[i]
            f.write(f"{t.id:7d}{t.up:7d}{t.dn:7d}{t.across:7d}")
            f.write(f"{t.pos[0]:10.4f}{t.pos[1]:10.4f}{t.pos[2]:10.4f}\n")


def _write_cndo(prob: ProbType, mesh: MeshType, dna: DNAType) -> None:
    path = Path(prob.path_work) / f"{prob.name_file}_{prob.n_edge_len}bp_16_cndo_format.cndo"

    def _fmt_val(value: float) -> str:
        s = f"{value:.3f}"
        if s.startswith("-0"):
            s = "-" + s[2:]
        elif s.startswith("0"):
            s = s[1:]
        return s

    def _fmt_id(value: int) -> str:
        return "-1" if value == -1 else str(value + 1)

    with path.open("w", encoding="utf-8") as f:
        f.write('"CanDo (.cndo) file format version 1.0"\n\n')
        f.write("dnaTop,id,up,down,across,seq\n")
        for i in range(dna.n_top):
            t = dna.top[i]
            f.write(
                f"{i+1},"
                f"{_fmt_id(t.id)},"
                f"{_fmt_id(t.dn)},"
                f"{_fmt_id(t.up)},"
                f"{_fmt_id(t.across)},"
                f"{t.seq}\n"
            )
        f.write("\n")

        min_pos = np.array(mesh.node[0].pos)
        for n in mesh.node:
            p = np.array(n.pos)
            min_pos = np.minimum(min_pos, p)
        fscale = 10.0
        min_pos = np.where(min_pos < -99.0, min_pos + 80.0, 0.0)

        f.write('dNode,"e0(1)","e0(2)","e0(3)"\n')
        for i, n in enumerate(mesh.node, start=1):
            pos = (np.array(n.pos) - min_pos) * fscale
            f.write(f"{i},{_fmt_val(pos[0])},{_fmt_val(pos[1])},{_fmt_val(pos[2])}\n")
        f.write("\n")

        f.write('triad,"e1(1)","e1(2)","e1(3)","e2(1)","e2(2)","e2(3)","e3(1)","e3(2)","e3(3)"\n')
        for i, n in enumerate(mesh.node, start=1):
            e1 = n.ori[0]; e2 = n.ori[1]; e3 = n.ori[2]
            f.write(
                f"{i},"
                f"{_fmt_val(e1[0])},{_fmt_val(e1[1])},{_fmt_val(e1[2])},"
                f"{_fmt_val(e2[0])},{_fmt_val(e2[1])},{_fmt_val(e2[2])},"
                f"{_fmt_val(e3[0])},{_fmt_val(e3[1])},{_fmt_val(e3[2])}\n"
            )
        f.write("\n")

        # Simplified nt mapping: scaffold/staple across
        node_nt = [[0, 0] for _ in range(mesh.n_node)]
        for i in range(dna.n_top):
            t = dna.top[i]
            if t.across != -1 and t.node != -1 and i < dna.n_base_scaf:
                node_nt[t.node][0] = t.id + 1
                node_nt[t.node][1] = t.across + 1

        f.write("id_nt,id1,id2\n")
        for i in range(mesh.n_node):
            f.write(f"{i+1},{node_nt[i][0]},{node_nt[i][1]}\n")
        f.write("\n")
        f.write("xy\n\n")
        f.write("    1.00\n\n")
        f.write(f"{prob.color[0]:12d}{prob.color[1]:12d}{prob.color[2]:12d}\n\n")
        for i in range(dna.n_top):
            strand = dna.top[i].strand
            if strand == -1:
                f.write("0\n")
            else:
                f.write("0\n" if dna.strand[strand - 1].type1 == "scaf" else "1\n")
        f.write("\n")


def _write_tecplot(prob: ProbType, mesh: MeshType) -> None:
    path = Path(prob.path_work) / f"{prob.name_file}_tecplot.dat"
    with path.open("w", encoding="utf-8") as f:
        for k in range(3):
            f.write("variables = 'X', 'Y', 'Z', 'e3', 'e2', 'e1'\n")
            f.write(f"ZONE F=FEPOINT, N={mesh.n_node}, E={mesh.n_ele}, ET=QUADRILATERAL\n")
            for n in mesh.node:
                f.write(f"{n.pos[0]:10.4f}{n.pos[1]:10.4f}{n.pos[2]:10.4f}")
                f.write(f"{n.ori[k][0]:10.4f}{n.ori[k][1]:10.4f}{n.ori[k][2]:10.4f}\n")
            for e in mesh.ele:
                f.write(f"{e.cn[0]:6d}{e.cn[1]:6d}{e.cn[0]:6d}{e.cn[1]:6d}\n")


def _write_adina(prob: ProbType, mesh: MeshType) -> None:
    path = Path(prob.path_work) / f"{prob.name_file}_adina.in"
    with path.open("w", encoding="utf-8") as f:
        f.write("*\n* Command file created from PERDIX for command import\n*\n")
        f.write("DATABASE NEW SAVE=NO PROMPT=NO\n")
        f.write("FEPROGRAM ADINA\n")
        f.write("COORDINATES POINT SYSTEM=0\n")
        f.write("@CLEAR\n\n")

        for i, n in enumerate(mesh.node, start=1):
            f.write(f"{i} {n.pos[0]:13.5f}{n.pos[1]:13.5f}{n.pos[2]:13.5f} 0\n")
        f.write("@\n")

        for i, e in enumerate(mesh.ele, start=1):
            f.write(f"LINE STRAIGHT NAME={i} P1={e.cn[0]} P2={e.cn[1]}\n*\n")

        f.write("*\n")
        f.write("EGROUP BEAM NAME=1 SUBTYPE=THREE-D DISPLACE=DEFAULT MATERIAL=1 RINT=5,\n")
        f.write("     SINT=DEFAULT TINT=DEFAULT RESULTS=STRESSES INITIALS=NONE,\n")
        f.write("     CMASS=DEFAULT RIGIDEND=NONE MOMENT-C=NO RIGIDITY=1,\n")
        f.write("     MULTIPLY=1000000.00000000 RUPTURE=ADINA OPTION=NONE,\n")
        f.write("     BOLT-TOL=0.00000000000000 DESCRIPT='NONE' SECTION=1,\n")
        f.write("     PRINT=DEFAULT SAVE=DEFAULT TBIRTH=0.00000000000000,\n")
        f.write("     TDEATH=0.00000000000000 SPOINT=2 BOLTFORC=0.00000000000000,\n")
        f.write("     BOLTNCUR=0 TMC-MATE=1 BOLT-NUM=0 BOLT-LOA=0.00000000000000,\n")
        f.write("     WARP=NO\n")

        f.write("*\nSUBDIVIDE LINE NAME=1 MODE=DIVISIONS NDIV=1 RATIO=1.00000000000000,\n")
        f.write("      PROGRESS=GEOMETRIC CBIAS=NO\n")
        f.write("@CLEAR\n")

        for i in range(1, mesh.n_ele + 1):
            f.write(f"{i}\n")
        f.write("@\n")


def _write_guide_bild(prob: ProbType, geom: GeomType, mesh: MeshType) -> None:
    from .naming import bild_path
    from .math_utils import is_same_vector

    path = bild_path(prob, "14_json_guide")
    with path.open("w", encoding="utf-8") as f:
        def _fmt_62(value: float) -> str:
            if abs(value) < 5e-4:
                value = 0.0
            return f"{value:6.2f}"

        f.write(f".color {_fmt_62(0.0/255.0)}{_fmt_62(114.0/255.0)}{_fmt_62(178.0/255.0)}\n")
        for p in geom.croP:
            f.write(f".sphere {p.pos[0]:9.3f}{p.pos[1]:9.3f}{p.pos[2]:9.3f}{0.20:9.3f}\n")

        f.write(f".color {_fmt_62(0.0/255.0)}{_fmt_62(114.0/255.0)}{_fmt_62(178.0/255.0)}\n")
        for l in geom.croL:
            p1 = geom.croP[l.poi[0]].pos
            p2 = geom.croP[l.poi[1]].pos
            f.write(
                f".cylinder {p1[0]:9.3f}{p1[1]:9.3f}{p1[2]:9.3f}"
                f"{p2[0]:9.3f}{p2[1]:9.3f}{p2[2]:9.3f}{0.10:9.3f}\n"
            )

        n_conn = 0
        f.write(f".color {_fmt_62(213.0/255.0)}{_fmt_62(94.0/255.0)}{_fmt_62(0.0/255.0)}\n")
        for j in range(geom.n_junc):
            for k in range(geom.n_sec * geom.junc[j].n_arm):
                a = geom.junc[j].conn[k][0]
                b = geom.junc[j].conn[k][1]
                if a == -1 or b == -1:
                    continue
                p1 = mesh.node[a].pos
                p2 = mesh.node[b].pos
                n_conn += 1
                if not is_same_vector(p1, p2):
                    f.write(
                        f".cylinder {p1[0]:9.3f}{p1[1]:9.3f}{p1[2]:9.3f}"
                        f"{p2[0]:9.3f}{p2[1]:9.3f}{p2[2]:9.3f}{0.04:9.3f}\n"
                    )

        f.write(".color red\n")
        for l in geom.croL:
            p1 = np.array(geom.croP[l.poi[0]].pos, dtype=float)
            p2 = np.array(geom.croP[l.poi[1]].pos, dtype=float)
            pc = (p1 + p2) / 2.0
            t1 = l.t[0]
            f.write(
                f".arrow {pc[0]:8.2f}{pc[1]:8.2f}{pc[2]:8.2f}"
                f"{pc[0] + t1[0]*1.6:8.2f}{pc[1] + t1[1]*1.6:8.2f}{pc[2] + t1[2]*1.6:8.2f}"
                f"{0.16:8.2f}{0.45:8.2f}{0.50:8.2f}\n"
            )

        add = -1
        f.write(".color dark green\n")
        denom = geom.n_croL // geom.n_sec if geom.n_sec > 0 else 0
        for i, l in enumerate(geom.croL, start=1):
            p1 = np.array(geom.croP[l.poi[0]].pos, dtype=float)
            p2 = np.array(geom.croP[l.poi[1]].pos, dtype=float)
            pc = (p1 + p2) / 2.0
            if denom > 0 and (i - 1) % denom == 0:
                add += 1
            f.write(f".cmov {pc[0] + 0.6:9.3f}{pc[1] + 0.6:9.3f}{pc[2] - 0.6:9.3f}\n")
            f.write(".font Helvetica 12 bold\n")
            label = (geom.n_sec * i - (geom.n_sec - 1)) % geom.n_croL + add - 1
            f.write(f"{label:7d}\n")

        if para.para_chimera_axis:
            from .mani import Mani_Set_Chimera_Axis

            Mani_Set_Chimera_Axis(f)


def _cadnano_color_palette() -> list[int]:
    return [
        13369344,
        16204552,
        16225054,
        11184640,
        5749504,
        29184,
        243362,
        1507550,
        7536862,
        12060012,
        3355443,
        8947848,
    ]


def _cadnano_max_unpaired(dna: DNAType) -> int:
    max_base = 0
    for i in range(dna.n_top):
        t = dna.top[i]
        if t.up == -1 or t.dn == -1 or t.across == -1:
            continue
        if dna.top[t.up].node != -1 or dna.top[t.dn].node == -1:
            continue
        base = t.id
        count = 0
        while True:
            base = dna.top[base].up
            if base == -1 or dna.top[base].across != -1:
                break
            count += 1
        max_base = max(max_base, count)
    return max_base


def _cadnano_dimensions(geom: GeomType, mesh: MeshType, max_unpaired: int) -> tuple[int, int, int, int]:
    max_bp = max(n.bp for n in mesh.node) + para.para_start_bp_ID - 1
    min_bp = min(n.bp for n in mesh.node) + para.para_start_bp_ID - 1
    shift = para.para_start_bp_ID + 42
    width = (max_bp - min_bp + 1 + max_unpaired) + 2 * para.para_start_bp_ID + 21
    width = (width // 21) * 21 + 42
    return width, geom.n_iniL, geom.n_sec, shift


def _cadnano_empty_conn() -> list[int]:
    return [-1, -1, -1, -1]


@dataclass
class _CadnanoGrid:
    scaf: list[list[list[list[int]]]]
    stap: list[list[list[list[int]]]]
    stap_col: list[list[list[list[int]]]]


@dataclass(frozen=True)
class _CadnanoCursor:
    edge: int
    sec: int
    bp: int


@dataclass(frozen=True)
class _CadnanoNeighborQuery:
    base_id: int
    cursor: _CadnanoCursor
    shift: int
    strand_type: str


@dataclass(frozen=True)
class _CadnanoLinkAssignment:
    strand_type: str
    cursor: _CadnanoCursor
    dn_pos: tuple[int, int, int] | None
    up_pos: tuple[int, int, int] | None


@dataclass(frozen=True)
class _CadnanoBuildContext:
    width: int
    n_sec: int
    shift: int
    colors: list[int]


def _build_cadnano_grid(width: int, n_edge: int, n_sec: int) -> _CadnanoGrid:
    scaf = [[[_cadnano_empty_conn() for _ in range(width)] for _ in range(n_sec)] for _ in range(n_edge)]
    stap = [[[_cadnano_empty_conn() for _ in range(width)] for _ in range(n_sec)] for _ in range(n_edge)]
    stap_col = [[[] for _ in range(n_sec)] for _ in range(n_edge)]
    return _CadnanoGrid(scaf=scaf, stap=stap, stap_col=stap_col)


def _cadnano_step_bp(sec: int, bp: int, strand_type: str) -> int:
    if sec % 2 == 0:
        return bp + 1 if strand_type == "scaf" else bp - 1
    return bp - 1 if strand_type == "scaf" else bp + 1


def _cadnano_pos_from_node(mesh: MeshType, node_id: int, shift: int) -> tuple[int, int, int]:
    node = mesh.node[node_id]
    return node.iniL, node.sec, node.bp + shift


def _cadnano_neighbor_pos(
    mesh: MeshType,
    dna: DNAType,
    query: _CadnanoNeighborQuery,
) -> tuple[int, int, int] | None:
    if query.base_id == -1:
        return None
    node_id = dna.top[query.base_id].node
    if node_id != -1:
        return _cadnano_pos_from_node(mesh, node_id, query.shift)
    return (
        query.cursor.edge,
        query.cursor.sec,
        _cadnano_step_bp(query.cursor.sec, query.cursor.bp, query.strand_type),
    )


def _cadnano_assign_links(
    grid: _CadnanoGrid,
    n_sec: int,
    assignment: _CadnanoLinkAssignment,
) -> None:
    idx = assignment.cursor.bp - 1
    target = grid.scaf if assignment.strand_type == "scaf" else grid.stap
    if assignment.dn_pos is not None:
        target[assignment.cursor.edge][assignment.cursor.sec][idx][0] = (
            assignment.dn_pos[0] * n_sec + assignment.dn_pos[1]
        )
        target[assignment.cursor.edge][assignment.cursor.sec][idx][1] = assignment.dn_pos[2] - 1
    if assignment.up_pos is not None:
        target[assignment.cursor.edge][assignment.cursor.sec][idx][2] = (
            assignment.up_pos[0] * n_sec + assignment.up_pos[1]
        )
        target[assignment.cursor.edge][assignment.cursor.sec][idx][3] = assignment.up_pos[2] - 1


def _populate_cadnano_grid(
    mesh: MeshType,
    dna: DNAType,
    grid: _CadnanoGrid,
    ctx: _CadnanoBuildContext,
) -> None:
    for strand_idx, strand in enumerate(dna.strand):
        c_base = Mani_Go_Start_Base(dna, strand_idx)
        if c_base == -1:
            continue
        c_edge = c_sec = c_bp = 0
        dn_edge = None
        dn_sec = None
        dn_bp = None
        up_edge = None
        up_sec = None
        up_bp = None
        for _ in range(strand.n_base):
            c_node = dna.top[c_base].node
            if c_node != -1:
                c_edge, c_sec, c_bp = _cadnano_pos_from_node(mesh, c_node, ctx.shift)
            else:
                if c_sec % 2 == 0:
                    c_bp = c_bp + 1 if strand.type1 == "scaf" else c_bp - 1
                else:
                    c_bp = c_bp - 1 if strand.type1 == "scaf" else c_bp + 1
            dn_base = dna.top[c_base].dn
            up_base = dna.top[c_base].up
            idx = c_bp - 1
            if 0 <= idx < ctx.width:
                target = grid.scaf if strand.type1 == "scaf" else grid.stap
                if dn_base != -1:
                    dn_node = dna.top[dn_base].node
                    if dn_node != -1:
                        dn_edge = mesh.node[dn_node].iniL
                        dn_sec = mesh.node[dn_node].sec
                        dn_bp = mesh.node[dn_node].bp + ctx.shift
                    else:
                        if dn_edge is None:
                            dn_edge = c_edge
                        if dn_sec is None:
                            dn_sec = c_sec
                            dn_bp = c_bp
                        if dn_sec % 2 == 0:
                            dn_bp = dn_bp + 1 if strand.type1 == "scaf" else dn_bp - 1
                        else:
                            dn_bp = dn_bp - 1 if strand.type1 == "scaf" else dn_bp + 1
                    target[c_edge][c_sec][idx][0] = dn_edge * ctx.n_sec + dn_sec
                    target[c_edge][c_sec][idx][1] = dn_bp - 1
                elif strand.type1 == "stap":
                    grid.stap_col[c_edge][c_sec].append([idx, ctx.colors[(strand_idx + 1) % 12]])

                if up_base != -1:
                    up_node = dna.top[up_base].node
                    if up_node != -1:
                        up_edge = mesh.node[up_node].iniL
                        up_sec = mesh.node[up_node].sec
                        up_bp = mesh.node[up_node].bp + ctx.shift
                    else:
                        if up_edge is None:
                            up_edge = c_edge
                        if up_sec is None:
                            up_sec = c_sec
                            up_bp = c_bp
                        if c_sec % 2 == 0:
                            up_bp = up_bp + 1 if strand.type1 == "scaf" else up_bp - 1
                        else:
                            up_bp = up_bp - 1 if strand.type1 == "scaf" else up_bp + 1
                    target[c_edge][c_sec][idx][2] = up_edge * ctx.n_sec + up_sec
                    target[c_edge][c_sec][idx][3] = up_bp - 1
            c_base = up_base


def _cadnano_vstrand_layout(i: int, j: int, n_sec: int, row_shift: int) -> tuple[int, int, int]:
    if n_sec == 2:
        edges_per_row = 15
        return 1 + 2 * (i % edges_per_row), 1 - j + 2 * (i // edges_per_row), row_shift

    col_shift = (i + 1) % 7
    if col_shift == 1 and j == 0:
        row_shift += 4
    if col_shift == 0:
        col_shift = 7
    positions = {0: (1, 1), 1: (1, 2), 2: (2, 2), 3: (3, 2), 4: (3, 1), 5: (2, 1)}
    pos_c, pos_r = positions.get(j, (1, 1))
    return 4 * col_shift - pos_c, row_shift - pos_r, row_shift


def _build_cadnano_vstrands(
    grid: _CadnanoGrid, width: int, n_edge: int, n_sec: int
) -> list[dict[str, object]]:
    vstrands: list[dict[str, object]] = []
    row_shift = 0
    for i in range(n_edge):
        for j in range(n_sec):
            col, row, row_shift = _cadnano_vstrand_layout(i, j, n_sec, row_shift)
            vstrands.append(
                {
                    "stap_colors": grid.stap_col[i][j],
                    "num": i * n_sec + j,
                    "scafLoop": [],
                    "stap": grid.stap[i][j],
                    "skip": [0] * width,
                    "scaf": grid.scaf[i][j],
                    "stapLoop": [],
                    "col": col,
                    "loop": [0] * width,
                    "row": row,
                }
            )
    return vstrands


def _write_cadnano_json(prob: ProbType, geom: GeomType, mesh: MeshType, dna: DNAType) -> None:
    path = Path(prob.path_work) / f"{prob.name_file}_{prob.n_edge_len}bp_15_json_caDNAno.json"
    name = prob.name_prob or prob.name_file

    if dna.n_base_scaf > 20000:
        return

    width, n_edge, n_sec, shift = _cadnano_dimensions(geom, mesh, _cadnano_max_unpaired(dna))
    grid = _build_cadnano_grid(width, n_edge, n_sec)
    _populate_cadnano_grid(
        mesh,
        dna,
        grid,
        _CadnanoBuildContext(
            width=width,
            n_sec=n_sec,
            shift=shift,
            colors=_cadnano_color_palette(),
        ),
    )
    vstrands = _build_cadnano_vstrands(grid, width, n_edge, n_sec)

    with path.open("w", encoding="utf-8") as f:
        json.dump({"name": name, "vstrands": vstrands}, f, separators=(",", ":"))


def _write_sequences_txt(prob: ProbType, dna: DNAType) -> None:
    path = Path(prob.path_work) / "TXT_Sequence.txt"
    with path.open("w", encoding="utf-8") as f:
        for idx, strand in enumerate(dna.strand, start=1):
            label = "scaf" if strand.type1 == "scaf" else "stap"
            f.write(f"{idx:6d} - [{label}] , # of nts: {strand.n_base}\n")
            seq = []
            for base_id in strand.base:
                if base_id < 0 or base_id >= dna.n_top:
                    seq.append("N")
                else:
                    seq.append(dna.top[base_id].seq or "N")
            f.write("".join(seq) + "\n")


def _write_sequences_csv(prob: ProbType, dna: DNAType) -> None:
    path = Path(prob.path_work) / f"{prob.name_file}_{prob.n_edge_len}bp_17_sequence.csv"
    name_prob = prob.name_prob or prob.name_file

    with path.open("w", encoding="utf-8") as f:
        f.write("\n")
        f.write("No, Type1, Type2, A12, Strand name, Length, Sequence\n")
        f.write("\n")

        n_scaf = 0
        n_stap = 0
        add = 0

        for idx, strand in enumerate(dna.strand):
            if strand.type1 == "scaf":
                n_scaf += 1
                f.write(f"{n_scaf}, ")
                f.write("Scaffold, -, -, -, ")
            else:
                n_stap += 1
                f.write(f"{n_stap}, ")
                type2 = strand.type2 or "-"
                f.write(f"Staple, {type2}, ")

                dozen = n_stap % 12
                if dozen == 0:
                    dozen = 12
                if dozen == 1:
                    add += 1
                a12 = f"{chr(65 + add - 1)}{dozen}"
                f.write(f"{a12}, ")

                f.write(f"{name_prob}_")
                if para.para_vertex_design == "flat":
                    f.write("F_")
                if para.para_vertex_design == "mitered":
                    f.write("M_")
                f.write(f"{prob.n_edge_len}bp_{n_stap}, ")

            f.write(f"{strand.n_base}, ")

            base = Mani_Go_Start_Base(dna, idx)
            for _ in range(strand.n_base):
                f.write(dna.top[base].seq)
                base = dna.top[base].up

            if strand.type1 == "scaf":
                f.write("\n")
            f.write("\n")


def _write_strand_base_txt(prob: ProbType, dna: DNAType) -> None:
    path = Path(prob.path_work) / "strand_base.txt"
    with path.open("w", encoding="utf-8") as f:
        for i, strand in enumerate(dna.strand, start=1):
            f.write(f"{i} {strand.type1} n_base={strand.n_base} circular={strand.b_circular}\n")
            f.write(" ".join(str(b) for b in strand.base) + "\n")

        f.write("*\nGLINE NODES=2 AUXPOINT=0 NCOINCID=ENDS NCENDS=12,\n")
        f.write("     NCTOLERA=1.00000000000000E-05 SUBSTRUC=0 GROUP=1 MIDNODES=CURVED,\n")
        f.write("     XO=0.00000000000000 YO=1.00000000000000 ZO=0.00000000000000,\n")
        f.write("     XYZOSYST=SKEW\n")
        f.write("@CLEAR\n")
        for i in range(1, mesh.n_ele + 1):
            f.write(f"{i}\n")
        f.write("@\n")
