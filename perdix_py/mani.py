from __future__ import annotations

"""Geometric manipulation helpers (Fortran: Mani.f90)."""

import sys
from dataclasses import replace
from typing import Iterable, TextIO

from .data_dna import BaseType, DNAType, StrandType
from .data_geom import GeomType, LineType, SecType
from .data_mesh import EleType, MeshType, NodeType
from .data_prob import ProbType
from .math_utils import deg2rad


def Mani_Set_Prob(prob: ProbType, color: tuple[int, int, int]) -> None:
    prob.name_file = f"{prob.name_prob}_DX_{prob.n_edge_len}bp"
    prob.color = color


def Mani_Set_Geo_Ori(geom: GeomType, pseudo: tuple[float, float, float], angle: float) -> None:
    # Rotate initial points about Z-axis and center geometry.
    import math

    rad = deg2rad(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    for p in geom.iniP:
        x, y, z = p.pos
        rx = cos_a * x - sin_a * y
        ry = sin_a * x + cos_a * y
        p.pos = (rx, ry, z)

    if not geom.iniP:
        return

    cx = sum(p.pos[0] for p in geom.iniP) / len(geom.iniP)
    cy = sum(p.pos[1] for p in geom.iniP) / len(geom.iniP)
    for p in geom.iniP:
        x, y, z = p.pos
        p.pos = (x - cx, y - cy, z)


def Space(n_unit: TextIO, num: int) -> None:
    n_unit.write(" " * num)


def Mani_To_Upper(str_in: str) -> str:
    return str_in.upper()


def Mani_Set_Chimera_Axis(fout: TextIO) -> None:
    fout.write(".translate 0.0 0.0 0.0\n")
    fout.write(".scale 0.5\n")
    fout.write(".color grey\n")
    fout.write(".sphere 0 0 0 0.5\n")
    fout.write(".color red\n")
    fout.write(".arrow 0 0 0 4 0 0 \n")
    fout.write(".color blue\n")
    fout.write(".arrow 0 0 0 0 4 0 \n")
    fout.write(".color yellow\n")
    fout.write(".arrow 0 0 0 0 0 4 \n")


def Mani_Progress_Bar(index: int, max_value: int) -> None:
    if max_value <= 0:
        return
    step = int(index * 10 / max_value)
    step = max(0, min(step, 10))
    bar = [" "] * 20
    for i in range(step):
        pos = 2 * i
        bar[pos] = "-"
        bar[pos + 1] = "-" if i < step - 1 or step == 10 else ">"
    percent = 10 * step
    ending = "" if step < 10 else "\n"
    sys.stdout.write(f"\r   * Progressing.... [{''.join(bar)}] {percent:3d}%{ending}")
    sys.stdout.flush()


def Mani_Init_LineType(n_line: int) -> list[LineType]:
    line: list[LineType] = []
    for _ in range(n_line):
        lt = LineType()
        lt.iniL = -1
        lt.sec = -1
        lt.poi = [-1, -1]
        lt.neiP = [[-1, -1], [-1, -1]]
        lt.neiL = [[-1, -1], [-1, -1]]
        lt.t = [(0.0, 0.0, 0.0)] * 3
        line.append(lt)
    return line


def Mani_Allocate_SecType(sec: SecType, n_sec: int) -> None:
    sec.id = [-1] * n_sec
    sec.posR = [-1] * n_sec
    sec.posC = [-1] * n_sec
    sec.conn = [-1] * n_sec


def Mani_Init_SecType(sec: SecType, n_sec: int, types: str) -> None:
    m_value = 9999999
    sec.types = types
    sec.maxR = -m_value
    sec.minR = m_value
    sec.maxC = -m_value
    sec.minC = m_value
    sec.n_row = -1
    sec.n_col = -1
    if not sec.id or len(sec.id) != n_sec:
        Mani_Allocate_SecType(sec, n_sec)
    for i in range(n_sec):
        sec.id[i] = -1
        sec.posR[i] = -1
        sec.posC[i] = -1
        sec.conn[i] = -1


def Mani_Init_MeshType(mesh: MeshType) -> None:
    for node in mesh.node:
        node.id = -1
        node.bp = -1
        node.up = -1
        node.dn = -1
        node.sec = -1
        node.iniL = -1
        node.croL = -1
        node.mitered = -1
        node.conn = -1
        node.ghost = -1
        node.pos = (0.0, 0.0, 0.0)
        node.ori = ((0.0, 0.0, 0.0),) * 3

    for ele in mesh.ele:
        ele.cn = (-1, -1)


def Mani_Init_Node(node: list[NodeType], n_node: int) -> None:
    for i in range(n_node):
        node[i].id = -1
        node[i].bp = -1
        node[i].up = -1
        node[i].dn = -1
        node[i].sec = -1
        node[i].iniL = -1
        node[i].croL = -1
        node[i].mitered = -1
        node[i].conn = -1
        node[i].ghost = -1
        node[i].pos = (0.0, 0.0, 0.0)
        node[i].ori = ((0.0, 0.0, 0.0),) * 3


def Mani_Init_Ele(ele: list[EleType], n_ele: int) -> None:
    for i in range(n_ele):
        ele[i].cn = (-1, -1)


def Mani_Copy_NodeType(ori: Iterable[NodeType], copy: list[NodeType], num: int) -> None:
    for i in range(num):
        copy[i] = replace(ori[i])


def Mani_Copy_EleType(ori: Iterable[EleType], copy: list[EleType], num: int) -> None:
    for i in range(num):
        copy[i] = replace(ori[i])


def Mani_Init_BaseType(base: list[BaseType], n_base: int) -> None:
    for i in range(n_base):
        base[i].id = -1
        base[i].node = -1
        base[i].up = -1
        base[i].dn = -1
        base[i].xover = -1
        base[i].across = -1
        base[i].strand = -1
        base[i].pos = (0.0, 0.0, 0.0)


def Mani_Init_StrandType(strand: list[StrandType], n_strand: int) -> None:
    for i in range(n_strand):
        strand[i].n_base = 0
        strand[i].b_circular = False
        strand[i].type1 = "NNNN"
        strand[i].type2 = "edge"


def Mani_Copy_BaseType(ori: Iterable[BaseType], copy: list[BaseType], num: int) -> None:
    for i in range(num):
        copy[i] = replace(ori[i])


def Mani_Go_Start_Base(dna: DNAType, strand: int) -> int:
    if strand < 0 or strand >= len(dna.strand):
        return -1
    if dna.strand[strand].n_base <= 0 or not dna.strand[strand].base:
        return -1
    base = dna.strand[strand].base[0]
    for _ in range(dna.strand[strand].n_base):
        if base == -1 or dna.top[base].dn == -1:
            break
        base = dna.top[base].dn
    return base


def Mani_Go_Base(dna: DNAType, strand: int, n: int) -> int:
    base = Mani_Go_Start_Base(dna, strand)
    if base == -1:
        return -1
    for _ in range(n):
        base = dna.top[base].up
        if base == -1:
            break
    return base
