from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from . import para
from .data_dna import DNAType
from .data_geom import GeomType
from .data_mesh import MeshType
from .data_prob import ProbType
from .validation_support import (
    ValidationError,
    ValidationIssue,
    add_issue as _add,
    check_count as _check_count,
    check_finite_vector as _check_finite_vector,
    check_ref as _check_ref,
    format_issue as _format_issue,
    format_issue_summary as _format_issue_summary,
    vector_norm as _vector_norm,
)


def validate_pipeline_state(
    stage: str,
    prob: ProbType,
    geom: GeomType,
    mesh: MeshType,
    dna: DNAType,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _validate_prob(stage, prob, issues)
    _validate_geom(stage, geom, len(mesh.node), issues)
    _validate_mesh(stage, mesh, issues)
    _validate_dna(stage, mesh, dna, issues)
    return issues


def enforce_pipeline_validation(
    stage: str,
    prob: ProbType,
    geom: GeomType,
    mesh: MeshType,
    dna: DNAType,
) -> list[ValidationIssue]:
    if not para.para_validate_pipeline:
        return []

    issues = validate_pipeline_state(stage, prob, geom, mesh, dna)
    for issue in issues:
        sys.stderr.write(f"{_format_issue(issue)}\n")

    fatal = [issue for issue in issues if issue.severity == "high"]
    if fatal and para.para_validate_pipeline_strict:
        raise ValidationError(fatal)
    return issues


def _validate_prob(stage: str, prob: ProbType, issues: list[ValidationIssue]) -> None:
    if stage != "initial" and not prob.name_file:
        _add(issues, stage, "medium", "prob.name_file.empty", "prob.name_file is empty")
    if stage not in {"initial", "input_initialize"} and prob.n_edge_len <= 0:
        _add(
            issues,
            stage,
            "high",
            "prob.n_edge_len.invalid",
            f"prob.n_edge_len must be positive, got {prob.n_edge_len}",
        )
    if stage not in {"initial"} and prob.name_file and not prob.path_work:
        _add(issues, stage, "medium", "prob.path_work.empty", "prob.path_work is empty")


def _validate_geom(
    stage: str,
    geom: GeomType,
    mesh_len: int,
    issues: list[ValidationIssue],
) -> None:
    _check_count(issues, stage, "geom.n_sec", geom.n_sec, geom.sec.id)
    _check_count(issues, stage, "geom.n_iniP", geom.n_iniP, geom.iniP)
    _check_count(issues, stage, "geom.n_iniL", geom.n_iniL, geom.iniL)
    _check_count(issues, stage, "geom.n_modP", geom.n_modP, geom.modP)
    _check_count(issues, stage, "geom.n_croP", geom.n_croP, geom.croP)
    _check_count(issues, stage, "geom.n_croL", geom.n_croL, geom.croL)
    _check_count(issues, stage, "geom.n_face", geom.n_face, geom.face)
    _check_count(issues, stage, "geom.n_junc", geom.n_junc, geom.junc)
    _validate_geom_lines(stage, geom, issues)
    _validate_geom_faces(stage, geom, issues)
    _validate_geom_sections(stage, geom, issues)
    _validate_geom_junctions(stage, geom, mesh_len, issues)


def _validate_geom_lines(stage: str, geom: GeomType, issues: list[ValidationIssue]) -> None:
    line_point_ref_len = len(geom.modP) if geom.modP else len(geom.iniP)
    for idx, line in enumerate(geom.iniL):
        for field_name, values, ref_len in (
            ("poi", line.poi, line_point_ref_len),
            ("iniP", line.iniP, len(geom.iniP)),
        ):
            if len(values) != 2:
                _add(
                    issues,
                    stage,
                    "high",
                    "geom.line.arity",
                    f"geom.iniL[{idx}].{field_name} must contain two point ids",
                )
                continue
            for pos, point_id in enumerate(values):
                _check_ref(
                    issues,
                    stage,
                    f"geom.iniL[{idx}].{field_name}[{pos}]",
                    point_id,
                    ref_len,
                    allow_sentinel=False,
                )


def _validate_geom_faces(stage: str, geom: GeomType, issues: list[ValidationIssue]) -> None:
    for idx, face in enumerate(geom.face):
        if face.n_poi != len(face.poi):
            _add(
                issues,
                stage,
                "high",
                "geom.face.count_mismatch",
                f"geom.face[{idx}].n_poi={face.n_poi}, len(poi)={len(face.poi)}",
            )
        for pos, point_id in enumerate(face.poi):
            _check_ref(
                issues,
                stage,
                f"geom.face[{idx}].poi[{pos}]",
                point_id,
                len(geom.iniP),
                allow_sentinel=False,
            )


def _validate_geom_sections(stage: str, geom: GeomType, issues: list[ValidationIssue]) -> None:
    if geom.n_sec:
        _check_count(issues, stage, "geom.sec.posR", geom.n_sec, geom.sec.posR)
        _check_count(issues, stage, "geom.sec.posC", geom.n_sec, geom.sec.posC)
        _check_count(issues, stage, "geom.sec.conn", geom.n_sec, geom.sec.conn)
        for idx, sec_id in enumerate(geom.sec.conn):
            _check_ref(
                issues,
                stage,
                f"geom.sec.conn[{idx}]",
                sec_id,
                geom.n_sec,
                allow_sentinel=True,
            )


def _validate_geom_junctions(
    stage: str,
    geom: GeomType,
    mesh_len: int,
    issues: list[ValidationIssue],
) -> None:
    for junc_idx, junc in enumerate(geom.junc):
        if junc.n_arm != len(junc.iniL):
            _add(
                issues,
                stage,
                "medium",
                "geom.junction.arm_count_mismatch",
                f"geom.junc[{junc_idx}].n_arm={junc.n_arm}, len(iniL)={len(junc.iniL)}",
            )
        for arm_idx, line_id in enumerate(junc.iniL):
            _check_ref(
                issues,
                stage,
                f"geom.junc[{junc_idx}].iniL[{arm_idx}]",
                line_id,
                len(geom.iniL),
                allow_sentinel=False,
            )
        for arm_idx, row in enumerate(junc.croP):
            for sec_idx, point_id in enumerate(row):
                _check_ref(
                    issues,
                    stage,
                    f"geom.junc[{junc_idx}].croP[{arm_idx}][{sec_idx}]",
                    point_id,
                    len(geom.croP),
                    allow_sentinel=True,
                )
        if mesh_len:
            for arm_idx, row in enumerate(junc.node):
                for sec_idx, node_id in enumerate(row):
                    _check_ref(
                        issues,
                        stage,
                        f"geom.junc[{junc_idx}].node[{arm_idx}][{sec_idx}]",
                        node_id,
                        mesh_len,
                        allow_sentinel=True,
                    )
            for conn_idx, pair in enumerate(junc.conn):
                if len(pair) != 2:
                    _add(
                        issues,
                        stage,
                        "high",
                        "geom.junction.conn_arity",
                        f"geom.junc[{junc_idx}].conn[{conn_idx}] must contain two node ids",
                    )
                    continue
                for pos, node_id in enumerate(pair):
                    _check_ref(
                        issues,
                        stage,
                        f"geom.junc[{junc_idx}].conn[{conn_idx}][{pos}]",
                        node_id,
                        mesh_len,
                        allow_sentinel=True,
                    )


@dataclass(frozen=True)
class _ReciprocalCheck:
    label: str
    idx: int
    field_name: str
    target_idx: int
    collection: list[Any]
    target_field: str


def _validate_mesh(stage: str, mesh: MeshType, issues: list[ValidationIssue]) -> None:
    _check_count(issues, stage, "mesh.n_node", mesh.n_node, mesh.node)
    _check_count(issues, stage, "mesh.n_ele", mesh.n_ele, mesh.ele)

    for idx, node in enumerate(mesh.node):
        if node.id != idx:
            _add(
                issues,
                stage,
                "high",
                "mesh.node.id_mismatch",
                f"mesh.node[{idx}].id={node.id}, expected {idx}",
            )
        for field_name in ("up", "dn", "mitered"):
            _check_ref(
                issues,
                stage,
                f"mesh.node[{idx}].{field_name}",
                getattr(node, field_name),
                len(mesh.node),
                allow_sentinel=True,
            )
        _check_finite_vector(issues, stage, f"mesh.node[{idx}].pos", node.pos)
        if stage in {"route_generation", "seqdesign_design", "output_generation"}:
            for axis_idx, axis in enumerate(node.ori):
                _check_finite_vector(issues, stage, f"mesh.node[{idx}].ori[{axis_idx}]", axis)
                if _vector_norm(axis) <= 1e-12:
                    _add(
                        issues,
                        stage,
                        "medium",
                        "mesh.node.orientation_zero",
                        f"mesh.node[{idx}].ori[{axis_idx}] is zero-length",
                    )

    for idx, ele in enumerate(mesh.ele):
        a, b = ele.cn
        _check_ref(issues, stage, f"mesh.ele[{idx}].cn[0]", a, len(mesh.node), allow_sentinel=False)
        _check_ref(issues, stage, f"mesh.ele[{idx}].cn[1]", b, len(mesh.node), allow_sentinel=False)
        if a == b:
            _add(
                issues,
                stage,
                "high",
                "mesh.element.self_edge",
                f"mesh.ele[{idx}] has identical endpoints {a}",
            )


def _validate_dna(
    stage: str,
    mesh: MeshType,
    dna: DNAType,
    issues: list[ValidationIssue],
) -> None:
    _check_count(issues, stage, "dna.n_base_scaf", dna.n_base_scaf, dna.base_scaf)
    _check_count(issues, stage, "dna.n_base_stap", dna.n_base_stap, dna.base_stap)
    _check_count(issues, stage, "dna.n_top", dna.n_top, dna.top)
    _check_count(issues, stage, "dna.n_strand", dna.n_strand, dna.strand)

    _validate_base_list(stage, "dna.base_scaf", dna.base_scaf, len(mesh.node), issues)
    _validate_base_list(stage, "dna.base_stap", dna.base_stap, len(mesh.node), issues)

    for idx, top in enumerate(dna.top):
        if top.id != idx:
            _add(
                issues,
                stage,
                "high",
                "dna.top.id_mismatch",
                f"dna.top[{idx}].id={top.id}, expected {idx}",
            )
        _check_ref(issues, stage, f"dna.top[{idx}].up", top.up, len(dna.top), allow_sentinel=True)
        _check_ref(issues, stage, f"dna.top[{idx}].dn", top.dn, len(dna.top), allow_sentinel=True)
        _check_ref(issues, stage, f"dna.top[{idx}].xover", top.xover, len(dna.top), allow_sentinel=True)
        _check_ref(issues, stage, f"dna.top[{idx}].across", top.across, len(dna.top), allow_sentinel=True)
        _check_ref(issues, stage, f"dna.top[{idx}].node", top.node, len(mesh.node), allow_sentinel=True)
        if top.strand != -1 and not (1 <= top.strand <= len(dna.strand)):
            _add(
                issues,
                stage,
                "high",
                "dna.top.strand.invalid",
                f"dna.top[{idx}].strand={top.strand}, valid range is 1..{len(dna.strand)} or -1",
            )
        _check_reciprocal(issues, stage, _ReciprocalCheck("dna.top", idx, "xover", top.xover, dna.top, "xover"))
        _check_reciprocal(issues, stage, _ReciprocalCheck("dna.top", idx, "across", top.across, dna.top, "across"))
        _check_finite_vector(issues, stage, f"dna.top[{idx}].pos", top.pos)

    for strand_idx, strand in enumerate(dna.strand):
        if strand.n_base != len(strand.base):
            _add(
                issues,
                stage,
                "high",
                "dna.strand.count_mismatch",
                f"dna.strand[{strand_idx}].n_base={strand.n_base}, len(base)={len(strand.base)}",
            )
        for base_pos, top_id in enumerate(strand.base):
            _check_ref(
                issues,
                stage,
                f"dna.strand[{strand_idx}].base[{base_pos}]",
                top_id,
                len(dna.top),
                allow_sentinel=False,
            )


def _validate_base_list(
    stage: str,
    label: str,
    bases: list[Any],
    mesh_len: int,
    issues: list[ValidationIssue],
) -> None:
    for idx, base in enumerate(bases):
        if base.id != idx:
            _add(
                issues,
                stage,
                "high",
                f"{label}.id_mismatch",
                f"{label}[{idx}].id={base.id}, expected {idx}",
            )
        _check_ref(issues, stage, f"{label}[{idx}].up", base.up, len(bases), allow_sentinel=True)
        _check_ref(issues, stage, f"{label}[{idx}].dn", base.dn, len(bases), allow_sentinel=True)
        _check_ref(issues, stage, f"{label}[{idx}].xover", base.xover, len(bases), allow_sentinel=True)
        _check_ref(issues, stage, f"{label}[{idx}].across", base.across, len(bases), allow_sentinel=True)
        _check_ref(issues, stage, f"{label}[{idx}].node", base.node, mesh_len, allow_sentinel=True)
        _check_finite_vector(issues, stage, f"{label}[{idx}].pos", base.pos)
        _check_reciprocal(issues, stage, _ReciprocalCheck(label, idx, "xover", base.xover, bases, "xover"))


def _check_reciprocal(
    issues: list[ValidationIssue],
    stage: str,
    check: _ReciprocalCheck,
) -> None:
    if check.target_idx == -1 or check.target_idx < 0 or check.target_idx >= len(check.collection):
        return
    actual = getattr(check.collection[check.target_idx], check.target_field)
    if actual != check.idx:
        _add(
            issues,
            stage,
            "high",
            "link.not_reciprocal",
            f"{check.label}[{check.idx}].{check.field_name}={check.target_idx}, "
            f"but {check.label}[{check.target_idx}].{check.target_field}={actual}",
        )
