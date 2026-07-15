from __future__ import annotations

import json
from pathlib import Path

from .data_dna import DNAType
from .data_geom import GeomType
from .data_mesh import MeshType
from .data_prob import ProbType
from . import para
from .reference_frame import compute_reference_frame, transform_position


def dump_mesh_snapshot(
    prob: ProbType,
    geom: GeomType,
    mesh: MeshType,
    stage: str,
    dna: DNAType | None = None,
) -> Path:
    """Write an in-memory mesh/DNA snapshot to JSON for debugging."""
    out_path = Path(prob.path_work) / f"{prob.name_file}_{prob.n_edge_len}bp_mesh_{stage}.json"
    rf = compute_reference_frame(prob, geom, mesh, mode=para.para_debug_frame_mode)
    payload = {
        "stage": stage,
        "reference_frame": {
            "mode": rf.mode,
            "origin": [float(rf.origin[0]), float(rf.origin[1]), float(rf.origin[2])],
            "axes": [
                [float(rf.axes[0][0]), float(rf.axes[0][1]), float(rf.axes[0][2])],
                [float(rf.axes[1][0]), float(rf.axes[1][1]), float(rf.axes[1][2])],
                [float(rf.axes[2][0]), float(rf.axes[2][1]), float(rf.axes[2][2])],
            ],
            "scale": float(rf.scale),
            "scale_basis": rf.scale_basis,
            "abs_center": [float(rf.abs_center[0]), float(rf.abs_center[1]), float(rf.abs_center[2])]
            if rf.abs_center is not None
            else None,
            "abs_total_scale": float(rf.abs_total_scale),
        },
        "n_node": mesh.n_node,
        "n_ele": mesh.n_ele,
        "n_mitered": mesh.n_mitered,
        "node": [
            {
                "id": n.id,
                "bp": n.bp,
                "up": n.up,
                "dn": n.dn,
                "sec": n.sec,
                "iniL": n.iniL,
                "croL": n.croL,
                "mitered": n.mitered,
                "conn": n.conn,
                "ghost": n.ghost,
                "pos": [float(n.pos[0]), float(n.pos[1]), float(n.pos[2])],
                "pos_ref": list(transform_position(n.pos, rf)),
                "ori": [
                    [float(n.ori[0][0]), float(n.ori[0][1]), float(n.ori[0][2])],
                    [float(n.ori[1][0]), float(n.ori[1][1]), float(n.ori[1][2])],
                    [float(n.ori[2][0]), float(n.ori[2][1]), float(n.ori[2][2])],
                ],
            }
            for n in mesh.node
        ],
        "ele": [{"cn": [int(e.cn[0]), int(e.cn[1])]} for e in mesh.ele],
        "dna": {
            "n_base_scaf": 0 if dna is None else int(dna.n_base_scaf),
            "n_base_stap": 0 if dna is None else int(dna.n_base_stap),
            "base_scaf": []
            if dna is None
            else [
                {
                    "id": int(b.id),
                    "node": int(b.node),
                    "pos": [float(b.pos[0]), float(b.pos[1]), float(b.pos[2])],
                    "pos_ref": list(transform_position(b.pos, rf)),
                }
                for b in dna.base_scaf
            ],
            "base_stap": []
            if dna is None
            else [
                {
                    "id": int(b.id),
                    "node": int(b.node),
                    "pos": [float(b.pos[0]), float(b.pos[1]), float(b.pos[2])],
                    "pos_ref": list(transform_position(b.pos, rf)),
                }
                for b in dna.base_stap
            ],
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
