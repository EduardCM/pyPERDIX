from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProbType:
    """Problem configuration (Fortran: Data_Prob.ProbType)."""

    sel_prob: int = 0
    sel_vertex: int = 0
    sel_edge_sec: int = 0
    sel_edge_len: int = 0
    sel_edge_ref: int = 0
    n_edge_len: int = 0

    p_mesh: float = 0.0

    scaf_seq: str = ""

    color: tuple[int, int, int] = (52, 152, 219)
    n_cng_min_stap: int = 0
    n_cng_max_stap: int = 0

    name_file: str = ""
    name_prob: str = ""
    type_file: str = ""
    type_geo: str = "closed"
    path_work: str = ""
    path_input: str = ""

    # Debug reference-frame bookkeeping
    input_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    input_init_scale: float = 1.0
    input_modgeo_scale: float = 1.0
