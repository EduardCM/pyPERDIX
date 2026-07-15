from __future__ import annotations

from pathlib import Path

from .data_prob import ProbType


def bild_filename(prob: ProbType, stem: str) -> str:
    return f"{prob.name_file}_{prob.n_edge_len}bp_{stem}.bild"


def bild_path(prob: ProbType, stem: str) -> Path:
    return Path(prob.path_work) / bild_filename(prob, stem)
