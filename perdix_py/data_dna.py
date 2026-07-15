from __future__ import annotations

from dataclasses import dataclass, field


Vector3 = tuple[float, float, float]


@dataclass
class BaseType:
    id: int = 0
    node: int = 0
    up: int = 0
    dn: int = 0
    xover: int = 0
    across: int = 0
    strand: int = 0
    pos: Vector3 = (0.0, 0.0, 0.0)


@dataclass
class TopType:
    id: int = 0
    node: int = 0
    up: int = 0
    dn: int = 0
    xover: int = 0
    across: int = 0
    strand: int = 0
    address: int = 0
    b_14nt: bool = False
    seq: str = ""
    status: str = "N"
    pos: Vector3 = (0.0, 0.0, 0.0)


@dataclass
class StrandType:
    n_base: int = 0
    n_14nt: int = 0
    n_4nt: int = 0
    b_circular: bool = False
    type1: str = ""
    type2: str = ""
    base: list[int] = field(default_factory=list)


@dataclass
class DNAType:
    n_base_scaf: int = 0
    n_base_stap: int = 0
    n_xover_scaf: int = 0
    n_xover_stap: int = 0
    n_sxover_stap: int = 0
    n_scaf: int = 0
    n_stap: int = 0
    n_top: int = 0
    n_strand: int = 0

    len_ave_stap: float = 0.0
    n_14nt: int = 0
    n_s14nt: int = 0
    n_4nt: int = 0
    n_only_4nt: int = 0
    n_nt_14nt: int = 0
    n_nt_4nt: int = 0
    n_tot_region: int = 0
    n_tot_14nt: int = 0
    n_tot_4nt: int = 0
    len_min_stap: int = 0
    len_max_stap: int = 0
    n_unpaired_scaf: int = 0
    n_nt_unpaired_scaf: int = 0
    n_unpaired_stap: int = 0
    n_nt_unpaired_stap: int = 0
    graph_node: int = 0
    graph_edge: int = 0
    min_xover_scaf: int = 0
    min_xover_stap: int = 0

    base_scaf: list[BaseType] = field(default_factory=list)
    base_stap: list[BaseType] = field(default_factory=list)
    top: list[TopType] = field(default_factory=list)
    strand: list[StrandType] = field(default_factory=list)
    order_stap: list[list[int]] = field(default_factory=list)

    # Graph data for spanning tree visualization (1-based indices, index 0 unused)
    graph_pos_node: list[Vector3] = field(default_factory=list)
    graph_tail: list[int] = field(default_factory=list)
    graph_head: list[int] = field(default_factory=list)
    graph_tree: list[int] = field(default_factory=list)
