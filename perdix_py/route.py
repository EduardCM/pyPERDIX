from . import _route_impl as _route_impl_module


globals().update(
    {
        name: value
        for name, value in vars(_route_impl_module).items()
        if not name.startswith("__")
    }
)


_apply_scaffold_xover_pair_impl = _apply_scaffold_xover_pair
_select_initial_centered_scaffold_split_impl = _select_initial_centered_scaffold_split
_select_best_centered_scaffold_split_impl = _select_best_centered_scaffold_split


def _apply_scaffold_xover_pair(
    dna,
    node_cur: int,
    node_com: int,
    neighbors=None,
    *,
    b_nei_up: bool | None = None,
    b_nei_dn: bool | None = None,
    up_cur: int | None = None,
    up_com: int | None = None,
    dn_cur: int | None = None,
    dn_com: int | None = None,
):
    if neighbors is None:
        neighbors = _ScaffoldNeighborPair(
            b_nei_up=bool(b_nei_up),
            b_nei_dn=bool(b_nei_dn),
            up_cur=-1 if up_cur is None else up_cur,
            up_com=-1 if up_com is None else up_com,
            dn_cur=-1 if dn_cur is None else dn_cur,
            dn_com=-1 if dn_com is None else dn_com,
        )
    return _apply_scaffold_xover_pair_impl(dna, node_cur, node_com, neighbors)


def _select_initial_centered_scaffold_split(
    geom,
    mesh,
    choice=None,
    bounds=None,
    *,
    node_cur: int | None = None,
    node_com: int | None = None,
    min_bp1: int | None = None,
    max_bp1: int | None = None,
    min_bp2: int | None = None,
    max_bp2: int | None = None,
):
    if choice is not None:
        node_cur = choice.node_cur
        node_com = choice.node_com
    if bounds is not None:
        min_bp1 = bounds.min_bp1
        max_bp1 = bounds.max_bp1
        min_bp2 = bounds.min_bp2
        max_bp2 = bounds.max_bp2
    if geom.sec.maxR == 1 and geom.sec.maxC == 2:
        step = 1 if para.para_set_xover_scaf == "center" else 5
    else:
        step = 1 if para.para_set_xover_scaf == "center" else 3
    b_fail, split_cur, split_com = _split_centered_scaf_xover(
        geom, mesh, node_cur, node_com, min_bp1, max_bp1, min_bp2, max_bp2, step
    )
    if step == 3 and b_fail:
        _b_fail, split_cur, split_com = _split_centered_scaf_xover(
            geom, mesh, split_cur, split_com, min_bp1, max_bp1, min_bp2, max_bp2, step + 1
        )
    return split_cur, split_com


def _select_best_centered_scaffold_split(
    geom,
    mesh,
    dna,
    choice=None,
    bounds=None,
    *,
    node_cur: int | None = None,
    node_com: int | None = None,
    min_bp1: int | None = None,
    max_bp1: int | None = None,
    min_bp2: int | None = None,
    max_bp2: int | None = None,
    max_gap: int | None = None,
    max_cur: int | None = None,
    max_com: int | None = None,
):
    if choice is not None:
        node_cur = choice.node_cur
        node_com = choice.node_com
        max_gap = choice.max_gap
        max_cur = choice.max_cur
        max_com = choice.max_com
    if bounds is not None:
        min_bp1 = bounds.min_bp1
        max_bp1 = bounds.max_bp1
        min_bp2 = bounds.min_bp2
        max_bp2 = bounds.max_bp2
    n_gap = max_gap
    split_cur = node_cur
    split_com = node_com
    for k in range(1, 6):
        split_cur = node_cur
        split_com = node_com
        b_fail, split_cur, split_com = _split_centered_scaf_xover(
            geom, mesh, split_cur, split_com, min_bp1, max_bp1, min_bp2, max_bp2, k
        )
        n_gap = _check_nei_xover(geom, mesh, dna, split_cur, split_com)
        if not b_fail and n_gap == para.para_gap_xover_two_scaf:
            break
        if not b_fail and n_gap != para.para_gap_xover_two_scaf:
            if n_gap >= max_gap:
                max_gap = n_gap
                max_cur = split_cur
                max_com = split_com
        if k == 5:
            split_cur = max_cur
            split_com = max_com
    return split_cur, split_com, n_gap
