from __future__ import annotations

from . import para
from .data_geom import GeomType
from .data_mesh import MeshType
from .data_prob import ProbType
from .basepair_edge_ops import (
    _edge_extension_count_from_lengths,
    _edge_increase_endpoint_pair,
    _mark_edge_endpoint_connection,
)
from .basepair_ghost import (
    _GhostNodeContext,
    _apply_ghost_node_replacement as _apply_ghost_node_replacement_impl,
    _delete_ghost_node,
    _delete_ghost_node_chain as _delete_ghost_node_chain_impl,
    _find_ghost_chain_replacement,
    _renumber_element_after_delete,
    _renumber_optional_node_ref_after_delete,
    _replace_junction_connection_node,
    _shift_junction_references_after_node_delete,
)
from .basepair_junction import (
    _apply_vertex_crash_strategy,
    _basepair_adjustment_endpoints,
    _decrease_basepair,
    _find_internal_junction_lines,
    _find_xover_nearby,
    _ghost_node_walk_start,
    _increase_basepair,
    _initial_xover_move,
    _mark_ghost_chain_until_second_xover,
    _make_ghost_node,
    _modify_junction,
    _should_increase_mitered_edge,
    _unique_junction_connections,
    _xover_search_starts_inside,
    _xover_search_starts_outside,
    _xover_search_steps,
)
from .basepair_mesh import (
    _apply_junction_self_connections,
    _apply_open_geometry_junction_fallback,
    _count_basepair,
    _generate_basepair,
    _round_mesh_positions,
    _set_conn_junction,
)
from .basepair_output import (
    _append_sticky_end_node,
    _copy_sticky_end_node_fields,
    _make_sticky_end,
    _place_sticky_end_node,
    _replace_sticky_end_junction_node,
    _reserve_mesh_node_and_element,
    _write_cylinder_bild,
    _write_edge_length,
    _write_mesh_bild,
)


def basepair_discretize(prob: ProbType, geom: GeomType, mesh: MeshType) -> None:
    _count_basepair(prob, geom, mesh)
    _generate_basepair(geom, mesh)
    _round_mesh_positions(mesh, digits=4)
    _set_conn_junction(geom, mesh)
    _write_cylinder_bild(prob, geom, mesh, "cylinder_prior")
    _modify_junction(prob, geom, mesh)
    _delete_ghost_node(geom, mesh)
    _make_sticky_end(geom, mesh)
    _write_cylinder_bild(prob, geom, mesh, "cylinder_final")
    _write_mesh_bild(prob, geom, mesh)
    _write_edge_length(prob, geom)
    _round_mesh_positions(mesh, digits=4)


def _apply_ghost_node_replacement(
    geom: GeomType,
    mesh: MeshType,
    junc_idx: int,
    arm_idx: int,
    sec_idx: int,
    old_node: int,
    replacement_node: int,
    break_field: str,
) -> int:
    return _apply_ghost_node_replacement_impl(
        geom,
        mesh,
        _GhostNodeContext(junc_idx, arm_idx, sec_idx),
        old_node,
        replacement_node,
        break_field,
    )


def _delete_ghost_node_chain(
    geom: GeomType,
    mesh: MeshType,
    junc_idx: int,
    arm_idx: int,
    sec_idx: int,
    start_node: int,
    replacement_node: int,
    break_field: str,
) -> None:
    _delete_ghost_node_chain_impl(
        geom,
        mesh,
        _GhostNodeContext(junc_idx, arm_idx, sec_idx),
        start_node,
        replacement_node,
        break_field,
    )


def _apply_vertex_crash_strategy(geom, mesh, node_cur: int, node_com: int) -> None:
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
