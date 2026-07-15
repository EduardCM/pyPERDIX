from __future__ import annotations

from dataclasses import dataclass

from .data_geom import GeomType
from .data_mesh import EleType, MeshType, NodeType


def _delete_ghost_node(geom: GeomType, mesh: MeshType) -> None:
    for i in range(geom.n_junc):
        for j in range(geom.junc[i].n_arm):
            for k in range(geom.n_sec):
                str_node = geom.junc[i].node[j][k]
                if mesh.node[str_node].ghost != 1:
                    continue
                replacement = _find_ghost_chain_replacement(mesh, str_node)
                if replacement is None:
                    continue
                replacement_node, break_field = replacement
                _delete_ghost_node_chain(geom, mesh, _GhostNodeContext(i, j, k), str_node, replacement_node, break_field)


def _find_ghost_chain_replacement(mesh: MeshType, start_node: int) -> tuple[int, str] | None:
    if mesh.node[start_node].dn == -1:
        walk_field = "up"
        break_field = "dn"
    else:
        walk_field = "dn"
        break_field = "up"
    node = start_node
    while True:
        node = getattr(mesh.node[node], walk_field)
        if node == -1:
            return None
        if mesh.node[node].ghost != 1:
            return node, break_field


@dataclass(frozen=True)
class _GhostNodeContext:
    junc_idx: int
    arm_idx: int
    sec_idx: int


def _delete_ghost_node_chain(geom: GeomType, mesh: MeshType, ghost_ctx: _GhostNodeContext, start_node: int, replacement_node: int, break_field: str) -> None:
    end_node = _apply_ghost_node_replacement(geom, mesh, ghost_ctx, start_node, replacement_node, break_field)
    min_idx, max_idx = (end_node, start_node) if start_node > end_node else (start_node, end_node)
    size = max_idx - min_idx + 1
    _delete_nodes(mesh, min_idx, max_idx)
    _shift_junction_references_after_node_delete(geom, max_idx, size)


def _apply_ghost_node_replacement(geom: GeomType, mesh: MeshType, ghost_ctx: _GhostNodeContext, old_node: int, replacement_node: int, break_field: str) -> int:
    end_node = getattr(mesh.node[replacement_node], break_field)
    setattr(mesh.node[replacement_node], break_field, -1)
    mesh.node[replacement_node].conn = 4
    geom.junc[ghost_ctx.junc_idx].node[ghost_ctx.arm_idx][ghost_ctx.sec_idx] = mesh.node[replacement_node].id
    geom.croP[geom.junc[ghost_ctx.junc_idx].croP[ghost_ctx.arm_idx][ghost_ctx.sec_idx]].pos = mesh.node[replacement_node].pos
    _replace_junction_connection_node(geom, old_node, mesh.node[replacement_node].id)
    return end_node


def _replace_junction_connection_node(geom: GeomType, old_node: int, new_node: int) -> None:
    for junc in geom.junc:
        for conn in junc.conn:
            if conn[0] == old_node:
                conn[0] = new_node
            if conn[1] == old_node:
                conn[1] = new_node


def _shift_junction_references_after_node_delete(geom: GeomType, max_idx: int, size: int) -> None:
    for junc in geom.junc:
        for arm_nodes in junc.node:
            for idx, node in enumerate(arm_nodes):
                if node > max_idx:
                    arm_nodes[idx] -= size
        for conn in junc.conn:
            if conn[0] > max_idx:
                conn[0] -= size
            if conn[1] > max_idx:
                conn[1] -= size


def _delete_nodes(mesh: MeshType, min_idx: int, max_idx: int) -> None:
    n_delete = max_idx - min_idx + 1
    new_nodes = [node for idx, node in enumerate(mesh.node) if not min_idx <= idx <= max_idx]
    _renumber_nodes_after_delete(new_nodes, min_idx, max_idx, n_delete)
    new_eles = []
    for ele in mesh.ele:
        a, b = ele.cn
        if min_idx <= a <= max_idx or min_idx <= b <= max_idx:
            continue
        new_eles.append(EleType(cn=_renumber_element_after_delete(a, b, max_idx, n_delete)))
    mesh.node = new_nodes
    mesh.ele = new_eles
    mesh.n_node = len(new_nodes)
    mesh.n_ele = len(new_eles)


def _renumber_nodes_after_delete(nodes: list[NodeType], min_idx: int, max_idx: int, n_delete: int) -> None:
    for i, node in enumerate(nodes):
        node.id = i
        node.up = _renumber_optional_node_ref_after_delete(node.up, min_idx, max_idx, n_delete)
        node.dn = _renumber_optional_node_ref_after_delete(node.dn, min_idx, max_idx, n_delete)


def _renumber_optional_node_ref_after_delete(node: int, min_idx: int, max_idx: int, n_delete: int) -> int:
    if node > max_idx:
        return node - n_delete
    if min_idx <= node <= max_idx:
        return -1
    return node


def _renumber_element_after_delete(node_a: int, node_b: int, max_idx: int, n_delete: int) -> tuple[int, int]:
    if node_a > max_idx:
        node_a -= n_delete
    if node_b > max_idx:
        node_b -= n_delete
    return node_a, node_b
