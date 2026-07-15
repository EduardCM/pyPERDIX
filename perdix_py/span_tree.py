from __future__ import annotations

"""Spanning tree routines (Fortran: SpanTree.f90)."""

import sys
from typing import Iterable, List, Tuple

import numpy as np


def _check_undirected(adj: np.ndarray) -> None:
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if not np.allclose(adj, adj.T):
        raise ValueError("adjacency matrix must be symmetric (undirected graph)")


def SpanTree_Count_Spanning_Trees(adj: Iterable[Iterable[int]]) -> int:
    mat = np.array(adj, dtype=float)
    _check_undirected(mat)

    n = mat.shape[0]
    deg = np.diag(np.sum(mat, axis=1))
    lap = deg - mat
    if n <= 1:
        return 1
    minor = lap[1:, 1:]
    det = np.linalg.det(minor)
    return int(round(det))


def SpanTree_Count_Spanning_Trees2(adj: Iterable[Iterable[int]]) -> int:
    # Same as Count_Spanning_Trees; kept for API parity.
    mat = np.array(adj, dtype=float)
    _check_undirected(mat)

    n = mat.shape[0]
    if n <= 1:
        return 1
    lap = np.diag(np.sum(mat, axis=1)) - mat
    det = np.linalg.det(lap[1:, 1:])
    return int(round(det))


def SpanTree_List2Adj(src: Iterable[int], dst: Iterable[int], n_node: int) -> np.ndarray:
    adj = np.zeros((n_node, n_node), dtype=int)
    for a, b in zip(src, dst):
        if a == b:
            continue
        adj[a, b] = 1
        adj[b, a] = 1
    return adj


def SpanTree_Adj2List(adj: Iterable[Iterable[int]]) -> Tuple[List[int], List[int]]:
    mat = np.array(adj, dtype=int)
    src: List[int] = []
    dst: List[int] = []
    n = mat.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] != 0:
                src.append(i)
                dst.append(j)
    return src, dst


def SpanTree_Kruskal_Algorithm(
    tail: Iterable[int],
    head: Iterable[int],
    cost: Iterable[int],
    n_node: int,
    n_edge: int,
    mode: str = "quick",
) -> Tuple[List[int], int]:
    edges = list(zip(tail, head, cost))
    edges.sort(key=lambda e: e[2])

    parent = list(range(n_node))
    rank = [0] * n_node

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    tree_idx: List[int] = []
    length = 0
    for idx, (u, v, w) in enumerate(edges):
        if union(u, v):
            tree_idx.append(idx)
            length += w
            if len(tree_idx) == n_node - 1:
                break
    return tree_idx, length


def SpanTree_Prim_Algorithm_1(
    tail: Iterable[int],
    head: Iterable[int],
    cost: Iterable[int],
    n_node: int,
    n_edge: int,
    mode: str = "quick",
) -> Tuple[List[Tuple[int, int]], int]:
    del n_edge, mode
    graph = _build_prim_graph(tail, head, cost, n_node)
    return _prim_from_graph(graph)


def SpanTree_Prim_Algorithm_2(
    tail: Iterable[int],
    head: Iterable[int],
    cost: Iterable[int],
    n_node: int,
    n_edge: int,
    mode: str = "quick",
) -> Tuple[List[Tuple[int, int]], int]:
    del n_edge, mode
    graph = _build_prim_graph(tail, head, cost, n_node)
    return _prim_from_graph(graph)


def _prim_from_edges(
    tail: Iterable[int],
    head: Iterable[int],
    cost: Iterable[int],
    n_node: int,
) -> Tuple[List[Tuple[int, int]], int]:
    graph = _build_prim_graph(tail, head, cost, n_node)
    return _prim_from_graph(graph)


def _build_prim_graph(
    tail: Iterable[int],
    head: Iterable[int],
    cost: Iterable[int],
    n_node: int,
) -> List[List[Tuple[int, int]]]:
    graph: List[List[Tuple[int, int]]] = [[] for _ in range(n_node)]
    for u, v, w in zip(tail, head, cost):
        graph[u].append((w, v))
        graph[v].append((w, u))
    return graph


def _prim_from_graph(graph: List[List[Tuple[int, int]]]) -> Tuple[List[Tuple[int, int]], int]:
    import heapq

    n_node = len(graph)

    visited = [False] * n_node
    mst_edges: List[Tuple[int, int]] = []
    length = 0

    visited[0] = True
    heap: List[Tuple[int, int, int]] = []
    for w, v in graph[0]:
        heapq.heappush(heap, (w, 0, v))

    while heap and len(mst_edges) < n_node - 1:
        w, u, v = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        mst_edges.append((u, v))
        length += w
        for w2, v2 in graph[v]:
            if not visited[v2]:
                heapq.heappush(heap, (w2, v, v2))

    return mst_edges, length


def SpanTree_Generate_Spanning_Trees(adj: Iterable[Iterable[int]]) -> List[List[Tuple[int, int]]]:
    mat = np.array(adj, dtype=int)
    _check_undirected(mat)
    n = mat.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] != 0:
                edges.append((i, j))

    trees: List[List[Tuple[int, int]]] = []

    def is_tree(edge_set: List[Tuple[int, int]]) -> bool:
        if len(edge_set) != n - 1:
            return False
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return False
            parent[rb] = ra
            return True

        for u, v in edge_set:
            if not union(u, v):
                return False
        root = find(0)
        return all(find(i) == root for i in range(n))

    def backtrack(idx: int, current: List[Tuple[int, int]]) -> None:
        if len(current) > n - 1:
            return
        if idx == len(edges):
            if is_tree(current):
                trees.append(list(current))
            return
        # choose edge
        current.append(edges[idx])
        backtrack(idx + 1, current)
        current.pop()
        # skip edge
        backtrack(idx + 1, current)

    backtrack(0, [])
    return trees


def SpanTree_Print_All_Trees(trees: Iterable[Iterable[Tuple[int, int]]]) -> None:
    for idx, tree in enumerate(trees, start=1):
        sys.stdout.write(f"Tree {idx}: {list(tree)}\n")


def SpanTree_Print_Matrix(mat: Iterable[Iterable[int]]) -> None:
    for row in mat:
        sys.stdout.write(" ".join(str(v) for v in row) + "\n")


def SpanTree_Print_Vector(vec: Iterable[int]) -> None:
    sys.stdout.write(" ".join(str(v) for v in vec) + "\n")
