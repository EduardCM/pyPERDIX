from __future__ import annotations

import math
from typing import List, Sequence, Tuple

pi: float = math.pi
eps: float = 1.0e-7


def rad2deg(rad: float) -> float:
    return rad * 180.0 / pi


def deg2rad(deg: float) -> float:
    return deg * pi / 180.0


int2str = str


def dble2str(val: float) -> str:
    return f"{val:.6f}".rstrip("0").rstrip(".")


def dble2str1(val: float) -> str:
    return f"{val:.1f}"


def dble2str2(val: float) -> str:
    return f"{val:.2f}"


def nint(val: float) -> int:
    if val >= 0.0:
        return int(math.floor(val + 0.5))
    return int(math.ceil(val - 0.5))


def norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def normalize(vec: Sequence[float]) -> Tuple[float, float, float]:
    n = norm(vec)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (vec[0] / n, vec[1] / n, vec[2] / n)


def cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def is_same_vector(a: Sequence[float], b: Sequence[float], tol: float = eps) -> bool:
    return all(abs(ai - bi) <= tol for ai, bi in zip(a, b))


def rotate_vector(vec: Sequence[float], axis: Sequence[float], ang_rad: float) -> Tuple[float, float, float]:
    """Rodrigues rotation formula."""
    kx, ky, kz = normalize(axis)
    vx, vy, vz = vec
    c = math.cos(ang_rad)
    s = math.sin(ang_rad)
    return (
        vx * c + (ky * vz - kz * vy) * s + kx * (kx * vx + ky * vy + kz * vz) * (1 - c),
        vy * c + (kz * vx - kx * vz) * s + ky * (kx * vx + ky * vy + kz * vz) * (1 - c),
        vz * c + (kx * vy - ky * vx) * s + kz * (kx * vx + ky * vy + kz * vz) * (1 - c),
    )


def inverse_22(mat: Sequence[Sequence[float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    if det == 0.0:
        raise ZeroDivisionError("Singular 2x2 matrix")
    inv = 1.0 / det
    return (
        (mat[1][1] * inv, -mat[0][1] * inv),
        (-mat[1][0] * inv, mat[0][0] * inv),
    )


def polygon_center(n: int, v: Sequence[Sequence[float]]) -> Tuple[float, float]:
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        ip1 = i + 1 if i + 1 < n else 0
        temp = v[i][0] * v[ip1][1] - v[ip1][0] * v[i][1]
        area += temp
        cx += (v[ip1][0] + v[i][0]) * temp
        cy += (v[ip1][1] + v[i][1]) * temp
    area *= 0.5
    if area == 0.0:
        return (v[0][0], v[0][1])
    return (cx / (6.0 * area), cy / (6.0 * area))


def polygon_contains_point(n: int, v: Sequence[Sequence[float]], p: Sequence[float]) -> bool:
    inside = False
    px1 = v[0][0]
    py1 = v[0][1]
    xints = p[0] - 1.0
    for i in range(n):
        px2 = v[(i + 1) % n][0]
        py2 = v[(i + 1) % n][1]
        if min(py1, py2) < p[1] <= max(py1, py2) and p[0] <= max(px1, px2):
            if py1 != py2:
                xints = (p[1] - py1) * (px2 - px1) / (py2 - py1) + px1
            if px1 == px2 or p[0] <= xints:
                inside = not inside
        px1 = px2
        py1 = py2
    return inside


def check_intersection(a1: Sequence[float], a2: Sequence[float], b1: Sequence[float], b2: Sequence[float]) -> bool:
    da = (a2[0] - a1[0], a2[1] - a1[1], a2[2] - a1[2])
    db = (b2[0] - b1[0], b2[1] - b1[1], b2[2] - b1[2])
    dc = (b1[0] - a1[0], b1[1] - a1[1], b1[2] - a1[2])
    return abs(sum(dc[i] * cross(da, db)[i] for i in range(3))) > 1.0e-5


def find_intersection(a1: Sequence[float], a2: Sequence[float], b1: Sequence[float], b2: Sequence[float]) -> Tuple[float, float, float]:
    mat = (
        (a2[0] - a1[0], -(b2[0] - b1[0])),
        (a2[1] - a1[1], -(b2[1] - b1[1])),
    )
    inv = inverse_22(mat)
    t1 = inv[0][0] * (b1[0] - a1[0]) + inv[0][1] * (b1[1] - a1[1])
    return (
        a1[0] + (a2[0] - a1[0]) * t1,
        a1[1] + (a2[1] - a1[1]) * t1,
        a1[2] + (a2[2] - a1[2]) * t1,
    )


def find_closest_point(
    a1: Sequence[float],
    a2: Sequence[float],
    b1: Sequence[float],
    b2: Sequence[float],
) -> Tuple[Tuple[float, float, float], bool]:
    small_num = 1.0e-8
    u = (a2[0] - a1[0], a2[1] - a1[1], a2[2] - a1[2])
    v = (b2[0] - b1[0], b2[1] - b1[1], b2[2] - b1[2])
    w = (a1[0] - b1[0], a1[1] - b1[1], a1[2] - b1[2])
    a = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
    b = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    c = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    d = u[0] * w[0] + u[1] * w[1] + u[2] * w[2]
    e = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]
    dd = a * c - b * b

    if dd < small_num:
        sc = 0.0
        if b > c:
            tc = d / b
        else:
            tc = e / c
        ok = False
    else:
        sc = (b * e - c * d) / dd
        tc = (a * e - b * d) / dd
        ok = True

    pos_u = (a1[0] + u[0] * sc, a1[1] + u[1] * sc, a1[2] + u[2] * sc)
    pos_v = (b1[0] + v[0] * tc, b1[1] + v[1] * tc, b1[2] + v[2] * tc)
    pos = ((pos_u[0] + pos_v[0]) * 0.5, (pos_u[1] + pos_v[1]) * 0.5, (pos_u[2] + pos_v[2]) * 0.5)
    return pos, ok


def find_minimum(arr: Sequence[float]) -> Tuple[float, int]:
    min_val = min(arr)
    return min_val, arr.index(min_val)


def swap(arr: List[int], i: int, j: int) -> None:
    arr[i], arr[j] = arr[j], arr[i]


sort = sorted


def sort2(keys: List[int], values: List[int]) -> Tuple[List[int], List[int]]:
    pairs = sorted(zip(keys, values), key=lambda kv: kv[0])
    if not pairs:
        return [], []
    k_sorted, v_sorted = zip(*pairs)
    return list(k_sorted), list(v_sorted)


def reallocate_int_1d(arr: List[int], n: int, fill: int = 0) -> List[int]:
    if len(arr) >= n:
        return arr[:n]
    return arr + [fill] * (n - len(arr))


def reallocate_int_2d(arr: List[List[int]], n: int, m: int, fill: int = 0) -> List[List[int]]:
    out = [row[:] for row in arr[:n]]
    for row in out:
        if len(row) < m:
            row.extend([fill] * (m - len(row)))
        else:
            del row[m:]
    while len(out) < n:
        out.append([fill] * m)
    return out
