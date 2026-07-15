from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_geom import GeomType
from .data_mesh import MeshType
from .data_prob import ProbType


@dataclass
class ReferenceFrame:
    origin: np.ndarray  # shape (3,)
    axes: np.ndarray  # shape (3, 3), rows are x/y/z unit axes
    scale: float  # multiply centered local coords by this scale
    scale_basis: str
    mode: str = "shape_local"
    abs_center: np.ndarray | None = None
    abs_total_scale: float = 1.0


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=float)
    return v / n


def _orient_canonical(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Flip vector so its dominant component is positive."""
    out = np.array(v, dtype=float)
    idx = int(np.argmax(np.abs(out)))
    if abs(out[idx]) < eps:
        return out
    if out[idx] < 0.0:
        out = -out
    return out


def _pick_x_axis_from_mesh(mesh: MeshType) -> np.ndarray:
    """Pick deterministic X-axis from shortest mesh element."""
    if mesh.n_ele <= 0 or not mesh.ele:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    best_len = None
    best_vec = None
    for i, e in enumerate(mesh.ele):
        a, b = e.cn
        if a < 0 or b < 0 or a >= mesh.n_node or b >= mesh.n_node:
            continue
        p1 = np.array(mesh.node[a].pos, dtype=float)
        p2 = np.array(mesh.node[b].pos, dtype=float)
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length < 1e-12:
            continue
        # Deterministic tie-break: keep later index to mirror other legacy patterns.
        if best_len is None or length < best_len - 1e-12 or abs(length - best_len) <= 1e-12:
            best_len = length
            best_vec = vec
    if best_vec is None:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return _orient_canonical(_normalize(best_vec))


def _pick_scale_from_mesh(mesh: MeshType) -> tuple[float, str]:
    """Use shortest non-zero mesh element as length scale."""
    lengths: list[float] = []
    for e in mesh.ele:
        a, b = e.cn
        if a < 0 or b < 0 or a >= mesh.n_node or b >= mesh.n_node:
            continue
        p1 = np.array(mesh.node[a].pos, dtype=float)
        p2 = np.array(mesh.node[b].pos, dtype=float)
        length = float(np.linalg.norm(p2 - p1))
        if length > 1e-12:
            lengths.append(length)
    if not lengths:
        return 1.0, "unit"
    min_len = min(lengths)
    return 1.0 / min_len, "min_mesh_edge"


def compute_reference_frame(prob: ProbType, geom: GeomType, mesh: MeshType, mode: str = "shape_local") -> ReferenceFrame:
    """Build a deterministic common reference frame for cross-shape comparison."""
    if mode == "input_absolute":
        total_scale = float(prob.input_init_scale * prob.input_modgeo_scale)
        if abs(total_scale) < 1e-12:
            total_scale = 1.0
        center = np.array(prob.input_center, dtype=float)
        return ReferenceFrame(
            origin=np.zeros(3, dtype=float),
            axes=np.eye(3, dtype=float),
            scale=1.0,
            scale_basis="input_absolute",
            mode="input_absolute",
            abs_center=center,
            abs_total_scale=total_scale,
        )

    if mesh.n_node <= 0 or not mesh.node:
        return ReferenceFrame(
            origin=np.zeros(3, dtype=float),
            axes=np.eye(3, dtype=float),
            scale=1.0,
            scale_basis="unit",
        )

    pts = np.array([n.pos for n in mesh.node], dtype=float)
    origin = np.mean(pts, axis=0)
    centered = pts - origin

    x_axis = _pick_x_axis_from_mesh(mesh)
    if float(np.linalg.norm(x_axis)) < 1e-12:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)

    # Secondary axis from PCA, projected orthogonal to x.
    cov = centered.T @ centered / max(1, centered.shape[0])
    evals, evecs = np.linalg.eigh(cov)
    pc2 = np.array(evecs[:, 1], dtype=float)
    pc3 = np.array(evecs[:, 0], dtype=float)

    y_axis = pc2 - float(np.dot(pc2, x_axis)) * x_axis
    if float(np.linalg.norm(y_axis)) < 1e-12:
        y_axis = pc3 - float(np.dot(pc3, x_axis)) * x_axis
    if float(np.linalg.norm(y_axis)) < 1e-12:
        fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        y_axis = fallback - float(np.dot(fallback, x_axis)) * x_axis
    y_axis = _orient_canonical(_normalize(y_axis))

    z_axis = _normalize(np.cross(x_axis, y_axis))
    if float(np.linalg.norm(z_axis)) < 1e-12:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    # Re-orthogonalize y to ensure an orthonormal right-handed basis.
    y_axis = _normalize(np.cross(z_axis, x_axis))

    scale, scale_basis = _pick_scale_from_mesh(mesh)
    axes = np.vstack([x_axis, y_axis, z_axis])
    return ReferenceFrame(origin=origin, axes=axes, scale=scale, scale_basis=scale_basis, mode="shape_local")


def transform_position(pos: tuple[float, float, float], rf: ReferenceFrame) -> tuple[float, float, float]:
    if rf.mode == "input_absolute":
        p = np.array(pos, dtype=float)
        p = p / rf.abs_total_scale + rf.abs_center
        return float(p[0]), float(p[1]), float(p[2])
    p = np.array(pos, dtype=float) - rf.origin
    # Project into local frame, then normalize scale.
    local = rf.axes @ p
    local *= rf.scale
    return float(local[0]), float(local[1]), float(local[2])
