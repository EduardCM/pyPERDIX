from __future__ import annotations

from dataclasses import dataclass, field


Vector3 = tuple[float, float, float]


@dataclass
class SecType:
    """Section data for cross-sections (Fortran: Data_Geom.SecType)."""

    types: str = ""
    dir: int = 0
    maxR: int = 0
    minR: int = 0
    maxC: int = 0
    minC: int = 0
    n_row: int = 0
    n_col: int = 0
    ref_row: int = 0
    ref_maxC: int = 0
    ref_minC: int = 0

    id: list[int] = field(default_factory=list)
    posR: list[int] = field(default_factory=list)
    posC: list[int] = field(default_factory=list)
    conn: list[int] = field(default_factory=list)


@dataclass
class PointType:
    pos: Vector3 = (0.0, 0.0, 0.0)
    ori_pos: Vector3 = (0.0, 0.0, 0.0)


@dataclass
class LineType:
    iniL: int = 0
    sec: int = 0
    iniP: list[int] = field(default_factory=lambda: [0, 0])
    poi: list[int] = field(default_factory=lambda: [0, 0])
    neiF: list[int] = field(default_factory=lambda: [-1, -1])
    neiP: list[list[int]] = field(default_factory=lambda: [[-1, -1], [-1, -1]])
    neiL: list[list[int]] = field(default_factory=lambda: [[-1, -1], [-1, -1]])
    n_xover: int = 0
    t: list[Vector3] = field(
        default_factory=lambda: [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ]
    )


@dataclass
class FaceType:
    n_poi: int = 0
    poi: list[int] = field(default_factory=list)


@dataclass
class JuncType:
    n_arm: int = 0
    poi_c: int = 0
    n_un_scaf: int = 0
    n_un_stap: int = 0
    ref_ang: float = 0.0
    tot_ang: float = 0.0
    gap: float = 0.0

    iniL: list[int] = field(default_factory=list)
    modP: list[int] = field(default_factory=list)
    croP: list[list[int]] = field(default_factory=list)
    node: list[list[int]] = field(default_factory=list)
    conn: list[list[int]] = field(default_factory=list)
    type_conn: list[int] = field(default_factory=list)


@dataclass
class GeomType:
    n_sec: int = 0
    n_iniP: int = 0
    n_modP: int = 0
    n_croP: int = 0
    n_iniL: int = 0
    n_croL: int = 0
    n_face: int = 0
    min_edge_length: int = 0
    max_edge_length: int = 0
    n_junc: int = 0
    # Optional SVG canvas rectangle (x, y, width, height) in imported units.
    canvas_rect: tuple[float, float, float, float] | None = None

    sec: SecType = field(default_factory=SecType)
    iniP: list[PointType] = field(default_factory=list)
    modP: list[PointType] = field(default_factory=list)
    croP: list[PointType] = field(default_factory=list)
    iniL: list[LineType] = field(default_factory=list)
    croL: list[LineType] = field(default_factory=list)
    face: list[FaceType] = field(default_factory=list)
    junc: list[JuncType] = field(default_factory=list)
