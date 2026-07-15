from __future__ import annotations

from dataclasses import dataclass, field


Vector3 = tuple[float, float, float]


@dataclass
class NodeType:
    id: int = 0
    bp: int = 0
    up: int = 0
    dn: int = 0
    sec: int = 0
    iniL: int = 0
    croL: int = 0
    mitered: int = -1
    conn: int = -1
    ghost: int = 0
    pos: Vector3 = (0.0, 0.0, 0.0)
    ori: tuple[Vector3, Vector3, Vector3] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )


@dataclass
class EleType:
    cn: tuple[int, int] = (0, 0)


@dataclass
class MeshType:
    n_node: int = 0
    n_ele: int = 0
    n_mitered: int = 0

    node: list[NodeType] = field(default_factory=list)
    ele: list[EleType] = field(default_factory=list)
