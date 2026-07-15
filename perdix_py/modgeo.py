from __future__ import annotations

from .data_geom import GeomType
from .data_prob import ProbType
from .modgeo_neighbors import (
    _round_geom_positions,
    _set_junction_data,
    _set_local_coordinate,
    _set_neighbor_line,
    _set_neighbor_point,
)
from .modgeo_scale import _find_scale_factor, _scale_geometry, _separate_line, _set_gap_junction


def modgeo_modification(prob: ProbType, geom: GeomType) -> None:
    _set_neighbor_point(prob, geom)
    _set_neighbor_line(geom)
    _set_junction_data(geom)
    _set_local_coordinate(geom)
    for p in geom.iniP:
        p.ori_pos = p.pos
    _separate_line(geom)
    scale = _find_scale_factor(prob, geom)
    prob.input_modgeo_scale = float(scale)
    _scale_geometry(geom, scale)
    _set_gap_junction(geom)
    _round_geom_positions(geom, digits=4)
