from __future__ import annotations

from math import cos, pi, sin

from .data_geom import GeomType
from .data_prob import ProbType
from .exam_perdix_shapes import (
    _annulus,
    _arrowhead,
    _circle_polygon,
    _cross_shape,
    _grid_faces_quad,
    _grid_faces_tri,
    _grid_points,
    _l_shape,
    _polygon_face,
    _polygon_points,
    _quarter_circle,
    _set_geom_from_points_faces,
    _set_prob_name,
    _star,
)


def _apply(prob: ProbType, geom: GeomType, name: str, points, faces) -> None:
    _set_prob_name(prob, name)
    _set_geom_from_points_faces(geom, points, faces)


def Exam_PERDIX_Square(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "01_Square", _grid_points(4, 4, 4.0, 4.0), _grid_faces_tri(4, 4))


def Exam_PERDIX_Honeycomb(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "02_Honeycomb", *_circle_polygon(6, 2.0))


def Exam_PERDIX_Circle(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "03_Circle", *_circle_polygon(36, 2.0))


def Exam_PERDIX_Wheel(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "04_Wheel", *_circle_polygon(24, 2.0))


def Exam_PERDIX_Ellipse(prob: ProbType, geom: GeomType) -> None:
    pts = [(3.0 * cos(2.0 * pi * i / 36), 1.5 * sin(2.0 * pi * i / 36), 0.0) for i in range(36)]
    _apply(prob, geom, "05_Ellipse", pts, [_polygon_face(len(pts))])


def Exam_PERDIX_Rhombic_Tiling(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "06_Rhombic_Tiling", _grid_points(4, 4, 4.0, 3.0), _grid_faces_quad(4, 4))


def Exam_PERDIX_Quarter_Circle(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "07_Quarter_Circle", *_quarter_circle(24, 3.0))


def Exam_PERDIX_Cross(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "08_Cross", *_cross_shape(2.0, 0.6))


def Exam_PERDIX_Arrowhead(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "09_Arrowhead", *_arrowhead(1.5))


def Exam_PERDIX_Annulus(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "10_Annulus", *_annulus(36, 3.0, 2.0))


def Exam_PERDIX_Cairo_Penta_Tiling(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "11_Cairo_Penta_Tiling", *_circle_polygon(5, 2.0))


def Exam_PERDIX_Lotus(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "12_Lotus", *_circle_polygon(12, 2.5))


def Exam_PERDIX_Hexagonal_Tiling(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "13_Hexagonal_Tiling", *_circle_polygon(6, 2.5))


def Exam_PERDIX_Prismatic_Penta_Tiling(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "14_Prismatic_Penta_Tiling", *_circle_polygon(5, 2.3))


def Exam_PERDIX_Hepta_Penta_Tiling(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "15_Hepta_Penta_Tiling", *_circle_polygon(7, 2.3))


def Exam_PERDIX_4_Sided_Polygon(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "16_4_Sided_Polygon", *_circle_polygon(4, 2.0))


def Exam_PERDIX_5_Sided_Polygon(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "17_5_Sided_Polygon", *_circle_polygon(5, 2.0))


def Exam_PERDIX_6_Sided_Polygon(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "18_6_Sided_Polygon", *_circle_polygon(6, 2.0))


def Exam_PERDIX_L_Shape_42bp(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "19_L_Shape_42bp", *_l_shape(3.0, 1.0))


def Exam_PERDIX_L_Shape_63bp(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "20_L_Shape_63bp", *_l_shape(4.0, 1.3))


def Exam_PERDIX_L_Shape_84bp(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "21_L_Shape_84bp", *_l_shape(5.0, 1.5))


def Exam_PERDIX_Curved_Arm_Quad(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "22_Curved_Arm_Quad", *_quarter_circle(20, 3.0))


def Exam_PERDIX_Curved_Arm_Tri(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "23_Curved_Arm_Tri", *_quarter_circle(20, 3.0))


def Exam_PERDIX_Curved_Arm_Mix(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "24_Curved_Arm_Mix", *_quarter_circle(20, 3.0))


def Exam_PERDIX_Pump_Quad(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Pump_Quad", _grid_points(6, 4, 6.0, 4.0), _grid_faces_quad(6, 4))


def Exam_PERDIX_Pump_Tri(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Pump_Tri", _grid_points(6, 4, 6.0, 4.0), _grid_faces_tri(6, 4))


def Exam_PERDIX_Pump_Eng(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Pump_Eng", _grid_points(6, 4, 6.0, 4.0), _grid_faces_tri(6, 4))


def Exam_PERDIX_S_Shape_Quad(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "S_Shape_Quad", _grid_points(6, 4, 6.0, 4.0), _grid_faces_quad(6, 4))


def Exam_PERDIX_S_Shape_Tri(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "S_Shape_Tri", _grid_points(6, 4, 6.0, 4.0), _grid_faces_tri(6, 4))


def Exam_PERDIX_S_Shape_Eng(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "S_Shape_Eng", _grid_points(6, 4, 6.0, 4.0), _grid_faces_tri(6, 4))


def Exam_PERDIX_Small_House_Quad(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Small_House_Quad", _polygon_points(5, 2.0), [_polygon_face(5)])


def Exam_PERDIX_Small_House_Tri(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Small_House_Tri", _polygon_points(5, 2.0), [_polygon_face(5)])


def Exam_PERDIX_Plate_3x4(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Plate_3x4", _grid_points(3, 4, 3.0, 4.0), _grid_faces_quad(3, 4))


def Exam_PERDIX_Pentagon(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Pentagon", *_circle_polygon(5, 2.0))


def Exam_PERDIX_Star(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Star", *_star(5, 2.0, 1.0))


def Exam_PERDIX_Plumeria(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Plumeria", *_circle_polygon(8, 2.0))


def Exam_PERDIX_Stickman(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Stickman", *_circle_polygon(6, 2.0))


def Exam_PERDIX_L_Shape_Tri(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "L_Shape_Tri", *_l_shape(3.0, 1.0))


def Exam_PERDIX_Quarter_Circle_Tri(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Quarter_Circle_Tri", *_quarter_circle(24, 3.0))


def Exam_PERDIX_Disk_Tri(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Disk_Tri", *_circle_polygon(24, 3.0))


def Exam_PERDIX_circle_Tri_Fine(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Circle_Tri_Fine", *_circle_polygon(60, 3.0))


def Exam_PERDIX_Ellipse_Tri_Fine(prob: ProbType, geom: GeomType) -> None:
    pts = [(3.0 * cos(2.0 * pi * i / 60), 1.5 * sin(2.0 * pi * i / 60), 0.0) for i in range(60)]
    _apply(prob, geom, "Ellipse_Tri_Fine", pts, [_polygon_face(len(pts))])


def Exam_PERDIX_L_Shape_Irregular(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "L_Shape_Irregular", *_l_shape(4.0, 1.2))


def Exam_PERDIX_Plate_Distorted_Quad(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Plate_Distorted_Quad", _grid_points(4, 4, 4.0, 3.0), _grid_faces_quad(4, 4))


def Exam_PERDIX_Plate_Distorted_Tri(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Plate_Distorted_Tri", _grid_points(4, 4, 4.0, 3.0), _grid_faces_tri(4, 4))


def Exam_PERDIX_Hyperbolic_Paraboloid_Quad(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Hyperbolic_Paraboloid_Quad", _grid_points(4, 4, 4.0, 4.0), _grid_faces_quad(4, 4))


def Exam_PERDIX_Hyperbolic_Paraboloid_Tri(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Hyperbolic_Paraboloid_Tri", _grid_points(4, 4, 4.0, 4.0), _grid_faces_tri(4, 4))


def Exam_PERDIX_N_Polygon(prob: ProbType, geom: GeomType, n: int) -> None:
    _apply(prob, geom, f"N_Polygon_{n}", *_circle_polygon(n, 2.0))


def Exam_PERDIX_N_Polygon_Mesh(prob: ProbType, geom: GeomType, n: int) -> None:
    _apply(prob, geom, f"N_Polygon_Mesh_{n}", *_circle_polygon(n, 2.0))


def Exam_PERDIX_Triangle(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Triangle", *_circle_polygon(3, 2.0))


def Exam_PERDIX_Hexagon(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Hexagon", *_circle_polygon(6, 2.0))


def Exam_PERDIX_Octagon(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Octagon", *_circle_polygon(8, 2.0))


def Exam_PERDIX_Octagon_Cross_1(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Octagon_Cross_1", *_circle_polygon(8, 2.0))


def Exam_PERDIX_Octagon_Cross_2(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Octagon_Cross_2", *_circle_polygon(8, 2.0))


def Exam_PERDIX_Controllable_Plate(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Controllable_Plate", _grid_points(4, 4, 4.0, 4.0), _grid_faces_quad(4, 4))


def Exam_PERDIX_Quarter_Hemi_Hole(prob: ProbType, geom: GeomType) -> None:
    _apply(prob, geom, "Quarter_Hemi_Hole", *_quarter_circle(24, 3.0))
