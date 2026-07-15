from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from perdix_py import para
from perdix_py.config import reset_para_defaults
from perdix_py.data_dna import BaseType, DNAType, TopType, StrandType
from perdix_py.data_geom import GeomType, LineType, PointType
from perdix_py.data_mesh import EleType, MeshType, NodeType
from perdix_py.data_prob import ProbType
from perdix_py.main import _run_pipeline
from perdix_py.validation import (
    ValidationError,
    enforce_pipeline_validation,
    validate_pipeline_state,
)


class TestValidationRules(unittest.TestCase):
    def setUp(self) -> None:
        reset_para_defaults()

    def tearDown(self) -> None:
        reset_para_defaults()

    def test_empty_initial_state_has_no_fatal_issues(self) -> None:
        issues = validate_pipeline_state(
            "initial",
            ProbType(),
            GeomType(),
            MeshType(),
            DNAType(),
        )

        self.assertEqual([issue for issue in issues if issue.severity == "high"], [])

    def test_count_mismatch_is_reported(self) -> None:
        geom = GeomType(n_iniP=2, iniP=[PointType()])

        issues = validate_pipeline_state("input_initialize", ProbType(name_file="x"), geom, MeshType(), DNAType())

        self.assertTrue(any(issue.code == "count.mismatch" for issue in issues))

    def test_negative_one_is_not_valid_for_required_point_reference(self) -> None:
        geom = GeomType(
            n_iniP=1,
            n_iniL=1,
            iniP=[PointType()],
            iniL=[LineType(poi=[0, -1], iniP=[0, -1])],
        )

        issues = validate_pipeline_state("input_initialize", ProbType(name_file="x"), geom, MeshType(), DNAType())

        self.assertTrue(
            any(
                issue.code == "reference.invalid"
                and "geom.iniL[0].poi[1]" in issue.message
                for issue in issues
            )
        )

    def test_line_poi_references_modified_points_after_modgeo(self) -> None:
        geom = GeomType(
            n_iniP=2,
            n_modP=4,
            n_iniL=1,
            iniP=[PointType(), PointType()],
            modP=[PointType(), PointType(), PointType(), PointType()],
            iniL=[LineType(poi=[2, 3], iniP=[0, 1])],
        )

        issues = validate_pipeline_state("modgeo_modification", ProbType(name_file="x"), geom, MeshType(), DNAType())

        self.assertFalse(
            any(
                issue.code == "reference.invalid"
                and "geom.iniL[0].poi" in issue.message
                for issue in issues
            )
        )

    def test_mesh_element_invalid_endpoint_is_reported(self) -> None:
        mesh = MeshType(
            n_node=1,
            n_ele=1,
            node=[NodeType(id=0)],
            ele=[EleType(cn=(0, -1))],
        )

        issues = validate_pipeline_state("basepair_discretize", ProbType(name_file="x"), GeomType(), mesh, DNAType())

        self.assertTrue(
            any(
                issue.code == "reference.invalid"
                and "mesh.ele[0].cn[1]" in issue.message
                for issue in issues
            )
        )

    def test_broken_base_xover_reciprocal_link_is_reported(self) -> None:
        dna = DNAType(
            n_base_scaf=2,
            base_scaf=[
                BaseType(id=0, xover=1),
                BaseType(id=1, xover=-1),
            ],
        )

        issues = validate_pipeline_state("route_generation", ProbType(name_file="x"), GeomType(), MeshType(), dna)

        self.assertTrue(any(issue.code == "link.not_reciprocal" for issue in issues))

    def test_invalid_strand_base_id_is_reported(self) -> None:
        dna = DNAType(
            n_top=1,
            n_strand=1,
            top=[TopType(id=0)],
            strand=[StrandType(n_base=1, base=[2])],
        )

        issues = validate_pipeline_state("seqdesign_design", ProbType(name_file="x"), GeomType(), MeshType(), dna)

        self.assertTrue(
            any(
                issue.code == "reference.invalid"
                and "dna.strand[0].base[0]" in issue.message
                for issue in issues
            )
        )


class TestValidationEnforcement(unittest.TestCase):
    def setUp(self) -> None:
        reset_para_defaults()

    def tearDown(self) -> None:
        reset_para_defaults()

    def test_disabled_validation_does_not_call_validator(self) -> None:
        para.para_validate_pipeline = False
        with mock.patch("perdix_py.validation.validate_pipeline_state") as validator:
            issues = enforce_pipeline_validation("stage", ProbType(), GeomType(), MeshType(), DNAType())

        self.assertEqual(issues, [])
        validator.assert_not_called()

    def test_strict_validation_raises_on_high_severity_issue(self) -> None:
        para.para_validate_pipeline = True
        para.para_validate_pipeline_strict = True
        geom = GeomType(n_iniP=1, iniP=[])

        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(ValidationError),
        ):
            enforce_pipeline_validation("input_initialize", ProbType(name_file="x"), geom, MeshType(), DNAType())

    def test_non_strict_validation_reports_and_continues(self) -> None:
        para.para_validate_pipeline = True
        para.para_validate_pipeline_strict = False
        geom = GeomType(n_iniP=1, iniP=[])

        with contextlib.redirect_stderr(io.StringIO()):
            issues = enforce_pipeline_validation("input_initialize", ProbType(name_file="x"), geom, MeshType(), DNAType())

        self.assertTrue(any(issue.severity == "high" for issue in issues))


class TestPipelineValidationHooks(unittest.TestCase):
    def setUp(self) -> None:
        reset_para_defaults()

    def tearDown(self) -> None:
        reset_para_defaults()

    def test_run_pipeline_calls_validation_after_each_stage(self) -> None:
        stages: list[str] = []

        with (
            mock.patch("perdix_py.main.input_initialize"),
            mock.patch("perdix_py.main.modgeo_modification"),
            mock.patch("perdix_py.main.section_generation"),
            mock.patch("perdix_py.main.basepair_discretize"),
            mock.patch("perdix_py.main.route_generation"),
            mock.patch("perdix_py.main.seqdesign_design"),
            mock.patch("perdix_py.main.output_generation"),
            mock.patch(
                "perdix_py.main.enforce_pipeline_validation",
                side_effect=lambda stage, *_args: stages.append(stage),
            ),
        ):
            _run_pipeline(
                svg_path="shape.svg",
                config_path=None,
                edge_len=None,
                frame_mode="legacy",
                summary=False,
                debug_mesh=False,
            )

        self.assertEqual(
            stages,
            [
                "input_initialize",
                "modgeo_modification",
                "section_generation",
                "basepair_discretize",
                "route_generation",
                "seqdesign_design",
            ],
        )


if __name__ == "__main__":
    unittest.main()
