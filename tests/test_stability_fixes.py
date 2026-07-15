from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from perdix_py import para
from perdix_py.basepair import (
    _append_sticky_end_node,
    _apply_ghost_node_replacement,
    _apply_junction_self_connections,
    _apply_open_geometry_junction_fallback,
    _apply_vertex_crash_strategy,
    _basepair_adjustment_endpoints,
    _copy_sticky_end_node_fields,
    _edge_extension_count_from_lengths,
    _edge_increase_endpoint_pair,
    _delete_ghost_node_chain,
    _find_ghost_chain_replacement,
    _find_internal_junction_lines,
    _ghost_node_walk_start,
    _initial_xover_move,
    _mark_ghost_chain_until_second_xover,
    _mark_edge_endpoint_connection,
    _place_sticky_end_node,
    _renumber_element_after_delete,
    _renumber_optional_node_ref_after_delete,
    _replace_junction_connection_node,
    _replace_sticky_end_junction_node,
    _reserve_mesh_node_and_element,
    _shift_junction_references_after_node_delete,
    _should_increase_mitered_edge,
    _unique_junction_connections,
    _xover_search_starts_inside,
    _xover_search_starts_outside,
    _xover_search_steps,
)
from perdix_py.config import load_config, reset_para_defaults
from perdix_py.data_dna import BaseType, DNAType, StrandType, TopType
from perdix_py.data_geom import GeomType, JuncType, LineType, PointType
from perdix_py.data_mesh import EleType, MeshType, NodeType
from perdix_py.data_prob import ProbType
from perdix_py.input import input_initialize
from perdix_py.route import (
    _accepted_staple_xover_neighbor_pair,
    _advance_scaffold_split_pair,
    _apply_scaffold_xover_pair,
    _clear_scaffold_xover_pair,
    _find_possible_stap_xover,
    _find_scaffold_neighbor_pair,
    _has_scaffold_strand_pair_record,
    _has_scaffold_xover_record,
    _iter_centered_scaffold_xover_candidates,
    _scaffold_split_strategy,
    _scaffold_split_hits_boundary_gap,
    _scan_scaffold_split_direction,
    _route_bild_enabled,
    _set_orientation,
    _select_best_centered_scaffold_split,
    _select_initial_centered_scaffold_split,
    _write_route_bild,
)
from perdix_py.seqdesign import (
    _advance_to_scaffold_nick_row_base,
    _apply_scaffold_nick,
    _apply_scaffold_sequence,
    _assign_short_scaffold_region_centers,
    _break_top_link_after_base,
    _can_cut_staple_short_xover,
    _collect_short_scaffold_regions,
    _count_scaffold_crossovers_by_edge,
    _cut_short_scaffold_regions,
    _cut_staple_after_base,
    _cut_staple_short_xover,
    _find_scaffold_nick_start_base,
    _is_valid_staple_cut_position,
    _is_scaffold_nick_boundary,
    _scan_scaffold_nick_candidate_run,
    _select_staple_cut_region,
    _skip_staple_cut_candidate,
    _update_max_stap_gap_change,
    RegionType,
    ScafRegionType,
    seqdesign_design,
)


class TestConfigState(unittest.TestCase):
    def setUp(self) -> None:
        reset_para_defaults()

    def tearDown(self) -> None:
        reset_para_defaults()

    def _write_config(self, root: Path, name: str, data: dict[str, object]) -> Path:
        path = root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_load_config_none_resets_para_defaults(self) -> None:
        para.para_write_102 = False
        para.para_scaf_seq = "rand"

        cfg = load_config(None)

        self.assertTrue(para.para_write_102)
        self.assertEqual(para.para_scaf_seq, "")
        self.assertFalse(cfg.para_scaf_seq_explicit)

    def test_load_config_does_not_leak_omitted_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_a = self._write_config(
                root,
                "a.json",
                {"para_scaf_seq": "rand", "para_write_102": False},
            )
            cfg_b = self._write_config(root, "b.json", {"para_cut_stap_method": "max"})

            load_config(str(cfg_a))
            self.assertEqual(para.para_scaf_seq, "rand")
            self.assertFalse(para.para_write_102)

            cfg = load_config(str(cfg_b))
            self.assertEqual(para.para_scaf_seq, "")
            self.assertTrue(para.para_write_102)
            self.assertFalse(cfg.para_scaf_seq_explicit)


class TestInputSequenceSelection(unittest.TestCase):
    def setUp(self) -> None:
        reset_para_defaults()
        self._cwd = os.getcwd()

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        reset_para_defaults()

    def _write_svg(self, root: Path) -> Path:
        path = root / "shape.svg"
        path.write_text(
            '<svg viewBox="0 0 10 10"><polygon points="0,0 10,0 0,10"/></svg>',
            encoding="utf-8",
        )
        return path

    def _run_input_initialize(self, root: Path, config: dict[str, object]) -> ProbType:
        cfg_path = root / "perdix.json"
        cfg_path.write_text(json.dumps(config), encoding="utf-8")
        svg_path = self._write_svg(root)
        prob = ProbType()
        geom = GeomType()

        with mock.patch("perdix_py.input._set_section_connectivity", return_value=None):
            input_initialize(prob, geom, svg_path=str(svg_path), config_path=str(cfg_path))
        return prob

    def test_configured_sequence_mode_is_not_overwritten_by_seq_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.chdir(root)
            (root / "seq.txt").write_text("user\nAAAA\n", encoding="utf-8")

            prob = self._run_input_initialize(
                root,
                {
                    "para_scaf_seq": "rand",
                    "para_write_102": False,
                    "output_dir": str(root / "out"),
                },
            )

            self.assertEqual(para.para_scaf_seq, "rand")
            self.assertEqual(prob.scaf_seq, "")

    def test_seq_txt_remains_legacy_fallback_when_config_omits_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.chdir(root)
            (root / "seq.txt").write_text("user\nacgt\n", encoding="utf-8")

            prob = self._run_input_initialize(
                root,
                {"para_write_102": False, "output_dir": str(root / "out")},
            )

            self.assertEqual(para.para_scaf_seq, "user")
            self.assertEqual(prob.scaf_seq, "ACGT")

    def test_name_prob_from_config_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.chdir(root)

            prob = self._run_input_initialize(
                root,
                {
                    "name_prob": "ConfiguredName",
                    "para_write_102": False,
                    "output_dir": str(root / "out"),
                },
            )

            self.assertEqual(prob.name_file, "shape")
            self.assertEqual(prob.name_prob, "ConfiguredName")


class TestSeqdesignValidation(unittest.TestCase):
    def setUp(self) -> None:
        reset_para_defaults()

    def tearDown(self) -> None:
        reset_para_defaults()

    def test_min_cut_method_fails_fast(self) -> None:
        para.para_cut_stap_method = "min"

        with self.assertRaisesRegex(ValueError, "Unsupported para_cut_stap_method"):
            seqdesign_design(ProbType(), GeomType(), MeshType(), DNAType())

    def test_staple_cut_prefilter_preserves_edge_thresholds(self) -> None:
        para.para_max_cut_stap = 32
        strand = StrandType(type1="stap", n_base=69, base=[0])
        dna = DNAType(top=[TopType(id=0, across=-1)])

        self.assertTrue(
            _skip_staple_cut_candidate(
                ProbType(sel_edge_sec=3),
                dna,
                strand,
                unpaired_edge_mode=3,
            )
        )

        strand.n_base = 70
        self.assertFalse(
            _skip_staple_cut_candidate(
                ProbType(sel_edge_sec=3),
                dna,
                strand,
                unpaired_edge_mode=3,
            )
        )

    def test_cut_staple_after_base_counts_even_when_no_upstream_base(self) -> None:
        dna = DNAType(
            n_stap=4,
            top=[TopType(id=0, up=-1)],
        )

        changed = _cut_staple_after_base(dna, 0)

        self.assertFalse(changed)
        self.assertEqual(dna.n_stap, 5)
        self.assertEqual(dna.top[0].up, -1)

    def test_break_top_link_after_base_does_not_update_counts(self) -> None:
        dna = DNAType(
            n_stap=4,
            top=[
                TopType(id=0, up=1),
                TopType(id=1, dn=0),
            ],
        )

        changed = _break_top_link_after_base(dna, 0)

        self.assertTrue(changed)
        self.assertEqual(dna.n_stap, 4)
        self.assertEqual(dna.top[0].up, -1)
        self.assertEqual(dna.top[1].dn, -1)

    def test_staple_cut_position_respects_min_max_and_tail_length(self) -> None:
        para.para_min_cut_stap = 10
        para.para_max_cut_stap = 20

        self.assertTrue(_is_valid_staple_cut_position(pre_pos=15, bgn_pos=0, strand_n_base=30))
        self.assertFalse(_is_valid_staple_cut_position(pre_pos=9, bgn_pos=0, strand_n_base=30))
        self.assertFalse(_is_valid_staple_cut_position(pre_pos=21, bgn_pos=0, strand_n_base=40))
        self.assertFalse(_is_valid_staple_cut_position(pre_pos=25, bgn_pos=10, strand_n_base=30))

    def test_update_max_stap_gap_change_advances_one_step_at_a_time(self) -> None:
        para.para_gap_xover_nick = 4
        prob = ProbType(n_cng_max_stap=0)

        _update_max_stap_gap_change(prob, cng_para=3)
        _update_max_stap_gap_change(prob, cng_para=1)
        self.assertEqual(prob.n_cng_max_stap, 1)

        _update_max_stap_gap_change(prob, cng_para=2)
        _update_max_stap_gap_change(prob, cng_para=1)
        self.assertEqual(prob.n_cng_max_stap, 3)

    def test_select_staple_cut_region_walks_back_and_relaxes_gap(self) -> None:
        para.para_gap_xover_nick = 4
        para.para_min_cut_stap = 4
        para.para_max_cut_stap = 20
        prob = ProbType()
        regions = [
            None,
            RegionType(types=2, length=3, cen_base=10, cen_pos=5),
            RegionType(types=2, length=10, cen_base=20, cen_pos=12),
        ]

        self.assertEqual(
            _select_staple_cut_region(
                prob,
                regions,
                start_idx=2,
                stop_idx=0,
                bgn_pos=0,
                strand_n_base=30,
                type_sensitive=False,
                allow_extended=True,
            ),
            (2, 20, 12, False),
        )

        self.assertEqual(
            _select_staple_cut_region(
                prob,
                regions,
                start_idx=1,
                stop_idx=0,
                bgn_pos=0,
                strand_n_base=30,
                type_sensitive=True,
                allow_extended=True,
            ),
            (1, 10, 5, True),
        )

    def test_short_xover_cut_predicate_checks_length_topology_and_mode(self) -> None:
        para.para_min_cut_stap = 10
        para.para_max_cut_stap = 30
        para.para_set_stap_sxover = "on"
        region = RegionType(length=12, end_pos=20)

        self.assertTrue(_can_cut_staple_short_xover(region, bgn_pos=0, strand_n_base=40, xover=1, upxover=2))
        self.assertFalse(_can_cut_staple_short_xover(region, bgn_pos=0, strand_n_base=40, xover=-1, upxover=2))

        para.para_set_stap_sxover = "off"
        self.assertFalse(_can_cut_staple_short_xover(region, bgn_pos=0, strand_n_base=40, xover=1, upxover=2))

    def test_cut_staple_short_xover_clears_topology_and_updates_counts(self) -> None:
        dna = DNAType(
            n_stap=1,
            n_xover_stap=2,
            n_sxover_stap=0,
            top=[
                TopType(id=0),
                TopType(id=1, up=2, dn=2, xover=2),
                TopType(id=2, up=1, dn=1, xover=1),
            ],
        )

        _cut_staple_short_xover(dna, up=1, xover=2)

        self.assertEqual(dna.n_stap, 2)
        self.assertEqual(dna.n_xover_stap, 1)
        self.assertEqual(dna.n_sxover_stap, 1)
        self.assertEqual(dna.top[1].up, -1)
        self.assertEqual(dna.top[1].dn, -1)
        self.assertEqual(dna.top[1].xover, -1)
        self.assertEqual(dna.top[2].up, -1)
        self.assertEqual(dna.top[2].dn, -1)
        self.assertEqual(dna.top[2].xover, -1)


class TestRouteSafety(unittest.TestCase):
    def setUp(self) -> None:
        reset_para_defaults()

    def tearDown(self) -> None:
        reset_para_defaults()

    def test_staple_xover_skips_boundary_nodes_without_negative_indexing(self) -> None:
        para.para_gap_xover_bound_stap = 0
        para.para_gap_xover_two = 2

        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, bp=10, up=-1, dn=-1, sec=0, iniL=0, croL=0),
                NodeType(id=1, bp=10, up=-1, dn=-1, sec=1, iniL=0, croL=1),
            ],
        )
        dna = DNAType(
            n_base_scaf=2,
            n_base_stap=2,
            base_scaf=[BaseType(id=0, xover=-1), BaseType(id=1, xover=-1)],
            base_stap=[BaseType(id=0, xover=-1), BaseType(id=1, xover=-1)],
        )
        geom = GeomType(n_croL=2, n_sec=2)

        with mock.patch("perdix_py.section.section_connection_stap", return_value=True):
            _find_possible_stap_xover(geom, mesh, dna)

        self.assertEqual(dna.n_xover_stap, 0)
        self.assertEqual([base.xover for base in dna.base_stap], [-1, -1])

    def test_centered_scaffold_candidates_require_matching_bp_and_edge(self) -> None:
        mesh = MeshType(
            n_node=4,
            node=[
                NodeType(id=0, bp=5, iniL=0, sec=0),
                NodeType(id=1, bp=5, iniL=0, sec=1),
                NodeType(id=2, bp=6, iniL=0, sec=2),
                NodeType(id=3, bp=5, iniL=1, sec=3),
            ],
        )

        self.assertEqual(list(_iter_centered_scaffold_xover_candidates(mesh)), [(0, 1)])

    def test_scaffold_neighbor_pair_skips_missing_neighbor_without_negative_indexing(self) -> None:
        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, up=-1, dn=0),
                NodeType(id=1, up=1, dn=0),
            ],
        )

        with mock.patch("perdix_py.section.section_connection_scaf", return_value=True):
            result = _find_scaffold_neighbor_pair(GeomType(), mesh, 0, 1)

        self.assertIsNone(result)

    def test_apply_scaffold_xover_pair_creates_reciprocal_links(self) -> None:
        dna = DNAType(
            n_base_scaf=4,
            base_scaf=[
                BaseType(id=0, xover=-1),
                BaseType(id=1, xover=-1),
                BaseType(id=2, xover=-1),
                BaseType(id=3, xover=-1),
            ],
        )

        _apply_scaffold_xover_pair(
            dna,
            node_cur=0,
            node_com=1,
            b_nei_up=True,
            b_nei_dn=False,
            up_cur=2,
            up_com=3,
            dn_cur=-1,
            dn_com=-1,
        )

        self.assertEqual(dna.n_xover_scaf, 2)
        self.assertEqual(dna.base_scaf[0].xover, 1)
        self.assertEqual(dna.base_scaf[1].xover, 0)
        self.assertEqual(dna.base_scaf[2].xover, 3)
        self.assertEqual(dna.base_scaf[3].xover, 2)

    def test_clear_scaffold_xover_pair_only_clears_requested_pair(self) -> None:
        dna = DNAType(
            base_scaf=[
                BaseType(id=0, xover=1),
                BaseType(id=1, xover=0),
                BaseType(id=2, xover=3),
                BaseType(id=3, xover=2),
            ],
        )

        _clear_scaffold_xover_pair(dna, 0, 1)

        self.assertEqual(dna.base_scaf[0].xover, -1)
        self.assertEqual(dna.base_scaf[1].xover, -1)
        self.assertEqual(dna.base_scaf[2].xover, 3)
        self.assertEqual(dna.base_scaf[3].xover, 2)

    def test_scaffold_xover_record_helpers_are_order_insensitive(self) -> None:
        records = [
            {
                "base1": (1, 4),
                "strd": (2, 7),
                "base2": (3, 6),
            }
        ]

        self.assertTrue(_has_scaffold_strand_pair_record(records, 7, 2))
        self.assertFalse(_has_scaffold_strand_pair_record(records, 7, 3))
        self.assertTrue(_has_scaffold_xover_record(records, 4, 1))
        self.assertTrue(_has_scaffold_xover_record(records, 6, 3))
        self.assertFalse(_has_scaffold_xover_record(records, 6, 4))

    def test_accepted_staple_xover_neighbor_pair_combines_candidate_filters(self) -> None:
        para.para_gap_xover_bound_stap = 0
        para.para_gap_xover_two = 0
        geom = GeomType(n_croL=2)
        mesh = MeshType(
            n_node=4,
            node=[
                NodeType(id=0, bp=5, sec=0, croL=0, dn=2, up=-1),
                NodeType(id=1, bp=5, sec=1, croL=1, up=3, dn=-1),
                NodeType(id=2, bp=4, sec=0, croL=0),
                NodeType(id=3, bp=4, sec=1, croL=1),
            ],
        )
        dna = DNAType(
            base_scaf=[BaseType(id=i, xover=-1) for i in range(4)],
            base_stap=[BaseType(id=i, xover=-1) for i in range(4)],
        )

        with mock.patch("perdix_py.section.section_connection_stap", return_value=True):
            self.assertEqual(
                _accepted_staple_xover_neighbor_pair(
                    geom,
                    mesh,
                    dna,
                    min_bp=[0, 0],
                    max_bp=[10, 10],
                    node_cur=0,
                    node_com=1,
                ),
                (True, 2, 3, -1, -1),
            )

    def test_scaffold_split_strategy_matches_center_and_split_modes(self) -> None:
        para.para_set_xover_scaf = "center"
        self.assertEqual(_scaffold_split_strategy(1), (0, "center"))
        self.assertEqual(_scaffold_split_strategy(4), (3, "down"))

        para.para_set_xover_scaf = "split"
        self.assertEqual(_scaffold_split_strategy(1), (3, "down"))
        self.assertEqual(_scaffold_split_strategy(5), (0, "center"))

    def test_scaffold_split_boundary_gap_checks_both_edges(self) -> None:
        para.para_gap_xover_bound_scaf = 2

        self.assertFalse(_scaffold_split_hits_boundary_gap(5, 5, 1, 10, 1, 10))
        self.assertTrue(_scaffold_split_hits_boundary_gap(2, 5, 1, 10, 1, 10))
        self.assertTrue(_scaffold_split_hits_boundary_gap(5, 9, 1, 10, 1, 10))

    def test_advance_scaffold_split_pair_stops_at_sentinel(self) -> None:
        mesh = MeshType(
            n_node=3,
            node=[
                NodeType(id=0, dn=1, up=-1),
                NodeType(id=1, dn=-1, up=2),
                NodeType(id=2, dn=1, up=-1),
            ],
        )

        self.assertEqual(_advance_scaffold_split_pair(mesh, 0, 2, "down", steps=1), (1, -1))
        self.assertEqual(_advance_scaffold_split_pair(mesh, 0, 2, "down", steps=2), (-1, -1))

    def test_scan_scaffold_split_direction_accepts_matching_connection(self) -> None:
        para.para_gap_xover_bound_scaf = 0
        geom = GeomType()
        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, bp=5, dn=-1, up=-1),
                NodeType(id=1, bp=6, dn=-1, up=-1),
            ],
        )

        with mock.patch("perdix_py.section.section_connection_scaf", return_value=True):
            self.assertEqual(
                _scan_scaffold_split_direction(
                    geom,
                    mesh,
                    node1=0,
                    node2=1,
                    sec1=2,
                    sec2=3,
                    n_cross=0,
                    direction="down",
                    min_bp1=1,
                    max_bp1=10,
                    min_bp2=1,
                    max_bp2=10,
                ),
                (False, 0, 1),
            )

    def test_scan_scaffold_split_direction_fails_at_boundary_or_sentinel(self) -> None:
        geom = GeomType()
        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, bp=2, dn=1, up=-1),
                NodeType(id=1, bp=5, dn=-1, up=-1),
            ],
        )
        para.para_gap_xover_bound_scaf = 2

        with mock.patch("perdix_py.section.section_connection_scaf", return_value=True):
            self.assertEqual(
                _scan_scaffold_split_direction(
                    geom,
                    mesh,
                    node1=0,
                    node2=1,
                    sec1=2,
                    sec2=3,
                    n_cross=0,
                    direction="down",
                    min_bp1=1,
                    max_bp1=10,
                    min_bp2=1,
                    max_bp2=10,
                ),
                (True, 0, 1),
            )

        para.para_gap_xover_bound_scaf = 0
        with mock.patch("perdix_py.section.section_connection_scaf", return_value=False):
            self.assertEqual(
                _scan_scaffold_split_direction(
                    geom,
                    mesh,
                    node1=0,
                    node2=1,
                    sec1=2,
                    sec2=3,
                    n_cross=0,
                    direction="down",
                    min_bp1=1,
                    max_bp1=10,
                    min_bp2=1,
                    max_bp2=10,
                ),
                (True, 1, -1),
            )

    def test_select_initial_centered_scaffold_split_retries_failed_step_three(self) -> None:
        para.para_set_xover_scaf = "split"
        geom = GeomType()
        geom.sec.maxR = 2
        geom.sec.maxC = 2

        with mock.patch(
            "perdix_py.route._split_centered_scaf_xover",
            side_effect=[(True, 10, 20), (False, 11, 21)],
        ) as split:
            result = _select_initial_centered_scaffold_split(
                geom,
                MeshType(),
                node_cur=1,
                node_com=2,
                min_bp1=0,
                max_bp1=10,
                min_bp2=0,
                max_bp2=10,
            )

        self.assertEqual(result, (11, 21))
        self.assertEqual([call.args[-1] for call in split.call_args_list], [3, 4])

    def test_select_best_centered_scaffold_split_keeps_best_gap_fallback(self) -> None:
        para.para_gap_xover_two_scaf = 4
        with (
            mock.patch(
                "perdix_py.route._split_centered_scaf_xover",
                side_effect=[
                    (False, 10, 20),
                    (False, 11, 21),
                    (True, 12, 22),
                    (False, 13, 23),
                    (False, 14, 24),
                ],
            ),
            mock.patch(
                "perdix_py.route._check_nei_xover",
                side_effect=[1, 3, 0, 2, 1],
            ),
        ):
            result = _select_best_centered_scaffold_split(
                GeomType(),
                MeshType(),
                DNAType(),
                node_cur=1,
                node_com=2,
                min_bp1=0,
                max_bp1=10,
                min_bp2=0,
                max_bp2=10,
                max_gap=0,
                max_cur=1,
                max_com=2,
            )

        self.assertEqual(result, (11, 21, 1))

    def test_orientation_skips_isolated_nodes(self) -> None:
        mesh = MeshType(
            n_node=1,
            node=[NodeType(id=0, up=-1, dn=-1, pos=(1.0, 2.0, 3.0))],
        )
        dna = DNAType(
            n_base_scaf=1,
            base_scaf=[BaseType(id=0, pos=(1.0, 2.0, 3.0))],
        )

        _set_orientation(mesh, dna)

        self.assertEqual(mesh.node[0].ori[0], (0.0, 0.0, 0.0))

    def test_route_bild_stops_unpaired_walk_at_sentinel(self) -> None:
        para.para_write_601_1 = True
        with tempfile.TemporaryDirectory() as tmp:
            prob = ProbType(path_work=tmp, name_file="native", n_edge_len=42)
            mesh = MeshType(
                n_node=1,
                node=[NodeType(id=0, pos=(0.0, 0.0, 0.0))],
            )
            dna = DNAType(
                n_base_scaf=2,
                n_base_stap=2,
                base_scaf=[
                    BaseType(id=0, node=0, up=1),
                    BaseType(id=1, node=-1, up=-1),
                ],
                base_stap=[
                    BaseType(id=0, node=0, up=1),
                    BaseType(id=1, node=-1, up=-1),
                ],
            )

            _write_route_bild(prob, GeomType(), mesh, dna, "route1")

            self.assertTrue((Path(tmp) / "native_42bp_route1_scaf.bild").exists())

    def test_route_bild_enabled_uses_step_specific_flags(self) -> None:
        para.para_write_601_1 = True
        para.para_write_601_2 = False

        self.assertTrue(_route_bild_enabled("route1"))
        self.assertFalse(_route_bild_enabled("route2"))
        self.assertFalse(_route_bild_enabled("unknown"))

    def test_route_bild_writes_xover_without_duplicate_backbone_segment(self) -> None:
        para.para_write_601_1 = True
        with tempfile.TemporaryDirectory() as tmp:
            prob = ProbType(path_work=tmp, name_file="native", n_edge_len=42)
            mesh = MeshType(
                n_node=2,
                node=[
                    NodeType(id=0, pos=(0.0, 0.0, 0.0)),
                    NodeType(id=1, pos=(1.0, 0.0, 0.0)),
                ],
            )
            dna = DNAType(
                n_base_scaf=2,
                n_base_stap=0,
                base_scaf=[
                    BaseType(id=0, node=0, up=1, xover=1),
                    BaseType(id=1, node=1, up=-1, xover=0),
                ],
            )

            _write_route_bild(prob, GeomType(), mesh, dna, "route1")

            text = (Path(tmp) / "native_42bp_route1_scaf.bild").read_text(encoding="utf-8")
            self.assertIn(".color tan\n", text)
            self.assertNotIn(".color steel blue\n", text)


class TestNativeSequenceTraversal(unittest.TestCase):
    def setUp(self) -> None:
        reset_para_defaults()

    def tearDown(self) -> None:
        reset_para_defaults()

    def test_scaffold_sequence_stops_at_sentinel(self) -> None:
        dna = DNAType()
        dna.top = [
            BaseType(id=0, up=1, dn=-1, across=-1),
            BaseType(id=1, up=-1, dn=0, across=-1),
        ]
        dna.strand = [
            type("Strand", (), {"type1": "scaf", "n_base": 4, "base": [0]})()
        ]

        _apply_scaffold_sequence(ProbType(), dna, "ACGT")

        self.assertEqual(dna.top[0].seq, "A")
        self.assertEqual(dna.top[1].seq, "C")

    def test_scaffold_xover_count_skips_unplaced_bases(self) -> None:
        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, croL=0),
                NodeType(id=1, croL=1),
            ],
        )
        dna = DNAType(
            strand=[StrandType(type1="scaf", n_base=3, base=[0, 1, 2])],
            top=[
                TopType(id=0, node=0, xover=1),
                TopType(id=1, node=1, xover=-1),
                TopType(id=2, node=-1, xover=0),
            ],
        )

        counts = _count_scaffold_crossovers_by_edge(GeomType(n_croL=2), mesh, dna)

        self.assertEqual(counts, [1, 0])

    def test_find_scaffold_nick_start_base_stops_at_boundary_node(self) -> None:
        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, dn=1),
                NodeType(id=1, dn=-1),
            ],
        )
        dna = DNAType(
            top=[
                TopType(id=0, node=0, dn=1),
                TopType(id=1, node=1, dn=-1),
            ],
        )
        strand = StrandType(n_base=2, base=[0])

        self.assertEqual(_find_scaffold_nick_start_base(mesh, dna, strand), 1)

    def test_advance_to_scaffold_nick_row_base_skips_unplaced_and_non_row_one_bases(self) -> None:
        geom = GeomType()
        geom.sec.posR = [2, 1]
        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, sec=0),
                NodeType(id=1, sec=1),
            ],
        )
        dna = DNAType(
            top=[
                TopType(id=0, node=-1, up=1),
                TopType(id=1, node=0, up=2),
                TopType(id=2, node=1, up=-1),
            ],
        )

        self.assertEqual(_advance_to_scaffold_nick_row_base(geom, mesh, dna, 0), 2)

    def test_scan_scaffold_nick_candidate_run_returns_longest_boundary_terminated_run(self) -> None:
        geom = GeomType(
            n_croL=1,
            iniL=[LineType(neiF=[-1, -1])],
            croL=[LineType(sec=2)],
        )
        geom.sec.posR = [1]
        mesh = MeshType(
            n_node=3,
            node=[
                NodeType(id=0, up=0, dn=0, sec=0, croL=0, iniL=0, mitered=-1),
                NodeType(id=1, up=1, dn=1, sec=0, croL=0, iniL=0, mitered=-1),
                NodeType(id=2, up=2, dn=2, sec=0, croL=0, iniL=0, mitered=-1),
            ],
        )
        dna = DNAType(
            top=[
                TopType(id=0, node=0, up=1, across=3, xover=-1),
                TopType(id=1, node=1, up=2, across=3, xover=-1),
                TopType(id=2, node=2, up=-1, across=3, xover=3),
                TopType(id=3, up=3, dn=3, xover=-1),
            ],
        )
        strand = StrandType(n_base=3, base=[0])

        self.assertEqual(
            _scan_scaffold_nick_candidate_run(
                geom,
                mesh,
                dna,
                strand,
                base=0,
                croL_n_xover=[0],
                b_inside=True,
            ),
            (0, 2),
        )

    def test_short_scaffold_region_collection_and_centers_follow_top_links(self) -> None:
        dna = DNAType(
            top=[
                TopType(id=0, up=1, dn=-1, across=-1),
                TopType(id=1, up=2, dn=0, across=5, xover=-1),
                TopType(id=2, up=3, dn=1, across=6, xover=-1),
                TopType(id=3, up=4, dn=2, across=7, xover=-1),
                TopType(id=4, up=-1, dn=3, across=-1),
                TopType(id=5, xover=-1),
                TopType(id=6, xover=-1),
                TopType(id=7, xover=-1),
            ],
        )

        regions = _collect_short_scaffold_regions(dna, base=0, n_base=5)
        _assign_short_scaffold_region_centers(dna, regions)

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].strt_base, 1)
        self.assertEqual(regions[0].length, 3)
        self.assertEqual(regions[0].end_pos, 4)
        self.assertEqual(regions[0].cntr_base, 2)
        self.assertEqual(regions[0].cntr_pos, 3)

    def test_cut_short_scaffold_regions_splits_only_before_following_region(self) -> None:
        para.para_max_cut_scaf = 2
        dna = DNAType(
            n_scaf=1,
            top=[
                TopType(id=0, up=1, dn=-1),
                TopType(id=1, up=2, dn=0),
                TopType(id=2, up=-1, dn=1),
            ],
        )
        regions = [
            ScafRegionType(cntr_base=1, cntr_pos=3),
            ScafRegionType(cntr_base=2, cntr_pos=5),
        ]

        _cut_short_scaffold_regions(dna, regions)

        self.assertEqual(dna.n_scaf, 2)
        self.assertEqual(dna.top[1].up, -1)
        self.assertEqual(dna.top[2].dn, -1)

    def test_apply_scaffold_nick_breaks_reciprocal_link_and_marks_once(self) -> None:
        dna = DNAType(
            top=[
                TopType(id=0, dn=1),
                TopType(id=1, up=0),
            ],
        )

        marked = _apply_scaffold_nick(dna, 0, 1, mark_status=True)

        self.assertTrue(marked)
        self.assertEqual(dna.top[0].dn, -1)
        self.assertEqual(dna.top[1].up, -1)
        self.assertEqual(dna.top[0].status, "nick_scaf")

    def test_scaffold_nick_boundary_predicate_is_read_only(self) -> None:
        geom = GeomType(
            n_croL=1,
            iniL=[LineType(neiF=[-1, -1])],
            croL=[LineType(sec=2)],
        )
        mesh = MeshType(
            n_node=1,
            node=[NodeType(id=0, up=0, dn=0, iniL=0, croL=0, mitered=-1)],
        )
        dna = DNAType(
            top=[
                TopType(id=0, across=1, xover=-1),
                TopType(id=1, up=1, dn=1, xover=-1),
            ],
        )

        self.assertFalse(_is_scaffold_nick_boundary(geom, mesh, dna, base=0, node=0, b_inside=True))

        dna.top[0].across = -1
        self.assertTrue(_is_scaffold_nick_boundary(geom, mesh, dna, base=0, node=0, b_inside=True))


class TestBasepairBookkeeping(unittest.TestCase):
    def test_open_geometry_junction_fallback_pairs_unmatched_same_grid_nodes(self) -> None:
        geom = GeomType(
            n_sec=2,
            n_junc=1,
            junc=[
                JuncType(
                    n_arm=1,
                    conn=[[0, -1], [1, -1]],
                )
            ],
        )
        geom.sec.posR = [1, 1]
        geom.sec.posC = [2, 2]
        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, sec=0),
                NodeType(id=1, sec=1),
            ],
        )

        _apply_open_geometry_junction_fallback(geom, mesh)

        self.assertEqual(geom.junc[0].conn, [[0, 1], [1, 0]])

    def test_junction_self_connections_respect_section_connectivity(self) -> None:
        geom = GeomType(
            n_sec=2,
            n_junc=1,
            junc=[
                JuncType(
                    n_arm=1,
                    node=[[0, 1]],
                    conn=[[-1, -1], [-1, -1]],
                    type_conn=[0, 0],
                )
            ],
        )
        geom.sec.conn = [1, 0]
        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, sec=0),
                NodeType(id=1, sec=1),
            ],
        )

        _apply_junction_self_connections(geom, mesh)

        self.assertEqual(geom.junc[0].conn, [[0, 1], [1, 0]])
        self.assertEqual(geom.junc[0].type_conn, [2, 2])

    def test_unique_junction_connections_drops_open_and_reversed_duplicates(self) -> None:
        junc = JuncType(
            conn=[
                [1, 2],
                [2, 1],
                [-1, 3],
                [3, 4],
            ],
            type_conn=[1, 2, 1, 2],
        )

        conn, type_conn = _unique_junction_connections(junc)

        self.assertEqual(conn, [[1, 2], [3, 4]])
        self.assertEqual(type_conn, [1, 2])

    def test_find_internal_junction_lines_counts_multi_arm_reuse_without_faces(self) -> None:
        geom = GeomType(
            n_iniL=4,
            junc=[
                JuncType(n_arm=3, iniL=[0, 1, 2]),
                JuncType(n_arm=3, iniL=[2, 3]),
                JuncType(n_arm=2, iniL=[0, 3]),
            ],
        )

        self.assertEqual(_find_internal_junction_lines(geom), {2})

    def test_should_increase_mitered_edge_matches_face_arm_and_open_filters(self) -> None:
        para.para_vertex_design = "mitered"
        geom = GeomType(
            n_sec=1,
            junc=[JuncType(n_arm=3, ref_ang=1.0)],
        )
        mesh = MeshType(
            n_node=1,
            node=[NodeType(id=0, sec=0, iniL=4)],
        )

        self.assertTrue(_should_increase_mitered_edge(geom, mesh, 0, 0, internal_lines=set()))
        self.assertFalse(_should_increase_mitered_edge(geom, mesh, 0, 0, internal_lines={4}))

        geom.junc[0].ref_ang = 1.3
        self.assertFalse(_should_increase_mitered_edge(geom, mesh, 0, 0, internal_lines=set()))

        geom.junc[0].n_arm = 2
        self.assertTrue(_should_increase_mitered_edge(geom, mesh, 0, 0, internal_lines={4}))

        geom.face = [object()]
        para.para_vertex_design = "plain"
        self.assertFalse(_should_increase_mitered_edge(geom, mesh, 0, 0, internal_lines={4}))

    def test_apply_vertex_crash_strategy_handles_const_and_mod2_without_policy_inline(self) -> None:
        geom = GeomType()
        mesh = MeshType(
            n_node=2,
            node=[NodeType(id=0), NodeType(id=1)],
        )

        para.para_vertex_crash = "const"
        with mock.patch("perdix_py.basepair._find_xover_nearby", return_value=0):
            _apply_vertex_crash_strategy(geom, mesh, 0, 1)
        self.assertEqual(mesh.node[0].conn, 2)
        self.assertEqual(mesh.node[1].conn, 2)

        para.para_vertex_crash = "mod2"
        with mock.patch("perdix_py.basepair._find_xover_nearby", return_value=0):
            _apply_vertex_crash_strategy(geom, mesh, 0, 1)
        self.assertEqual(mesh.node[0].conn, 4)
        self.assertEqual(mesh.node[1].conn, 4)

    def test_apply_vertex_crash_strategy_mod1_uses_row_threshold_or_ghost_node(self) -> None:
        para.para_vertex_crash = "mod1"
        geom = GeomType()
        geom.sec.posR = [2]
        geom.sec.ref_row = 1
        mesh = MeshType(
            n_node=2,
            node=[NodeType(id=0, sec=0), NodeType(id=1, sec=0)],
        )

        _apply_vertex_crash_strategy(geom, mesh, 0, 1)
        self.assertEqual(mesh.node[0].conn, 2)
        self.assertEqual(mesh.node[1].conn, 2)

        geom.sec.posR = [0]
        with mock.patch("perdix_py.basepair._make_ghost_node") as make_ghost:
            _apply_vertex_crash_strategy(geom, mesh, 0, 1)
        make_ghost.assert_called_once_with(geom, mesh, 0, 1)

    def test_xover_search_direction_helpers_match_endpoint_orientation(self) -> None:
        mesh = MeshType(
            n_node=4,
            node=[
                NodeType(id=0, sec=0, up=-1, dn=1),
                NodeType(id=1, sec=0, up=0, dn=-1),
                NodeType(id=2, sec=1, up=-1, dn=3),
                NodeType(id=3, sec=1, up=2, dn=-1),
            ],
        )

        self.assertTrue(_xover_search_starts_inside(mesh, 0))
        self.assertEqual(_xover_search_steps(mesh, 0), (-1, 1))
        self.assertTrue(_xover_search_starts_outside(mesh, 1))
        self.assertEqual(_xover_search_steps(mesh, 1), (1, -1))
        self.assertTrue(_xover_search_starts_outside(mesh, 2))
        self.assertEqual(_xover_search_steps(mesh, 2), (1, -1))
        self.assertTrue(_xover_search_starts_inside(mesh, 3))
        self.assertEqual(_xover_search_steps(mesh, 3), (-1, 1))

    def test_initial_xover_move_maps_previous_connection_to_signed_move(self) -> None:
        geom = GeomType()
        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, sec=0, up=-1, dn=1),
                NodeType(id=1, sec=0, up=0, dn=-1),
            ],
        )

        with mock.patch("perdix_py.section.section_connection_scaf", return_value=True):
            self.assertEqual(_initial_xover_move(geom, mesh, 0, 0, 1, 5), -1)
            self.assertEqual(_initial_xover_move(geom, mesh, 1, 0, 1, 5), 0)

        with mock.patch("perdix_py.section.section_connection_scaf", return_value=False):
            self.assertEqual(_initial_xover_move(geom, mesh, 0, 0, 1, 5), 0)
            self.assertEqual(_initial_xover_move(geom, mesh, 1, 0, 1, 5), -1)

    def test_basepair_adjustment_endpoints_select_down_and_up_sides(self) -> None:
        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, up=-1, dn=1),
                NodeType(id=1, up=0, dn=-1),
            ],
        )

        self.assertEqual(_basepair_adjustment_endpoints(mesh, 0, 1), (0, 1))

        mesh.node[0].up = 1
        mesh.node[0].dn = -1
        self.assertEqual(_basepair_adjustment_endpoints(mesh, 0, 1), (1, 0))

        mesh.node[0].dn = 1
        self.assertIsNone(_basepair_adjustment_endpoints(mesh, 0, 1))

    def test_ghost_node_walk_start_selects_parallel_neighbors(self) -> None:
        mesh = MeshType(
            n_node=4,
            node=[
                NodeType(id=0, up=-1, dn=2),
                NodeType(id=1, up=3, dn=-1),
                NodeType(id=2),
                NodeType(id=3),
            ],
        )

        self.assertEqual(_ghost_node_walk_start(mesh, 0, 1), (3, 2))

        mesh.node[0].up = 2
        mesh.node[0].dn = -1
        mesh.node[1].dn = 3
        self.assertEqual(_ghost_node_walk_start(mesh, 0, 1), (2, 3))

        mesh.node[1].dn = -1
        self.assertIsNone(_ghost_node_walk_start(mesh, 0, 1))

    def test_mark_ghost_chain_stops_at_second_xover_or_sentinel(self) -> None:
        geom = GeomType()
        mesh = MeshType(
            n_node=4,
            node=[
                NodeType(id=0, up=1, dn=-1, sec=0, bp=0),
                NodeType(id=1, up=-1, dn=0, sec=0, bp=1),
                NodeType(id=2, up=3, dn=-1, sec=1, bp=0),
                NodeType(id=3, up=-1, dn=2, sec=1, bp=1),
            ],
        )

        with mock.patch("perdix_py.section.section_connection_scaf", side_effect=[False, True]):
            _mark_ghost_chain_until_second_xover(geom, mesh, node_up=0, node_dn=2)

        self.assertEqual(mesh.node[0].ghost, 1)
        self.assertEqual(mesh.node[2].ghost, 1)
        self.assertEqual(mesh.node[1].ghost, 0)
        self.assertEqual(mesh.node[3].ghost, 0)

        with mock.patch("perdix_py.section.section_connection_scaf", return_value=False):
            _mark_ghost_chain_until_second_xover(geom, mesh, node_up=1, node_dn=3)

    def test_edge_increase_endpoint_pair_classifies_open_ends(self) -> None:
        mesh = MeshType(
            n_node=2,
            node=[
                NodeType(id=0, up=-1, dn=1),
                NodeType(id=1, up=0, dn=2),
            ],
        )

        self.assertEqual(_edge_increase_endpoint_pair(mesh, 0, 1), (0, 1))

        mesh.node[0].up = 1
        mesh.node[0].dn = -1
        self.assertEqual(_edge_increase_endpoint_pair(mesh, 0, 1), (1, 0))

        mesh.node[0].dn = 1
        mesh.node[1].up = -1
        self.assertEqual(_edge_increase_endpoint_pair(mesh, 0, 1), (1, 0))

        mesh.node[1].up = 0
        mesh.node[1].dn = -1
        self.assertEqual(_edge_increase_endpoint_pair(mesh, 0, 1), (0, 1))

    def test_edge_extension_count_from_lengths_rejects_impossible_geometry(self) -> None:
        self.assertIsNone(_edge_extension_count_from_lengths(a=1.0, b=5.0, c=5.0, y=1.0))
        self.assertEqual(_edge_extension_count_from_lengths(a=1.0, b=5.0, c=7.0, y=1.0), 2)

    def test_mark_edge_endpoint_connection_marks_length_mismatch_as_three(self) -> None:
        para.para_dist_bp = 1.0
        geom = GeomType(
            n_croP=2,
            n_croL=1,
            croP=[
                PointType(pos=(0.0, 0.0, 0.0)),
                PointType(pos=(4.0, 0.0, 0.0)),
            ],
            croL=[LineType(poi=[0, 1])],
        )
        mesh = MeshType(
            n_node=1,
            node=[NodeType(id=0, croL=0)],
        )

        _mark_edge_endpoint_connection(ProbType(n_edge_len=4), geom, mesh, node=0)

        self.assertEqual(mesh.node[0].conn, 3)

    def test_reserve_mesh_node_and_element_appends_placeholders(self) -> None:
        mesh = MeshType(n_node=1, n_ele=1, node=[NodeType(id=0)], ele=[EleType(cn=(0, 0))])

        new_id = _reserve_mesh_node_and_element(mesh)

        self.assertEqual(new_id, 1)
        self.assertEqual(mesh.n_node, 2)
        self.assertEqual(mesh.n_ele, 2)
        self.assertEqual(len(mesh.node), 2)
        self.assertEqual(len(mesh.ele), 2)

    def test_append_sticky_end_node_adds_downstream_node(self) -> None:
        para.para_sticky_self = "on"
        geom = GeomType()
        geom.sec.conn = [-1]
        mesh = MeshType(
            n_node=1,
            n_ele=0,
            node=[NodeType(id=0, bp=5, sec=0, up=0, dn=-1)],
            ele=[],
        )

        result = _append_sticky_end_node(geom, mesh, node=0, sec=0)

        self.assertEqual(result, (1, 0))
        self.assertEqual(mesh.node[1].bp, 4)
        self.assertEqual(mesh.node[1].up, 0)
        self.assertEqual(mesh.node[0].dn, 1)

    def test_copy_and_place_sticky_end_node_use_existing_geometry(self) -> None:
        para.para_dist_bp = 2.0
        mesh = MeshType(
            n_node=3,
            node=[
                NodeType(id=0, pos=(1.0, 0.0, 0.0), sec=4, iniL=5, croL=6, conn=7, ghost=8),
                NodeType(id=1, pos=(0.0, 0.0, 0.0)),
                NodeType(id=2),
            ],
        )

        _copy_sticky_end_node_fields(mesh, node=0, new_id=2)
        _place_sticky_end_node(mesh, node=0, new_id=2, pre=1)

        self.assertEqual(mesh.node[2].id, 2)
        self.assertEqual(mesh.node[2].sec, 4)
        self.assertEqual(mesh.node[2].iniL, 5)
        self.assertEqual(mesh.node[2].croL, 6)
        self.assertEqual(mesh.node[2].conn, 7)
        self.assertEqual(mesh.node[2].ghost, 8)
        self.assertEqual(mesh.node[2].pos, (3.0, 0.0, 0.0))

    def test_replace_sticky_end_junction_node_updates_owner_and_connections(self) -> None:
        geom = GeomType(
            n_sec=2,
            junc=[
                JuncType(
                    n_arm=1,
                    node=[[4, 5]],
                    conn=[[4, 10], [11, 4]],
                ),
                JuncType(
                    n_arm=1,
                    node=[[6, 7]],
                    conn=[[12, 4], [13, 14]],
                ),
            ],
        )

        _replace_sticky_end_junction_node(
            geom,
            old_node=4,
            new_id=20,
            junc_index=0,
            arm_index=0,
            sec_index=0,
        )

        self.assertEqual(geom.junc[0].node[0][0], 20)
        self.assertEqual(geom.junc[0].conn, [[20, 10], [11, 20]])
        self.assertEqual(geom.junc[1].conn, [[12, 20], [13, 14]])

    def test_apply_ghost_node_replacement_updates_endpoint_and_connections(self) -> None:
        geom = GeomType(
            n_sec=1,
            croP=[PointType(pos=(0.0, 0.0, 0.0))],
            junc=[
                JuncType(
                    n_arm=1,
                    croP=[[0]],
                    node=[[2]],
                    conn=[[2, 5]],
                )
            ],
        )
        mesh = MeshType(
            n_node=6,
            node=[
                NodeType(id=0),
                NodeType(id=1),
                NodeType(id=2),
                NodeType(id=3, dn=2, pos=(1.0, 2.0, 3.0)),
                NodeType(id=4),
                NodeType(id=5),
            ],
        )

        end_node = _apply_ghost_node_replacement(
            geom,
            mesh,
            junc_idx=0,
            arm_idx=0,
            sec_idx=0,
            old_node=2,
            replacement_node=3,
            break_field="dn",
        )

        self.assertEqual(end_node, 2)
        self.assertEqual(mesh.node[3].dn, -1)
        self.assertEqual(mesh.node[3].conn, 4)
        self.assertEqual(geom.junc[0].node, [[3]])
        self.assertEqual(geom.croP[0].pos, (1.0, 2.0, 3.0))
        self.assertEqual(geom.junc[0].conn, [[3, 5]])

    def test_find_ghost_chain_replacement_walks_up_down_and_stops_at_sentinel(self) -> None:
        mesh = MeshType(
            n_node=3,
            node=[
                NodeType(id=0, up=1, dn=-1, ghost=1),
                NodeType(id=1, up=2, dn=0, ghost=1),
                NodeType(id=2, up=-1, dn=1, ghost=-1),
            ],
        )

        self.assertEqual(_find_ghost_chain_replacement(mesh, 0), (2, "dn"))

        mesh.node[0].dn = 1
        mesh.node[1].dn = 2
        self.assertEqual(_find_ghost_chain_replacement(mesh, 0), (2, "up"))

        mesh.node[0].dn = -1
        mesh.node[0].up = -1
        self.assertIsNone(_find_ghost_chain_replacement(mesh, 0))

    def test_delete_ghost_node_chain_replaces_then_renumbers_survivors(self) -> None:
        geom = GeomType(
            n_sec=1,
            croP=[PointType(pos=(0.0, 0.0, 0.0))],
            junc=[
                JuncType(
                    n_arm=1,
                    croP=[[0]],
                    node=[[1]],
                    conn=[[1, 4]],
                )
            ],
        )
        mesh = MeshType(
            n_node=4,
            n_ele=2,
            node=[
                NodeType(id=0),
                NodeType(id=1, ghost=1),
                NodeType(id=2, ghost=1),
                NodeType(id=3, up=2, pos=(3.0, 0.0, 0.0)),
            ],
            ele=[EleType(cn=(0, 1)), EleType(cn=(0, 3))],
        )

        _delete_ghost_node_chain(
            geom,
            mesh,
            junc_idx=0,
            arm_idx=0,
            sec_idx=0,
            start_node=1,
            replacement_node=3,
            break_field="up",
        )

        self.assertEqual(mesh.n_node, 2)
        self.assertEqual([node.id for node in mesh.node], [0, 1])
        self.assertEqual(mesh.node[1].up, -1)
        self.assertEqual(mesh.node[1].conn, 4)
        self.assertEqual(mesh.ele[0].cn, (0, 1))
        self.assertEqual(geom.junc[0].node, [[1]])
        self.assertEqual(geom.junc[0].conn, [[1, 2]])
        self.assertEqual(geom.croP[0].pos, (3.0, 0.0, 0.0))

    def test_replace_junction_connection_node_updates_both_sides(self) -> None:
        geom = GeomType(
            n_sec=2,
            junc=[
                JuncType(
                    n_arm=1,
                    conn=[[3, 4], [5, 3]],
                )
            ],
        )

        _replace_junction_connection_node(geom, old_node=3, new_node=9)

        self.assertEqual(geom.junc[0].conn, [[9, 4], [5, 9]])

    def test_shift_junction_references_after_delete_preserves_lower_ids(self) -> None:
        geom = GeomType(
            n_sec=2,
            junc=[
                JuncType(
                    n_arm=1,
                    node=[[1, 5]],
                    conn=[[0, 4], [6, 2]],
                )
            ],
        )

        _shift_junction_references_after_node_delete(geom, max_idx=3, size=2)

        self.assertEqual(geom.junc[0].node, [[1, 3]])
        self.assertEqual(geom.junc[0].conn, [[0, 2], [4, 2]])

    def test_renumber_optional_node_ref_after_delete_handles_deleted_and_shifted_refs(self) -> None:
        self.assertEqual(_renumber_optional_node_ref_after_delete(2, min_idx=2, max_idx=4, n_delete=3), -1)
        self.assertEqual(_renumber_optional_node_ref_after_delete(7, min_idx=2, max_idx=4, n_delete=3), 4)
        self.assertEqual(_renumber_optional_node_ref_after_delete(1, min_idx=2, max_idx=4, n_delete=3), 1)
        self.assertEqual(_renumber_optional_node_ref_after_delete(-1, min_idx=2, max_idx=4, n_delete=3), -1)

    def test_renumber_element_after_delete_shifts_surviving_endpoints(self) -> None:
        self.assertEqual(_renumber_element_after_delete(1, 7, max_idx=4, n_delete=3), (1, 4))


if __name__ == "__main__":
    unittest.main()
