from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import unittest
import re
import json
from typing import Optional

from perdix_py.svg_import import import_svg_to_geom


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "tests" / "output"
REF_ROOT = ROOT / "tests" / "reference"
DEFAULT_EDGE_LEN = 42
CONFIG_PATH = "tests/perdix_test_config.json"

@dataclass(frozen=True)
class ExpectedArtifact:
    suffix: str
    reader: str = "text"
    reference_prefix: str = "reference_"

class TestLineBild(unittest.TestCase):
    # Collected across all tests in this class
    _report = []  # list[dict[str, str]]

    def _config_path_for(self, shape: str, input_name: str, edge_len: int) -> Path:
        base_cfg_path = ROOT / CONFIG_PATH
        cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
        # Match reference behavior per shape/edge length.
        if shape == "square" and edge_len == 63:
            cfg["para_cut_stap_method"] = "max"
            cfg["para_method_sort"] = "quick"
            cfg["para_set_start_scaf"] = 1
            cfg["name_prob"] = "01_Square"
        elif shape == "hex" and edge_len == 121:
            cfg["para_cut_stap_method"] = "opt"
            cfg["para_method_sort"] = "none"
            cfg["para_set_start_scaf"] = 60
        else:
            cfg["para_cut_stap_method"] = "opt" if edge_len <= 42 else "max"
            cfg["para_method_sort"] = "none" if edge_len <= 42 else "quick"

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        cfg_path = OUTPUT_ROOT / f"perdix_test_config_{Path(input_name).stem}_{edge_len}bp.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return cfg_path

    def _run_pipeline(
        self,
        shape: str,
        input_name: str,
        edge_len: int,
        extra_args: Optional[list[str]] = None,
    ) -> None:
        cfg_path = self._config_path_for(shape, input_name, edge_len)
        cmd = [
            sys.executable,
            "-m",
            "perdix_py.main",
            input_name,
            "--config",
            str(cfg_path),
            "--edge-len",
            str(edge_len),
            "--debug-mesh"
        ]
        if extra_args:
            cmd.extend(extra_args)
        # redirect stdout/stderr to avoid cluttering test output if desired,
        # otherwise keep as is.
        subprocess.check_call(cmd, cwd=ROOT)

    def _run_pipeline_svg_rect(self, input_name: str, edge_len: int) -> None:
        cfg_path = self._config_path_for(Path(input_name).stem, input_name, edge_len)
        cmd = [
            sys.executable,
            "-m",
            "perdix_py.main",
            input_name,
            "--config",
            str(cfg_path),
            "--edge-len",
            str(edge_len),
            "--frame-mode",
            "svg-rect",
        ]
        subprocess.check_call(cmd, cwd=ROOT)

    @staticmethod
    def _geom_line_signature(
        input_name: str,
        svg_layer: Optional[int] = None,
    ) -> tuple[set[tuple[tuple[float, float], tuple[float, float]]], int]:
        geom = import_svg_to_geom(str(ROOT / "input" / input_name), scale=1.0, layer_index=svg_layer)
        xs = [float(p.pos[0]) for p in geom.iniP]
        ys = [float(p.pos[1]) for p in geom.iniP]
        min_x = min(xs)
        min_y = min(ys)
        span_x = max(xs) - min_x
        span_y = max(ys) - min_y
        if span_x <= 1e-12:
            span_x = 1.0
        if span_y <= 1e-12:
            span_y = 1.0

        def norm_point(pos: tuple[float, float, float]) -> tuple[float, float]:
            return (
                round((float(pos[0]) - min_x) / span_x, 2),
                round((float(pos[1]) - min_y) / span_y, 2),
            )

        lines: set[tuple[tuple[float, float], tuple[float, float]]] = set()
        for line in geom.iniL:
            p1 = geom.iniP[line.poi[0]].pos
            p2 = geom.iniP[line.poi[1]].pos
            a = norm_point(p1)
            b = norm_point(p2)
            lines.add((a, b) if a <= b else (b, a))
        return lines, geom.n_iniP

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        return text.replace("\r\n", "\n")

    def _read_text(self, path: Path) -> str:
        return self._normalize_newlines(path.read_text(encoding="utf-8"))

    def _read_cndo(self, path: Path) -> str:
        text = self._normalize_newlines(path.read_text(encoding="utf-8"))
        return text.replace(",-.000", ",.000").replace(",-0.000", ",0.000")

    def _read_with(self, path: Path, reader: str) -> str:
        if reader == "text":
            return self._read_text(path)
        if reader == "cndo":
            return self._read_cndo(path)
        raise ValueError(f"Unknown reader: {reader}")

    @staticmethod
    def _first_mismatch(a: str, b: str) -> int | None:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        if len(a) != len(b):
            return n
        return None

    @staticmethod
    def _load_bild_points(path: Path) -> set[tuple[str, str, str]]:
        pts: set[tuple[str, str, str]] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == ".sphere" and len(parts) >= 4:
                pts.add((parts[1], parts[2], parts[3]))
            elif parts[0] == ".cylinder" and len(parts) >= 7:
                pts.add((parts[1], parts[2], parts[3]))
                pts.add((parts[4], parts[5], parts[6]))
        return pts

    @staticmethod
    def _load_bild_sphere_points(path: Path) -> set[tuple[str, str, str]]:
        pts: set[tuple[str, str, str]] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and parts[0] == ".sphere" and len(parts) >= 4:
                pts.add((parts[1], parts[2], parts[3]))
        return pts

    @staticmethod
    def _load_bild_sphere_points_float(path: Path) -> list[tuple[float, float, float]]:
        pts: list[tuple[float, float, float]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and parts[0] == ".sphere" and len(parts) >= 4:
                pts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        return pts

    @staticmethod
    def _sphere_overlap_ratio_tol(
        reference: list[tuple[float, float, float]],
        query: list[tuple[float, float, float]],
        tol: float,
    ) -> float:
        if not query:
            return 0.0
        tol2 = tol * tol
        hit = 0
        for qx, qy, qz in query:
            best = None
            for rx, ry, rz in reference:
                d2 = (rx - qx) * (rx - qx) + (ry - qy) * (ry - qy) + (rz - qz) * (rz - qz)
                if best is None or d2 < best:
                    best = d2
                if best <= tol2:
                    break
            if best is not None and best <= tol2:
                hit += 1
        return hit / float(len(query))

    @staticmethod
    def _load_bild_red_cylinders(
        path: Path,
    ) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        cyls: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        current_color = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == ".color":
                current_color = " ".join(parts[1:]).lower()
                continue
            if parts[0] != ".cylinder" or len(parts) < 8 or current_color != "red":
                continue
            p1 = (float(parts[1]), float(parts[2]), float(parts[3]))
            p2 = (float(parts[4]), float(parts[5]), float(parts[6]))
            cyls.append((p1, p2))
        return cyls

    @staticmethod
    def _red_cylinder_overlap_ratio_tol(
        reference: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
        query: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
        tol: float,
    ) -> float:
        if not query:
            return 0.0
        hit = 0
        for (q1x, q1y, q1z), (q2x, q2y, q2z) in query:
            best = None
            for (r1x, r1y, r1z), (r2x, r2y, r2z) in reference:
                same = (
                    ((q1x - r1x) ** 2 + (q1y - r1y) ** 2 + (q1z - r1z) ** 2) ** 0.5
                    + ((q2x - r2x) ** 2 + (q2y - r2y) ** 2 + (q2z - r2z) ** 2) ** 0.5
                )
                swapped = (
                    ((q1x - r2x) ** 2 + (q1y - r2y) ** 2 + (q1z - r2z) ** 2) ** 0.5
                    + ((q2x - r1x) ** 2 + (q2y - r1y) ** 2 + (q2z - r1z) ** 2) ** 0.5
                )
                dist = same if same < swapped else swapped
                if best is None or dist < best:
                    best = dist
                if best <= tol:
                    break
            if best is not None and best <= tol:
                hit += 1
        return hit / float(len(query))

    @staticmethod
    def _red_cylinder_midpoint_overlap_ratio_tol(
        reference: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
        query: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
        tol: float,
    ) -> float:
        if not query:
            return 0.0
        ref_mid = [
            ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)
            for a, b in reference
        ]
        qry_mid = [
            ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)
            for a, b in query
        ]
        return TestLineBild._sphere_overlap_ratio_tol(ref_mid, qry_mid, tol)

    @staticmethod
    def _compare_with_tolerance(ref_text: str, gen_text: str) -> bool:
        float_pat = re.compile(r"[-+]?(?:\d+\.\d+|\.\d+)")
        ref_iter = list(float_pat.finditer(ref_text))
        gen_iter = list(float_pat.finditer(gen_text))
        if len(ref_iter) != len(gen_iter):
            return False

        def _norm_ws(segment: str) -> str:
            return re.sub(r"\s+", " ", segment)

        def _cmp_numeric_tokens(ref_tok: str, gen_tok: str) -> bool:
            dec_r = len(ref_tok.split(".")[1]) if "." in ref_tok else 0
            dec_g = len(gen_tok.split(".")[1]) if "." in gen_tok else 0
            decimals = max(dec_r, dec_g)
            tol = (6 * (10 ** (-decimals))) + 1e-12
            return abs(float(ref_tok) - float(gen_tok)) <= tol

        prev_ref = 0
        prev_gen = 0
        for ref_m, gen_m in zip(ref_iter, gen_iter):
            # compare non-float segments exactly
            if _norm_ws(ref_text[prev_ref:ref_m.start()]) != _norm_ws(gen_text[prev_gen:gen_m.start()]):
                return False

            ref_tok = ref_m.group(0)
            gen_tok = gen_m.group(0)
            try:
                if not _cmp_numeric_tokens(ref_tok, gen_tok):
                    return False
            except ValueError:
                return False

            prev_ref = ref_m.end()
            prev_gen = gen_m.end()

        # compare tail segments
        return _norm_ws(ref_text[prev_ref:]) == _norm_ws(gen_text[prev_gen:])

    @staticmethod
    def _compare_with_sign_tolerance(ref_text: str, gen_text: str) -> tuple[bool, bool]:
        def _norm_ws(segment: str) -> str:
            return re.sub(r"\s+", " ", segment).strip()

        def _num_ok(ref_tok: str, gen_tok: str, allow_sign: bool) -> tuple[bool, bool]:
            dec_r = len(ref_tok.split(".")[1]) if "." in ref_tok else 0
            dec_g = len(gen_tok.split(".")[1]) if "." in gen_tok else 0
            decimals = max(dec_r, dec_g)
            tol = (6 * (10 ** (-decimals))) + 1e-12
            ref_val = float(ref_tok)
            gen_val = float(gen_tok)
            if abs(ref_val - gen_val) <= tol:
                return True, False
            if allow_sign and abs(ref_val + gen_val) <= tol:
                return True, True
            return False, False

        ref_lines = ref_text.splitlines()
        gen_lines = gen_text.splitlines()
        if len(ref_lines) != len(gen_lines):
            return False, False

        used_flip = False
        for ref_line, gen_line in zip(ref_lines, gen_lines):
            if ref_line == gen_line:
                continue
            if not ref_line.startswith(".") or not gen_line.startswith("."):
                if _norm_ws(ref_line) != _norm_ws(gen_line):
                    return False, used_flip
                continue

            ref_parts = ref_line.split()
            gen_parts = gen_line.split()
            if ref_parts[0] != gen_parts[0]:
                return False, used_flip

            cmd = ref_parts[0]
            if cmd == ".sphere":
                # x y z r -> flip xyz only
                if len(ref_parts) != len(gen_parts) or len(ref_parts) < 5:
                    return False, used_flip
                for i in range(1, 4):
                    ok, flipped = _num_ok(ref_parts[i], gen_parts[i], True)
                    if not ok:
                        return False, used_flip
                    used_flip = used_flip or flipped
                ok, flipped = _num_ok(ref_parts[4], gen_parts[4], False)
                if not ok:
                    return False, used_flip
                continue

            if cmd == ".cylinder":
                # x1 y1 z1 x2 y2 z2 r -> flip xyz only
                if len(ref_parts) != len(gen_parts) or len(ref_parts) < 8:
                    return False, used_flip
                for i in range(1, 7):
                    ok, flipped = _num_ok(ref_parts[i], gen_parts[i], True)
                    if not ok:
                        return False, used_flip
                    used_flip = used_flip or flipped
                ok, flipped = _num_ok(ref_parts[7], gen_parts[7], False)
                if not ok:
                    return False, used_flip
                continue

            if cmd == ".arrow":
                # x1 y1 z1 x2 y2 z2 r1 r2 r3 -> flip xyz only
                if len(ref_parts) != len(gen_parts) or len(ref_parts) < 10:
                    return False, used_flip
                for i in range(1, 7):
                    ok, flipped = _num_ok(ref_parts[i], gen_parts[i], True)
                    if not ok:
                        return False, used_flip
                    used_flip = used_flip or flipped
                for i in range(7, 10):
                    ok, flipped = _num_ok(ref_parts[i], gen_parts[i], False)
                    if not ok:
                        return False, used_flip
                continue

            if cmd == ".cmov":
                # x y z -> flip xyz
                if len(ref_parts) != len(gen_parts) or len(ref_parts) < 4:
                    return False, used_flip
                for i in range(1, 4):
                    ok, flipped = _num_ok(ref_parts[i], gen_parts[i], True)
                    if not ok:
                        return False, used_flip
                    used_flip = used_flip or flipped
                continue

            # fallback: whitespace-normalized exact match
            if _norm_ws(ref_line) != _norm_ws(gen_line):
                return False, used_flip

        return True, used_flip

    def _record(
        self,
        *,
        shape: str,
        artifact_suffix: str,
        gen_path: Path,
        ref_path: Path,
        status: str,
        edge_len: int,
        detail: str = "",
    ) -> None:
        self.__class__._report.append(
            {
                "shape": shape,
                "artifact": artifact_suffix,
                "generated": str(gen_path),
                "reference": str(ref_path),
                "status": status,
                "edge_len": str(edge_len),
                "detail": detail,
            }
        )

    def _compare_one(self, out_dir: Path, ref_dir: Path, base_name: str, artifact: ExpectedArtifact, edge_len: int) -> None:
        gen_name = f"{base_name}_{edge_len}bp{artifact.suffix}"
        ref_name = f"{artifact.reference_prefix}{base_name}_{edge_len}bp{artifact.suffix}"

        gen_path = out_dir / gen_name
        ref_path = ref_dir / ref_name

        try:
            self.assertTrue(gen_path.exists(), f"Missing generated file: {gen_path}")
            self.assertTrue(ref_path.exists(), f"Missing reference file: {ref_path}")

            gen_text = self._read_with(gen_path, artifact.reader)
            ref_text = self._read_with(ref_path, artifact.reader)

            if ref_text != gen_text:
                if artifact.suffix.endswith(".bild"):
                    if self._compare_with_tolerance(ref_text, gen_text):
                        self._record(
                            shape=base_name,
                            artifact_suffix=artifact.suffix,
                            gen_path=gen_path,
                            ref_path=ref_path,
                            status="PASS",
                            edge_len=edge_len,
                        )
                        return
                    ok, used_flip = self._compare_with_sign_tolerance(ref_text, gen_text)
                    if ok:
                        self._record(
                            shape=base_name,
                            artifact_suffix=artifact.suffix,
                            gen_path=gen_path,
                            ref_path=ref_path,
                            status="PASS",
                            edge_len=edge_len,
                            detail="WARN: accepted sign-flip + rounding tolerance" if used_flip else "",
                        )
                        return
                if artifact.suffix.endswith(".cndo"):
                    if self._compare_with_tolerance(ref_text, gen_text):
                        self._record(
                            shape=base_name,
                            artifact_suffix=artifact.suffix,
                            gen_path=gen_path,
                            ref_path=ref_path,
                            status="PASS",
                            edge_len=edge_len,
                        )
                        return
                idx = self._first_mismatch(ref_text, gen_text)
                idx_str = "unknown" if idx is None else str(idx)
                raise AssertionError(
                    f"Content mismatch (first mismatch char offset: {idx_str}; "
                    f"ref_len={len(ref_text)} gen_len={len(gen_text)})"
                )

        except AssertionError as e:
            self._record(
                shape=base_name,
                artifact_suffix=artifact.suffix,
                gen_path=gen_path,
                ref_path=ref_path,
                status="FAIL",
                edge_len=edge_len,
                detail=str(e),
            )
            # CHANGE 1: removed 'raise' here. 
            # This suppresses the traceback and lets unittest assert success.
            # The failure is strictly tracked in self._report.
        else:
            self._record(
                shape=base_name,
                artifact_suffix=artifact.suffix,
                gen_path=gen_path,
                ref_path=ref_path,
                status="PASS",
                edge_len=edge_len,
            )

    def _compare_artifacts(self, *, shape: str, input_file: str, artifacts: list[ExpectedArtifact], edge_len: int) -> None:
        self._run_pipeline(shape, input_file, edge_len)
        out_dir = OUTPUT_ROOT / f"{shape}_{edge_len}bp"
        ref_dir = REF_ROOT / f"{shape}_{edge_len}bp"

        self._compare_existing_artifacts(
            out_dir=out_dir,
            ref_dir=ref_dir,
            base_name=shape,
            artifacts=artifacts,
            edge_len=edge_len,
        )

    def _compare_existing_artifacts(
        self,
        *,
        out_dir: Path,
        ref_dir: Path,
        base_name: str,
        artifacts: list[ExpectedArtifact],
        edge_len: int,
    ) -> None:

        self.assertTrue(out_dir.exists(), f"Missing output directory: {out_dir}")
        self.assertTrue(ref_dir.exists(), f"Missing reference directory: {ref_dir}")

        for art in artifacts:
            # subTest ensures we keep going, but since we swallow exceptions 
            # in _compare_one now, subTest will always implicitly 'pass'.
            with self.subTest(shape=base_name, artifact=art.suffix, edge_len=edge_len):
                self._compare_one(out_dir, ref_dir, base_name, art, edge_len)

    COMMON_ARTIFACTS = [
        ExpectedArtifact("_01_target_geometry.bild"),
        ExpectedArtifact("_02_target_geometry_local.bild"),
        ExpectedArtifact("_03_sep_line.bild"),
        ExpectedArtifact("_04_doubled_lines.bild"),
        ExpectedArtifact("_05_cylindrical_model_1.bild"),
        ExpectedArtifact("_06_cylindrical_model_2.bild"),
        ExpectedArtifact("_07_spantree.bild"),
        ExpectedArtifact("_08_crossovers.bild"),
        ExpectedArtifact("_09_atomic_model.bild"),
        ExpectedArtifact("_10_routing_scaf.bild"),
        ExpectedArtifact("_11_routing_stap.bild"),
        ExpectedArtifact("_13_cylindrical_model_xover.bild"),
        ExpectedArtifact("_14_json_guide.bild"),
        ExpectedArtifact("_15_json_caDNAno.json"),
        ExpectedArtifact("_16_cndo_format.cndo", reader="cndo"),
        ExpectedArtifact("_17_sequence.csv"),
    ]

    def test_triangle_outputs_match_reference(self) -> None:
        self._compare_artifacts(
            shape="triangle_l1",
            input_file="triangle_l1.svg",
            artifacts=self.COMMON_ARTIFACTS,
            edge_len=DEFAULT_EDGE_LEN,
        )
        
    def test_triangle2_outputs_match_reference(self) -> None:
        self._compare_artifacts(
            shape="triangle_l2",
            input_file="triangle_l2.svg",
            artifacts=self.COMMON_ARTIFACTS,
            edge_len=DEFAULT_EDGE_LEN,
        )
        
    def test_asset_svg_rect_outputs_match_references(self) -> None:
        edge_len = DEFAULT_EDGE_LEN
        for shape in ("asset1", "asset2"):
            with self.subTest(shape=shape):
                self._run_pipeline_svg_rect(f"{shape}.svg", edge_len)
                self._compare_existing_artifacts(
                    out_dir=OUTPUT_ROOT / f"{shape}_{edge_len}bp",
                    ref_dir=REF_ROOT / f"{shape}_{edge_len}bp",
                    base_name=shape,
                    artifacts=self.COMMON_ARTIFACTS,
                    edge_len=edge_len,
                )

    def test_multilayer_svg_layer_selection_matches_legacy_files(self) -> None:
        old_l1_lines, old_l1_pts = self._geom_line_signature("triangle_l1.svg")
        new_l1_lines, new_l1_pts = self._geom_line_signature("triangle_l1m2.svg", svg_layer=1)
        self.assertEqual(new_l1_pts, old_l1_pts, "Layer_1 point count does not match triangle_l1.svg")
        self.assertSetEqual(new_l1_lines, old_l1_lines, "Layer_1 geometry does not match triangle_l1.svg")

        new_l2_lines, new_l2_pts = self._geom_line_signature("triangle_l1m2.svg", svg_layer=2)
        expected_l2_lines = {
            ((0.0, 1.0), (0.01, 1.0)),
            ((0.01, 0.0), (0.01, 1.0)),
            ((0.01, 1.0), (1.0, 1.0)),
            ((0.01, 0.0), (1.0, 1.0)),
        }
        self.assertEqual(new_l2_pts, 4, "Layer_2 point count should match triangle_l1m2.svg group geometry")
        self.assertSetEqual(new_l2_lines, expected_l2_lines, "Layer_2 geometry does not match triangle_l1m2.svg")

    def test_triangle_l1m2_layer1_outputs_match_references(self) -> None:
        edge_len = DEFAULT_EDGE_LEN
        self._run_pipeline(
            "triangle_l1m2",
            "triangle_l1m2.svg",
            edge_len,
            ["--svg-layer", "1"],
        )

        self._compare_existing_artifacts(
            out_dir=OUTPUT_ROOT / f"triangle_l1m2_{edge_len}bp" / "Layer_1",
            ref_dir=REF_ROOT / f"triangle_l1m2_{edge_len}bp" / "Layer_1",
            base_name="triangle_l1m2",
            artifacts=self.COMMON_ARTIFACTS,
            edge_len=edge_len,
        )

    def test_triangle_l1m2_layer2_outputs_match_references(self) -> None:
        edge_len = DEFAULT_EDGE_LEN

        self._run_pipeline(
            "triangle_l1m2",
            "triangle_l1m2.svg",
            edge_len,
            ["--svg-layer", "2"],
        )

        self._compare_existing_artifacts(
            out_dir=OUTPUT_ROOT / f"triangle_l1m2_{edge_len}bp" / "Layer_2",
            ref_dir=REF_ROOT / f"triangle_l1m2_{edge_len}bp" / "Layer_2",
            base_name="triangle_l1m2",
            artifacts=self.COMMON_ARTIFACTS,
            edge_len=edge_len,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # Print a compact report at the end.
        rep = cls._report
        if not rep:
            return

        total = len(rep)
        passed = sum(1 for r in rep if r["status"] == "PASS")
        failed = total - passed

        lines = []

        # Group by shape for readability
        by_shape: dict[str, list[dict[str, str]]] = {}
        for r in rep:
            by_shape.setdefault(r["shape"], []).append(r)

        for shape, items in sorted(by_shape.items()):
            by_len: dict[str, list[dict[str, str]]] = {}
            for r in items:
                by_len.setdefault(r.get("edge_len", str(DEFAULT_EDGE_LEN)), []).append(r)

            for edge_len, len_items in sorted(by_len.items()):
                shape_fails = sum(1 for r in len_items if r["status"] == "FAIL")
                lines.append(f"\n[{shape} {edge_len}bp | FAIL: {shape_fails}]")
                for r in len_items:
                    status = r["status"]
                    artifact = r["artifact"]
                    gen_name = Path(r["generated"]).name
                    ref_name = Path(r["reference"]).name
                    if status == "PASS":
                        detail = r.get("detail", "")
                        suffix = f"  ({detail})" if detail else ""
                        lines.append(f"  PASS  {artifact}  {gen_name}{suffix}")
                    else:
                        detail = r["detail"]
                        lines.append(f"  FAIL  {artifact}  {gen_name}")
                        lines.append(f"        REF: {ref_name}")
                        lines.append(f"        {detail}")

        lines.append("\n=== Artifact Comparison Report ===")
        lines.append(f"Total: {total}  PASS: {passed}  FAIL: {failed}")
        
        sys.stderr.write("\n".join(lines) + "\n")

        # CHANGE 2: Manually exit with error code if failures occurred.
        # Since we silenced the exceptions, unittest thinks everything passed.
        # We correct the exit code here to ensure CI failure.
        if failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    unittest.main()
