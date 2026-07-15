from __future__ import annotations

import json
import re
from pathlib import Path

from . import para
from .data_prob import ProbType
from .data_geom import GeomType
from .data_mesh import MeshType
from .data_dna import DNAType
from .input import input_initialize
from .modgeo import modgeo_modification
from .section import section_generation
from .basepair import basepair_discretize
from .route import route_generation
from .seqdesign import seqdesign_design
from .output import output_generation, print_summary
from .debug_dump import dump_mesh_snapshot
from .config import load_config
from .exam_perdix import select_exam_prob
from .validation import enforce_pipeline_validation


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        resp = input(f"{text}{suffix}: ").strip()
        if resp == "" and default is not None:
            return default
        if resp != "":
            return resp


def _prompt_int(text: str, default: int | None = None) -> int:
    while True:
        resp = _prompt(text, str(default) if default is not None else None)
        if re.fullmatch(r"[+-]?\d+", resp):
            return int(resp)
        print(f"  Enter a valid integer, got: {resp!r}.")


def _prompt_float(text: str, default: float | None = None) -> float:
    while True:
        resp = _prompt(text, str(default) if default is not None else None)
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", resp):
            return float(resp)
        print(f"  Enter a valid number, got: {resp!r}.")


def _prompt_bool(text: str, default: bool) -> bool:
    d = "y" if default else "n"
    while True:
        resp = _prompt(text + " (y/n)", d).lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("  Enter y or n.")


def _print_banner() -> None:
    print()
    print("  +===================================================================+")
    print("  |                                                                   |")
    print("  |          PERDIX for the 2D DNA wireframe with DX edges            |")
    print("  |                                        by Hyungmin Jun            |")
    print("  |                                                                   |")
    print("  +===================================================================+")
    print()


def _print_prob_menu() -> None:
    print("  A. First input - Pre-defined 2D target geometries")
    print("  =================================================")
    print()
    print("   [ Triangular-mesh objects ]")
    print("      1. Square,               2. Honeycomb")
    print("      3. Circle,               4. Wheel,                    5. Ellipse")
    print()
    print("   [ Quadrilateral-mesh objects ]")
    print("      6. Rhombic Tiling,       7. Quarter Circle")
    print("      8. Cross,                9. Arrowhead,               10. Annulus")
    print()
    print("   [ N-polygon-mesh objects ]")
    print("     11. Cairo Penta Tiling,  12. Lotus")
    print("     13. Hexagonal Tiling,    14. Prismatic Penta Tiling,  15. Hepta Penta Tiling")
    print()
    print("   [ Variable vertex-number, edge-length, and internal mesh ]")
    print("     16. 4-Sided Polygon,     17. 5-Sided Polygon,         18. 6-Sided Polygon")
    print("     19. L-Shape [42-bp],     20. L-Shape [63-bp],         21. L-Shape [84-bp]")
    print("     22. Curved Arm [Quad],   23. Curved Arm [Tri],        24. Curved Arm [Mixed]")
    print()
    print("  Select or type geometry file (*.svg) [Enter]")


def _print_edge_len_menu() -> None:
    print()
    print("  B. Second input - Pre-defined minimum edge lengths")
    print("  ==================================================")
    print()
    print("  * 1.  42 bp =  4 turn * 10.5 bp/turn ->  42 bp * 0.34nm/bp = 14.28nm")
    print("    2.  52 bp =  5 turn * 10.5 bp/turn ->  52 bp * 0.34nm/bp = 17.85nm")
    print("  * 3.  63 bp =  6 turn * 10.5 bp/turn ->  63 bp * 0.34nm/bp = 21.42nm")
    print("    4.  73 bp =  7 turn * 10.5 bp/turn ->  73 bp * 0.34nm/bp = 24.99nm")
    print("  * 5.  84 bp =  8 turn * 10.5 bp/turn ->  84 bp * 0.34nm/bp = 28.56nm")
    print("    6.  94 bp =  9 turn * 10.5 bp/turn ->  94 bp * 0.34nm/bp = 32.13nm")
    print("  * 7. 105 bp = 10 turn * 10.5 bp/turn -> 105 bp * 0.34nm/bp = 35.70nm")
    print("    8. 115 bp = 11 turn * 10.5 bp/turn -> 115 bp * 0.34nm/bp = 39.27nm")
    print("  * 9. 126 bp = 12 turn * 10.5 bp/turn -> 126 bp * 0.34nm/bp = 42.84nm")
    print()
    print("    0. If needed to choose the specific edge to assign the length")
    print()


def _edge_len_from_sel(sel: int) -> int:
    mapping = {
        1: 42,
        2: 52,
        3: 63,
        4: 73,
        5: 84,
        6: 94,
        7: 105,
        8: 115,
        9: 126,
    }
    if sel in mapping:
        return mapping[sel]
    if 10 <= sel <= 37:
        raise ValueError("The minimum edge length should be over 38-bp.")
    return sel


def _select_geometry_source() -> tuple[Path | None, int | None]:
    svg_path: Path | None = None
    sel_prob: int | None = None
    while svg_path is None:
        resp = _prompt("  Geometry selection or file")
        if resp.isdigit():
            sel = int(resp)
            if sel <= 0:
                print("  Exit.")
                return
            sel_prob = sel
            break
        path = Path(resp)
        if not path.exists():
            print("  The file does not exist.")
            continue
        ext = path.suffix.lower().lstrip(".")
        if ext != "svg":
            print("  Only .svg is supported in the Python port.")
            continue
        svg_path = path
    return svg_path, sel_prob


def _prompt_edge_config() -> tuple[int, int]:
    _print_edge_len_menu()
    sel_edge_len = _prompt_int("  Select the number or type the min. edge length", 1)
    if sel_edge_len < 0:
        raise SystemExit(0)
    edge_ref = 0
    edge_len = 42
    if sel_edge_len == 0:
        edge_ref = _prompt_int("  Type the specific edge ID", 0)
        edge_len = _prompt_int("  Type the minimum edge length for the edge ID", 42)
    else:
        edge_len = _edge_len_from_sel(sel_edge_len)
    return edge_ref, edge_len


def _prompt_config_values(edge_len: int, edge_ref: int) -> tuple[Path, dict[str, object]]:
    svg_scale = _prompt_float("  SVG scale", 100.0)
    output_dir = _prompt("  Output directory (empty = current dir)", "")
    sec_type = _prompt("  Section type (square/honeycomb)", "square").lower()
    if sec_type not in {"square", "honeycomb"}:
        raise ValueError("Section type must be 'square' or 'honeycomb'")
    sec_n_row = _prompt_int("  Section rows", 1)
    sec_n_col = _prompt_int("  Section cols", 2)
    sec_ref_row = _prompt_int("  Section ref row", 1)
    sec_ref_minC = _prompt_int("  Section ref min col", 1)
    sec_ref_maxC = _prompt_int("  Section ref max col", max(sec_ref_minC, sec_n_col))

    para_dist_bp = _prompt_float("  Basepair distance (para_dist_bp)", para.para_dist_bp)
    para_rad_helix = _prompt_float("  Helix radius (para_rad_helix)", para.para_rad_helix)

    print()
    print("  Output flags (default shown).")
    flags = {
        "para_write_102": _prompt_bool("  Write _01_target_geometry.bild", para.para_write_102),
        "para_write_302": _prompt_bool("  Write _02_geometry_local.bild", para.para_write_302),
        "para_write_303": _prompt_bool("  Write _03_separate_lines.bild", para.para_write_303),
        "para_write_504": _prompt_bool("  Write _04_doubled_lines.bild", para.para_write_504),
        "para_write_502": _prompt_bool("  Write _05/_06_cylinder*.bild", para.para_write_502),
        "para_write_606": _prompt_bool("  Write _07_spantree.bild", para.para_write_606),
        "para_write_607": _prompt_bool("  Write _08_crossovers.bild", para.para_write_607),
        "para_write_609": _prompt_bool("  Write _13_cylindrical_model_xover.bild", para.para_write_609),
        "para_write_702": _prompt_bool("  Write _09_atomic_model.bild", para.para_write_702),
        "para_write_703": _prompt_bool("  Write _10/_11_routing*.bild", para.para_write_703),
        "para_write_705": _prompt_bool("  Write _12_routing_all.bild", para.para_write_705),
        "para_write_701": _prompt_bool("  Write TXT_Sequence.txt", para.para_write_701),
        "para_chimera_102_info": _prompt_bool("  Show IDs on _01_target_geometry", para.para_chimera_102_info),
        "para_chimera_axis": _prompt_bool("  Draw global axis", para.para_chimera_axis),
        "para_debug_mesh": _prompt_bool("  Dump mesh before/after routing", para.para_debug_mesh),
        "para_validate_pipeline": _prompt_bool("  Validate pipeline state", para.para_validate_pipeline),
        "para_validate_pipeline_strict": _prompt_bool(
            "  Fail on validation errors",
            para.para_validate_pipeline_strict,
        ),
    }

    cfg_path = Path(_prompt("  Config file path", "perdix_config.json"))
    cfg = {
        "svg_scale": svg_scale,
        "output_dir": output_dir,
        "sec_type": sec_type,
        "sec_n_row": sec_n_row,
        "sec_n_col": sec_n_col,
        "sec_ref_row": sec_ref_row,
        "sec_ref_minC": sec_ref_minC,
        "sec_ref_maxC": sec_ref_maxC,
        "edge_len": edge_len,
        "edge_ref": edge_ref,
        "para_dist_bp": para_dist_bp,
        "para_rad_helix": para_rad_helix,
    }
    cfg.update(flags)
    return cfg_path, cfg


def _configure_predefined_geometry(prob: ProbType, geom: GeomType, cfg_path: Path) -> None:
    cfg_loaded = load_config(str(cfg_path))
    out_base = Path(cfg_loaded.output_dir) if cfg_loaded.output_dir else (Path.cwd() / "output")
    out_dir = out_base / f"{prob.name_file}_{prob.n_edge_len}bp"
    out_dir.mkdir(parents=True, exist_ok=True)
    prob.path_work = str(out_dir)
    prob.path_input = str(Path.cwd() / "input") + "/"
    prob.type_file = "primitive"
    geom.sec.types = cfg_loaded.sec_type
    geom.sec.n_row = cfg_loaded.sec_n_row
    geom.sec.n_col = cfg_loaded.sec_n_col
    geom.sec.minR = 1
    geom.sec.maxR = cfg_loaded.sec_n_row
    geom.sec.minC = 1
    geom.sec.maxC = cfg_loaded.sec_n_col
    geom.sec.ref_row = cfg_loaded.sec_ref_row
    geom.sec.ref_minC = cfg_loaded.sec_ref_minC
    geom.sec.ref_maxC = cfg_loaded.sec_ref_maxC
    if geom.sec.id:
        return
    geom.n_sec = cfg_loaded.sec_n_row * cfg_loaded.sec_n_col
    geom.sec.id = list(range(geom.n_sec))
    geom.sec.posR = []
    geom.sec.posC = []
    for r in range(1, cfg_loaded.sec_n_row + 1):
        for c in range(1, cfg_loaded.sec_n_col + 1):
            geom.sec.posR.append(r)
            geom.sec.posC.append(c)
    geom.sec.conn = [-1 for _ in range(geom.n_sec)]


def _run_pipeline(
    sel_prob: int | None,
    svg_path: Path | None,
    cfg_path: Path,
    edge_len: int,
) -> None:
    print("  Running pipeline...")
    prob = ProbType()
    geom = GeomType()
    mesh = MeshType()
    dna = DNAType()
    if sel_prob is not None:
        select_exam_prob(sel_prob, prob, geom)
        prob.n_edge_len = edge_len
        _configure_predefined_geometry(prob, geom, cfg_path)
    else:
        input_initialize(prob, geom, svg_path=str(svg_path), config_path=str(cfg_path))
    enforce_pipeline_validation("input_initialize", prob, geom, mesh, dna)
    modgeo_modification(prob, geom)
    enforce_pipeline_validation("modgeo_modification", prob, geom, mesh, dna)
    section_generation(prob, geom)
    enforce_pipeline_validation("section_generation", prob, geom, mesh, dna)
    basepair_discretize(prob, geom, mesh)
    enforce_pipeline_validation("basepair_discretize", prob, geom, mesh, dna)
    if para.para_debug_mesh:
        dump_mesh_snapshot(prob, geom, mesh, "pre_route", dna)
    route_generation(prob, geom, mesh, dna)
    enforce_pipeline_validation("route_generation", prob, geom, mesh, dna)
    if para.para_debug_mesh:
        dump_mesh_snapshot(prob, geom, mesh, "post_route", dna)
    seqdesign_design(prob, geom, mesh, dna)
    enforce_pipeline_validation("seqdesign_design", prob, geom, mesh, dna)
    output_generation(prob, geom, mesh, dna)
    if para.para_print_summary:
        print_summary(prob, geom, mesh, dna)
    print("  Done.")


def run() -> None:
    _print_banner()
    _print_prob_menu()
    svg_path, sel_prob = _select_geometry_source()
    try:
        edge_ref, edge_len = _prompt_edge_config()
    except SystemExit:
        return
    cfg_path, cfg = _prompt_config_values(edge_len, edge_ref)

    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"  Wrote config: {cfg_path}")
    _run_pipeline(sel_prob, svg_path, cfg_path, edge_len)


if __name__ == "__main__":
    run()
