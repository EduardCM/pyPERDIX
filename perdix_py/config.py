from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import para


_PARA_DEFAULTS = {
    name: getattr(para, name)
    for name in dir(para)
    if (name.startswith("para_") or name in {"p_redir", "p_detail"})
    and not name.startswith("__")
}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value!r}")
    return bool(value)


@dataclass
class Config:
    svg_scale: float = 1.0
    svg_layer_snap_compat: bool = False
    output_dir: str = ""
    sec_type: str = "honeycomb"
    sec_n_row: int = 1
    sec_n_col: int = 2
    sec_ref_row: int = 1
    sec_ref_minC: int = 1
    sec_ref_maxC: int = 2
    edge_len: int = 42
    edge_ref: int = 0
    para_cut_stap_method: str = "max"
    para_scaf_seq: str = "user"
    para_set_start_scaf: int = 1
    para_method_sort: str = "quick"
    name_prob: str = ""
    para_scaf_seq_explicit: bool = False


def reset_para_defaults() -> None:
    for key, value in _PARA_DEFAULTS.items():
        setattr(para, key, value)


def load_config(path: str | None) -> Config:
    reset_para_defaults()
    if not path:
        return Config()
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))

    cfg = Config()
    cfg.svg_scale = float(data.get("svg_scale", cfg.svg_scale))
    cfg.svg_layer_snap_compat = _as_bool(data.get("svg_layer_snap_compat", cfg.svg_layer_snap_compat))
    cfg.output_dir = str(data.get("output_dir", cfg.output_dir))
    cfg.sec_type = str(data.get("sec_type", cfg.sec_type))
    cfg.sec_n_row = int(data.get("sec_n_row", cfg.sec_n_row))
    cfg.sec_ref_row = int(data.get("sec_ref_row", cfg.sec_ref_row))
    cfg.sec_ref_minC = int(data.get("sec_ref_minC", cfg.sec_ref_minC))
    cfg.sec_ref_maxC = int(data.get("sec_ref_maxC", cfg.sec_ref_maxC))
    cfg.sec_n_col = int(data.get("sec_n_col", cfg.sec_n_col))
    cfg.edge_len = int(data.get("edge_len", cfg.edge_len))
    cfg.edge_ref = int(data.get("edge_ref", cfg.edge_ref))
    cfg.para_cut_stap_method = str(data.get("para_cut_stap_method", cfg.para_cut_stap_method))
    cfg.para_scaf_seq_explicit = "para_scaf_seq" in data
    cfg.para_scaf_seq = str(data.get("para_scaf_seq", cfg.para_scaf_seq))
    cfg.para_set_start_scaf = int(data.get("para_set_start_scaf", cfg.para_set_start_scaf))
    cfg.para_method_sort = str(data.get("para_method_sort", cfg.para_method_sort))
    cfg.name_prob = str(data.get("name_prob", cfg.name_prob))

    para_overrides = {
        "para_write_102": _as_bool,
        "para_write_302": _as_bool,
        "para_write_303": _as_bool,
        "para_write_504": _as_bool,
        "para_write_502": _as_bool,
        "para_write_606": _as_bool,
        "para_write_607": _as_bool,
        "para_write_609": _as_bool,
        "para_write_701": _as_bool,
        "para_write_702": _as_bool,
        "para_write_703": _as_bool,
        "para_write_705": _as_bool,
        "para_chimera_102_info": _as_bool,
        "para_chimera_axis": _as_bool,
        "para_print_summary": _as_bool,
        "para_debug_mesh": _as_bool,
        "para_debug_frame_mode": str,
        "para_validate_pipeline": _as_bool,
        "para_validate_pipeline_strict": _as_bool,
        "para_cut_stap_method": str,
        "para_scaf_seq": str,
        "para_set_start_scaf": int,
        "para_method_sort": str,
        "para_dist_bp": float,
        "para_dist_pp": float,
        "para_rad_helix": float,
    }
    for key, coerce in para_overrides.items():
        if key in data:
            setattr(para, key, coerce(data[key]))

    return cfg
