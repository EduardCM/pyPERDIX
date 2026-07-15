"""Parameters translated from Para.f90."""

p_redir: int = 0
p_detail: bool = False

para_scaf_seq: str = ""

para_vertex_design: str = "mitered"
para_vertex_angle: str = "opt"
para_vertex_crash: str = "const"
para_const_edge_mesh: str = "off"
para_sticky_self: str = "off"
para_unpaired_scaf: str = "on"

para_dist_pp: float = 0.42
para_dist_bp: float = 0.34
para_rad_helix: float = 1.0
para_gap_helix: float = 0.25
para_ang_minor: float = 150.0
para_ang_correct: float = 0.0
para_n_base_tn: int = -1
para_start_bp_ID: int = -1

para_weight_edge: str = "on"
para_method_MST: str = "prim"
para_method_sort: str = "quick"
para_adjacent_list: str = "off"
para_all_spanning: str = "off"

para_cut_stap_method: str = "max"
para_set_stap_sxover: str = "off"
para_output_design: str = "arrow"
para_set_xover_scaf: str = "split"

para_gap_xover_two_scaf: int = 3
para_gap_xover_bound_scaf: int = 7
para_gap_xover_bound_stap: int = 6
para_gap_xover_two: int = 6
para_gap_xover_nick1: int = 10
para_gap_xover_nick: int = 3

para_max_cut_scaf: int = 0
para_min_cut_stap: int = 20
para_mid_cut_stap: int = 40
para_max_cut_stap: int = 60
para_set_start_scaf: int = 1

# Output/control flags (Fortran-style numeric naming kept for compatibility).
# Flag meaning in this Python port:
# - para_write_102: write 01_target_geometry.bild
# - para_write_302: write 02_target_geometry_local.bild
# - para_write_303: write 03_sep_line.bild
# - para_write_504: write 04_doubled_lines.bild
# - para_write_502: write 05_cylindrical_model_1.bild and 06_cylindrical_model_2.bild
# - para_write_606: write 07_spantree.bild
# - para_write_607: write 08_crossovers.bild
# - para_write_702: write 09_atomic_model.bild
# - para_write_703: write 10_routing_scaf.bild and 11_routing_stap.bild
# - para_write_705: write 12_routing_all.bild
# - para_write_609: write 13_cylindrical_model_xover.bild
# - para_write_701: write 17_sequence.csv
# Some legacy flags are retained but are currently unused/partial in this port.
para_tecplot: bool = False
para_write_101: bool = False
para_write_102: bool = True
para_write_103: bool = False
para_write_104: bool = False
para_write_301: bool = False
para_write_302: bool = True
para_write_303: bool = True
para_write_401: bool = False
para_write_501: bool = False
para_write_502: bool = True
para_write_503: bool = False
para_write_504: bool = True
para_write_505: bool = True
para_write_601_1: bool = False
para_write_601_2: bool = False
para_write_601_3: bool = False
para_write_601_4: bool = False
para_write_601_5: bool = False
para_write_606: bool = True
para_write_607: bool = True
para_write_608: bool = False
para_write_609: bool = True
para_write_610: bool = False
para_write_701: bool = True
para_write_711: bool = False
para_write_702: bool = True
para_write_703: bool = True
para_write_705: bool = True
para_write_706: bool = False
para_write_710: bool = False
para_write_801: bool = False
para_write_802: bool = False
para_write_803: bool = True
para_write_804: bool = False
para_write_805: bool = False
para_write_808: bool = False

para_chimera_axis: bool = False
para_chimera_102_info: bool = True
para_chimera_301_info: bool = False
para_chimera_302_info: bool = True
para_chimera_303_info: bool = True
para_chimera_401_info: bool = False
para_chimera_502_ori: bool = False
para_chimera_503_mod: bool = False
para_chimera_504_info: bool = True
para_chimera_601_dir: bool = False
para_chimera_609_cyl: bool = False
para_chimera_609_dir: bool = False

# Console summary output
para_print_summary: bool = False
para_debug_mesh: bool = False
para_debug_frame_mode: str = "input_absolute"
para_validate_pipeline: bool = False
para_validate_pipeline_strict: bool = True
