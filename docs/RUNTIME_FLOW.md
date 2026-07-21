# PERDIX Python Runtime Flow

This document describes the current execution flow of the Python implementation in `perdix_py`. It is a runtime and package-structure guide for the code that exists today, not a porting plan.

## Entry Point

Primary CLI entry:

```bash
python -m perdix_py.main <input.svg> [--config path/to/config.json]
```

The CLI in `perdix_py/main.py` supports these notable switches:

- `--config`: load JSON config overrides
- `--edge-len`: override `edge_len` without editing config
- `--summary`: print runtime summary after output generation
- `--debug-mesh`: dump JSON snapshots before and after routing
- `--frame-mode legacy|svg-rect`: choose SVG coordinate normalization mode
- `--svg-layer`: run only one `Layer_N` group from a multi-layer SVG
- `--shared-crossover-indices LAYER`: route one layer normally and map its compatible scaffold crossover indices to the other layers
- `--shared-start smallest|exterior`: align the scaffold nick at an exact common node selected by one of two policies

## Top-Level Runtime Sequence

For a normal single-run invocation, `perdix_py.main._run_pipeline()` executes:

1. `input_initialize`
2. `enforce_pipeline_validation("input_initialize", ...)`
3. `modgeo_modification`
4. `enforce_pipeline_validation("modgeo_modification", ...)`
5. `section_generation`
6. `enforce_pipeline_validation("section_generation", ...)`
7. `basepair_discretize`
8. `enforce_pipeline_validation("basepair_discretize", ...)`
9. optional `dump_mesh_snapshot(..., "pre_route", ...)`
10. `route_generation`
11. `enforce_pipeline_validation("route_generation", ...)`
12. optional `dump_mesh_snapshot(..., "post_route", ...)`
13. `seqdesign_design`
14. `enforce_pipeline_validation("seqdesign_design", ...)`
15. optional shared scaffold nick relocation and sequence reassignment
16. `output_generation`
17. optional `print_summary`

The four long-running computational stages are still exposed through the legacy-style module names:

- `perdix_py.modgeo`
- `perdix_py.basepair`
- `perdix_py.route`
- `perdix_py.seqdesign`

Internally, some of those modules now delegate to split implementation files such as:

- `perdix_py/basepair_edge_ops.py`
- `perdix_py/basepair_mesh.py`
- `perdix_py/basepair_junction.py`
- `perdix_py/basepair_ghost.py`
- `perdix_py/basepair_output.py`
- `perdix_py/input_geom.py`
- `perdix_py/input_svg.py`
- `perdix_py/modgeo_neighbors.py`
- `perdix_py/modgeo_scale.py`
- `perdix_py/validation_support.py`

`perdix_py/output.py`, `perdix_py/route.py`, and `perdix_py/seqdesign.py` now import their implementation modules directly.

## Layered SVG Execution

`perdix_py.main.main()` detects SVG layer groups with `list_svg_layers()`.

If the input SVG contains `Layer_N` groups and `--svg-layer` is not provided, the CLI runs the full pipeline once per layer through `_run_layered_pipelines()`. In that mode it first computes shared alignment state:

- `compute_svg_shared_frame(...)`
- `compute_svg_shared_route_start(...)` internally when either shared-routing option is active

Each layer then runs through the normal pipeline with:

- `svg_import_layer=<layer index>`
- `svg_output_subdir=Layer_<N>`

The shared boundary route start is an internal normalization step. It is not exposed as a CLI option; users select either shared crossover indices, a shared scaffold start, or both.

### Shared crossover indices

With `--shared-crossover-indices LAYER`, the named layer runs first. Its scaffold crossovers are exported using stable keys composed of original edge geometry, base-pair index, and section pair. Compatible crossovers are then forced onto the other layers. Forced crossover fields are followed by scaffold traversal reconstruction so the final `up/dn` topology remains complete.

### Shared scaffold starts

`--shared-start` runs after sequence design and supports two policies:

- `smallest`: select the layer with the lowest total scaffold nucleotide count and reuse its native scaffold start
- `exterior`: intersect edges that have exactly one adjacent face in every layer, intersect strict safe nick sites on those edges, choose the longest contiguous common run, and use its center

Both policies require an exact edge/base-pair/section node present in every layer. The relocation reconnects the old scaffold nick and cuts the new scaffold bond without modifying staple `up/dn`, staple crossovers, or reciprocal `across` pairing. Scaffold strand order and sequence assignment are then rebuilt before output files are rewritten.

Examples:

```bash
python -m perdix_py.main input/levels_Lm.svg \
  --config perdix_config.json \
  --shared-crossover-indices 1 \
  --shared-start smallest
```

```bash
python -m perdix_py.main input/levels_Lm.svg \
  --config perdix_config.json \
  --shared-crossover-indices 1 \
  --shared-start exterior
```

That means a multi-layer SVG writes into:

```text
output/<name>_<edge_len>bp/Layer_1/
output/<name>_<edge_len>bp/Layer_2/
...
```

If `--svg-layer N` is provided, only that layer is imported and only one output directory is written.

## Input and Configuration

### Input handling

`input_initialize()` in `perdix_py/input.py` is responsible for:

- resolving the SVG path directly or under `input/`
- rejecting non-SVG inputs
- loading config via `perdix_py.config.load_config()`
- importing SVG geometry with `import_svg_to_geom()`
- polygonizing linework if faces are missing
- converting faces back to line representations used downstream
- applying optional per-layer compatibility normalization
- populating section-grid defaults
- scaling and rounding geometry
- creating the output directory
- optionally writing `01_target_geometry.bild`

### Config model

`perdix_py/config.py` exposes a `Config` dataclass. Important runtime fields are:

- `svg_scale`
- `svg_layer_snap_compat`
- `output_dir`
- `sec_type`
- `sec_n_row`
- `sec_n_col`
- `sec_ref_row`
- `sec_ref_minC`
- `sec_ref_maxC`
- `edge_len`
- `edge_ref`
- `para_cut_stap_method`
- `para_scaf_seq`
- `para_set_start_scaf`
- `para_method_sort`
- `name_prob`

`load_config()` also resets and reapplies `para` module globals on every run. That is how output toggles, validation toggles, scaffold sequence mode, and geometric constants are controlled at runtime.

Built-in scaffold sequence data is loaded from packaged resources under `perdix_py/resources/`. A local `seq.txt` in the working directory still acts as an explicit user override when present.

The test config at `tests/perdix_test_config.json` is a minimal example:

```json
{
  "para_cut_stap_method": "max",
  "para_set_start_scaf": 60,
  "para_scaf_seq": "m13",
  "output_dir": "tests/output"
}
```

## Validation and Debugging

Pipeline validation is first-class in the current codebase.

- `perdix_py.validation.enforce_pipeline_validation()` runs after each major stage.
- Validation can be disabled with `para.para_validate_pipeline = False`.
- High-severity issues raise `ValidationError` when `para.para_validate_pipeline_strict` is enabled.
- Validation coverage spans `ProbType`, `GeomType`, `MeshType`, and `DNAType`.

Debug helpers:

- `--debug-mesh` or `para.para_debug_mesh` writes mesh snapshots through `perdix_py.debug_dump.dump_mesh_snapshot()`
- `--summary` or `para.para_print_summary` prints a runtime summary after generation

## Outputs

Artifact naming goes through `perdix_py.naming.bild_path()` and uses this pattern:

```text
<input_stem>_<edge_len>bp_<artifact>.bild
```

The output root is:

- `output/<input_stem>_<edge_len>bp/` by default
- or `<output_dir>/<input_stem>_<edge_len>bp/` if `output_dir` is set in config

For layered runs, `Layer_<N>/` is appended under that directory.

Common generated artifacts include:

- `01_target_geometry.bild`
- `02_target_geometry_local.bild`
- `03_sep_line.bild`
- `04_doubled_lines.bild`
- `05_cylindrical_model_1.bild`
- `06_cylindrical_model_2.bild`
- `07_spantree.bild`
- `08_crossovers.bild`
- `09_atomic_model.bild`
- `09_atomic_model_scaf.bild`
- `09_atomic_model_stap.bild`
- `09_atomic_model_all.bild`
- `10_routing_scaf.bild`
- `11_routing_stap.bild`
- `13_cylindrical_model_xover.bild`
- `14_json_guide.bild`
- `15_json_caDNAno.json`
- `16_cndo_format.cndo`
- `17_sequence.csv`

Some outputs are conditional on `para.para_write_*` flags. The active golden-reference coverage in `tests/test_bild.py` currently checks the core BILD, JSON, CNDO, and sequence outputs for selected fixtures.

## Core Data Flow

The main runtime data objects are:

- `ProbType`: problem naming, edge length, output paths, selection anchors
- `GeomType`: imported, modified, cross-section, face, section, and junction geometry
- `MeshType`: discretized nodes and elements
- `DNAType`: scaffold/staple topology and per-base sequence assignments

`dna.top` is the canonical topology after sequence design. Final scaffold/staple strand order, nicks, crossovers, pairing, and sequence assignments are read from `dna.top` and `dna.strand`. Legacy `base_scaf` and `base_stap` connectivity is synchronized from that state before output generation. Consequently, `09_atomic_model.bild`, its scaffold/staple split files, and `09_atomic_model_all.bild` emit identical directed backbone segments for the same strands.

Conceptually:

```text
SVG input
  -> imported geometry
  -> modified geometry
  -> section geometry
  -> discretized mesh
  -> routed DNA topology
  -> sequence assignment
  -> output artifacts
```

## Package Map

Useful files when navigating the current codebase:

- `perdix_py/main.py`: CLI orchestration
- `perdix_py/input.py`: run setup and geometry preparation
- `perdix_py/input_geom.py`: geometry cleanup, polygonization, scaling helpers
- `perdix_py/input_svg.py`: layered SVG frame and route-start coordination
- `perdix_py/modgeo.py`: modified-geometry generation
- `perdix_py/section.py`: section/cross-section generation
- `perdix_py/basepair.py`: base-pair discretization entrypoint
- `perdix_py/route.py`: routing entrypoint
- `perdix_py/seqdesign.py`: sequence assignment entrypoint
- `perdix_py/output.py`: artifact emission entrypoint
- `perdix_py/validation.py`: stage validation
- `tests/test_stability_fixes.py`: regression coverage for stability-sensitive behavior
- `tests/test_validation.py`: validation-specific tests
- `tests/test_bild.py`: active golden-reference artifact tests
