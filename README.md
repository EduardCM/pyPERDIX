# PERDIX Python

Python implementation of the PERDIX DNA origami pipeline.

The current codebase is centered on the `perdix_py` package and runs from SVG input geometry to routed and sequenced output artifacts such as BILD, caDNAno JSON, CNDO, and sequence CSV files.

## What It Does

- imports SVG geometry from `input/`
- builds modified and section geometry
- discretizes the structure into base-pair mesh data
- generates scaffold and staple routing
- assigns scaffold and staple sequences
- writes visualization and downstream export artifacts

## Run

Primary entrypoint:

```bash
python -m perdix_py.main <input.svg> [--config path/to/config.json]
```

Example:

```bash
python -m perdix_py.main input/triangle_l1.svg --config perdix_config.json --summary
```

Useful CLI flags:

- `--config`: load JSON config overrides
- `--edge-len`: override edge length in base pairs
- `--summary`: print a console summary after the run
- `--debug-mesh`: dump mesh snapshots before and after routing
- `--frame-mode legacy|svg-rect`: choose SVG coordinate handling mode
- `--svg-layer`: run only one `Layer_N` group from a multi-layer SVG
- `--route-start default|shared-boundary`: coordinate route-start selection across layered runs
- `--shared-crossover-indices LAYER`: use one layer's scaffold crossover indices on all compatible layers
- `--shared-start smallest|exterior`: align scaffold starts at an exact node shared by every layer

### Layered routing alignment

Use `smallest` to align every scaffold to the native start of the layer with the lowest total scaffold nucleotide count:

```bash
python -m perdix_py.main input/levels_Lm.svg \
  --config perdix_config.json \
  --shared-crossover-indices 1 \
  --shared-start smallest
```

Use `exterior` to choose a safe common position on an edge that is topologically exterior in every layer:

```bash
python -m perdix_py.main input/levels_Lm.svg \
  --config perdix_config.json \
  --shared-crossover-indices 1 \
  --shared-start exterior
```

`--shared-crossover-indices 1` routes `Layer_1` normally and maps compatible scaffold crossovers to the other layers by shared edge geometry, base-pair index, and section pair.

Both shared-start modes require a multi-layer SVG and cannot be combined with `--svg-layer`. Start alignment moves only the scaffold nick. Staple connectivity and scaffold-staple pairing remain attached to the same physical nucleotides, and scaffold/staple sequences are reassigned from the relocated start.

## Install From GitHub

This project can be installed directly from a tagged GitHub revision without publishing to PyPI.

Install from a tag:

```bash
pip install git+https://github.com/EduardCM/pyPERDIX.git@v0.1.0
```

Install from a specific commit:

```bash
pip install git+https://github.com/EduardCM/pyPERDIX.git@6418888
```

After installation, the console entry point is:

```bash
perdix --help
```

Recommended release workflow:

1. Commit and push the release-ready package state.
2. Create a git tag such as `v0.1.0`.
3. Push the tag to GitHub.
4. Ask users to install from that tag, not from a moving branch.

## Inputs and Outputs

Inputs are SVG files, typically stored under `input/`.

By default, outputs are written under:

```text
output/<input_stem>_<edge_len>bp/
```

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

For multi-layer SVGs, the runtime can emit separate `Layer_<N>/` subdirectories under the output root.

After sequence design, `dna.top` is the canonical final topology. The split scaffold, staple, and combined atomic-model BILD files follow the same final strand ordering and connectivity as `09_atomic_model.bild`.

## Configuration

Runtime configuration is loaded from JSON through `perdix_py.config.load_config()`.

Common fields:

- `svg_scale`
- `output_dir`
- `sec_type`
- `sec_n_row`
- `sec_n_col`
- `edge_len`
- `edge_ref`
- `para_cut_stap_method`
- `para_scaf_seq`
- `para_set_start_scaf`
- `name_prob`

Minimal test-oriented example:

```json
{
  "para_cut_stap_method": "max",
  "para_set_start_scaf": 60,
  "para_scaf_seq": "m13",
  "output_dir": "tests/output"
}
```

## Repository Layout

- `perdix_py/`: main runtime package
- `input/`: example SVG inputs
- `docs/RUNTIME_FLOW.md`: stage-by-stage runtime documentation
- `tests/`: regression, validation, and golden-reference tests
- `perdix_config.json`: default config example
- `perdix_py/resources/`: packaged scaffold sequence resources and default `seq.txt`

## Tests

Active focused test suite:

```bash
python -m unittest -q tests.test_stability_fixes tests.test_validation tests.test_bild
```

`tests/test_bild.py` is the active golden-reference artifact suite and compares generated outputs against files under `tests/reference/`.

Packaging/install smoke coverage also exists in `tests/test_package_install.py`, which verifies fresh-venv installation and packaged resource availability.

## Notes

- The runtime includes stage validation hooks after each major pipeline step.
- Some large legacy-style modules are now split internally, but the public stage entrypoints remain under `perdix_py.input`, `perdix_py.modgeo`, `perdix_py.basepair`, `perdix_py.route`, `perdix_py.seqdesign`, and `perdix_py.output`.
- For a deeper walkthrough of current execution flow, see [docs/RUNTIME_FLOW.md](docs/RUNTIME_FLOW.md).
