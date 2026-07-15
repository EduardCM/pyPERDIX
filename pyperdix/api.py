from __future__ import annotations

from perdix_py.config import Config, load_config, reset_para_defaults
from perdix_py.main import _run_pipeline, main


def run_pipeline(
    svg_path: str,
    *,
    config_path: str | None = None,
    edge_len: int | None = None,
    frame_mode: str = "legacy",
    summary: bool = False,
    debug_mesh: bool = False,
    svg_layer: int | None = None,
    svg_import_layer: int | None = None,
    svg_output_subdir: str = "",
    svg_shared_frame=None,
    svg_shared_route_start=None,
) -> None:
    _run_pipeline(
        svg_path=svg_path,
        config_path=config_path,
        edge_len=edge_len,
        frame_mode=frame_mode,
        summary=summary,
        debug_mesh=debug_mesh,
        svg_layer=svg_layer,
        svg_import_layer=svg_import_layer,
        svg_output_subdir=svg_output_subdir,
        svg_shared_frame=svg_shared_frame,
        svg_shared_route_start=svg_shared_route_start,
    )


def cli_main() -> None:
    main()
