from __future__ import annotations

from importlib import resources


_RESOURCE_PACKAGE = "perdix_py.resources"


def read_packaged_text(name: str) -> str:
    return resources.files(_RESOURCE_PACKAGE).joinpath(name).read_text(encoding="utf-8")
