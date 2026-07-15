from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestPackageInstall(unittest.TestCase):
    def test_fresh_venv_install_and_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory(prefix="perdix-pkg-test-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            venv_dir = tmp_root / "venv"

            subprocess.check_call(
                [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
                cwd=ROOT,
            )

            vpy = self._venv_python(venv_dir)
            cli = self._venv_cli(venv_dir, "perdix")
            env = os.environ.copy()
            env["PYTHONPYCACHEPREFIX"] = str(tmp_root / "pycache")

            subprocess.check_call(
                [str(vpy), "-m", "pip", "install", "--no-build-isolation", str(ROOT)],
                cwd=ROOT,
                env=env,
            )

            smoke = subprocess.check_output(
                [
                    str(vpy),
                    "-c",
                    (
                        "import importlib.resources as r, json, pathlib, sys; "
                        "import perdix_py; "
                        "resource_root = r.files('perdix_py.resources'); "
                        "payload = {"
                        "'package': perdix_py.__name__, "
                        "'m13': resource_root.joinpath('m13mp18_perdix.txt').is_file(), "
                        "'lamda': resource_root.joinpath('lamda_perdix.txt').is_file(), "
                        "'seq': resource_root.joinpath('seq.txt').is_file() "
                        "}; "
                        "print(json.dumps(payload))"
                    ),
                ],
                cwd=ROOT,
                env=env,
                text=True,
            )
            payload = json.loads(smoke.strip())

            self.assertEqual(payload["package"], "perdix_py")
            self.assertTrue(payload["m13"])
            self.assertTrue(payload["lamda"])
            self.assertTrue(payload["seq"])

            cli_help = subprocess.check_output(
                [str(cli), "--help"],
                cwd=ROOT,
                env=env,
                text=True,
            )
            self.assertIn("PERDIX Python port", cli_help)

    @staticmethod
    def _venv_python(venv_dir: Path) -> Path:
        if os.name == "nt":
            return venv_dir / "Scripts" / "python.exe"
        return venv_dir / "bin" / "python"

    @staticmethod
    def _venv_cli(venv_dir: Path, name: str) -> Path:
        if os.name == "nt":
            return venv_dir / "Scripts" / f"{name}.exe"
        return venv_dir / "bin" / name


if __name__ == "__main__":
    unittest.main()
