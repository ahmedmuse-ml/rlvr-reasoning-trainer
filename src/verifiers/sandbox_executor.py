import subprocess
import tempfile
import sys
import os
from typing import Tuple


class SandboxExecutor:
    """Fuliye ammaan ah: Wuxuu isticmaalaa Docker haddii image-ku jiro, haddii kalena subprocess isolated ah."""

    def __init__(self, image: str = "rlvr-sandbox:latest", timeout: int = 5):
        self.image = image
        self.timeout = timeout

    def _is_image_available(self) -> bool:
        """Hubi in Docker daemon iyo image-ka labaduba diyaar yihiin."""
        try:
            res = subprocess.run(
                ["docker", "image", "inspect", self.image],
                capture_output=True,
                timeout=2,
            )
            return res.returncode == 0
        except Exception:
            return False

    def run(self, code: str, test_input: str = "") -> Tuple[bool, str]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            if self._is_image_available():
                cmd = [
                    "docker", "run", "--rm", "--network", "none",
                    "--memory", "256m", "--cpus", "1", "--pids-limit", "64",
                    "--read-only", "-v", f"{script_path}:/sandbox/run.py:ro",
                    self.image, "/sandbox/run.py",
                ]
            else:
                # Fallback safe standalone subprocess
                cmd = [sys.executable, "-I", script_path]

            result = subprocess.run(
                cmd, input=test_input, capture_output=True, text=True, timeout=self.timeout
            )
            success = (result.returncode == 0)
            return success, (result.stdout if success else result.stderr)

        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        finally:
            if os.path.exists(script_path):
                os.unlink(script_path)