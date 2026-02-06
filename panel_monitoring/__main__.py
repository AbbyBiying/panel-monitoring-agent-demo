# panel_monitoring/__main__.py

import os
import sys
import subprocess
from pathlib import Path
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        logger.error(
            "Usage: uv run python -m panel_monitoring <file-to-run> [-- <args-for-script>]"
        )
        sys.exit(1)

    file_to_run = sys.argv[1]
    extra_args = sys.argv[2:] if len(sys.argv) > 2 else []

    pkg_root = Path(__file__).resolve().parent  # .../panel_monitoring
    repo_root = pkg_root.parent  # repo root

    # Resolve the target path (accept relative to cwd or repo root)
    file_path = Path(file_to_run)
    if not file_path.exists():
        candidate = repo_root / file_to_run
        if candidate.exists():
            file_path = candidate
        else:
            logger.error("File not found: %s", file_to_run)
            sys.exit(1)

    resolved = file_path.resolve()

    # If target lives inside the package, run it as a module so absolute imports work
    if resolved.suffix == ".py" and pkg_root in resolved.parents:
        # e.g. repo_root/panel_monitoring/panel_agent_openai.py -> "panel_monitoring.panel_agent_openai"
        rel = resolved.relative_to(repo_root).with_suffix("")
        module_name = ".".join(rel.parts)
        cmd = [sys.executable, "-m", module_name, *extra_args]
    else:
        cmd = [sys.executable, str(resolved), *extra_args]

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
