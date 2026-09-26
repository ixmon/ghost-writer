"""Start GhostWriter from a clone when its packages are not installed yet.

Running `python3 server.py` or `python3 ghostwriter.py` calls ensure_runtime()
before the third-party imports. If those imports already work, the current
Python is used and nothing is created. Otherwise a `.venv` is created in the
clone, the project is installed into it once, and the process is replaced
with that interpreter.

Importing this module does not install anything. The entry scripts opt in
only when they are the main program.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class RuntimePlan:
    kind: str  # "run", "reexec", or "fail"
    setup: str = "none"  # "none", "create", or "install"
    message: str | None = None
    python: Path | None = None
    venv_dir: Path | None = None
    script: Path | None = None
    argv: tuple[str, ...] = ()
    error: str | None = None


def ensure_runtime(modules: tuple[str, ...]) -> None:
    """Run *modules* with this interpreter, or switch into the project .venv."""
    main = sys.modules.get("__main__")
    origin_file = getattr(main, "__file__", None)
    if not origin_file:
        sys.exit("GhostWriter could not find its own script path.")
    origin = Path(origin_file).resolve()
    plan = plan_runtime(
        modules=modules,
        executable=Path(sys.executable).resolve(),
        origin=origin,
        argv=tuple(sys.argv[1:]),
        version=sys.version_info[:2],
        can_import=_can_import,
        venv_can_import=_interpreter_can_import,
    )
    if plan.kind == "run":
        return
    if plan.kind == "fail":
        sys.exit(plan.error or "GhostWriter could not start.")
    if plan.message:
        print(plan.message, file=sys.stderr, flush=True)
    if plan.setup == "create":
        _create_venv(sys.executable, plan.venv_dir)
    if plan.setup in ("create", "install"):
        _pip_install(plan.python, project_root(origin))
    _reexec(plan)


def plan_runtime(
    *,
    modules: tuple[str, ...],
    executable: Path,
    origin: Path,
    argv: tuple[str, ...],
    version: tuple[int, int],
    can_import: Callable[[tuple[str, ...]], bool],
    venv_can_import: Callable[[Path, tuple[str, ...]], bool],
) -> RuntimePlan:
    """Decide whether to run, install, or stop. No filesystem changes."""
    if can_import(modules):
        return RuntimePlan("run")

    if version < (3, 11):
        return RuntimePlan(
            "fail",
            error=(
                "GhostWriter needs Python 3.11 or newer. "
                f"This interpreter is {version[0]}.{version[1]}."
            ),
        )

    root = project_root(origin)
    if root is None:
        missing = ", ".join(modules)
        return RuntimePlan(
            "fail",
            error=(
                f"GhostWriter is missing Python packages ({missing}).\n"
                "Install them with: pip install ."
            ),
        )

    venv_dir = root / ".venv"
    venv_python = venv_interpreter(root)
    script = origin.resolve()
    if _same_path(executable, venv_python):
        missing = ", ".join(modules)
        return RuntimePlan(
            "fail",
            error=(
                f"GhostWriter's .venv is still missing packages ({missing}).\n"
                "Delete .venv and run this again."
            ),
        )

    if not venv_python.is_file():
        return RuntimePlan(
            "reexec",
            setup="create",
            message=(
                "GhostWriter: creating .venv and installing dependencies.\n"
                "This needs network access once. Later runs start directly."
            ),
            python=venv_python,
            venv_dir=venv_dir,
            script=script,
            argv=argv,
        )

    if not venv_can_import(venv_python, modules):
        return RuntimePlan(
            "reexec",
            setup="install",
            message=(
                "GhostWriter: installing dependencies into .venv.\n"
                "This needs network access once. Later runs start directly."
            ),
            python=venv_python,
            venv_dir=venv_dir,
            script=script,
            argv=argv,
        )

    return RuntimePlan(
        "reexec",
        setup="none",
        python=venv_python,
        venv_dir=venv_dir,
        script=script,
        argv=argv,
    )


def project_root(origin: Path) -> Path | None:
    """The clone that contains this script, or None for an installed copy."""
    directory = origin.resolve().parent
    if (directory / "pyproject.toml").is_file():
        return directory
    return None


def venv_interpreter(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _can_import(modules: tuple[str, ...]) -> bool:
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError:
            return False
    return True


def _interpreter_can_import(python: Path, modules: tuple[str, ...]) -> bool:
    code = "import " + ", ".join(modules)
    result = subprocess.run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _create_venv(python: str, dest: Path | None) -> None:
    if dest is None:
        sys.exit("GhostWriter: no location for .venv.")
    result = subprocess.run(
        [python, "-m", "venv", str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "").strip()
    hint = ""
    lowered = detail.lower()
    if "ensurepip" in lowered or "no module named venv" in lowered:
        hint = "\nOn Debian/Ubuntu, install the python3-venv package and run again."
    sys.exit(f"GhostWriter: could not create .venv.\n{detail}{hint}")


def _pip_install(python: Path | None, root: Path | None) -> None:
    if python is None or root is None:
        sys.exit("GhostWriter: could not find the project to install.")
    result = subprocess.run([str(python), "-m", "pip", "install", "-e", str(root)])
    if result.returncode != 0:
        sys.exit(
            "GhostWriter: dependency install failed. "
            "Fix the error above and run again."
        )


def _reexec(plan: RuntimePlan) -> None:
    if plan.python is None or plan.script is None or plan.venv_dir is None:
        sys.exit("GhostWriter: could not switch into .venv.")
    python = str(plan.python)
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(plan.venv_dir)
    argv = [python, str(plan.script), *plan.argv]
    try:
        os.execve(python, argv, env)
    except OSError as exc:
        sys.exit(f"GhostWriter: could not switch to {python}: {exc}")
