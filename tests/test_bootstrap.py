"""First-run launcher decisions. These tests do not create a virtual environment."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bootstrap import plan_runtime, venv_interpreter  # noqa: E402


class PlanTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = Path(self._td.name)
        (self.root / "pyproject.toml").write_text("[project]\nname = 'ghostwriter'\n")
        self.origin = self.root / "server.py"
        self.origin.write_text("# entry\n")
        self.venv_python = venv_interpreter(self.root)
        self.modules = ("fastapi", "yaml")

    def _plan(self, *, can_import, executable=None, venv_can=None, version=(3, 12)):
        return plan_runtime(
            modules=self.modules,
            executable=Path(sys.executable if executable is None else executable).resolve(),
            origin=self.origin,
            argv=("--host", "0.0.0.0"),
            version=version,
            can_import=can_import,
            venv_can_import=venv_can or (lambda python, modules: False),
        )

    def test_current_interpreter_is_used_when_packages_import(self):
        plan = self._plan(can_import=lambda modules: True)
        self.assertEqual(plan.kind, "run")
        self.assertIsNone(plan.message)

    def test_missing_packages_outside_a_clone_ask_for_pip(self):
        loose = self.root / "installed" / "server.py"
        loose.parent.mkdir()
        loose.write_text("# no project\n")
        plan = plan_runtime(
            modules=self.modules,
            executable=Path(sys.executable).resolve(),
            origin=loose,
            argv=(),
            version=(3, 12),
            can_import=lambda modules: False,
            venv_can_import=lambda python, modules: False,
        )
        self.assertEqual(plan.kind, "fail")
        self.assertIn("pip install .", plan.error)

    def test_old_python_is_rejected_before_creating_a_venv(self):
        plan = self._plan(can_import=lambda modules: False, version=(3, 10))
        self.assertEqual(plan.kind, "fail")
        self.assertIn("3.11", plan.error)
        self.assertEqual(plan.setup, "none")

    def test_first_run_creates_a_venv_and_keeps_arguments(self):
        plan = self._plan(can_import=lambda modules: False)
        self.assertEqual(plan.kind, "reexec")
        self.assertEqual(plan.setup, "create")
        self.assertIn("creating .venv", plan.message)
        self.assertIn("network access once", plan.message)
        self.assertEqual(plan.python, self.venv_python)
        self.assertEqual(plan.venv_dir, self.root / ".venv")
        self.assertEqual(plan.script, self.origin.resolve())
        self.assertEqual(plan.argv, ("--host", "0.0.0.0"))

    def test_existing_venv_without_packages_installs(self):
        self.venv_python.parent.mkdir(parents=True)
        self.venv_python.write_text("")
        plan = self._plan(can_import=lambda modules: False, venv_can=lambda python, modules: False)
        self.assertEqual(plan.kind, "reexec")
        self.assertEqual(plan.setup, "install")
        self.assertIn("installing dependencies", plan.message)

    def test_ready_venv_reexecs_quietly(self):
        self.venv_python.parent.mkdir(parents=True)
        self.venv_python.write_text("")
        plan = self._plan(can_import=lambda modules: False, venv_can=lambda python, modules: True)
        self.assertEqual(plan.kind, "reexec")
        self.assertEqual(plan.setup, "none")
        self.assertIsNone(plan.message)
        self.assertEqual(plan.argv, ("--host", "0.0.0.0"))

    def test_broken_project_venv_does_not_reexec(self):
        self.venv_python.parent.mkdir(parents=True)
        self.venv_python.write_text("")
        plan = self._plan(
            can_import=lambda modules: False,
            executable=self.venv_python,
            venv_can=lambda python, modules: False,
        )
        self.assertEqual(plan.kind, "fail")
        self.assertIn("Delete .venv", plan.error)


if __name__ == "__main__":
    unittest.main()
