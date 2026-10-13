"""Guards the package layering: lower layers must not import higher layers at module level.

Function-local imports are allowed, which is how convenience wrappers such as
``Trajectory.from_file`` delegate to :mod:`trajectopy.io` without creating an import cycle.
"""

import ast
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "trajectopy"

# Sub-packages each layer may import from (in addition to itself).
ALLOWED_DEPENDENCIES: dict[str, set[str]] = {
    "exceptions": set(),
    "utils": {"exceptions"},
    "core": {"utils", "exceptions"},
    "results": {"core", "utils", "exceptions"},
    "processing": {"core", "results", "utils", "exceptions"},
    "io": {"core", "results", "utils", "exceptions"},
    "readers": {"io"},  # deprecated re-export shims
    "visualization": {"core", "results", "utils", "exceptions"},
}


def _layer_of(module_path: Path) -> str:
    relative = module_path.relative_to(PACKAGE_ROOT)
    return relative.parts[0].removesuffix(".py")


def _module_level_imports(module_path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
    return imports


class TestArchitecture(unittest.TestCase):
    def test_layer_dependencies(self):
        violations = []
        for module_path in PACKAGE_ROOT.rglob("*.py"):
            layer = _layer_of(module_path)
            if layer not in ALLOWED_DEPENDENCIES:
                continue

            allowed = ALLOWED_DEPENDENCIES[layer] | {layer}
            for lineno, module in _module_level_imports(module_path):
                parts = module.split(".")
                if parts[0] != "trajectopy" or len(parts) < 2 or parts[1] == "__version__":
                    continue
                if parts[1] not in allowed:
                    violations.append(f"{module_path.relative_to(PACKAGE_ROOT)}:{lineno} imports {module}")

        self.assertEqual(violations, [], "Layering violations:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
