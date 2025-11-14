import ast
from pathlib import Path

import pytest

ROOT_PACKAGE = "tabarena"

# Files that are allowed to import the root package (adjust as needed)
ALLOWED_FILES = {
    "tabarena/__init__.py",
    # Add more if you explicitly want to allow them, e.g.:
    # "tabarena/__main__.py",
}


def _find_repo_root() -> Path:
    """Find the repository root by walking up until we see the package dir."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ROOT_PACKAGE).is_dir():
            return parent
    raise RuntimeError(
        f"Could not find package directory '{ROOT_PACKAGE}' above {here}"
    )


def test_no_root_level_imports_inside_package():
    """
    Ensure no internal module does 'from tabarena import ...' or 'import tabarena'.

    This enforces the rule: internal code should import submodules directly
    (absolute or relative), not via the package root, to avoid circular imports.
    """
    repo_root = _find_repo_root()
    pkg_dir = repo_root / ROOT_PACKAGE

    violations: list[str] = []

    for path in pkg_dir.rglob("*.py"):
        rel_path = path.relative_to(repo_root).as_posix()
        if rel_path in ALLOWED_FILES:
            continue

        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            pytest.fail(f"Syntax error while parsing {rel_path}: {e}")  # sanity check

        for node in ast.walk(tree):
            # from tabarena import X
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module == ROOT_PACKAGE:
                    violations.append(
                        f"{rel_path}:{node.lineno} uses 'from {ROOT_PACKAGE} import ...'"
                    )

            # import tabarena
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == ROOT_PACKAGE:
                        violations.append(
                            f"{rel_path}:{node.lineno} uses 'import {ROOT_PACKAGE}'"
                        )

    if violations:
        message = "Forbidden root-level imports of the package from inside itself:\n"
        message += "\n".join(violations)
        pytest.fail(message)
