from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import cast


class TestGdbpModelInit(unittest.TestCase):
    def test___all___is_strings_and_matches_imports(self) -> None:
        """Ensure __all__ lists only imported symbols without duplicates."""
        root = Path(__file__).resolve().parents[3]
        init_path = root / "pocket_flow" / "gdbp_model" / "__init__.py"
        module = ast.parse(init_path.read_text(encoding="utf-8"))

        imported_names: set[str] = set()
        for node in module.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)

        all_list: list[str] | None = None
        for node in module.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                self.assertIsInstance(node.value, ast.List)
                list_value = cast(ast.List, node.value)
                all_list = []
                for elt in list_value.elts:
                    self.assertIsInstance(elt, ast.Constant)
                    constant_elt = cast(ast.Constant, elt)
                    self.assertIsInstance(constant_elt.value, str)
                    all_list.append(cast(str, constant_elt.value))
                break

        self.assertIsNotNone(all_list, "__all__ not found")
        assert all_list is not None  # Type narrowing for type checker
        self.assertEqual(len(all_list), len(set(all_list)), "__all__ contains duplicates")

        missing = sorted(set(all_list) - imported_names)
        self.assertEqual(missing, [], "__all__ contains names not imported in __init__.py")
