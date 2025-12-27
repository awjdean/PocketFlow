from __future__ import annotations

import ast
import unittest
from pathlib import Path


class TestTypes(unittest.TestCase):
    def test_type_aliases_exist_and_match_expected_forms(self) -> None:
        """Ensure type aliases are defined with expected shapes."""
        root = Path(__file__).resolve().parents[3]
        path = root / "pocket_flow" / "gdbp_model" / "types.py"
        module = ast.parse(path.read_text(encoding="utf-8"))

        aliases: dict[str, str] = {}
        for node in module.body:
            if isinstance(node, ast.TypeAlias):
                aliases[node.name.id] = ast.unparse(node.value)

        self.assertIn("ScalarVectorFeatures", aliases)
        self.assertIn("BottleneckSpec", aliases)
        self.assertEqual(aliases["ScalarVectorFeatures"], "tuple[Tensor, Tensor]")
        self.assertEqual(aliases["BottleneckSpec"], "int | tuple[int, int]")
