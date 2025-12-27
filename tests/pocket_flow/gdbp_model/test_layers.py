from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


class TestLayers(unittest.TestCase):
    def test_module_defines_expected_classes(self) -> None:
        """Ensure all core layer classes are defined."""
        root = Path(__file__).resolve().parents[3]
        path = root / "pocket_flow" / "gdbp_model" / "layers.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        class_names = {n.name for n in module.body if isinstance(n, ast.ClassDef)}
        expected = {
            "VNLinear",
            "VNLeakyReLU",
            "GDBLinear",
            "GDBPerceptronVN",
            "ST_GDBP_Exp",
            "MessageAttention",
            "MessageModule",
            "AttentionInteractionBlockVN",
            "AttentionBias",
            "AttentionEdges",
        }
        self.assertTrue(expected.issubset(class_names))

    def test_st_gdbp_exp_output_shapes_and_bounded_scale(self) -> None:
        """Validate ST_GDBP_Exp shapes and tanh-bounded scale."""
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("torch_scatter"):
            self.skipTest("requires torch + torch_scatter")

        import torch

        from pocket_flow.gdbp_model.layers import ST_GDBP_Exp

        torch.manual_seed(0)
        layer = ST_GDBP_Exp(in_scalar=6, in_vector=4, out_scalar=5, out_vector=4)
        x = (torch.randn(3, 6), torch.randn(3, 4, 3))
        s, t = layer(x)
        self.assertEqual(tuple(s.shape), (3, 5))
        self.assertEqual(tuple(t.shape), (3, 5))
        # Rescale is initialized to identity and `s` goes through `tanh`.
        self.assertTrue((s.abs() <= 1.0 + 1e-6).all())
