from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


class TestContextEncoder(unittest.TestCase):
    def test_module_defines_ContextEncoder(self) -> None:
        """Confirm the encoder class is declared in the module."""
        root = Path(__file__).resolve().parents[3]
        path = root / "pocket_flow" / "gdbp_model" / "encoder.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        class_names = {n.name for n in module.body if isinstance(n, ast.ClassDef)}
        self.assertIn("ContextEncoder", class_names)

    def test_invalid_edge_feature_channels_raises(self) -> None:
        """Assert mismatched edge feature channels raise an error."""
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("torch_scatter"):
            self.skipTest("requires torch + torch_scatter")

        import torch

        from pocket_flow.gdbp_model.encoder import ContextEncoder

        encoder = ContextEncoder(
            hidden_channels=(8, 4), edge_channels=6, num_edge_types=2, num_interactions=1
        )
        n, e = 4, 4
        node_attr = (torch.randn(n, 8), torch.randn(n, 4, 3))
        pos = torch.randn(n, 3)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        bad_edge_feature = torch.randn(e, 3)  # should be num_edge_types=2

        with self.assertRaises(AssertionError):
            encoder(node_attr, pos, edge_index, bad_edge_feature)

    def test_forward_shape_smoke(self) -> None:
        """Sanity-check output shapes for a small random graph."""
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("torch_scatter"):
            self.skipTest("requires torch + torch_scatter")

        import torch

        from pocket_flow.gdbp_model.encoder import ContextEncoder

        torch.manual_seed(0)
        encoder = ContextEncoder(
            hidden_channels=(8, 4), edge_channels=6, num_edge_types=2, num_interactions=2
        )
        n, e = 6, 10
        node_attr = (torch.randn(n, 8), torch.randn(n, 4, 3))
        pos = torch.randn(n, 3)
        # destination=row, source=col
        edge_index = torch.randint(0, n, (2, e), dtype=torch.long)
        edge_feature = torch.randn(e, 2)
        out_sca, out_vec = encoder(node_attr, pos, edge_index, edge_feature, annealing=False)
        self.assertEqual(tuple(out_sca.shape), (n, 8))
        self.assertEqual(tuple(out_vec.shape), (n, 4, 3))
