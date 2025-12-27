from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


class TestFocalNet(unittest.TestCase):
    def test_module_defines_FocalNet(self) -> None:
        """Verify the focal head class is present in the module."""
        root = Path(__file__).resolve().parents[3]
        path = root / "pocket_flow" / "gdbp_model" / "focal_net.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        class_names = {n.name for n in module.body if isinstance(n, ast.ClassDef)}
        self.assertIn("FocalNet", class_names)

    def test_forward_shapes_for_index_and_mask(self) -> None:
        """Check output shapes for index and mask selection."""
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("torch_scatter"):
            self.skipTest("requires torch + torch_scatter")

        import torch

        from pocket_flow.gdbp_model.focal_net import FocalNet

        torch.manual_seed(0)
        model = FocalNet(in_sca=6, in_vec=4, hidden_dim_sca=8, hidden_dim_vec=4)
        n = 5
        h_att = (torch.randn(n, 6), torch.randn(n, 4, 3))

        idx = torch.tensor([0, 3, 4], dtype=torch.long)
        out = model(h_att, idx)
        self.assertEqual(tuple(out.shape), (len(idx), 1))

        mask = torch.tensor([True, False, True, False, True])
        out_mask = model(h_att, mask)
        self.assertEqual(tuple(out_mask.shape), (int(mask.sum().item()), 1))
