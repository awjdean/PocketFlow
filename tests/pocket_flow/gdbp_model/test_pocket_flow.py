from __future__ import annotations

import ast
import importlib.util
import types
import unittest
from pathlib import Path


class TestPocketFlowModule(unittest.TestCase):
    def test_module_defines_PocketFlow_class(self) -> None:
        """Confirm the PocketFlow class exists."""
        root = Path(__file__).resolve().parents[3]
        path = root / "pocket_flow" / "gdbp_model" / "pocket_flow.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        class_names = {n.name for n in module.body if isinstance(n, ast.ClassDef)}
        self.assertIn("PocketFlow", class_names)

    def test_parameter_count_method(self) -> None:
        """Check get_parameter_number returns sane counts."""
        if (
            not importlib.util.find_spec("torch")
            or not importlib.util.find_spec("torch_scatter")
            or not importlib.util.find_spec("easydict")
        ):
            self.skipTest("requires torch + torch_scatter + easydict")

        import torch

        from pocket_flow.gdbp_model.pocket_flow import PocketFlow

        # Minimal config object satisfying PocketFlow.__init__ access patterns.
        cfg = types.SimpleNamespace(
            hidden_channels=8,
            hidden_channels_vec=4,
            protein_atom_feature_dim=6,
            ligand_atom_feature_dim=6,
            num_atom_type=5,
            num_bond_types=4,  # includes "no-bond" channel for encoder edge features
            msg_annealing=False,
            deq_coeff=0.0,
            bottleneck=1,
            use_conv1d=False,
            encoder=types.SimpleNamespace(
                edge_channels=8, num_interactions=1, knn=4, cutoff=10.0, num_heads=1
            ),
            focal_net=types.SimpleNamespace(hidden_dim_sca=8, hidden_dim_vec=4),
            atom_flow=types.SimpleNamespace(hidden_dim_sca=8, hidden_dim_vec=4, num_flow_layers=1),
            pos_predictor=types.SimpleNamespace(num_filters=(8, 4), n_component=2),
            edge_flow=types.SimpleNamespace(
                edge_channels=8,
                num_filters=(8, 4),
                num_bond_types=3,
                num_heads=1,
                cutoff=10.0,
                num_flow_layers=1,
            ),
        )

        model = PocketFlow(cfg)
        counts = model.get_parameter_number()
        self.assertEqual(set(counts.keys()), {"Total", "Trainable"})
        self.assertGreater(counts["Total"], 0)
        self.assertGreater(counts["Trainable"], 0)
        self.assertLessEqual(counts["Trainable"], counts["Total"])
        self.assertIsInstance(counts["Total"], int)

        # Smoke: parameters are torch tensors.
        self.assertTrue(all(isinstance(p, torch.Tensor) for p in model.parameters()))
