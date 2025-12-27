from __future__ import annotations

import ast
import importlib.util
import math
import unittest
from pathlib import Path


class TestPositionPredictor(unittest.TestCase):
    def test_module_defines_PositionPredictor(self) -> None:
        """Confirm the PositionPredictor class is declared."""
        root = Path(__file__).resolve().parents[3]
        path = root / "pocket_flow" / "gdbp_model" / "position_predictor.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        class_names = {n.name for n in module.body if isinstance(n, ast.ClassDef)}
        self.assertIn("PositionPredictor", class_names)

    def test_probability_matches_isotropic_unit_gaussian_when_single_component(self) -> None:
        """Validate MDN probability for a unit Gaussian."""
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("torch_scatter"):
            self.skipTest("requires torch + torch_scatter")

        import torch

        from pocket_flow.gdbp_model.position_predictor import PositionPredictor

        # Use helper methods without instantiating the heavy internal nets.
        predictor = PositionPredictor(in_sca=2, in_vec=1, num_filters=(4, 2), n_component=1)
        mu = torch.zeros(2, 1, 3)
        sigma = torch.ones(2, 1, 3)
        pi = torch.ones(2, 1)
        pos_target = torch.zeros(2, 3)
        p = predictor.get_mdn_probability(mu, sigma, pi, pos_target)

        expected = torch.full((2,), (1.0 / math.sqrt(2.0 * math.pi)) ** 3)
        self.assertTrue(torch.allclose(p, expected, atol=1e-6, rtol=1e-6))

    def test_forward_and_sampling_shapes(self) -> None:
        """Check shapes for forward outputs and sampled points."""
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("torch_scatter"):
            self.skipTest("requires torch + torch_scatter")

        import torch

        from pocket_flow.gdbp_model.position_predictor import PositionPredictor

        torch.manual_seed(0)
        in_sca, in_vec = 6, 4
        n_component = 3
        predictor = PositionPredictor(
            in_sca=in_sca, in_vec=in_vec, num_filters=(8, 4), n_component=n_component
        )

        n_total = 5
        idx_focal = torch.tensor([1, 4], dtype=torch.long)
        h_compose = [torch.randn(n_total, in_sca), torch.randn(n_total, in_vec, 3)]
        pos_compose = torch.randn(n_total, 3)
        atom_type_emb = torch.randn(len(idx_focal), in_sca)

        rel_mu, abs_mu, sigma, pi = predictor(h_compose, idx_focal, pos_compose, atom_type_emb=atom_type_emb)
        self.assertEqual(tuple(rel_mu.shape), (len(idx_focal), n_component, 3))
        self.assertEqual(tuple(abs_mu.shape), (len(idx_focal), n_component, 3))
        self.assertEqual(tuple(sigma.shape), (len(idx_focal), n_component, 3))
        self.assertEqual(tuple(pi.shape), (len(idx_focal), n_component))
        self.assertTrue(torch.allclose(pi.sum(dim=1), torch.ones(len(idx_focal)), atol=1e-6, rtol=1e-6))
        self.assertTrue((sigma > 0).all())

        samples = predictor.sample_batch(abs_mu, sigma, pi, num=7)
        self.assertEqual(tuple(samples.shape), (len(idx_focal), 7, 3))
