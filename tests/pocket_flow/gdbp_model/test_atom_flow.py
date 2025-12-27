from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


class TestAtomFlow(unittest.TestCase):
    def test_module_defines_AtomFlow_class(self) -> None:
        """Confirm the AtomFlow class exists in the module."""
        root = Path(__file__).resolve().parents[3]
        path = root / "pocket_flow" / "gdbp_model" / "atom_flow.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        class_names = {n.name for n in module.body if isinstance(n, ast.ClassDef)}
        self.assertIn("AtomFlow", class_names)

    def test_round_trip_and_empty_focal_behavior(self) -> None:
        """Check empty-focal behavior and forward/reverse consistency."""
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("torch_scatter"):
            self.skipTest("requires torch + torch_scatter")

        import torch

        from pocket_flow.gdbp_model.atom_flow import AtomFlow

        torch.manual_seed(0)
        in_sca, in_vec = 6, 4
        hidden_sca, hidden_vec = 8, 6
        k = 5
        flow = AtomFlow(
            in_sca=in_sca,
            in_vec=in_vec,
            hidden_dim_sca=hidden_sca,
            hidden_dim_vec=hidden_vec,
            num_lig_atom_type=k,
            num_flow_layers=3,
        )

        # Empty focal indices: identity and zero log-Jacobian.
        z_atom = torch.randn(0, k)
        compose_features = (torch.randn(7, in_sca), torch.randn(7, in_vec, 3))
        focal_idx = torch.empty(0, dtype=torch.long)
        z_latent, log_j = flow(z_atom, compose_features, focal_idx)
        self.assertEqual(tuple(z_latent.shape), (0, k))
        self.assertTrue(torch.equal(z_latent, z_atom))
        self.assertTrue(torch.equal(log_j, torch.zeros_like(z_atom)))

        # Non-empty: reverse(forward(x)) == x and matches manual affine composition.
        focal_idx = torch.tensor([1, 4], dtype=torch.long)
        x = torch.randn(len(focal_idx), k)
        z, log_j = flow(x.clone(), compose_features, focal_idx)
        x_hat = flow.reverse(z.clone(), compose_features, focal_idx)
        self.assertTrue(torch.allclose(x_hat, x, atol=1e-5, rtol=1e-5))

        sca_focal = compose_features[0][focal_idx]
        vec_focal = compose_features[1][focal_idx]
        sca_focal, vec_focal = flow.net([sca_focal, vec_focal])
        z_manual = x.clone()
        log_j_manual = torch.zeros_like(x)
        for layer in flow.flow_layers:
            log_s, t = layer([sca_focal, vec_focal])
            z_manual = (z_manual + t) * torch.exp(log_s)
            log_j_manual = log_j_manual + log_s
        self.assertTrue(torch.allclose(z_manual, z, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(log_j_manual, log_j, atol=1e-6, rtol=1e-6))
