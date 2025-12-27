from __future__ import annotations

import ast
import importlib.util
import math
import unittest
from pathlib import Path


class TestNetUtils(unittest.TestCase):
    def test_module_defines_expected_symbols(self) -> None:
        """Confirm helper functions and classes are declared."""
        root = Path(__file__).resolve().parents[3]
        path = root / "pocket_flow" / "gdbp_model" / "net_utils.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        func_names = {n.name for n in module.body if isinstance(n, ast.FunctionDef)}
        class_names = {n.name for n in module.body if isinstance(n, ast.ClassDef)}
        self.assertTrue(
            {
                "reset_parameters",
                "freeze_parameters",
                "flow_forward",
                "flow_reverse",
                "embed_compose",
            }.issubset(func_names)
        )
        self.assertTrue(
            {
                "GaussianSmearing",
                "EdgeExpansion",
                "Scalarize",
                "Rescale",
                "AtomEmbedding",
                "SmoothCrossEntropyLoss",
            }.issubset(class_names)
        )

    def test_flow_forward_reverse_are_inverses_for_constant_layer(self) -> None:
        """Check flow_forward/flow_reverse invert each other."""
        if not importlib.util.find_spec("torch"):
            self.skipTest("requires torch")

        import torch
        from torch import nn

        from pocket_flow.gdbp_model.net_utils import flow_forward, flow_reverse

        class ConstantLayer(nn.Module):
            def __init__(self, log_s: float, t: float) -> None:
                super().__init__()
                self.log_s = log_s
                self.t = t

            def forward(
                self, feat: tuple[torch.Tensor, torch.Tensor]
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                batch = feat[0].shape[0]
                log_s = torch.full((batch, 3), self.log_s, dtype=feat[0].dtype, device=feat[0].device)
                t = torch.full((batch, 3), self.t, dtype=feat[0].dtype, device=feat[0].device)
                return log_s, t, feat[1]

        layers = nn.ModuleList([ConstantLayer(log_s=0.1, t=0.2), ConstantLayer(log_s=-0.3, t=-0.1)])
        x = torch.randn(4, 3)
        feat = (torch.randn(4, 5), torch.randn(4, 2, 3))

        z, log_j, _ = flow_forward(layers, x.clone(), feat)
        x_hat, _ = flow_reverse(layers, z.clone(), feat)
        self.assertTrue(torch.allclose(x_hat, x, atol=1e-6, rtol=1e-6))

        # log-Jacobian should equal sum(log_s) per layer (since scale=exp(log_s) > 0).
        expected = torch.full_like(x, 0.1 + (-0.3))
        self.assertTrue(torch.allclose(log_j, expected, atol=1e-6, rtol=1e-6))

    def test_gaussian_smearing_clamps_to_stop(self) -> None:
        """Ensure distances above stop clamp to the same RBF."""
        if not importlib.util.find_spec("torch"):
            self.skipTest("requires torch")

        import torch

        from pocket_flow.gdbp_model.net_utils import GaussianSmearing

        smear = GaussianSmearing(start=0.0, stop=2.0, num_gaussians=5)
        d = torch.tensor([0.0, 1.0, 3.0])
        out = smear(d)
        self.assertEqual(tuple(out.shape), (3, 5))

        out_stop = smear(torch.tensor([2.0]))
        self.assertTrue(torch.allclose(out[2:3], out_stop, atol=0.0, rtol=0.0))

    def test_reset_and_freeze_parameters_by_name_substring(self) -> None:
        """Validate reset/freeze uses parameter name substrings."""
        if not importlib.util.find_spec("torch"):
            self.skipTest("requires torch")

        import torch
        from torch import nn

        from pocket_flow.gdbp_model.net_utils import Rescale, freeze_parameters, reset_parameters

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layernorm = nn.LayerNorm(3)
                self.rescale = Rescale()
                self.linear = nn.Linear(3, 3)

        model = Tiny()
        reset_parameters(model, keys=["layernorm", "rescale", "linear"])

        self.assertTrue(torch.allclose(model.layernorm.weight, torch.ones_like(model.layernorm.weight)))
        self.assertTrue(torch.allclose(model.layernorm.bias, torch.zeros_like(model.layernorm.bias)))
        self.assertTrue(torch.allclose(model.rescale.weight, torch.zeros_like(model.rescale.weight)))
        self.assertTrue(torch.allclose(model.linear.bias, torch.zeros_like(model.linear.bias)))

        freeze_parameters(model, keys=["linear"])
        self.assertFalse(model.linear.weight.requires_grad)
        self.assertFalse(model.linear.bias.requires_grad)

    def test_embed_compose_scatter_back(self) -> None:
        """Verify embeddings are scattered back to composed tensors."""
        if not importlib.util.find_spec("torch"):
            self.skipTest("requires torch")

        import torch

        from pocket_flow.gdbp_model.net_utils import AtomEmbedding, embed_compose

        torch.manual_seed(0)
        n_total = 5
        idx_ligand = torch.tensor([0, 2], dtype=torch.long)
        idx_protein = torch.tensor([1, 3, 4], dtype=torch.long)

        in_scalar = 6
        emb_dim = (8, 4)
        ligand_emb = AtomEmbedding(
            in_scalar=in_scalar, in_vector=1, out_scalar=emb_dim[0], out_vector=emb_dim[1]
        )
        protein_emb = AtomEmbedding(
            in_scalar=in_scalar, in_vector=1, out_scalar=emb_dim[0], out_vector=emb_dim[1]
        )

        compose_feature = torch.randn(n_total, in_scalar)
        compose_pos = torch.randn(n_total, 3)

        h_sca, h_vec = embed_compose(
            compose_feature=compose_feature,
            compose_pos=compose_pos,
            idx_ligand=idx_ligand,
            idx_protein=idx_protein,
            ligand_atom_emb=ligand_emb,
            protein_atom_emb=protein_emb,
            emb_dim=emb_dim,
        )

        h_ligand = ligand_emb(compose_feature[idx_ligand], compose_pos[idx_ligand])
        h_protein = protein_emb(compose_feature[idx_protein], compose_pos[idx_protein])

        self.assertTrue(torch.allclose(h_sca[idx_ligand], h_ligand[0], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(h_vec[idx_ligand], h_ligand[1], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(h_sca[idx_protein], h_protein[0], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(h_vec[idx_protein], h_protein[1], atol=1e-6, rtol=1e-6))

    def test_rescale_raises_on_nan_weight(self) -> None:
        """Rescale should raise if its scale becomes NaN."""
        if not importlib.util.find_spec("torch"):
            self.skipTest("requires torch")

        import torch

        from pocket_flow.gdbp_model.net_utils import Rescale

        r = Rescale()
        with torch.no_grad():
            r.weight[:] = float("nan")
        with self.assertRaises(RuntimeError):
            _ = r(torch.ones(1))

    def test_smooth_cross_entropy_matches_known_value(self) -> None:
        """Match CE against the uniform-logit closed form."""
        if not importlib.util.find_spec("torch"):
            self.skipTest("requires torch")

        import torch

        from pocket_flow.gdbp_model.net_utils import SmoothCrossEntropyLoss

        logits = torch.tensor([[0.0, 0.0, 0.0]])  # uniform => p=1/3
        targets = torch.tensor([1])
        loss = SmoothCrossEntropyLoss(smoothing=0.0, reduction="mean")(logits, targets)
        self.assertTrue(torch.allclose(loss, torch.tensor([math.log(3.0)]), atol=1e-6, rtol=1e-6))
