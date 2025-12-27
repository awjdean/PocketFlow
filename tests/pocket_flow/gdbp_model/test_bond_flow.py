from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


class TestBondFlow(unittest.TestCase):
    def test_module_defines_expected_symbols(self) -> None:
        """Verify BondFlow, PositionEncoder, and _has_edges exist."""
        root = Path(__file__).resolve().parents[3]
        path = root / "pocket_flow" / "gdbp_model" / "bond_flow.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        class_names = {n.name for n in module.body if isinstance(n, ast.ClassDef)}
        self.assertIn("BondFlow", class_names)
        self.assertIn("PositionEncoder", class_names)
        func_names = {n.name for n in module.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("_has_edges", func_names)

    def test_has_edges_and_empty_edge_short_circuit(self) -> None:
        """Ensure edge detection and empty-edge fast paths behave."""
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("torch_scatter"):
            self.skipTest("requires torch + torch_scatter")

        import torch

        from pocket_flow.gdbp_model.bond_flow import BondFlow, _has_edges

        self.assertFalse(_has_edges(torch.empty(0)))
        self.assertFalse(_has_edges(torch.empty(2, 0, dtype=torch.long)))
        self.assertTrue(_has_edges(torch.zeros(2, 1, dtype=torch.long)))

        in_sca, in_vec = 6, 4
        edge_channels = 8
        num_filters = (8, 4)
        num_bond_types = 3
        model = BondFlow(
            in_sca=in_sca,
            in_vec=in_vec,
            edge_channels=edge_channels,
            num_filters=num_filters,
            num_bond_types=num_bond_types,
            num_heads=2,
            num_st_layers=2,
        )

        z_edge = torch.randn(0, num_bond_types + 1)
        pos_query = torch.randn(2, 3)
        empty_edge_index_query = torch.empty(2, 0, dtype=torch.long)
        cpx_pos = torch.randn(3, 3)
        node_attr_compose = (torch.randn(3, in_sca), torch.randn(3, in_vec, 3))
        edge_index_q_cps_knn = torch.empty(2, 0, dtype=torch.long)
        atom_type_emb = torch.randn(2, in_sca)
        index_real = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        tri_edge_index = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        tri_edge_feat = torch.empty(0, num_bond_types + 2)

        z_out, log_j = model(
            z_edge=z_edge,
            pos_query=pos_query,
            edge_index_query=empty_edge_index_query,
            cpx_pos=cpx_pos,
            node_attr_compose=node_attr_compose,
            edge_index_q_cps_knn=edge_index_q_cps_knn,
            atom_type_emb=atom_type_emb,
            index_real_cps_edge_for_atten=index_real,
            tri_edge_index=tri_edge_index,
            tri_edge_feat=tri_edge_feat,
        )
        self.assertEqual(tuple(z_out.shape), (0, num_bond_types + 1))
        self.assertEqual(tuple(log_j.shape), (0, num_bond_types + 1))

        x_out = model.reverse(
            edge_latent=z_out,
            pos_query=pos_query,
            edge_index_query=empty_edge_index_query,
            cpx_pos=cpx_pos,
            node_attr_compose=node_attr_compose,
            edge_index_q_cps_knn=edge_index_q_cps_knn,
            atom_type_emb=atom_type_emb,
            index_real_cps_edge_for_atten=index_real,
            tri_edge_index=tri_edge_index,
            tri_edge_feat=tri_edge_feat,
        )
        self.assertEqual(tuple(x_out.shape), (0, num_bond_types + 1))

    def test_round_trip_small_graph(self) -> None:
        """Check forward/reverse consistency on a tiny graph."""
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("torch_scatter"):
            self.skipTest("requires torch + torch_scatter")

        import torch

        from pocket_flow.gdbp_model.bond_flow import BondFlow

        torch.manual_seed(0)
        in_sca, in_vec = 6, 4
        edge_channels = 8
        num_filters = (8, 4)
        num_bond_types = 3
        model = BondFlow(
            in_sca=in_sca,
            in_vec=in_vec,
            edge_channels=edge_channels,
            num_filters=num_filters,
            num_bond_types=num_bond_types,
            num_heads=2,
            num_st_layers=2,
        )

        n_query = 2
        n_cpx = 3
        e_query = 2
        n_attn = 3

        pos_query = torch.randn(n_query, 3)
        cpx_pos = torch.randn(n_cpx, 3)
        node_attr_compose = (torch.randn(n_cpx, in_sca), torch.randn(n_cpx, in_vec, 3))
        atom_type_emb = torch.randn(n_query, in_sca)

        # Two queried edges: each query connects to one complex node.
        edge_index_query = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        z_edge = torch.randn(e_query, num_bond_types + 1)

        # KNN edges query -> complex for PositionEncoder
        edge_index_q_cps_knn = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], dtype=torch.long)

        # Edge attention: attend from each queried edge to itself and the other edge.
        index_real = (torch.tensor([0, 0, 1], dtype=torch.long), torch.tensor([0, 1, 1], dtype=torch.long))

        # Triangle-edge features: pairs of complex nodes with extra edge-type channels.
        tri_edge_index = (
            torch.tensor([0, 1, 2], dtype=torch.long),
            torch.tensor([1, 2, 0], dtype=torch.long),
        )
        tri_edge_feat = torch.randn(n_attn, num_bond_types + 2)

        z, log_j = model(
            z_edge=z_edge.clone(),
            pos_query=pos_query,
            edge_index_query=edge_index_query,
            cpx_pos=cpx_pos,
            node_attr_compose=node_attr_compose,
            edge_index_q_cps_knn=edge_index_q_cps_knn,
            atom_type_emb=atom_type_emb,
            index_real_cps_edge_for_atten=index_real,
            tri_edge_index=tri_edge_index,
            tri_edge_feat=tri_edge_feat,
            annealing=False,
        )
        self.assertEqual(tuple(z.shape), (e_query, num_bond_types + 1))
        self.assertEqual(tuple(log_j.shape), (e_query, num_bond_types + 1))
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(torch.isfinite(log_j).all())

        x_hat = model.reverse(
            edge_latent=z.clone(),
            pos_query=pos_query,
            edge_index_query=edge_index_query,
            cpx_pos=cpx_pos,
            node_attr_compose=node_attr_compose,
            edge_index_q_cps_knn=edge_index_q_cps_knn,
            atom_type_emb=atom_type_emb,
            index_real_cps_edge_for_atten=index_real,
            tri_edge_index=tri_edge_index,
            tri_edge_feat=tri_edge_feat,
            annealing=False,
        )
        self.assertTrue(torch.allclose(x_hat, z_edge, atol=1e-5, rtol=1e-5))
