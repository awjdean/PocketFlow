import torch
from torch import Tensor, nn

from pocket_flow.gdbp_model.layers import GDBLinear, GDBPerceptronVN, ST_GDBP_Exp


class AtomFlow(nn.Module):
    """Normalizing flow module for atom type prediction.

    Uses a series of affine coupling layers to transform between
    the atom latent space and the observed atom type distribution.
    """

    net: nn.Sequential
    flow_layers: nn.ModuleList

    def __init__(
        self,
        in_sca: int,
        in_vec: int,
        hidden_dim_sca: int,
        hidden_dim_vec: int,
        num_lig_atom_type: int = 10,
        num_flow_layers: int = 6,
        bottleneck: int = 1,
        use_conv1d: bool = False,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            GDBPerceptronVN(
                in_sca,
                in_vec,
                hidden_dim_sca,
                hidden_dim_vec,
                bottleneck=bottleneck,
                use_conv1d=use_conv1d,
            ),
            GDBLinear(
                hidden_dim_sca,
                hidden_dim_vec,
                hidden_dim_sca,
                hidden_dim_vec,
                bottleneck=bottleneck,
                use_conv1d=use_conv1d,
            ),
        )

        self.flow_layers = nn.ModuleList()
        for _ in range(num_flow_layers):
            layer = ST_GDBP_Exp(
                hidden_dim_sca,
                hidden_dim_vec,
                num_lig_atom_type,
                hidden_dim_vec,
                bottleneck=bottleneck,
                use_conv1d=use_conv1d,
            )
            self.flow_layers.append(layer)

    def forward(
        self,
        z_atom: Tensor,
        compose_features: tuple[Tensor, Tensor],
        focal_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass: transform atom latent to observed space.

        Args:
            z_atom: Atom latent representation (N_focal, num_lig_atom_type).
            compose_features: Tuple of (scalar_features, vector_features) from encoder.
                - scalar_features: (N_compose, hidden_dim_sca)
                - vector_features: (N_compose, hidden_dim_vec, 3)
            focal_idx: Indices of focal atoms (N_focal,).

        Returns:
            Tuple of (transformed_z_atom, atom_log_jacob):
                - transformed_z_atom: (N_focal, num_lig_atom_type)
                - atom_log_jacob: Log Jacobian determinant (N_focal, num_lig_atom_type)
        """
        sca_focal, vec_focal = compose_features[0][focal_idx], compose_features[1][focal_idx]
        sca_focal, vec_focal = self.net([sca_focal, vec_focal])

        atom_log_jacob = torch.zeros_like(z_atom)
        for flow_layer in self.flow_layers:
            s, t = flow_layer([sca_focal, vec_focal])
            s = s.exp()
            z_atom = (z_atom + t) * s
            atom_log_jacob = atom_log_jacob + (torch.abs(s) + 1e-20).log()

        return z_atom, atom_log_jacob

    def reverse(
        self,
        atom_latent: Tensor,
        compose_features: tuple[Tensor, Tensor],
        focal_idx: Tensor,
    ) -> Tensor:
        """Reverse pass: transform observed atom type back to latent space.

        Args:
            atom_latent: Atom type representation in observed space (N_focal, num_lig_atom_type).
            compose_features: Tuple of (scalar_features, vector_features) from encoder.
                - scalar_features: (N_compose, hidden_dim_sca)
                - vector_features: (N_compose, hidden_dim_vec, 3)
            focal_idx: Indices of focal atoms (N_focal,).

        Returns:
            Atom latent in base distribution space (N_focal, num_lig_atom_type).
        """
        sca_focal, vec_focal = compose_features[0][focal_idx], compose_features[1][focal_idx]
        sca_focal, vec_focal = self.net([sca_focal, vec_focal])

        for flow_layer in self.flow_layers:
            s, t = flow_layer([sca_focal, vec_focal])
            atom_latent = (atom_latent / s.exp()) - t

        return atom_latent
