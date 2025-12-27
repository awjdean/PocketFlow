"""Mixture-density position prediction for focal atoms.

This module implements a small network that predicts parameters of a
mixture-density network (MDN) over 3D positions. Given scalar/vector features
for a subset of "focal" atoms, the model outputs a mixture of diagonal 3D
Gaussians:

- component means `mu` in R^3 (predicted as *relative* offsets from the current
  focal position and optionally converted to absolute coordinates),
- component scales `sigma` in R^3 (per-axis standard deviations, `sigma > 0`),
- mixture weights `pi` on the simplex (softmax over components).

The helper methods in this file provide utilities to evaluate the mixture
likelihood at target coordinates and to sample coordinates from the predicted
distribution.
"""

import math
from typing import override

import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import functional as F

from pocket_flow.gdbp_model.layers import GDBLinear, GDBPerceptronVN
from pocket_flow.gdbp_model.types import BottleneckSpec, ScalarVectorFeatures

GAUSSIAN_COEF: float = 1.0 / math.sqrt(2 * math.pi)
EPS_SIGMA: float = 1e-16


class PositionPredictor(Module):
    """Predict 3D coordinates using a Gaussian mixture (MDN).

    The model consumes scalar/vector atom features (in the same "GDB" scalar +
    vector representation used throughout the codebase) and predicts parameters
    of a per-atom mixture of `n_component` 3D Gaussians with diagonal covariance.

    Notes:
        - All Gaussians are factorized across x/y/z by design (diagonal sigma),
          and the returned densities are computed as a product of 1D Normal pdfs.
        - `mu_net` and `logsigma_net` produce *vector* outputs of shape
          `(N, n_component, 3)`. `pi_net` produces *scalar* logits of shape
          `(N, n_component)` which are normalized with softmax.
    """

    n_component: int
    mlp: Sequential
    mu_net: GDBLinear
    logsigma_net: GDBLinear
    pi_net: GDBLinear

    def __init__(
        self,
        in_sca: int,
        in_vec: int,
        num_filters: tuple[int, int],
        n_component: int,
        bottleneck: BottleneckSpec = 1,
        use_conv1d: bool = False,
    ) -> None:
        """Initialize a position predictor.

        Args:
            in_sca: Input scalar feature dimension per atom.
            in_vec: Input vector feature channels per atom (each channel is a
                3D vector).
            num_filters: Hidden dimensions `(hidden_sca, hidden_vec)` used by the
                internal MLP and subsequent heads.
            n_component: Number of mixture components per predicted position.
            bottleneck: Bottleneck configuration forwarded to `GDB*` layers.
            use_conv1d: Whether to use 1D convolutional variants in `GDB*`
                layers (passed through).
        """
        super().__init__()
        self.n_component = n_component
        self.mlp = Sequential(
            GDBPerceptronVN(
                in_sca * 2,
                in_vec,
                num_filters[0],
                num_filters[1],
                bottleneck=bottleneck,
                use_conv1d=use_conv1d,
            ),
            GDBLinear(
                num_filters[0],
                num_filters[1],
                num_filters[0],
                num_filters[1],
                bottleneck=bottleneck,
                use_conv1d=use_conv1d,
            ),
        )
        self.mu_net = GDBLinear(
            num_filters[0],
            num_filters[1],
            n_component,
            n_component,
            bottleneck=bottleneck,
            use_conv1d=use_conv1d,
        )
        self.logsigma_net = GDBLinear(
            num_filters[0],
            num_filters[1],
            n_component,
            n_component,
            bottleneck=bottleneck,
            use_conv1d=use_conv1d,
        )
        self.pi_net = GDBLinear(
            num_filters[0],
            num_filters[1],
            n_component,
            1,
            bottleneck=bottleneck,
            use_conv1d=use_conv1d,
        )

    @override
    def forward(
        self,
        h_compose: list[Tensor],
        idx_focal: Tensor,
        pos_compose: Tensor,
        atom_type_emb: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute mixture parameters for focal atoms.

        Args:
            h_compose: Scalar/vector feature list `[h_sca, h_vec]` for the full
                composed system. Expected shapes:
                - `h_sca`: `(N_total, F_sca)`
                - `h_vec`: `(N_total, F_vec, 3)`
            idx_focal: 1D integer indices selecting focal atoms from the composed
                tensors, shape `(N_focal,)`.
            pos_compose: Cartesian coordinates for the composed system, shape
                `(N_total, 3)`.
            atom_type_emb: Optional per-focal scalar embedding to concatenate to
                the focal scalar features, shape `(N_focal, E)`. If provided, the
                first MLP layer expects `in_sca * 2` scalar channels (the
                concatenation is performed here).

        Returns:
            A tuple `(relative_mu, abs_mu, sigma, pi)` where:
            - `relative_mu`: Predicted component means as offsets from the focal
              coordinates, shape `(N_focal, n_component, 3)`.
            - `abs_mu`: `relative_mu` shifted by `pos_focal` to absolute
              coordinates, shape `(N_focal, n_component, 3)`.
            - `sigma`: Per-axis standard deviations (positive), shape
              `(N_focal, n_component, 3)`.
            - `pi`: Mixture weights summing to 1 over components, shape
              `(N_focal, n_component)`.
        """
        h_focal: list[Tensor] = [h[idx_focal] for h in h_compose]
        pos_focal: Tensor = pos_compose[idx_focal]
        if isinstance(atom_type_emb, Tensor):
            h_focal[0] = torch.cat([h_focal[0], atom_type_emb], dim=1)

        feat_focal: ScalarVectorFeatures = self.mlp(h_focal)
        relative_mu: Tensor = self.mu_net(feat_focal)[1]  # (N_focal, n_component, 3)
        logsigma: Tensor = self.logsigma_net(feat_focal)[1]  # (N_focal, n_component, 3)
        sigma: Tensor = torch.exp(logsigma)
        pi: Tensor = self.pi_net(feat_focal)[0]  # (N_focal, n_component)
        pi = F.softmax(pi, dim=1)

        abs_mu: Tensor = relative_mu + pos_focal.unsqueeze(dim=1).expand_as(relative_mu)
        return relative_mu, abs_mu, sigma, pi

    def get_mdn_probability(
        self,
        mu: Tensor,
        sigma: Tensor,
        pi: Tensor,
        pos_target: Tensor,
    ) -> Tensor:
        """Evaluate the mixture density at target coordinates.

        Args:
            mu: Component means, shape `(N, n_component, 3)`.
            sigma: Component per-axis standard deviations, shape
                `(N, n_component, 3)`.
            pi: Mixture weights, shape `(N, n_component)`.
            pos_target: Target coordinates, shape `(N, 3)`.

        Returns:
            Mixture density `p(pos_target)` for each item, shape `(N,)`.
        """
        prob_gauss: Tensor = self._get_gaussian_probability(mu, sigma, pos_target)
        prob_mdn: Tensor = pi * prob_gauss
        prob_mdn = torch.sum(prob_mdn, dim=1)
        return prob_mdn

    def _get_gaussian_probability(
        self,
        mu: Tensor,
        sigma: Tensor,
        pos_target: Tensor,
    ) -> Tensor:
        """Compute per-component diagonal 3D Gaussian pdf values.

        This computes, for each sample and mixture component:
        `prod_{d in {x,y,z}} Normal(pos_target[d] | mu[d], sigma[d])`.

        Args:
            mu: Component means, shape `(N, n_component, 3)`.
            sigma: Component per-axis standard deviations, shape
                `(N, n_component, 3)`.
            pos_target: Target coordinates, shape `(N, 3)`.

        Returns:
            Per-component densities, shape `(N, n_component)`.
        """
        # mu: (N, n_component, 3)
        # sigma: (N, n_component, 3)
        # pos_target: (N, 3)
        target: Tensor = pos_target.unsqueeze(1).expand_as(mu)
        errors: Tensor = target - mu
        sigma = sigma.clamp_min(EPS_SIGMA)
        p: Tensor = GAUSSIAN_COEF * torch.exp(-0.5 * (errors / sigma) ** 2) / sigma
        p = torch.prod(p, dim=2)
        return p  # (N, n_component)

    def sample_batch(
        self,
        mu: Tensor,
        sigma: Tensor,
        pi: Tensor,
        num: int,
    ) -> Tensor:
        """Sample coordinates from a batch of mixtures.

        Sampling is performed by first drawing component indices from `pi` using
        `torch.multinomial`, then drawing from the corresponding Gaussian via
        `torch.normal(mu_k, sigma_k)`.

        Args:
            mu: Component means, shape `(N_batch, n_component, 3)`.
            sigma: Component per-axis standard deviations, shape
                `(N_batch, n_component, 3)`.
            pi: Mixture weights, shape `(N_batch, n_component)`.
            num: Number of samples to draw per batch element.

        Returns:
            Sampled coordinates, shape `(N_batch, num, 3)`.
        """
        # mu: (N_batch, n_cat, 3)
        # sigma: (N_batch, n_cat, 3)
        # pi: (N_batch, n_cat)
        # Returns: (N_batch, num, 3)
        index_cats: Tensor = torch.multinomial(pi, num, replacement=True)  # (N_batch, num)
        index_batch: Tensor = torch.arange(len(mu)).unsqueeze(-1).expand(-1, num)  # (N_batch, num)
        mu_sample: Tensor = mu[index_batch, index_cats]  # (N_batch, num, 3)
        sigma_sample: Tensor = sigma[index_batch, index_cats]
        values: Tensor = torch.normal(mu_sample, sigma_sample)  # (N_batch, num, 3)
        return values

    def get_maximum(
        self,
        mu: Tensor,
        sigma: Tensor,  # noqa: ARG002
        pi: Tensor,  # noqa: ARG002
    ) -> Tensor:
        """Return a simple "maximum" estimate for each component.

        This helper currently returns `mu` directly (the per-component means)
        and ignores `sigma` and `pi`. It can be used as a deterministic
        representative location for each mixture component.

        Args:
            mu: Component means, shape `(N_batch, n_component, 3)`.
            sigma: Component standard deviations (unused).
            pi: Mixture weights (unused).

        Returns:
            Component means, shape `(N_batch, n_component, 3)`.
        """
        # mu: (N_batch, n_cat, 3)
        # sigma: (N_batch, n_cat, 3)
        # pi: (N_batch, n_cat)
        # Returns: (N_batch, n_cat, 3)
        return mu
