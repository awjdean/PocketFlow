"""Core scalar/vector layers for the GDBP model.

This module defines the building blocks used across the encoder and the normalising
flows in `pocket_flow.gdbp_model`. The model uses *paired* features:

- **Scalar features**: invariant channels with shape `(N, F_sca)`.
- **Vector features**: equivariant channels with shape `(N, F_vec, 3)`, where the
  last dimension corresponds to a 3D Cartesian vector.

Throughout this file, such features are passed as a tuple `(scalar, vector)`.

Notes:
    - Many layers are "vector-neuron" style layers: they apply learned linear maps
      across channels while preserving the 3D vector structure.
    - Several modules operate on *messages* indexed by `edge_index`-like tensors
      and use `torch_scatter` for segment softmax/sums.
"""

from math import pi as PI
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm, LeakyReLU, Linear, Module
from torch_scatter import scatter_softmax, scatter_sum

from pocket_flow.gdbp_model.net_utils import EdgeExpansion, GaussianSmearing, Rescale
from pocket_flow.gdbp_model.types import BottleneckSpec, ScalarVectorFeatures

EPS: float = 1e-6


class VNLinear(Module):
    """Linear projection for vector features.

    This layer applies a `torch.nn.Linear` over the *channel* dimension of vector
    features shaped `(..., C, 3)`, preserving the trailing 3D coordinate axis.

    In other words, it maps `C_in -> C_out` while keeping the per-channel vector
    structure equivariant to rotations of the last axis.

    Args:
        in_channels: Number of input vector channels (`C_in`).
        out_channels: Number of output vector channels (`C_out`).
        bias: Whether to include a bias term in the underlying linear map.
    """

    map_to_feat: Linear

    def __init__(self, in_channels: int, out_channels: int, *, bias: bool = True) -> None:
        super().__init__()
        self.map_to_feat = Linear(in_channels, out_channels, bias=bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Project vector channels.

        Args:
            x: Vector features of shape `(..., in_channels, 3)`.

        Returns:
            Vector features of shape `(..., out_channels, 3)`.
        """
        return self.map_to_feat(x.transpose(-2, -1)).transpose(-2, -1)


class VNLeakyReLU(Module):
    """LeakyReLU-like nonlinearity for vector features.

    This implements the "Vector Neuron" directional nonlinearity: for each
    channel, it learns a direction `d` and reflects components of `x` that have
    negative dot product with `d`. The behavior is analogous to LeakyReLU, but
    operates on 3D vectors.

    Args:
        in_channels: Number of input vector channels.
        share_nonlinearity: If True, use a single learned direction shared across
            channels (maps channels to 1); otherwise learn one direction per
            channel.
        negative_slope: Multiplicative slope for the "negative" branch.
    """

    map_to_dir: Linear
    negative_slope: float

    def __init__(
        self,
        in_channels: int,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        out_channels = 1 if share_nonlinearity else in_channels
        self.map_to_dir = Linear(in_channels, out_channels, bias=False)
        self.negative_slope = negative_slope

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the vector nonlinearity.

        Args:
            x: Vector features of shape `(..., in_channels, 3)`.

        Returns:
            Vector features with the same shape as `x`.
        """
        d = self.map_to_dir(x.transpose(-2, -1)).transpose(-2, -1)
        dotprod = (x * d).sum(-1, keepdim=True)
        mask = (dotprod >= 0).to(x.dtype)
        d_norm_sq = (d * d).sum(-1, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class GDBLinear(Module):
    """Scalar/vector linear mixing with scalar-to-vector gating.

    `GDBLinear` consumes a `(scalar, vector)` feature pair and produces a new pair.
    Vector features are projected via `VNLinear`; scalar features are produced
    from a concatenation of:

    - the L2 norms of intermediate vector channels, and
    - a linear projection of the input scalar channels.

    The produced scalar output is then used to gate the produced vector channels
    via a learned sigmoid gate.

    Shape conventions:
        - `feat_scalar`: `(N, in_scalar)`
        - `feat_vector`: `(N, in_vector, 3)`
        - output scalar: `(N, out_scalar)`
        - output vector: `(N, out_vector, 3)`

    Args:
        in_scalar: Number of input scalar channels.
        in_vector: Number of input vector channels.
        out_scalar: Number of output scalar channels.
        out_vector: Number of output vector channels.
        bottleneck:
            Bottleneck factor(s) controlling intermediate widths.
            If an `int`, the same factor is used for scalar and vector branches.
            If a `(sca_bottleneck, vec_bottleneck)` tuple, the two branches use
            different factors.
        use_conv1d:
            Kept for weight-loading backward compatibility. This flag is stored
            but does not affect computation in this implementation.

    Raises:
        AssertionError: If `in_scalar` is not divisible by `sca_bottleneck` or
            `in_vector` is not divisible by `vec_bottleneck` when the respective
            bottleneck is > 1.
    """

    sca_hidden_dim: int
    hidden_dim: int
    out_vector: int
    lin_vector: VNLinear
    lin_vector2: VNLinear
    use_conv1d: bool
    scalar_to_vector_gates: Linear
    lin_scalar_1: Linear
    lin_scalar_2: Linear

    def __init__(
        self,
        in_scalar: int,
        in_vector: int,
        out_scalar: int,
        out_vector: int,
        bottleneck: BottleneckSpec = (1, 1),
        # Note: use_conv1d does nothing but must be kept for model weight loading compatibility
        use_conv1d: bool = False,  # noqa: ARG002
    ) -> None:
        super().__init__()

        if isinstance(bottleneck, int):
            sca_bottleneck = bottleneck
            vec_bottleneck = bottleneck
        else:
            sca_bottleneck, vec_bottleneck = bottleneck

        assert in_vector % vec_bottleneck == 0, (
            f"in_vector ({in_vector}) must be divisible by vec_bottleneck ({vec_bottleneck})"
        )
        assert in_scalar % sca_bottleneck == 0, (
            f"in_scalar ({in_scalar}) must be divisible by sca_bottleneck ({sca_bottleneck})"
        )

        if sca_bottleneck > 1:
            self.sca_hidden_dim = in_scalar // sca_bottleneck
        else:
            self.sca_hidden_dim = max(in_vector, out_vector)

        if vec_bottleneck > 1:
            self.hidden_dim = in_vector // vec_bottleneck
        else:
            self.hidden_dim = max(in_vector, out_vector)

        self.out_vector = out_vector
        self.lin_vector = VNLinear(in_vector, self.hidden_dim, bias=False)
        self.lin_vector2 = VNLinear(self.hidden_dim, out_vector, bias=False)

        self.use_conv1d = use_conv1d
        self.scalar_to_vector_gates = Linear(out_scalar, out_vector)
        self.lin_scalar_1 = Linear(in_scalar, self.sca_hidden_dim, bias=False)
        self.lin_scalar_2 = Linear(self.hidden_dim + self.sca_hidden_dim, out_scalar, bias=False)

    @override
    def forward(self, features: ScalarVectorFeatures) -> ScalarVectorFeatures:
        """Transform a scalar/vector feature pair.

        Args:
            features: Tuple `(feat_scalar, feat_vector)` with shapes
                `(N, in_scalar)` and `(N, in_vector, 3)`.

        Returns:
            Tuple `(out_scalar, out_vector)` with shapes `(N, out_scalar)` and
            `(N, out_vector, 3)`.
        """
        feat_scalar, feat_vector = features
        feat_vector_inter = self.lin_vector(feat_vector)
        feat_vector_norm = torch.norm(feat_vector_inter, p=2, dim=-1)
        z_sca = self.lin_scalar_1(feat_scalar)
        feat_scalar_cat = torch.cat([feat_vector_norm, z_sca], dim=-1)

        out_scalar = self.lin_scalar_2(feat_scalar_cat)
        gating = torch.sigmoid(self.scalar_to_vector_gates(out_scalar)).unsqueeze(-1)

        out_vector = self.lin_vector2(feat_vector_inter)
        out_vector = gating * out_vector

        return out_scalar, out_vector


class GDBPerceptronVN(Module):
    """A 2-branch perceptron over scalar/vector features.

    This layer applies:

    1) a `GDBLinear` projection producing `(out_scalar, out_vector)`, then
    2) a scalar `LeakyReLU` and a vector `VNLeakyReLU` nonlinearity.

    Args:
        in_scalar: Number of input scalar channels.
        in_vector: Number of input vector channels.
        out_scalar: Number of output scalar channels.
        out_vector: Number of output vector channels.
        bottleneck: Bottleneck specification forwarded to `GDBLinear`.
        use_conv1d: Compatibility flag forwarded to `GDBLinear`.
    """

    gb_linear: GDBLinear
    act_sca: LeakyReLU
    act_vec: VNLeakyReLU

    def __init__(
        self,
        in_scalar: int,
        in_vector: int,
        out_scalar: int,
        out_vector: int,
        bottleneck: BottleneckSpec = 1,
        use_conv1d: bool = False,
    ) -> None:
        super().__init__()
        self.gb_linear = GDBLinear(
            in_scalar, in_vector, out_scalar, out_vector, bottleneck=bottleneck, use_conv1d=use_conv1d
        )
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(out_vector)

    @override
    def forward(self, x: ScalarVectorFeatures) -> ScalarVectorFeatures:
        """Apply the perceptron.

        Args:
            x: `(scalar, vector)` input pair.

        Returns:
            `(scalar, vector)` output pair with channels defined by the module
            construction.
        """
        sca, vec = self.gb_linear(x)
        vec = self.act_vec(vec)
        sca = self.act_sca(sca)
        return sca, vec


class ST_GDBP_Exp(Module):
    """Scale/translate predictor for affine flows (GDBP-based).

    This module is used as a flow layer in the atom/bond normalising flows. It
    predicts per-dimension log-scale `s` and translation `t` from conditioning
    features.

    Internally:
        - Applies a first `GDBLinear` + nonlinearities to produce an intermediate
          representation.
        - Applies a second `GDBLinear` to produce `2 * out_scalar` scalar outputs,
          which are split into `(s, t)`.
        - Applies `tanh` and a learned `Rescale` to `s` to stabilize training.

    Args:
        in_scalar: Number of input scalar channels for conditioning.
        in_vector: Number of input vector channels for conditioning.
        out_scalar: Dimensionality of the predicted scale/shift (per sample).
        out_vector: Number of output vector channels from the internal projection.
            (This is mainly used to preserve a vector stream for downstream flow
            layers.)
        bottleneck: Bottleneck specification forwarded to internal `GDBLinear`s.
        use_conv1d: Compatibility flag forwarded to internal `GDBLinear`s.
    """

    in_scalar: int
    in_vector: int
    out_scalar: int
    out_vector: int
    gb_linear1: GDBLinear
    gb_linear2: GDBLinear
    act_sca: nn.Tanh
    act_vec: VNLeakyReLU
    rescale: Rescale

    def __init__(
        self,
        in_scalar: int,
        in_vector: int,
        out_scalar: int,
        out_vector: int,
        bottleneck: BottleneckSpec = 1,
        use_conv1d: bool = False,
    ) -> None:
        super().__init__()
        self.in_scalar = in_scalar
        self.in_vector = in_vector
        self.out_scalar = out_scalar
        self.out_vector = out_vector

        self.gb_linear1 = GDBLinear(
            in_scalar, in_vector, in_scalar, in_vector, bottleneck=bottleneck, use_conv1d=use_conv1d
        )
        self.gb_linear2 = GDBLinear(
            in_scalar, in_vector, out_scalar * 2, out_vector, bottleneck=bottleneck, use_conv1d=use_conv1d
        )
        self.act_sca = nn.Tanh()
        self.act_vec = VNLeakyReLU(out_vector)
        self.rescale = Rescale()

    @override
    def forward(self, x: ScalarVectorFeatures) -> tuple[Tensor, Tensor]:
        """Predict log-scale and translation.

        Args:
            x: Conditioning features `(scalar, vector)`.

        Returns:
            `(s, t)` where both tensors have shape `(N, out_scalar)`.

        Notes:
            The returned `s` is intended to be used as a log-scale (i.e. callers
            typically apply `exp(s)`), but this module itself does not exponentiate.
        """
        sca, vec = self.gb_linear1(x)
        sca = self.act_sca(sca)
        vec = self.act_vec(vec)
        sca, vec = self.gb_linear2((sca, vec))
        s = sca[:, : self.out_scalar]
        t = sca[:, self.out_scalar :]
        s = self.rescale(torch.tanh(s))
        return s, t


class MessageAttention(Module):
    """Multi-head attention for message aggregation over indexed groups.

    This module aggregates per-message "query" features into per-node outputs,
    attending messages that share the same destination index `edge_index_i`.

    The attention score is computed separately for scalar and vector streams:
      - scalar score: dot product over scalar channel dimension
      - vector score: dot product over both vector channel and xyz dimensions

    Segment-wise normalization and aggregation use `torch_scatter`:
      - `scatter_softmax(..., edge_index_i)` normalizes over messages per node
      - `scatter_sum(..., edge_index_i)` aggregates into `N` node outputs

    Args:
        in_sca: Scalar channel count for `x`.
        in_vec: Vector channel count for `x`.
        out_sca: Scalar channel count for the output.
        out_vec: Vector channel count for the output.
        bottleneck: Bottleneck specification for internal projections.
        num_heads: Number of attention heads. All scalar/vector channel counts
            must be divisible by `num_heads`.
        use_conv1d: Compatibility flag forwarded to internal `GDBLinear`s.
    """

    num_heads: int
    lin_v: GDBLinear
    lin_k: GDBLinear

    def __init__(
        self,
        in_sca: int,
        in_vec: int,
        out_sca: int,
        out_vec: int,
        bottleneck: BottleneckSpec = 1,
        num_heads: int = 1,
        use_conv1d: bool = False,
    ) -> None:
        super().__init__()

        assert in_sca % num_heads == 0 and in_vec % num_heads == 0
        assert out_sca % num_heads == 0 and out_vec % num_heads == 0

        self.num_heads = num_heads
        self.lin_v = GDBLinear(in_sca, in_vec, out_sca, out_vec, bottleneck=bottleneck, use_conv1d=use_conv1d)
        self.lin_k = GDBLinear(in_sca, in_vec, out_sca, out_vec, bottleneck=bottleneck, use_conv1d=use_conv1d)

    @override
    def forward(
        self,
        x: ScalarVectorFeatures,
        query: ScalarVectorFeatures,
        edge_index_i: Tensor,
    ) -> ScalarVectorFeatures:
        """Aggregate `query` messages into node outputs.

        Args:
            x:
                Node features `(scalar, vector)` for `N` nodes:
                  - scalar: `(N, in_sca)`
                  - vector: `(N, in_vec, 3)`
            query:
                Message/query features for `N_msg` messages:
                  - scalar: `(N_msg, out_sca)`
                  - vector: `(N_msg, out_vec, 3)`
                (The channel sizes must match the module configuration so they
                can be reshaped into `num_heads`.)
            edge_index_i:
                Destination node index for each message, shape `(N_msg,)`. Values
                must be in `[0, N)`.

        Returns:
            Aggregated `(out_sca, out_vec)` node features with shapes
            `(N, out_sca)` and `(N, out_vec, 3)`.
        """
        N = x[0].size(0)
        N_msg = len(edge_index_i)
        msg = [query[0].view(N_msg, self.num_heads, -1), query[1].view(N_msg, self.num_heads, -1, 3)]
        k = self.lin_k(x)
        x_i = [
            k[0][edge_index_i].view(N_msg, self.num_heads, -1),
            k[1][edge_index_i].view(N_msg, self.num_heads, -1, 3),
        ]
        alpha = [
            (msg[0] * x_i[0]).sum(-1),
            (msg[1] * x_i[1]).sum(-1).sum(-1),
        ]
        alpha = [
            scatter_softmax(alpha[0], edge_index_i, dim=0),
            scatter_softmax(alpha[1], edge_index_i, dim=0),
        ]
        msg = [
            (alpha[0].unsqueeze(-1) * msg[0]).view(N_msg, -1),
            (alpha[1].unsqueeze(-1).unsqueeze(-1) * msg[1]).view(N_msg, -1, 3),
        ]
        sca_msg = scatter_sum(msg[0], edge_index_i, dim=0, dim_size=N)
        vec_msg = scatter_sum(msg[1], edge_index_i, dim=0, dim_size=N)
        root_sca, root_vec = self.lin_v(x)
        out_sca = sca_msg + root_sca
        out_vec = vec_msg + root_vec
        return out_sca, out_vec


class MessageModule(Module):
    """Edge-conditioned message construction between node and edge features.

    This module combines projected node features (indexed per edge) with edge
    features to produce a new edge message in the scalar/vector paired format.

    Optionally, when `annealing=True`, it applies a cosine cutoff envelope based
    on inter-node distance `dist_ij` (commonly used as a training-time annealing
    or cutoff mechanism).

    Args:
        node_sca: Scalar channel count for node features.
        node_vec: Vector channel count for node features.
        edge_sca: Scalar channel count for edge features.
        edge_vec: Vector channel count for edge features.
        out_sca: Scalar channel count for output messages.
        out_vec: Vector channel count for output messages.
        bottleneck: Bottleneck specification for internal projections.
        cutoff: Distance cutoff (Å) used when `annealing=True`.
        use_conv1d: Compatibility flag forwarded to internal `GDBLinear`s.
    """

    cutoff: float
    node_gblinear: GDBLinear
    edge_gbp: GDBPerceptronVN
    sca_linear: Linear
    e2n_linear: Linear
    n2e_linear: Linear
    edge_vnlinear: VNLinear
    out_gblienar: GDBLinear

    def __init__(
        self,
        node_sca: int,
        node_vec: int,
        edge_sca: int,
        edge_vec: int,
        out_sca: int,
        out_vec: int,
        bottleneck: BottleneckSpec = 1,
        cutoff: float = 10.0,
        use_conv1d: bool = False,
    ) -> None:
        super().__init__()
        hid_sca, hid_vec = edge_sca, edge_vec
        self.cutoff = cutoff
        self.node_gblinear = GDBLinear(
            node_sca, node_vec, out_sca, out_vec, bottleneck=bottleneck, use_conv1d=use_conv1d
        )
        self.edge_gbp = GDBPerceptronVN(
            edge_sca, edge_vec, hid_sca, hid_vec, bottleneck=bottleneck, use_conv1d=use_conv1d
        )

        self.sca_linear = Linear(hid_sca, out_sca)
        self.e2n_linear = Linear(hid_sca, out_vec)
        self.n2e_linear = Linear(out_sca, out_vec)
        self.edge_vnlinear = VNLinear(hid_vec, out_vec)

        self.out_gblienar = GDBLinear(
            out_sca, out_vec, out_sca, out_vec, bottleneck=bottleneck, use_conv1d=use_conv1d
        )

    @override
    def forward(
        self,
        node_features: ScalarVectorFeatures,
        edge_features: ScalarVectorFeatures,
        edge_index_node: Tensor,
        dist_ij: Tensor | None = None,
        annealing: bool = False,
    ) -> ScalarVectorFeatures:
        """Construct edge messages.

        Args:
            node_features:
                Node features `(scalar, vector)` with shapes `(N_nodes, node_sca)`
                and `(N_nodes, node_vec, 3)`.
            edge_features:
                Edge features `(scalar, vector)` with shapes `(N_edges, edge_sca)`
                and `(N_edges, edge_vec, 3)`.
            edge_index_node:
                Node index per edge, shape `(N_edges,)`. This selects the
                per-edge node context as `node_features[..., edge_index_node]`.
            dist_ij:
                Optional per-edge distance tensor, shape `(N_edges,)`. Required
                when `annealing=True`.
            annealing:
                If True, apply a cosine cutoff envelope using `dist_ij` and
                `self.cutoff`.

        Returns:
            Edge message features `(scalar, vector)` with shapes `(N_edges, out_sca)`
            and `(N_edges, out_vec, 3)`.
        """
        node_scalar, node_vector = self.node_gblinear(node_features)
        node_scalar, node_vector = node_scalar[edge_index_node], node_vector[edge_index_node]
        edge_scalar, edge_vector = self.edge_gbp(edge_features)

        y_scalar = node_scalar * self.sca_linear(edge_scalar)
        y_node_vector = self.e2n_linear(edge_scalar).unsqueeze(-1) * node_vector
        y_edge_vector = self.n2e_linear(node_scalar).unsqueeze(-1) * self.edge_vnlinear(edge_vector)
        y_vector = y_node_vector + y_edge_vector

        output = self.out_gblienar((y_scalar, y_vector))

        if annealing:
            assert dist_ij is not None
            cutoff_coeff = 0.5 * (torch.cos(dist_ij * PI / self.cutoff) + 1.0)
            cutoff_coeff = cutoff_coeff * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
            output = (output[0] * cutoff_coeff.view(-1, 1), output[1] * cutoff_coeff.view(-1, 1, 1))
        return output


class AttentionInteractionBlockVN(Module):
    """Message passing + attention interaction block (vector neuron variant).

    Given node features and edge geometry/features, this block:

    1) expands edge distances and directions into scalar/vector edge features,
    2) computes edge-conditioned messages with `MessageModule`,
    3) aggregates messages back to nodes with `MessageAttention`,
    4) applies layer normalization and a final `GDBLinear` transform.

    Args:
        hidden_channels: Tuple `(F_sca, F_vec)` for node feature widths.
        edge_channels: Total edge channel count used for distance/feature expansion.
        num_edge_types: Number of discrete edge-type channels already present in
            `edge_feature` (these are concatenated after the distance expansion).
        bottleneck: Bottleneck specification for internal projections.
        num_heads: Number of attention heads.
        cutoff: Distance cutoff (Å) used for the Gaussian smearing and (optional)
            annealing envelope inside `MessageModule`.
        use_conv1d: Compatibility flag forwarded to internal `GDBLinear`s.
    """

    num_heads: int
    distance_expansion: GaussianSmearing
    vector_expansion: EdgeExpansion
    message_module: MessageModule
    msg_att: MessageAttention
    act_sca: LeakyReLU
    act_vec: VNLeakyReLU
    out_transform: GDBLinear
    layernorm_sca: LayerNorm
    layernorm_vec: LayerNorm

    def __init__(
        self,
        hidden_channels: tuple[int, int],
        edge_channels: int,
        num_edge_types: int,
        bottleneck: BottleneckSpec = 1,
        num_heads: int = 1,
        cutoff: float = 10.0,
        use_conv1d: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels - num_edge_types)
        self.vector_expansion = EdgeExpansion(edge_channels)

        self.message_module = MessageModule(
            hidden_channels[0],
            hidden_channels[1],
            edge_channels,
            edge_channels,
            hidden_channels[0],
            hidden_channels[1],
            bottleneck=bottleneck,
            cutoff=cutoff,
            use_conv1d=use_conv1d,
        )
        self.msg_att = MessageAttention(
            hidden_channels[0],
            hidden_channels[1],
            hidden_channels[0],
            hidden_channels[1],
            bottleneck=bottleneck,
            num_heads=num_heads,
            use_conv1d=use_conv1d,
        )

        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(hidden_channels[1], share_nonlinearity=True)
        self.out_transform = GDBLinear(
            hidden_channels[0],
            hidden_channels[1],
            hidden_channels[0],
            hidden_channels[1],
            use_conv1d=use_conv1d,
            bottleneck=bottleneck,
        )
        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])

    @override
    def forward(
        self,
        x: ScalarVectorFeatures,
        edge_index: Tensor,
        edge_feature: Tensor,
        edge_vector: Tensor,
        edge_dist: Tensor,
        annealing: bool = False,
    ) -> ScalarVectorFeatures:
        """Update node features given edge information.

        Args:
            x:
                Node features `(scalar, vector)` with shapes `(N, F_sca)` and
                `(N, F_vec, 3)`.
            edge_index:
                Edge indices as a `(2, E)` tensor where `row, col = edge_index`.
                This block uses `col` as the per-edge node index for message
                construction and `row` as the destination index for aggregation.
            edge_feature:
                Per-edge scalar features to concatenate after distance expansion,
                shape `(E, num_edge_types)`.
            edge_vector:
                Raw per-edge direction vectors, shape `(E, 3)`.
            edge_dist:
                Per-edge distances, shape `(E,)`.
            annealing:
                If True, apply the cosine cutoff envelope inside `MessageModule`.

        Returns:
            Updated node features `(scalar, vector)` with shapes `(N, F_sca)` and
            `(N, F_vec, 3)`.
        """
        row, col = edge_index

        edge_sca_feat = torch.cat([self.distance_expansion(edge_dist), edge_feature], dim=-1)
        edge_vec_feat = self.vector_expansion(edge_vector)

        msg_j_sca, msg_j_vec = self.message_module(
            x, (edge_sca_feat, edge_vec_feat), col, edge_dist, annealing=annealing
        )
        out_sca, out_vec = self.msg_att(x, (msg_j_sca, msg_j_vec), row)
        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out = self.out_transform((self.act_sca(out_sca), self.act_vec(out_vec)))
        return out


class AttentionBias(Module):
    """Compute additive attention biases from triangle-edge geometry.

    This module builds a scalar/vector feature pair for each "triangle edge"
    (pair of nodes `(a, b)` provided by `tri_edge_index`) using:
      - distance RBF features of `||pos[a] - pos[b]||`, and
      - an external per-triangle-edge feature tensor `tri_edge_feat`.

    The result is projected to per-head biases via `GDBLinear`.

    Args:
        num_heads: Number of attention heads (output bias channels).
        hidden_channels: Tuple `(F_sca, F_vec)` used to size the internal feature
            expansions and the `GDBLinear`.
        cutoff: Maximum distance used for Gaussian smearing.
        num_bond_types: Number of bond types (used to infer edge-type channels as
            `num_bond_types + 1` including "no-bond").
        bottleneck: Bottleneck specification for the `GDBLinear`.
        use_conv1d: Compatibility flag forwarded to `GDBLinear`.
    """

    num_bond_types: int
    distance_expansion: GaussianSmearing
    vector_expansion: EdgeExpansion
    gblinear: GDBLinear

    def __init__(
        self,
        num_heads: int,
        hidden_channels: tuple[int, int],
        cutoff: float = 10.0,
        num_bond_types: int = 3,
        bottleneck: BottleneckSpec = 1,
        use_conv1d: bool = False,
    ) -> None:
        super().__init__()
        num_edge_types = num_bond_types + 1
        self.num_bond_types = num_bond_types
        self.distance_expansion = GaussianSmearing(
            stop=cutoff, num_gaussians=hidden_channels[0] - num_edge_types - 1
        )
        self.vector_expansion = EdgeExpansion(hidden_channels[1])
        self.gblinear = GDBLinear(
            hidden_channels[0],
            hidden_channels[1],
            num_heads,
            num_heads,
            bottleneck=bottleneck,
            use_conv1d=use_conv1d,
        )

    @override
    def forward(
        self,
        tri_edge_index: Tensor,
        tri_edge_feat: Tensor,
        pos_compose: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute per-head scalar/vector attention biases.

        Args:
            tri_edge_index: Index tensor of shape `(2, E_tri)` giving `(a, b)`
                pairs used to form vectors `pos[a] - pos[b]`.
            tri_edge_feat: Additional per-triangle-edge scalar features, shape
                `(E_tri, K)`.
            pos_compose: Node positions, shape `(N, 3)`.

        Returns:
            `(bias_sca, bias_vec)` where:
              - `bias_sca`: shape `(E_tri, num_heads)`
              - `bias_vec`: shape `(E_tri, num_heads)`, derived from the squared
                norm of per-head vector outputs.
        """
        node_a, node_b = tri_edge_index
        pos_a = pos_compose[node_a]
        pos_b = pos_compose[node_b]
        vector = pos_a - pos_b
        dist = torch.norm(vector, p=2, dim=-1)

        dist_feat = self.distance_expansion(dist)
        sca_feat = torch.cat([dist_feat, tri_edge_feat], dim=-1)
        vec_feat = self.vector_expansion(vector)
        output_sca, output_vec = self.gblinear((sca_feat, vec_feat))
        output_vec = (output_vec * output_vec).sum(-1)
        return output_sca, output_vec


class AttentionEdges(Module):
    """Edge-to-edge self-attention with learned geometric bias.

    This module performs attention among edges using:
      - queries/keys/values computed from `edge_attr` via `GDBLinear`, and
      - additive per-head biases from `AttentionBias` based on node geometry and
        triangle-edge features.

    The attention is computed over a sparse set of edge pairs given by
    `index_real_cps_edge_for_atten = (edge_i, edge_j)`, and aggregated back to a
    per-edge output using `scatter_softmax`/`scatter_sum` with `edge_i` as the
    segment index.

    Args:
        hidden_channels: Tuple `(F_sca, F_vec)` for input/output edge feature widths.
        key_channels: Tuple `(K_sca, K_vec)` for query/key widths.
        num_heads: Number of attention heads. Channel counts must be divisible by
            `num_heads`.
        num_bond_types: Number of bond types used in `AttentionBias`.
        bottleneck: Bottleneck specification for internal projections.
        use_conv1d: Compatibility flag forwarded to internal `GDBLinear`s.
    """

    hidden_channels: tuple[int, int]
    key_channels: tuple[int, int]
    num_heads: int
    q_lin: GDBLinear
    k_lin: GDBLinear
    v_lin: GDBLinear
    atten_bias_lin: AttentionBias
    layernorm_sca: LayerNorm
    layernorm_vec: LayerNorm

    def __init__(
        self,
        hidden_channels: tuple[int, int],
        key_channels: tuple[int, int],
        num_heads: int = 1,
        num_bond_types: int = 3,
        bottleneck: BottleneckSpec = 1,
        use_conv1d: bool = False,
    ) -> None:
        super().__init__()

        assert hidden_channels[0] % num_heads == 0 and hidden_channels[1] % num_heads == 0
        assert key_channels[0] % num_heads == 0 and key_channels[1] % num_heads == 0

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        self.q_lin = GDBLinear(
            hidden_channels[0],
            hidden_channels[1],
            key_channels[0],
            key_channels[1],
            bottleneck=bottleneck,
            use_conv1d=use_conv1d,
        )
        self.k_lin = GDBLinear(
            hidden_channels[0],
            hidden_channels[1],
            key_channels[0],
            key_channels[1],
            bottleneck=bottleneck,
            use_conv1d=use_conv1d,
        )
        self.v_lin = GDBLinear(
            hidden_channels[0],
            hidden_channels[1],
            hidden_channels[0],
            hidden_channels[1],
            bottleneck=bottleneck,
            use_conv1d=use_conv1d,
        )

        self.atten_bias_lin = AttentionBias(
            self.num_heads,
            hidden_channels,
            num_bond_types=num_bond_types,
            bottleneck=bottleneck,
            use_conv1d=use_conv1d,
        )

        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])

    @override
    def forward(
        self,
        edge_attr: ScalarVectorFeatures,
        edge_index: Tensor,
        pos_compose: Tensor,
        index_real_cps_edge_for_atten: tuple[Tensor, Tensor],
        tri_edge_index: Tensor,
        tri_edge_feat: Tensor,
    ) -> ScalarVectorFeatures:
        """Compute attended edge features.

        Args:
            edge_attr:
                Edge features `(scalar, vector)` with shapes `(N_edges, F_sca)`
                and `(N_edges, F_vec, 3)`.
            edge_index:
                Edge index tensor of shape `(2, N_edges)`. (This argument is kept
                for interface consistency; the attention computation uses the
                provided index lists instead.)
            pos_compose: Node positions, shape `(N_nodes, 3)`.
            index_real_cps_edge_for_atten:
                Tuple `(edge_i, edge_j)` of index tensors, each of shape
                `(N_attn,)`, defining which edge pairs participate in attention.
                The output is aggregated over pairs sharing the same `edge_i`.
            tri_edge_index:
                Triangle-edge node indices passed to `AttentionBias`, shape
                `(2, N_attn)` (or another shape consistent with `tri_edge_feat`).
            tri_edge_feat:
                Triangle-edge scalar features passed to `AttentionBias`, shape
                `(N_attn, K)`.

        Returns:
            A tuple `(scalar, vector)` containing the updated edge features with
            shapes `(N_edges, F_sca)` and `(N_edges, F_vec, 3)`.
        """
        scalar, _ = edge_attr
        N = scalar.size(0)

        h_queries = self.q_lin(edge_attr)
        h_queries = (
            h_queries[0].view(N, self.num_heads, -1),
            h_queries[1].view(N, self.num_heads, -1, 3),
        )
        h_keys = self.k_lin(edge_attr)
        h_keys = (
            h_keys[0].view(N, self.num_heads, -1),
            h_keys[1].view(N, self.num_heads, -1, 3),
        )
        h_values = self.v_lin(edge_attr)
        h_values = (
            h_values[0].view(N, self.num_heads, -1),
            h_values[1].view(N, self.num_heads, -1, 3),
        )

        index_edge_i_list, index_edge_j_list = index_real_cps_edge_for_atten

        atten_bias = self.atten_bias_lin(tri_edge_index, tri_edge_feat, pos_compose)

        queries_i = [h_queries[0][index_edge_i_list], h_queries[1][index_edge_i_list]]
        keys_j = [h_keys[0][index_edge_j_list], h_keys[1][index_edge_j_list]]

        qk_ij = [
            (queries_i[0] * keys_j[0]).sum(-1),
            (queries_i[1] * keys_j[1]).sum(-1).sum(-1),
        ]

        alpha = [atten_bias[0] + qk_ij[0], atten_bias[1] + qk_ij[1]]
        alpha = [
            scatter_softmax(alpha[0], index_edge_i_list, dim=0),
            scatter_softmax(alpha[1], index_edge_i_list, dim=0),
        ]

        values_j = [h_values[0][index_edge_j_list], h_values[1][index_edge_j_list]]
        num_attens = len(index_edge_j_list)
        output = [
            scatter_sum(
                (alpha[0].unsqueeze(-1) * values_j[0]).view(num_attens, -1),
                index_edge_i_list,
                dim=0,
                dim_size=N,
            ),
            scatter_sum(
                (alpha[1].unsqueeze(-1).unsqueeze(-1) * values_j[1]).view(num_attens, -1, 3),
                index_edge_i_list,
                dim=0,
                dim_size=N,
            ),
        ]

        output = [edge_attr[0] + output[0], edge_attr[1] + output[1]]
        output = [self.layernorm_sca(output[0]), self.layernorm_vec(output[1])]

        return (output[0], output[1])
