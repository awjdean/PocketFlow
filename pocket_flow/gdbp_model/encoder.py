"""Context encoder for the GDBP model.

This module defines :class:`~pocket_flow.gdbp_model.encoder.ContextEncoder`, a
stacked message-passing encoder operating on paired scalar/vector features:

- **Scalar features**: invariant channels with shape ``(N, F_sca)``.
- **Vector features**: equivariant channels with shape ``(N, F_vec, 3)``.

The encoder applies a sequence of
:class:`~pocket_flow.gdbp_model.layers.AttentionInteractionBlockVN` interaction
blocks and accumulates their outputs as residual updates.

Edge indexing convention:
    ``edge_index`` is a ``(2, E)`` tensor. This code treats ``edge_index[0]`` as
    the **destination** ("row") node indices and ``edge_index[1]`` as the
    **source/context** ("col") node indices. The per-edge direction vector is
    computed as ``pos[row] - pos[col]`` (pointing from source to destination).
"""

from typing import override

import torch
from torch import Tensor, nn

from pocket_flow.gdbp_model.layers import AttentionInteractionBlockVN
from pocket_flow.gdbp_model.types import BottleneckSpec, ScalarVectorFeatures


class ContextEncoder(nn.Module):
    """Stacked attention-based interaction encoder over a context graph.

    This encoder repeatedly updates node features using geometric message passing
    conditioned on 3D coordinates and per-edge features.

    Architecture:
        For ``num_interactions`` steps, run an
        :class:`~pocket_flow.gdbp_model.layers.AttentionInteractionBlockVN` and
        add its output to the running node state (residual accumulation).

    Shape conventions:
        - ``node_attr[0]`` (scalar): ``(N, F_sca)``
        - ``node_attr[1]`` (vector): ``(N, F_vec, 3)``
        - ``pos``: ``(N, 3)``
        - ``edge_index``: ``(2, E)``
        - ``edge_feature``: ``(E, num_edge_types)``

    Args:
        hidden_channels: Tuple ``(F_sca, F_vec)`` giving node feature widths.
        edge_channels: Total edge channel count used inside interaction blocks.
        num_edge_types: Number of per-edge scalar feature channels supplied via
            ``edge_feature``.
        key_channels:
            Kept for historical/config compatibility. This module does not use
            it directly.
        num_heads: Number of attention heads used in each interaction block.
        num_interactions: Number of stacked interaction blocks.
        k:
            Kept for historical/config compatibility. This module stores it but
            does not use it directly.
        cutoff:
            Distance cutoff (Å) used by interaction blocks for distance
            expansion and optional cosine cutoff annealing.
        bottleneck: Bottleneck specification forwarded to internal projections.
        use_conv1d: Compatibility flag forwarded to internal layers.
    """

    hidden_channels: tuple[int, int]
    edge_channels: int
    key_channels: int
    num_heads: int
    num_interactions: int
    k: int
    cutoff: float
    interactions: nn.ModuleList

    def __init__(
        self,
        hidden_channels: tuple[int, int] = (256, 64),
        edge_channels: int = 64,
        num_edge_types: int = 4,
        key_channels: int = 128,
        num_heads: int = 4,
        num_interactions: int = 6,
        k: int = 32,
        cutoff: float = 10.0,
        bottleneck: BottleneckSpec = 1,
        use_conv1d: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels  # stored for compatibility; unused by this module
        self.num_heads = num_heads  # stored for introspection; used to construct blocks
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlockVN(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                num_edge_types=num_edge_types,
                num_heads=num_heads,
                cutoff=cutoff,
                bottleneck=bottleneck,
                use_conv1d=use_conv1d,
            )
            self.interactions.append(block)

    @property
    def out_sca(self) -> int:
        """Number of scalar output channels (``F_sca``)."""
        return self.hidden_channels[0]

    @property
    def out_vec(self) -> int:
        """Number of vector output channels (``F_vec``)."""
        return self.hidden_channels[1]

    @override
    def forward(
        self,
        node_attr: ScalarVectorFeatures,
        pos: Tensor,
        edge_index: Tensor,
        edge_feature: Tensor,
        *,
        annealing: bool = True,
    ) -> ScalarVectorFeatures:
        """Encode context node features via stacked geometric interactions.

        Args:
            node_attr:
                Node features ``(scalar, vector)`` with shapes ``(N, F_sca)`` and
                ``(N, F_vec, 3)``.
            pos: Node coordinates with shape ``(N, 3)``.
            edge_index:
                Edge indices with shape ``(2, E)`` following the convention
                described in the module docstring: ``edge_index[0]`` are
                destination indices and ``edge_index[1]`` are source indices.
            edge_feature:
                Per-edge scalar features of shape ``(E, num_edge_types)``. These
                are concatenated with distance-expanded features inside each
                interaction block.
            annealing:
                If True, enable the cosine cutoff envelope inside each
                interaction block's message module. When False, messages are not
                distance-annealed (but distance features are still computed).

        Returns:
            Updated node features ``(scalar, vector)`` with shapes ``(N, F_sca)``
            and ``(N, F_vec, 3)``.
        """
        edge_vector: Tensor = pos[edge_index[0]] - pos[edge_index[1]]
        edge_dist: Tensor = torch.norm(edge_vector, dim=-1, p=2)
        h: list[Tensor] = list(node_attr)
        for interaction in self.interactions:
            delta_h: ScalarVectorFeatures = interaction(
                h, edge_index, edge_feature, edge_vector, edge_dist, annealing=annealing
            )
            h[0] = h[0] + delta_h[0]
            h[1] = h[1] + delta_h[1]
        return (h[0], h[1])
