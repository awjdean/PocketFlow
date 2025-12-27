"""Focal/frontier prediction head for the GDBP model.

This module defines :class:`~pocket_flow.gdbp_model.focal_net.FocalNet`, a small
MLP-like head that maps paired scalar/vector node features to a single scalar
logit per selected node.

In PocketFlow training this head is used for "focal" supervision, e.g. predicting
which context nodes are on the ligand frontier or which apo protein surface nodes
are candidate focal sites.

The head operates on the same paired features used throughout the GDBP model:

- **Scalar features**: invariant channels with shape ``(N, F_sca)``.
- **Vector features**: equivariant channels with shape ``(N, F_vec, 3)``.

Only a subset of nodes is scored in
:meth:`~pocket_flow.gdbp_model.focal_net.FocalNet.forward`, selected by indexing
tensors (typically a 1D LongTensor of node indices).
"""

from typing import override

from torch import Tensor
from torch.nn import Module, Sequential

from .layers import GDBLinear, GDBPerceptronVN
from .types import BottleneckSpec, ScalarVectorFeatures


class FocalNet(Module):
    """Predict a per-node focal logit from scalar/vector features.

    Architecture:
        ``GDBPerceptronVN -> GDBLinear`` operating on a ``(scalar, vector)``
        feature pair. The final output is taken from the *scalar* branch only
        and has width 1.

    Shape conventions:
        - Input scalar features: ``(N, in_sca)``
        - Input vector features: ``(N, in_vec, 3)``
        - Selected node count: ``M`` (via ``idx_ligand``)
        - Output logits: ``(M, 1)``

    Args:
        in_sca: Number of input scalar channels (``in_sca``).
        in_vec: Number of input vector channels (``in_vec``).
        hidden_dim_sca: Hidden scalar width used inside the head.
        hidden_dim_vec: Hidden vector width used inside the head.
        bottleneck:
            Bottleneck specification forwarded to internal projections.
            See :data:`~pocket_flow.gdbp_model.types.BottleneckSpec`.
        use_conv1d: Compatibility flag forwarded to internal layers.
    """

    net: Sequential

    def __init__(
        self,
        in_sca: int,
        in_vec: int,
        hidden_dim_sca: int,
        hidden_dim_vec: int,
        bottleneck: BottleneckSpec = 1,
        use_conv1d: bool = False,
    ) -> None:
        super().__init__()
        self.net = Sequential(
            GDBPerceptronVN(
                in_sca, in_vec, hidden_dim_sca, hidden_dim_vec, bottleneck=bottleneck, use_conv1d=use_conv1d
            ),
            GDBLinear(hidden_dim_sca, hidden_dim_vec, 1, 1, bottleneck=bottleneck, use_conv1d=use_conv1d),
        )

    @override
    def forward(self, h_att: ScalarVectorFeatures, idx_ligand: Tensor) -> Tensor:
        """Score a subset of nodes with a focal logit.

        Args:
            h_att:
                Context node features ``(scalar, vector)`` with shapes
                ``(N, in_sca)`` and ``(N, in_vec, 3)``.
            idx_ligand:
                Indexing tensor used to select which nodes to score. In typical
                usage this is a 1D integer tensor of length ``M`` containing node
                indices into ``h_att`` (e.g. ligand-context indices), but any
                advanced-indexing tensor supported by PyTorch for the first
                dimension (e.g. a boolean mask) will work.

        Returns:
            A tensor of shape ``(M, 1)`` containing unnormalised logits suitable
            for `torch.nn.functional.binary_cross_entropy_with_logits`.
        """
        h_att_ligand: ScalarVectorFeatures = (h_att[0][idx_ligand], h_att[1][idx_ligand])
        pred: ScalarVectorFeatures = self.net(h_att_ligand)
        return pred[0]
