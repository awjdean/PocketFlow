from torch import Tensor

# Type alias for scalar/vector feature tuple: (scalar: [N, F_sca], vector: [N, F_vec, 3])
type ScalarVectorFeatures = tuple[Tensor, Tensor]

type BottleneckSpec = int | tuple[int, int]
