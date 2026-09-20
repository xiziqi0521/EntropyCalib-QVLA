"""Entropy-guided calibration sampling for W4A4 quantization of VLA policies.

This package ports the architecture-agnostic core of DemoSpeedup's
entropy/redundancy signal (github.com/lingxiao-guo/DemoSpeedup) and applies it
to Omega-QVLA-official-baseline's GPTQ/DuQuant calibration sampling. Neither
source repo is modified; both are imported read-only via `sys.path`.
"""

from .entropy_utils import KDE
from .labeling import label_precision_segments
from .multi_sample_policy import compute_entropy_trace
from .multi_sample_policy_pi05 import compute_entropy_trace_pi05
from .sample_selection import entropy_guided_sample_indices

__all__ = [
    "KDE",
    "label_precision_segments",
    "compute_entropy_trace",
    "compute_entropy_trace_pi05",
    "entropy_guided_sample_indices",
]

# Both multi_sample_policy variants take the policy/loader functions as
# plain arguments (dependency injection) rather than importing Omega-QVLA
# or openpi themselves, so importing this package needs only torch/numpy/
# scipy/hdbscan regardless of which policy backend you actually use.
