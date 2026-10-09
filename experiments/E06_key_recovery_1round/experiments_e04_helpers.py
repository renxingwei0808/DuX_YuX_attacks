"""Re-export the affine-subspace helpers of experiments/E04_zero_sum_F2n/run.py
so that the key-recovery scripts can build the same structures."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "E04_zero_sum_F2n"))
from run import gf2_rank, random_basis, span_values  # noqa: F401,E402
