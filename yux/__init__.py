"""Reference + vectorised model of the YuX block cipher family for cryptanalysis.

Liu, Wang, Zhang, Wen, Zhang, Yu, "YuX: Finite Field Multiplication Based
Block Ciphers for Efficient FHE Evaluation", IEEE TIT 70(5), 2024.
Instances: Yu2X-8, Yu2X-16 (F_{2^n}) and YupX-65537 (F_p), plus three toys.
"""
from .cipher import YuX
from .params import INSTANCES, FIELDS, ROT_INV, ROT_FWD_BIN, V_P_65537
from .linear import LinearLayer

__all__ = ["YuX", "LinearLayer", "INSTANCES", "FIELDS",
           "ROT_INV", "ROT_FWD_BIN", "V_P_65537"]
