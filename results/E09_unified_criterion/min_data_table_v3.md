# E09 · O7 框架内的最小数据表 v3

v2（`min_data_table_v2.md`）**原样保留**；v3 把搜索扩到**全部 15 个平衡位置类**，
并为每个类打印 `tools/cheap_rows.py` 的 dim K 与它对密钥恢复的可用性。
搜索空间与 v2 相同：0 或 1 个自由块（O11）× 其余字的每个子集 × 每层，
活跃集按块轮转规范化；特征 2 允许混合子空间维数（凸性 ⇒ 极端分配最优）。

## 1. 与早先手工搜索表的逐格对照

| 格 | 备忘 | v3 | dim K | 可用性 | 一致 |
|---|---|---|---|---|---|
| Yu2X-8 / layer 6 / 1100 | 2^40.0 | 2^31 | 0 | distinguisher only | **不一致** |
| Yu2X-8 / layer 6 / 1111 | 2^56.0 | 2^56 | 4 | dim K = 4 + plain rows | ✅ |
| Yu2X-8 / layer 6 / 1110 | 2^32.0 | 2^32 | 3 | dim K = 3 | ✅ |
| Yu2X-8 / layer 7 / any class | unreachable (<= 6 extra words) | unreachable | — | — | ✅ |
| Yu2X-16 / layer 10 / 1100 | 2^80.0 | 2^63 | 0 | distinguisher only | **不一致** |
| Yu2X-16 / layer 10 / 1110 | 2^64.0 | 2^64 | 3 | dim K = 3 | ✅ |
| Yu2X-16 / layer 10 / 1111 | 2^112.0 | 2^112 | 4 | dim K = 4 + plain rows | ✅ |
| Yu2X-16 / layer 11 / any class | unreachable (<= 6 extra words) | unreachable | — | — | ✅ |
| YupX-65537 / layer 11 / any class | unreachable (<= 6 extra words) | unreachable | — | — | ✅ |
| DuX(65537) / layer 12 / any class | unreachable (<= 6 extra words) | unreachable | — | — | ✅ |

## 2. 每个实例、每层、每类的最小数据

只列出可达的格；`—` 的层在 O7 框架内不可达。

### DuX(65537) · 解密（CCA）

| 层 | 类 | dim K | 可用性 | 最小数据 | 自由块 | 额外字 | 维数 | 需要的次数 |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1 |
| 1 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1 |
| 1 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `0001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 1 |
| 1 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1 |
| 1 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `1001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 1 |
| 1 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `0101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 1 |
| 1 | `0011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 2 |
| 1 | `1110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `1101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 1 |
| 1 | `1011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 2 |
| 1 | `0111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 2 |
| 1 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 2 |
| 2 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 4 |
| 2 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5 |
| 2 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7 |
| 2 | `0001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 3 |
| 2 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5 |
| 2 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7 |
| 2 | `1001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 4 |
| 2 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7 |
| 2 | `0101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 5 |
| 2 | `0011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 7 |
| 2 | `1110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7 |
| 2 | `1101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 5 |
| 2 | `1011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 7 |
| 2 | `0111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 7 |
| 2 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 7 |
| 3 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 14 |
| 3 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 19 |
| 3 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26 |
| 3 | `0001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 12 |
| 3 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 19 |
| 3 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26 |
| 3 | `1001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 14 |
| 3 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26 |
| 3 | `0101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 19 |
| 3 | `0011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 26 |
| 3 | `1110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26 |
| 3 | `1101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 19 |
| 3 | `1011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 26 |
| 3 | `0111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 26 |
| 3 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 26 |
| 4 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 52 |
| 4 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 71 |
| 4 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 97 |
| 4 | `0001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 45 |
| 4 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 71 |
| 4 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 97 |
| 4 | `1001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 52 |
| 4 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 97 |
| 4 | `0101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 71 |
| 4 | `0011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 97 |
| 4 | `1110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 97 |
| 4 | `1101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 71 |
| 4 | `1011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 97 |
| 4 | `0111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 97 |
| 4 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 97 |
| 5 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 194 |
| 5 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 265 |
| 5 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 362 |
| 5 | `0001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 168 |
| 5 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 265 |
| 5 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 362 |
| 5 | `1001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 194 |
| 5 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 362 |
| 5 | `0101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 265 |
| 5 | `0011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 362 |
| 5 | `1110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 362 |
| 5 | `1101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 265 |
| 5 | `1011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 362 |
| 5 | `0111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 362 |
| 5 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 362 |
| 6 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 724 |
| 6 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 989 |
| 6 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1351 |
| 6 | `0001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 627 |
| 6 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 989 |
| 6 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1351 |
| 6 | `1001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 724 |
| 6 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1351 |
| 6 | `0101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 989 |
| 6 | `0011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 1351 |
| 6 | `1110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1351 |
| 6 | `1101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 989 |
| 6 | `1011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 1351 |
| 6 | `0111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 1351 |
| 6 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 1351 |
| 7 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2702 |
| 7 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 3691 |
| 7 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5042 |
| 7 | `0001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 2340 |
| 7 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 3691 |
| 7 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5042 |
| 7 | `1001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 2702 |
| 7 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5042 |
| 7 | `0101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 3691 |
| 7 | `0011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 5042 |
| 7 | `1110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5042 |
| 7 | `1101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 3691 |
| 7 | `1011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 5042 |
| 7 | `0111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 5042 |
| 7 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 5042 |
| 8 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 10084 |
| 8 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 13775 |
| 8 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 18817 |
| 8 | `0001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 8733 |
| 8 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 13775 |
| 8 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 18817 |
| 8 | `1001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 10084 |
| 8 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 18817 |
| 8 | `0101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 13775 |
| 8 | `0011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 18817 |
| 8 | `1110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 18817 |
| 8 | `1101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 13775 |
| 8 | `1011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 18817 |
| 8 | `0111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 18817 |
| 8 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 18817 |
| 9 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 37634 |
| 9 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 51409 |
| 9 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 40545 |
| 9 | `0001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 32592 |
| 9 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 51409 |
| 9 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 40545 |
| 9 | `1001` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 37634 |
| 9 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 40545 |
| 9 | `0101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 51409 |
| 9 | `0011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 40545 |
| 9 | `1110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 40545 |
| 9 | `1101` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 51409 |
| 9 | `1011` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 40545 |
| 9 | `0111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 40545 |
| 9 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 40545 |
| 10 | `1000` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 81090 |
| 10 | `0100` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 110771 |
| 10 | `0010` | 0 | distinguisher only | 2^48.0 | — | 3 | None | 151316 |
| 10 | `0001` | 1 | dim K = 1 | 2^32.0 | — | 2 | None | 121635 |
| 10 | `1100` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 110771 |
| 10 | `1010` | 0 | distinguisher only | 2^48.0 | — | 3 | None | 151316 |
| 10 | `1001` | 1 | dim K = 1 | 2^32.0 | — | 2 | None | 81090 |
| 10 | `0110` | 0 | distinguisher only | 2^48.0 | — | 3 | None | 151316 |
| 10 | `0101` | 1 | dim K = 1 | 2^32.0 | — | 2 | None | 110771 |
| 10 | `0011` | 4 | dim K = 4 + plain rows | 2^48.0 | — | 3 | None | 151316 |
| 10 | `1110` | 0 | distinguisher only | 2^48.0 | — | 3 | None | 151316 |
| 10 | `1101` | 1 | dim K = 1 | 2^32.0 | — | 2 | None | 110771 |
| 10 | `1011` | 4 | dim K = 4 + plain rows | 2^48.0 | — | 3 | None | 151316 |
| 10 | `0111` | 4 | dim K = 4 + plain rows | 2^48.0 | — | 3 | None | 151316 |
| 10 | `1111` | 4 | dim K = 4 + plain rows | 2^48.0 | — | 3 | None | 151316 |
| 11 | `1000` | 0 | distinguisher only | 2^80.0 | [0] | 1 | None | 302632 |
| 11 | `0100` | 0 | distinguisher only | 2^112.0 | [0] | 3 | None | 413403 |
| 11 | `0001` | 1 | dim K = 1 | 2^64.0 | — | 4 | None | 262087 |
| 11 | `1100` | 0 | distinguisher only | 2^112.0 | [0] | 3 | None | 413403 |
| 11 | `1001` | 1 | dim K = 1 | 2^80.0 | [0] | 1 | None | 302632 |
| 11 | `0101` | 1 | dim K = 1 | 2^112.0 | [0] | 3 | None | 413403 |
| 11 | `1101` | 1 | dim K = 1 | 2^112.0 | [0] | 3 | None | 413403 |

### DuX(2^16) · 解密（CCA）

| 层 | 类 | dim K | 可用性 | 最小数据 | 自由块 | 额外字 | 维数 | 需要的次数 |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^1 | — | 1 | [1] | 0 |
| 1 | `0100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0010` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^1 | — | 1 | [1] | 0 |
| 1 | `1100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `1010` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 1 |
| 1 | `0110` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 1 |
| 1 | `0011` | 4 | dim K = 4 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 1 | `1110` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 1 |
| 1 | `1011` | 4 | dim K = 4 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 1 | `0111` | 4 | dim K = 4 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 1 | `1111` | 4 | dim K = 4 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 2 | `1000` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 2 | `0100` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 2 | `0010` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 2 |
| 2 | `1100` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 2 | `1010` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 2 |
| 2 | `0110` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^3 | — | 1 | [3] | 5 |
| 2 | `0011` | 4 | dim K = 4 + plain rows | 2^3 | — | 1 | [3] | 4 |
| 2 | `1110` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^3 | — | 1 | [3] | 5 |
| 2 | `1011` | 4 | dim K = 4 + plain rows | 2^3 | — | 1 | [3] | 4 |
| 2 | `0111` | 4 | dim K = 4 + plain rows | 2^3 | — | 1 | [3] | 4 |
| 2 | `1111` | 4 | dim K = 4 + plain rows | 2^3 | — | 1 | [3] | 4 |
| 3 | `1000` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 14 |
| 3 | `0100` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 11 |
| 3 | `0010` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^4 | — | 1 | [4] | 12 |
| 3 | `1100` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 11 |
| 3 | `1010` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^4 | — | 1 | [4] | 14 |
| 3 | `0110` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^4 | — | 1 | [4] | 11 |
| 3 | `0011` | 4 | dim K = 4 + plain rows | 2^5 | — | 1 | [5] | 26 |
| 3 | `1110` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^4 | — | 1 | [4] | 11 |
| 3 | `1011` | 4 | dim K = 4 + plain rows | 2^5 | — | 1 | [5] | 26 |
| 3 | `0111` | 4 | dim K = 4 + plain rows | 2^5 | — | 1 | [5] | 26 |
| 3 | `1111` | 4 | dim K = 4 + plain rows | 2^5 | — | 1 | [5] | 26 |
| 4 | `1000` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 30 |
| 4 | `0100` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 41 |
| 4 | `0010` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 56 |
| 4 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^5 | — | 1 | [5] | 26 |
| 4 | `1100` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 41 |
| 4 | `1010` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 56 |
| 4 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^5 | — | 1 | [5] | 30 |
| 4 | `0110` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 56 |
| 4 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^6 | — | 1 | [6] | 41 |
| 4 | `0011` | 4 | dim K = 4 + plain rows | 2^6 | — | 1 | [6] | 56 |
| 4 | `1110` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 56 |
| 4 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^6 | — | 1 | [6] | 41 |
| 4 | `1011` | 4 | dim K = 4 + plain rows | 2^6 | — | 1 | [6] | 56 |
| 4 | `0111` | 4 | dim K = 4 + plain rows | 2^6 | — | 1 | [6] | 56 |
| 4 | `1111` | 4 | dim K = 4 + plain rows | 2^6 | — | 1 | [6] | 56 |
| 5 | `1000` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 112 |
| 5 | `0100` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 153 |
| 5 | `0010` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 209 |
| 5 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^7 | — | 1 | [7] | 97 |
| 5 | `1100` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 153 |
| 5 | `1010` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 209 |
| 5 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^7 | — | 1 | [7] | 112 |
| 5 | `0110` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 209 |
| 5 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^8 | — | 1 | [8] | 153 |
| 5 | `0011` | 4 | dim K = 4 + plain rows | 2^8 | — | 1 | [8] | 209 |
| 5 | `1110` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 209 |
| 5 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^8 | — | 1 | [8] | 153 |
| 5 | `1011` | 4 | dim K = 4 + plain rows | 2^8 | — | 1 | [8] | 209 |
| 5 | `0111` | 4 | dim K = 4 + plain rows | 2^8 | — | 1 | [8] | 209 |
| 5 | `1111` | 4 | dim K = 4 + plain rows | 2^8 | — | 1 | [8] | 209 |
| 6 | `1000` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 418 |
| 6 | `0100` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 989 |
| 6 | `0010` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 780 |
| 6 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^9 | — | 1 | [9] | 362 |
| 6 | `1100` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 989 |
| 6 | `1010` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 780 |
| 6 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^9 | — | 1 | [9] | 418 |
| 6 | `0110` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 780 |
| 6 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^10 | — | 1 | [10] | 989 |
| 6 | `0011` | 4 | dim K = 4 + plain rows | 2^10 | — | 1 | [10] | 780 |
| 6 | `1110` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 780 |
| 6 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^10 | — | 1 | [10] | 989 |
| 6 | `1011` | 4 | dim K = 4 + plain rows | 2^10 | — | 1 | [10] | 780 |
| 6 | `0111` | 4 | dim K = 4 + plain rows | 2^10 | — | 1 | [10] | 780 |
| 6 | `1111` | 4 | dim K = 4 + plain rows | 2^10 | — | 1 | [10] | 780 |
| 7 | `1000` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1560 |
| 7 | `0100` | 0 | distinguisher only | 2^12 | — | 1 | [12] | 3691 |
| 7 | `0010` | 0 | distinguisher only | 2^12 | — | 1 | [12] | 2911 |
| 7 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^11 | — | 1 | [11] | 1351 |
| 7 | `1100` | 0 | distinguisher only | 2^12 | — | 1 | [12] | 3691 |
| 7 | `1010` | 0 | distinguisher only | 2^12 | — | 1 | [12] | 2911 |
| 7 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^11 | — | 1 | [11] | 1560 |
| 7 | `0110` | 0 | distinguisher only | 2^12 | — | 1 | [12] | 2911 |
| 7 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^12 | — | 1 | [12] | 3691 |
| 7 | `0011` | 4 | dim K = 4 + plain rows | 2^12 | — | 1 | [12] | 2911 |
| 7 | `1110` | 0 | distinguisher only | 2^12 | — | 1 | [12] | 2911 |
| 7 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^12 | — | 1 | [12] | 3691 |
| 7 | `1011` | 4 | dim K = 4 + plain rows | 2^12 | — | 1 | [12] | 2911 |
| 7 | `0111` | 4 | dim K = 4 + plain rows | 2^12 | — | 1 | [12] | 2911 |
| 7 | `1111` | 4 | dim K = 4 + plain rows | 2^12 | — | 1 | [12] | 2911 |
| 8 | `1000` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 5822 |
| 8 | `0100` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 7953 |
| 8 | `0010` | 0 | distinguisher only | 2^14 | — | 1 | [14] | 10864 |
| 8 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^13 | — | 1 | [13] | 5042 |
| 8 | `1100` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 7953 |
| 8 | `1010` | 0 | distinguisher only | 2^14 | — | 1 | [14] | 10864 |
| 8 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^13 | — | 1 | [13] | 5822 |
| 8 | `0110` | 0 | distinguisher only | 2^14 | — | 1 | [14] | 10864 |
| 8 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^13 | — | 1 | [13] | 7953 |
| 8 | `0011` | 4 | dim K = 4 + plain rows | 2^14 | — | 1 | [14] | 10864 |
| 8 | `1110` | 0 | distinguisher only | 2^14 | — | 1 | [14] | 10864 |
| 8 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^13 | — | 1 | [13] | 7953 |
| 8 | `1011` | 4 | dim K = 4 + plain rows | 2^14 | — | 1 | [14] | 10864 |
| 8 | `0111` | 4 | dim K = 4 + plain rows | 2^14 | — | 1 | [14] | 10864 |
| 8 | `1111` | 4 | dim K = 4 + plain rows | 2^14 | — | 1 | [14] | 10864 |
| 9 | `1000` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 21728 |
| 9 | `0100` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 29681 |
| 9 | `0010` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 40545 |
| 9 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^15 | — | 1 | [15] | 32592 |
| 9 | `1100` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 29681 |
| 9 | `1010` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 40545 |
| 9 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^15 | — | 1 | [15] | 21728 |
| 9 | `0110` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 40545 |
| 9 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^15 | — | 1 | [15] | 29681 |
| 9 | `0011` | 4 | dim K = 4 + plain rows | 2^16 | — | 1 | [16] | 40545 |
| 9 | `1110` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 40545 |
| 9 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^15 | — | 1 | [15] | 29681 |
| 9 | `1011` | 4 | dim K = 4 + plain rows | 2^16 | — | 1 | [16] | 40545 |
| 9 | `0111` | 4 | dim K = 4 + plain rows | 2^16 | — | 1 | [16] | 40545 |
| 9 | `1111` | 4 | dim K = 4 + plain rows | 2^16 | — | 1 | [16] | 40545 |
| 10 | `1000` | 0 | distinguisher only | 2^30 | — | 2 | [16, 14] | 81090 |
| 10 | `0100` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 110771 |
| 10 | `0010` | 0 | distinguisher only | 2^47 | — | 3 | [16, 16, 15] | 151316 |
| 10 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^29 | — | 2 | [16, 13] | 70226 |
| 10 | `1100` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 110771 |
| 10 | `1010` | 0 | distinguisher only | 2^47 | — | 3 | [16, 16, 15] | 151316 |
| 10 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^30 | — | 2 | [16, 14] | 81090 |
| 10 | `0110` | 0 | distinguisher only | 2^47 | — | 3 | [16, 16, 15] | 151316 |
| 10 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^32 | — | 2 | [16, 16] | 110771 |
| 10 | `0011` | 4 | dim K = 4 + plain rows | 2^47 | — | 3 | [16, 16, 15] | 151316 |
| 10 | `1110` | 0 | distinguisher only | 2^47 | — | 3 | [16, 16, 15] | 151316 |
| 10 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^32 | — | 2 | [16, 16] | 110771 |
| 10 | `1011` | 4 | dim K = 4 + plain rows | 2^47 | — | 3 | [16, 16, 15] | 151316 |
| 10 | `0111` | 4 | dim K = 4 + plain rows | 2^47 | — | 3 | [16, 16, 15] | 151316 |
| 10 | `1111` | 4 | dim K = 4 + plain rows | 2^47 | — | 3 | [16, 16, 15] | 151316 |
| 11 | `1000` | 0 | distinguisher only | 2^80 | [0] | 1 | [16] | 302632 |
| 11 | `0100` | 0 | distinguisher only | 2^111 | [0] | 3 | [16, 16, 15] | 413403 |
| 11 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^64 | — | 4 | [16, 16, 16, 16] | 262087 |
| 11 | `1100` | 0 | distinguisher only | 2^111 | [0] | 3 | [16, 16, 15] | 413403 |
| 11 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^80 | [0] | 1 | [16] | 302632 |
| 11 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^111 | [0] | 3 | [16, 16, 15] | 413403 |
| 11 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^111 | [0] | 3 | [16, 16, 15] | 413403 |

### DuX(2^8) · 解密（CCA）

| 层 | 类 | dim K | 可用性 | 最小数据 | 自由块 | 额外字 | 维数 | 需要的次数 |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^1 | — | 1 | [1] | 0 |
| 1 | `0100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0010` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^1 | — | 1 | [1] | 0 |
| 1 | `1100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `1010` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 1 |
| 1 | `0110` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 1 |
| 1 | `0011` | 4 | dim K = 4 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 1 | `1110` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 1 |
| 1 | `1011` | 4 | dim K = 4 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 1 | `0111` | 4 | dim K = 4 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 1 | `1111` | 4 | dim K = 4 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 2 | `1000` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 2 | `0100` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 2 | `0010` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 2 |
| 2 | `1100` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 2 | `1010` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 2 |
| 2 | `0110` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^3 | — | 1 | [3] | 5 |
| 2 | `0011` | 4 | dim K = 4 + plain rows | 2^3 | — | 1 | [3] | 4 |
| 2 | `1110` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^3 | — | 1 | [3] | 5 |
| 2 | `1011` | 4 | dim K = 4 + plain rows | 2^3 | — | 1 | [3] | 4 |
| 2 | `0111` | 4 | dim K = 4 + plain rows | 2^3 | — | 1 | [3] | 4 |
| 2 | `1111` | 4 | dim K = 4 + plain rows | 2^3 | — | 1 | [3] | 4 |
| 3 | `1000` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 14 |
| 3 | `0100` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 11 |
| 3 | `0010` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^4 | — | 1 | [4] | 12 |
| 3 | `1100` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 11 |
| 3 | `1010` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^4 | — | 1 | [4] | 14 |
| 3 | `0110` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^4 | — | 1 | [4] | 11 |
| 3 | `0011` | 4 | dim K = 4 + plain rows | 2^5 | — | 1 | [5] | 26 |
| 3 | `1110` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^4 | — | 1 | [4] | 11 |
| 3 | `1011` | 4 | dim K = 4 + plain rows | 2^5 | — | 1 | [5] | 26 |
| 3 | `0111` | 4 | dim K = 4 + plain rows | 2^5 | — | 1 | [5] | 26 |
| 3 | `1111` | 4 | dim K = 4 + plain rows | 2^5 | — | 1 | [5] | 26 |
| 4 | `1000` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 30 |
| 4 | `0100` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 41 |
| 4 | `0010` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 56 |
| 4 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^5 | — | 1 | [5] | 26 |
| 4 | `1100` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 41 |
| 4 | `1010` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 56 |
| 4 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^5 | — | 1 | [5] | 30 |
| 4 | `0110` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 56 |
| 4 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^6 | — | 1 | [6] | 41 |
| 4 | `0011` | 4 | dim K = 4 + plain rows | 2^6 | — | 1 | [6] | 56 |
| 4 | `1110` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 56 |
| 4 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^6 | — | 1 | [6] | 41 |
| 4 | `1011` | 4 | dim K = 4 + plain rows | 2^6 | — | 1 | [6] | 56 |
| 4 | `0111` | 4 | dim K = 4 + plain rows | 2^6 | — | 1 | [6] | 56 |
| 4 | `1111` | 4 | dim K = 4 + plain rows | 2^6 | — | 1 | [6] | 56 |
| 5 | `1000` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 112 |
| 5 | `0100` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 153 |
| 5 | `0010` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 209 |
| 5 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^7 | — | 1 | [7] | 97 |
| 5 | `1100` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 153 |
| 5 | `1010` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 209 |
| 5 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^7 | — | 1 | [7] | 112 |
| 5 | `0110` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 209 |
| 5 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^8 | — | 1 | [8] | 153 |
| 5 | `0011` | 4 | dim K = 4 + plain rows | 2^8 | — | 1 | [8] | 209 |
| 5 | `1110` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 209 |
| 5 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^8 | — | 1 | [8] | 153 |
| 5 | `1011` | 4 | dim K = 4 + plain rows | 2^8 | — | 1 | [8] | 209 |
| 5 | `0111` | 4 | dim K = 4 + plain rows | 2^8 | — | 1 | [8] | 209 |
| 5 | `1111` | 4 | dim K = 4 + plain rows | 2^8 | — | 1 | [8] | 209 |
| 6 | `1000` | 0 | distinguisher only | 2^16 | — | 2 | [8, 8] | 418 |
| 6 | `0100` | 0 | distinguisher only | 2^22 | — | 3 | [8, 8, 6] | 571 |
| 6 | `0010` | 0 | distinguisher only | 2^29 | — | 4 | [8, 8, 8, 5] | 780 |
| 6 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^15 | — | 2 | [8, 7] | 362 |
| 6 | `1100` | 0 | distinguisher only | 2^22 | — | 3 | [8, 8, 6] | 571 |
| 6 | `1010` | 0 | distinguisher only | 2^29 | — | 4 | [8, 8, 8, 5] | 780 |
| 6 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^16 | — | 2 | [8, 8] | 418 |
| 6 | `0110` | 0 | distinguisher only | 2^29 | — | 4 | [8, 8, 8, 5] | 780 |
| 6 | `0101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^22 | — | 3 | [8, 8, 6] | 571 |
| 6 | `0011` | 4 | dim K = 4 + plain rows | 2^29 | — | 4 | [8, 8, 8, 5] | 780 |
| 6 | `1110` | 0 | distinguisher only | 2^29 | — | 4 | [8, 8, 8, 5] | 780 |
| 6 | `1101` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^22 | — | 3 | [8, 8, 6] | 571 |
| 6 | `1011` | 4 | dim K = 4 + plain rows | 2^29 | — | 4 | [8, 8, 8, 5] | 780 |
| 6 | `0111` | 4 | dim K = 4 + plain rows | 2^29 | — | 4 | [8, 8, 8, 5] | 780 |
| 6 | `1111` | 4 | dim K = 4 + plain rows | 2^29 | — | 4 | [8, 8, 8, 5] | 780 |
| 7 | `1000` | 0 | distinguisher only | 2^53 | [0] | 3 | [8, 8, 5] | 1560 |
| 7 | `0001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^47 | [0] | 2 | [8, 7] | 1351 |
| 7 | `1001` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^53 | [0] | 3 | [8, 8, 5] | 1560 |

### YupX-65537 · 解密（CCA）

| 层 | 类 | dim K | 可用性 | 最小数据 | 自由块 | 额外字 | 维数 | 需要的次数 |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1 |
| 1 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1 |
| 1 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1 |
| 1 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1 |
| 1 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1 |
| 1 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1 |
| 1 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `1110` | 3 | dim K = 3 | 2^16.0 | — | 1 | None | 1 |
| 1 | `1101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `1011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `0111` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 2 |
| 2 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 3 |
| 2 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 4 |
| 2 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5 |
| 2 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7 |
| 2 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 4 |
| 2 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5 |
| 2 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7 |
| 2 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5 |
| 2 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7 |
| 2 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7 |
| 2 | `1110` | 3 | dim K = 3 | 2^16.0 | — | 1 | None | 5 |
| 2 | `1101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7 |
| 2 | `1011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7 |
| 2 | `0111` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7 |
| 2 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 7 |
| 3 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 13 |
| 3 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 14 |
| 3 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 20 |
| 3 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26 |
| 3 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 14 |
| 3 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 20 |
| 3 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26 |
| 3 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 20 |
| 3 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26 |
| 3 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26 |
| 3 | `1110` | 3 | dim K = 3 | 2^16.0 | — | 1 | None | 20 |
| 3 | `1101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26 |
| 3 | `1011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26 |
| 3 | `0111` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26 |
| 3 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 26 |
| 4 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 52 |
| 4 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 52 |
| 4 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 78 |
| 4 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 104 |
| 4 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 52 |
| 4 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 78 |
| 4 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 104 |
| 4 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 78 |
| 4 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 104 |
| 4 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 104 |
| 4 | `1110` | 3 | dim K = 3 | 2^16.0 | — | 1 | None | 78 |
| 4 | `1101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 104 |
| 4 | `1011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 104 |
| 4 | `0111` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 104 |
| 4 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 104 |
| 5 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 204 |
| 5 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 208 |
| 5 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 308 |
| 5 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 412 |
| 5 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 208 |
| 5 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 308 |
| 5 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 412 |
| 5 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 308 |
| 5 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 412 |
| 5 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 412 |
| 5 | `1110` | 3 | dim K = 3 | 2^16.0 | — | 1 | None | 308 |
| 5 | `1101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 412 |
| 5 | `1011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 412 |
| 5 | `0111` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 412 |
| 5 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 412 |
| 6 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 816 |
| 6 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 824 |
| 6 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1228 |
| 6 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1633 |
| 6 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 824 |
| 6 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1228 |
| 6 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1633 |
| 6 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1228 |
| 6 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1633 |
| 6 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1633 |
| 6 | `1110` | 3 | dim K = 3 | 2^16.0 | — | 1 | None | 1228 |
| 6 | `1101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1633 |
| 6 | `1011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1633 |
| 6 | `0111` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1633 |
| 6 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 1633 |
| 7 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 3265 |
| 7 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 3266 |
| 7 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 4898 |
| 7 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 6530 |
| 7 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 3266 |
| 7 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 4898 |
| 7 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 6530 |
| 7 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 4898 |
| 7 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 6530 |
| 7 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 6530 |
| 7 | `1110` | 3 | dim K = 3 | 2^16.0 | — | 1 | None | 4898 |
| 7 | `1101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 6530 |
| 7 | `1011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 6530 |
| 7 | `0111` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 6530 |
| 7 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 6530 |
| 8 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 13023 |
| 8 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 13060 |
| 8 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 19553 |
| 8 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26083 |
| 8 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 13060 |
| 8 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 19553 |
| 8 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26083 |
| 8 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 19553 |
| 8 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26083 |
| 8 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26083 |
| 8 | `1110` | 3 | dim K = 3 | 2^16.0 | — | 1 | None | 19553 |
| 8 | `1101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26083 |
| 8 | `1011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26083 |
| 8 | `0111` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 26083 |
| 8 | `1111` | 4 | dim K = 4 + plain rows | 2^16.0 | — | 1 | None | 26083 |
| 9 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 52041 |
| 9 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 52166 |
| 9 | `0010` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 78124 |
| 9 | `0001` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 104178 |
| 9 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 52166 |
| 9 | `1010` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 78124 |
| 9 | `1001` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 104178 |
| 9 | `0110` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 78124 |
| 9 | `0101` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 104178 |
| 9 | `0011` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 104178 |
| 9 | `1110` | 3 | dim K = 3 | 2^32.0 | — | 2 | None | 78124 |
| 9 | `1101` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 104178 |
| 9 | `1011` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 104178 |
| 9 | `0111` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 104178 |
| 9 | `1111` | 4 | dim K = 4 + plain rows | 2^32.0 | — | 2 | None | 104178 |
| 10 | `1000` | 0 | distinguisher only | 2^64.0 | — | 4 | None | 256803 |
| 10 | `0100` | 0 | distinguisher only | 2^64.0 | — | 4 | None | 256832 |
| 10 | `0010` | 0 | distinguisher only | 2^64.0 | [0] | 0 | None | 196608 |
| 10 | `0001` | 0 | distinguisher only | 2^112.0 | [0] | 3 | None | 436907 |
| 10 | `1100` | 0 | distinguisher only | 2^64.0 | — | 4 | None | 256832 |
| 10 | `1010` | 0 | distinguisher only | 2^64.0 | [0] | 0 | None | 196608 |
| 10 | `1001` | 0 | distinguisher only | 2^112.0 | [0] | 3 | None | 436907 |
| 10 | `0110` | 0 | distinguisher only | 2^64.0 | [0] | 0 | None | 196608 |
| 10 | `0101` | 0 | distinguisher only | 2^112.0 | [0] | 3 | None | 436907 |
| 10 | `0011` | 0 | distinguisher only | 2^112.0 | [0] | 3 | None | 436907 |
| 10 | `1110` | 3 | dim K = 3 | 2^64.0 | [0] | 0 | None | 196608 |
| 10 | `1101` | 0 | distinguisher only | 2^112.0 | [0] | 3 | None | 436907 |
| 10 | `1011` | 0 | distinguisher only | 2^112.0 | [0] | 3 | None | 436907 |
| 10 | `0111` | 0 | distinguisher only | 2^112.0 | [0] | 3 | None | 436907 |
| 10 | `1111` | 4 | dim K = 4 + plain rows | 2^112.0 | [0] | 3 | None | 436907 |

### Yu2X-16 · 解密（CCA）

| 层 | 类 | dim K | 可用性 | 最小数据 | 自由块 | 额外字 | 维数 | 需要的次数 |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0010` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0001` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `1010` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `1001` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0110` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0101` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0011` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1110` | 3 | dim K = 3 | 2^2 | — | 1 | [2] | 1 |
| 1 | `1101` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1011` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0111` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1111` | 4 | dim K = 4 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 2 | `1000` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 3 |
| 2 | `0100` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `0010` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 2 | `0001` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `1100` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `1010` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 2 | `1001` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `0110` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 2 | `0101` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `0011` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `1110` | 3 | dim K = 3 | 2^3 | — | 1 | [3] | 5 |
| 2 | `1101` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `1011` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `0111` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `1111` | 4 | dim K = 4 + plain rows | 2^4 | — | 1 | [4] | 7 |
| 3 | `1000` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 13 |
| 3 | `0100` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 14 |
| 3 | `0010` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 20 |
| 3 | `0001` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1100` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 14 |
| 3 | `1010` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 20 |
| 3 | `1001` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `0110` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 20 |
| 3 | `0101` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `0011` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1110` | 3 | dim K = 3 | 2^5 | — | 1 | [5] | 20 |
| 3 | `1101` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1011` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `0111` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1111` | 4 | dim K = 4 + plain rows | 2^5 | — | 1 | [5] | 26 |
| 4 | `1000` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 52 |
| 4 | `0100` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 52 |
| 4 | `0010` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 78 |
| 4 | `0001` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `1100` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 52 |
| 4 | `1010` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 78 |
| 4 | `1001` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `0110` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 78 |
| 4 | `0101` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `0011` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `1110` | 3 | dim K = 3 | 2^7 | — | 1 | [7] | 78 |
| 4 | `1101` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `1011` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `0111` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `1111` | 4 | dim K = 4 + plain rows | 2^7 | — | 1 | [7] | 104 |
| 5 | `1000` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 204 |
| 5 | `0100` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 208 |
| 5 | `0010` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 308 |
| 5 | `0001` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 412 |
| 5 | `1100` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 208 |
| 5 | `1010` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 308 |
| 5 | `1001` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 412 |
| 5 | `0110` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 308 |
| 5 | `0101` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 412 |
| 5 | `0011` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 412 |
| 5 | `1110` | 3 | dim K = 3 | 2^9 | — | 1 | [9] | 308 |
| 5 | `1101` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 412 |
| 5 | `1011` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 412 |
| 5 | `0111` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 412 |
| 5 | `1111` | 4 | dim K = 4 + plain rows | 2^9 | — | 1 | [9] | 412 |
| 6 | `1000` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 816 |
| 6 | `0100` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 824 |
| 6 | `0010` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1228 |
| 6 | `0001` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1633 |
| 6 | `1100` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 824 |
| 6 | `1010` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1228 |
| 6 | `1001` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1633 |
| 6 | `0110` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1228 |
| 6 | `0101` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1633 |
| 6 | `0011` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1633 |
| 6 | `1110` | 3 | dim K = 3 | 2^11 | — | 1 | [11] | 1228 |
| 6 | `1101` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1633 |
| 6 | `1011` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1633 |
| 6 | `0111` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1633 |
| 6 | `1111` | 4 | dim K = 4 + plain rows | 2^11 | — | 1 | [11] | 1633 |
| 7 | `1000` | 0 | distinguisher only | 2^12 | — | 1 | [12] | 3265 |
| 7 | `0100` | 0 | distinguisher only | 2^12 | — | 1 | [12] | 3266 |
| 7 | `0010` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 4898 |
| 7 | `0001` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 6530 |
| 7 | `1100` | 0 | distinguisher only | 2^12 | — | 1 | [12] | 3266 |
| 7 | `1010` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 4898 |
| 7 | `1001` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 6530 |
| 7 | `0110` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 4898 |
| 7 | `0101` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 6530 |
| 7 | `0011` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 6530 |
| 7 | `1110` | 3 | dim K = 3 | 2^13 | — | 1 | [13] | 4898 |
| 7 | `1101` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 6530 |
| 7 | `1011` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 6530 |
| 7 | `0111` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 6530 |
| 7 | `1111` | 4 | dim K = 4 + plain rows | 2^13 | — | 1 | [13] | 6530 |
| 8 | `1000` | 0 | distinguisher only | 2^14 | — | 1 | [14] | 13023 |
| 8 | `0100` | 0 | distinguisher only | 2^14 | — | 1 | [14] | 13060 |
| 8 | `0010` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 19553 |
| 8 | `0001` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 26083 |
| 8 | `1100` | 0 | distinguisher only | 2^14 | — | 1 | [14] | 13060 |
| 8 | `1010` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 19553 |
| 8 | `1001` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 26083 |
| 8 | `0110` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 19553 |
| 8 | `0101` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 26083 |
| 8 | `0011` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 26083 |
| 8 | `1110` | 3 | dim K = 3 | 2^15 | — | 1 | [15] | 19553 |
| 8 | `1101` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 26083 |
| 8 | `1011` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 26083 |
| 8 | `0111` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 26083 |
| 8 | `1111` | 4 | dim K = 4 + plain rows | 2^15 | — | 1 | [15] | 26083 |
| 9 | `1000` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 52041 |
| 9 | `0100` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 52166 |
| 9 | `0010` | 0 | distinguisher only | 2^30 | — | 2 | [16, 14] | 78124 |
| 9 | `0001` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 104178 |
| 9 | `1100` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 52166 |
| 9 | `1010` | 0 | distinguisher only | 2^30 | — | 2 | [16, 14] | 78124 |
| 9 | `1001` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 104178 |
| 9 | `0110` | 0 | distinguisher only | 2^30 | — | 2 | [16, 14] | 78124 |
| 9 | `0101` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 104178 |
| 9 | `0011` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 104178 |
| 9 | `1110` | 3 | dim K = 3 | 2^30 | — | 2 | [16, 14] | 78124 |
| 9 | `1101` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 104178 |
| 9 | `1011` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 104178 |
| 9 | `0111` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 104178 |
| 9 | `1111` | 4 | dim K = 4 + plain rows | 2^32 | — | 2 | [16, 16] | 104178 |
| 10 | `1000` | 0 | distinguisher only | 2^63 | — | 4 | [16, 16, 16, 15] | 218453 |
| 10 | `0100` | 0 | distinguisher only | 2^63 | — | 4 | [16, 16, 16, 15] | 218454 |
| 10 | `0010` | 0 | distinguisher only | 2^64 | [0] | 0 | [] | 196608 |
| 10 | `0001` | 0 | distinguisher only | 2^112 | [0] | 3 | [16, 16, 16] | 436907 |
| 10 | `1100` | 0 | distinguisher only | 2^63 | — | 4 | [16, 16, 16, 15] | 218454 |
| 10 | `1010` | 0 | distinguisher only | 2^64 | [0] | 0 | [] | 196608 |
| 10 | `1001` | 0 | distinguisher only | 2^112 | [0] | 3 | [16, 16, 16] | 436907 |
| 10 | `0110` | 0 | distinguisher only | 2^64 | [0] | 0 | [] | 196608 |
| 10 | `0101` | 0 | distinguisher only | 2^112 | [0] | 3 | [16, 16, 16] | 436907 |
| 10 | `0011` | 0 | distinguisher only | 2^112 | [0] | 3 | [16, 16, 16] | 436907 |
| 10 | `1110` | 3 | dim K = 3 | 2^64 | [0] | 0 | [] | 196608 |
| 10 | `1101` | 0 | distinguisher only | 2^112 | [0] | 3 | [16, 16, 16] | 436907 |
| 10 | `1011` | 0 | distinguisher only | 2^112 | [0] | 3 | [16, 16, 16] | 436907 |
| 10 | `0111` | 0 | distinguisher only | 2^112 | [0] | 3 | [16, 16, 16] | 436907 |
| 10 | `1111` | 4 | dim K = 4 + plain rows | 2^112 | [0] | 3 | [16, 16, 16] | 436907 |

### Yu2X-8 · 解密（CCA）

| 层 | 类 | dim K | 可用性 | 最小数据 | 自由块 | 额外字 | 维数 | 需要的次数 |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0010` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0001` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `1010` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `1001` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0110` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0101` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0011` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1110` | 3 | dim K = 3 | 2^2 | — | 1 | [2] | 1 |
| 1 | `1101` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1011` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0111` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1111` | 4 | dim K = 4 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 2 | `1000` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 3 |
| 2 | `0100` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `0010` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 2 | `0001` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `1100` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 4 |
| 2 | `1010` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 2 | `1001` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `0110` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 2 | `0101` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `0011` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `1110` | 3 | dim K = 3 | 2^3 | — | 1 | [3] | 5 |
| 2 | `1101` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `1011` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `0111` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 7 |
| 2 | `1111` | 4 | dim K = 4 + plain rows | 2^4 | — | 1 | [4] | 7 |
| 3 | `1000` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 13 |
| 3 | `0100` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 14 |
| 3 | `0010` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 20 |
| 3 | `0001` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1100` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 14 |
| 3 | `1010` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 20 |
| 3 | `1001` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `0110` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 20 |
| 3 | `0101` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `0011` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1110` | 3 | dim K = 3 | 2^5 | — | 1 | [5] | 20 |
| 3 | `1101` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1011` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `0111` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 26 |
| 3 | `1111` | 4 | dim K = 4 + plain rows | 2^5 | — | 1 | [5] | 26 |
| 4 | `1000` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 52 |
| 4 | `0100` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 52 |
| 4 | `0010` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 78 |
| 4 | `0001` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `1100` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 52 |
| 4 | `1010` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 78 |
| 4 | `1001` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `0110` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 78 |
| 4 | `0101` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `0011` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `1110` | 3 | dim K = 3 | 2^7 | — | 1 | [7] | 78 |
| 4 | `1101` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `1011` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `0111` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 104 |
| 4 | `1111` | 4 | dim K = 4 + plain rows | 2^7 | — | 1 | [7] | 104 |
| 5 | `1000` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 204 |
| 5 | `0100` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 208 |
| 5 | `0010` | 0 | distinguisher only | 2^14 | — | 2 | [8, 6] | 308 |
| 5 | `0001` | 0 | distinguisher only | 2^16 | — | 2 | [8, 8] | 412 |
| 5 | `1100` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 208 |
| 5 | `1010` | 0 | distinguisher only | 2^14 | — | 2 | [8, 6] | 308 |
| 5 | `1001` | 0 | distinguisher only | 2^16 | — | 2 | [8, 8] | 412 |
| 5 | `0110` | 0 | distinguisher only | 2^14 | — | 2 | [8, 6] | 308 |
| 5 | `0101` | 0 | distinguisher only | 2^16 | — | 2 | [8, 8] | 412 |
| 5 | `0011` | 0 | distinguisher only | 2^16 | — | 2 | [8, 8] | 412 |
| 5 | `1110` | 3 | dim K = 3 | 2^14 | — | 2 | [8, 6] | 308 |
| 5 | `1101` | 0 | distinguisher only | 2^16 | — | 2 | [8, 8] | 412 |
| 5 | `1011` | 0 | distinguisher only | 2^16 | — | 2 | [8, 8] | 412 |
| 5 | `0111` | 0 | distinguisher only | 2^16 | — | 2 | [8, 8] | 412 |
| 5 | `1111` | 4 | dim K = 4 + plain rows | 2^16 | — | 2 | [8, 8] | 412 |
| 6 | `1000` | 0 | distinguisher only | 2^31 | — | 4 | [8, 8, 8, 7] | 853 |
| 6 | `0100` | 0 | distinguisher only | 2^31 | — | 4 | [8, 8, 8, 7] | 854 |
| 6 | `0010` | 0 | distinguisher only | 2^32 | [0] | 0 | [] | 768 |
| 6 | `0001` | 0 | distinguisher only | 2^56 | [0] | 3 | [8, 8, 8] | 1707 |
| 6 | `1100` | 0 | distinguisher only | 2^31 | — | 4 | [8, 8, 8, 7] | 854 |
| 6 | `1010` | 0 | distinguisher only | 2^32 | [0] | 0 | [] | 768 |
| 6 | `1001` | 0 | distinguisher only | 2^56 | [0] | 3 | [8, 8, 8] | 1707 |
| 6 | `0110` | 0 | distinguisher only | 2^32 | [0] | 0 | [] | 768 |
| 6 | `0101` | 0 | distinguisher only | 2^56 | [0] | 3 | [8, 8, 8] | 1707 |
| 6 | `0011` | 0 | distinguisher only | 2^56 | [0] | 3 | [8, 8, 8] | 1707 |
| 6 | `1110` | 3 | dim K = 3 | 2^32 | [0] | 0 | [] | 768 |
| 6 | `1101` | 0 | distinguisher only | 2^56 | [0] | 3 | [8, 8, 8] | 1707 |
| 6 | `1011` | 0 | distinguisher only | 2^56 | [0] | 3 | [8, 8, 8] | 1707 |
| 6 | `0111` | 0 | distinguisher only | 2^56 | [0] | 3 | [8, 8, 8] | 1707 |
| 6 | `1111` | 4 | dim K = 4 + plain rows | 2^56 | [0] | 3 | [8, 8, 8] | 1707 |

