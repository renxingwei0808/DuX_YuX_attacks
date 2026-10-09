# E13 / T1 — the cells the class x1x1 opens (characteristic 2, DuX)

`min_data_v3.search_all` run twice, with the paper's cheap set {2} and
with T1's {1, 2}.  A cell counts as NEWLY USABLE when its class has
positions 1 and 3 balanced (0101, 0111, 1101, 1111) and its dim K goes
from <= 1 (where the characteristic-2 orbit-sum theorem kills the
combined row, W19-B) to >= 2.

## 1. The deepest newly-usable cell per instance

With `rank_max_b = 14400` -- the ROW COUNT at which the extended template determines the sixteen key words (step 3), not the rank there -- a cell needs `ceil(rank_max_b / (dim K x N_w))` structures.

| instance | layer | class | structure | points | usable weights | rows/structure | structures | **total data** | dim K {2} -> {1,2} |
|---|---|---|---|---|---|---|---|---|---|
| DuX(2^16) | 10 | `0101` | words [2, 6], dims [16, 16] | 2^32.0 | 20299 | 81196 | 1 | **2^32.0** | 1 -> 4 |
| DuX(2^8) | 6 | `0101` | words [2, 6, 10], dims [8, 8, 6] | 2^22.0 | 2 | 8 | 1800 | **2^32.81** | 1 -> 4 |

## 2. Every newly-usable cell with a weight budget

| instance | layer | class | structure | points | usable weights | structures | total data |
|---|---|---|---|---|---|---|---|
| DuX(2^16) | 10 | `0101` | words [2, 6], dims [16, 16] | 2^32.0 | 20299 | 1 | **2^32.0** |
| DuX(2^16) | 10 | `1101` | words [2, 6], dims [16, 16] | 2^32.0 | 20299 | 1 | **2^32.0** |
| DuX(2^16) | 9 | `0101` | words [2], dims [15] | 2^15.0 | 3086 | 2 | **2^16.0** |
| DuX(2^16) | 9 | `1101` | words [2], dims [15] | 2^15.0 | 3086 | 2 | **2^16.0** |
| DuX(2^16) | 8 | `0101` | words [2], dims [13] | 2^13.0 | 238 | 16 | **2^17.0** |
| DuX(2^16) | 8 | `1101` | words [2], dims [13] | 2^13.0 | 238 | 16 | **2^17.0** |
| DuX(2^16) | 7 | `0101` | words [0], dims [12] | 2^12.0 | 404 | 9 | **2^15.17** |
| DuX(2^16) | 7 | `1101` | words [0], dims [12] | 2^12.0 | 404 | 9 | **2^15.17** |
| DuX(2^16) | 6 | `0101` | words [0], dims [10] | 2^10.0 | 34 | 106 | **2^16.73** |
| DuX(2^16) | 6 | `1101` | words [0], dims [10] | 2^10.0 | 34 | 106 | **2^16.73** |
| DuX(2^16) | 5 | `0101` | words [2], dims [8] | 2^8.0 | 102 | 36 | **2^13.17** |
| DuX(2^16) | 5 | `1101` | words [2], dims [8] | 2^8.0 | 102 | 36 | **2^13.17** |
| DuX(2^16) | 4 | `0101` | words [2], dims [6] | 2^6.0 | 22 | 164 | **2^13.36** |
| DuX(2^16) | 4 | `1101` | words [2], dims [6] | 2^6.0 | 22 | 164 | **2^13.36** |
| DuX(2^16) | 3 | `0101` | words [2], dims [4] | 2^4.0 | 4 | 900 | **2^13.81** |
| DuX(2^16) | 3 | `1101` | words [2], dims [4] | 2^4.0 | 4 | 900 | **2^13.81** |
| DuX(2^16) | 2 | `0101` | words [0], dims [3] | 2^3.0 | 2 | 1800 | **2^13.81** |
| DuX(2^16) | 2 | `1101` | words [0], dims [3] | 2^3.0 | 2 | 1800 | **2^13.81** |
| DuX(2^16) | 1 | `0101` | words [0], dims [2] | 2^2.0 | 2 | 1800 | **2^12.81** |
| DuX(2^16) | 1 | `1101` | words [0], dims [2] | 2^2.0 | 2 | 1800 | **2^12.81** |
| DuX(2^8) | 6 | `0101` | words [2, 6, 10], dims [8, 8, 6] | 2^22.0 | 2 | 1800 | **2^32.81** |
| DuX(2^8) | 6 | `1101` | words [2, 6, 10], dims [8, 8, 6] | 2^22.0 | 2 | 1800 | **2^32.81** |
| DuX(2^8) | 5 | `0101` | words [2], dims [8] | 2^8.0 | 102 | 36 | **2^13.17** |
| DuX(2^8) | 5 | `1101` | words [2], dims [8] | 2^8.0 | 102 | 36 | **2^13.17** |
| DuX(2^8) | 4 | `0101` | words [2], dims [6] | 2^6.0 | 22 | 164 | **2^13.36** |
| DuX(2^8) | 4 | `1101` | words [2], dims [6] | 2^6.0 | 22 | 164 | **2^13.36** |
| DuX(2^8) | 3 | `0101` | words [2], dims [4] | 2^4.0 | 4 | 900 | **2^13.81** |
| DuX(2^8) | 3 | `1101` | words [2], dims [4] | 2^4.0 | 4 | 900 | **2^13.81** |
| DuX(2^8) | 2 | `0101` | words [0], dims [3] | 2^3.0 | 2 | 1800 | **2^13.81** |
| DuX(2^8) | 2 | `1101` | words [0], dims [3] | 2^3.0 | 2 | 1800 | **2^13.81** |
| DuX(2^8) | 1 | `0101` | words [0], dims [2] | 2^2.0 | 2 | 1800 | **2^12.81** |
| DuX(2^8) | 1 | `1101` | words [0], dims [2] | 2^2.0 | 2 | 1800 | **2^12.81** |

Cells whose class is newly usable but whose POSITION-1 margin is negative are distinguishers the combined row cannot read, and layers above 10 would need more rounds than the cipher has (r = l + 2, and both instances have twelve); together they are 2 of the 34 newly-usable cells and are listed in the JSON only.

## 3. YuX, for the record

The same trick gives YuX the cheap set {2, 3}; `kernel_table.py --cipher yux` shows `1110` going from dim K = 3 to 4 and `1111`
from 4 to 8.  The classes that BECOME nonzero (xx11, 11x1) all
require position 3 balanced, which is what `1111` already requires,
so no new route opens -- only the existing twelve-round route's
weight budget drops by about a quarter (4 rows per weight instead
of 3).

