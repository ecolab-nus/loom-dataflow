# Resolved Time Model — Wormhole 2D-Mesh Matmul

Unit: **cycles** (1 cycle ≈ 1 ns at 1 GHz)

Symbols: `BM`, `BN`, `BK` ∈ {32, 64, 128} — tile block dimensions.
`E` denotes an element count (product of matrix dims, e.g. `BM×BK`).

---

## Hardware Parameters

| Parameter | Value |
|-----------|-------|
| FPU peak throughput (fp16) | 1 TFLOPS/core = 1000 FLOPs/cycle |
| FPU per-unit throughput | 125 FLOPs/cycle (= 1000 / 8 units) |
| FPU max parallel tile pairs | 8 |
| Matrix engine efficiency | 1.0 (BM == BN) · 0.7 (BM ≠ BN) |
| Element size (f16) | 2 bytes |
| DRAM base latency `lat_d` | 454 cycles |
| DRAM bandwidth `BW_d` | 15 bytes/cycle |
| NoC H/V broadcast latency `lat_hv` | 344 cycles |
| NoC H/V broadcast bandwidth `BW_hv` | 27.88 bytes/cycle (≈ ×1000/27880) |
| All-broadcast latency `lat_a` | 586 cycles |
| All-broadcast bandwidth `BW_a` | 18.235 bytes/cycle (≈ ×10000/182350) |

---

## Operation Latency Models

### §A — FPU Matmul (`linalg.matmul`)

**Formula:**
```
P = min(8, (BM/32) × (BN/32))

Balanced (BM == BN):
  T_compute = 2 × BM × BN × BK / (125 × P)

Unbalanced (BM ≠ BN):
  T_compute = 14 × BM × BN × BK / (1250 × P)
```

**Expr ADT:**
```
IfElse(Eq(Sym("BM"), Sym("BN")),
  then: Div(Mul(Const(2), Mul(Mul(Sym("BM"), Sym("BN")), Sym("BK"))),
            Mul(Const(125), Min(Const(8), Div(Mul(Sym("BM"), Sym("BN")), Const(1024))))),
  else: Div(Mul(Const(14), Mul(Mul(Sym("BM"), Sym("BN")), Sym("BK"))),
            Mul(Const(1250), Min(Const(8), Div(Mul(Sym("BM"), Sym("BN")), Const(1024)))))
)
```

**JSON Expr:**
```json
{"IfElse": {
  "cond": {"Eq": [{"Sym": "BM"}, {"Sym": "BN"}]},
  "then_expr": {"Div": [
    {"Mul": [{"Const": 2}, {"Mul": [{"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}, {"Sym": "BK"}]}]},
    {"Mul": [{"Const": 125}, {"Min": [{"Const": 8}, {"Div": [{"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}, {"Const": 1024}]}]}]}
  ]},
  "else_expr": {"Div": [
    {"Mul": [{"Const": 14}, {"Mul": [{"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}, {"Sym": "BK"}]}]},
    {"Mul": [{"Const": 1250}, {"Min": [{"Const": 8}, {"Div": [{"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}, {"Const": 1024}]}]}]}
  ]}
}}
```

**Concrete example** (BM=BN=64, BK=32):
```
P = min(8, (64/32)×(64/32)) = min(8, 4) = 4
T = 2×64×64×32 / (125×4) = 262144 / 500 = 524 cycles
```

---

### §B — DRAM Fetch (`d`) — occupies NoC_H **and** NoC_V

A `d` workload fetches data from DRAM via both horizontal and vertical NoC.

**Formula (N workloads, S total elements):**
```
T_d = 454 × N + S × 2 / 15
```

**Expr ADT (single workload, E elements):**
```
Add(Const(454), Div(Mul(Sym("E"), Const(2)), Const(15)))
```

**JSON Expr (single workload on dims BM×BK):**
```json
{"Add": [{"Const": 454},
         {"Div": [{"Mul": [{"Mul": [{"Sym": "BM"}, {"Sym": "BK"}]}, {"Const": 2}]}, {"Const": 15}]}]}
```

**Concrete example** (BM=64, BK=32 → E=2048 elements):
```
T_d = 454 + 2048×2/15 = 454 + 273 = 727 cycles
```

---

### §C — All-Broadcast (`a`) — occupies NoC_H **and** NoC_V

An `a` workload broadcasts data to all cores via both NoC directions.

**Formula (N workloads, S total elements):**
```
T_a = 586 × N + S × 2 × 10000 / 182350
```

**Expr ADT (single workload, E elements):**
```
Add(Const(586), Div(Mul(Mul(Sym("E"), Const(2)), Const(10000)), Const(182350)))
```

**JSON Expr (single workload on dims BK×BN):**
```json
{"Add": [{"Const": 586},
         {"Div": [{"Mul": [{"Mul": [{"Sym": "BK"}, {"Sym": "BN"}]}, {"Const": 20000}]}, {"Const": 182350}]}]}
```

**Concrete example** (BK=32, BN=64 → E=2048 elements):
```
T_a = 586 + 2048×20000/182350 = 586 + 224 = 810 cycles
```

---

### §D — Horizontal Broadcast (`h`) — occupies **NoC_H only**

An `h` workload broadcasts data along rows (horizontal direction only).

**Formula (N workloads, S total elements):**
```
T_h = 344 × N + S × 2 × 1000 / 27880
```

**Expr ADT (single workload, E elements):**
```
Add(Const(344), Div(Mul(Mul(Sym("E"), Const(2)), Const(1000)), Const(27880)))
```

**JSON Expr (single workload on dims BK×BN):**
```json
{"Add": [{"Const": 344},
         {"Div": [{"Mul": [{"Mul": [{"Sym": "BK"}, {"Sym": "BN"}]}, {"Const": 2000}]}, {"Const": 27880}]}]}
```

**Concrete example** (BK=32, BN=64 → E=2048 elements):
```
T_h = 344 + 2048×2000/27880 = 344 + 147 = 491 cycles
```

---

### §E — Vertical Broadcast (`v`) — occupies **NoC_V only**

A `v` workload broadcasts data along columns (vertical direction only).
Same formula structure as `h`.

**Formula (N workloads, S total elements):**
```
T_v = 344 × N + S × 2 × 1000 / 27880
```

**Expr ADT (single workload, E elements):**
```
Add(Const(344), Div(Mul(Mul(Sym("E"), Const(2)), Const(1000)), Const(27880)))
```

**JSON Expr (single workload on dims BM×BK):**
```json
{"Add": [{"Const": 344},
         {"Div": [{"Mul": [{"Mul": [{"Sym": "BM"}, {"Sym": "BK"}]}, {"Const": 2000}]}, {"Const": 27880}]}]}
```

**Concrete example** (BM=64, BK=32 → E=2048 elements):
```
T_v = 344 + 2048×2000/27880 = 344 + 147 = 491 cycles
```

---

## Queue Composition Rule

When a queue holds multiple workloads (possibly of different transfer types), the total `resolved_time` is the **serial sum** of each workload's latency:

```
resolved_time(Q) = Σᵢ T_{opᵢ}(Eᵢ)
```

Note: `d` and `a` transfers occupy **both** NoC_H and NoC_V simultaneously; `h` occupies only NoC_H; `v` occupies only NoC_V. This determines which queue(s) each workload contributes to.

---

## Variant Suffix → Queue `resolved_time` Mapping

Variant names end in `_A_B` where A = transfer mode for matrix A (dims BM×BK) and B = transfer mode for matrix B (dims BK×BN).

| Suffix | NoC_H `resolved_time` | NoC_V `resolved_time` |
|--------|----------------------|-----------------------|
| `d_d` | `T_d(BM×BK) + T_d(BK×BN)` | same as NoC_H |
| `d_a` | `T_d(BM×BK) + T_a(BK×BN)` | same as NoC_H |
| `d_h` | `T_d(BM×BK) + T_h(BK×BN)` | `T_d(BM×BK)` only |
| `d_v` | `T_d(BM×BK)` only | `T_d(BM×BK) + T_v(BK×BN)` |
| `v_d` | `T_d(BK×BN)` only | `T_v(BM×BK) + T_d(BK×BN)` |
| `v_h` | `T_h(BK×BN)` only | `T_v(BM×BK)` only |

### JSON Expr for each pattern

**`d_d` — both queues identical:**
```json
{"Add": [{"Const": 908},
         {"Div": [{"Mul": [{"Const": 2},
                           {"Add": [{"Mul": [{"Sym": "BM"}, {"Sym": "BK"}]},
                                    {"Mul": [{"Sym": "BK"}, {"Sym": "BN"}]}]}]},
                  {"Const": 15}]}]}
```

**`d_a` — both queues identical:**
```json
{"Add": [{"Add": [{"Const": 454},
                  {"Div": [{"Mul": [{"Mul": [{"Sym": "BM"}, {"Sym": "BK"}]}, {"Const": 2}]}, {"Const": 15}]}]},
         {"Add": [{"Const": 586},
                  {"Div": [{"Mul": [{"Mul": [{"Sym": "BK"}, {"Sym": "BN"}]}, {"Const": 20000}]}, {"Const": 182350}]}]}]}
```

**`d_h` — NoC_H (d on A + h on B):**
```json
{"Add": [{"Add": [{"Const": 454},
                  {"Div": [{"Mul": [{"Mul": [{"Sym": "BM"}, {"Sym": "BK"}]}, {"Const": 2}]}, {"Const": 15}]}]},
         {"Add": [{"Const": 344},
                  {"Div": [{"Mul": [{"Mul": [{"Sym": "BK"}, {"Sym": "BN"}]}, {"Const": 2000}]}, {"Const": 27880}]}]}]}
```

**`d_h` — NoC_V (d on A only):**
```json
{"Add": [{"Const": 454},
         {"Div": [{"Mul": [{"Mul": [{"Sym": "BM"}, {"Sym": "BK"}]}, {"Const": 2}]}, {"Const": 15}]}]}
```

**`d_v` — NoC_H (d on A only):** same as `d_h` NoC_V above.

**`d_v` — NoC_V (d on A + v on B):**
```json
{"Add": [{"Add": [{"Const": 454},
                  {"Div": [{"Mul": [{"Mul": [{"Sym": "BM"}, {"Sym": "BK"}]}, {"Const": 2}]}, {"Const": 15}]}]},
         {"Add": [{"Const": 344},
                  {"Div": [{"Mul": [{"Mul": [{"Sym": "BK"}, {"Sym": "BN"}]}, {"Const": 2000}]}, {"Const": 27880}]}]}]}
```

**`v_d` — NoC_H (d on B only):**
```json
{"Add": [{"Const": 454},
         {"Div": [{"Mul": [{"Mul": [{"Sym": "BK"}, {"Sym": "BN"}]}, {"Const": 2}]}, {"Const": 15}]}]}
```

**`v_d` — NoC_V (v on A + d on B):**
```json
{"Add": [{"Add": [{"Const": 344},
                  {"Div": [{"Mul": [{"Mul": [{"Sym": "BM"}, {"Sym": "BK"}]}, {"Const": 2000}]}, {"Const": 27880}]}]},
         {"Add": [{"Const": 454},
                  {"Div": [{"Mul": [{"Mul": [{"Sym": "BK"}, {"Sym": "BN"}]}, {"Const": 2}]}, {"Const": 15}]}]}]}
```

**`v_h` — NoC_H (h on B only):**
```json
{"Add": [{"Const": 344},
         {"Div": [{"Mul": [{"Mul": [{"Sym": "BK"}, {"Sym": "BN"}]}, {"Const": 2000}]}, {"Const": 27880}]}]}
```

**`v_h` — NoC_V (v on A only):**
```json
{"Add": [{"Const": 344},
         {"Div": [{"Mul": [{"Mul": [{"Sym": "BM"}, {"Sym": "BK"}]}, {"Const": 2000}]}, {"Const": 27880}]}]}
```

---

## Verification Examples

| Case | Parameters | Formula | Result |
|------|-----------|---------|--------|
| FPU balanced | BM=BN=64, BK=32 | `2×64×64×32/(125×4)` | **524 cycles** |
| FPU unbalanced | BM=32, BN=64, BK=32 | `14×32×64×32/(1250×2)` | **732 cycles** |
| `d_d` NoC | BM=BN=64, BK=32 | `908+(64×32+32×64)×2/15` | **1454 cycles** |
| `d` single | BM=64, BK=32 | `454+2048×2/15` | **727 cycles** |
| `h` single | BK=32, BN=64 | `344+2048×2000/27880` | **491 cycles** |
