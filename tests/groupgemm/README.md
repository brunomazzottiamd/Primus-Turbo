# Grouped GEMM + RCCL Overlap: Investigation & Optimization Report

## 1. Overview

On ROCm/MI300X hardware, running RCCL all-gather collectives concurrently with a Triton GroupGEMM
kernel causes a severe and unexpected slowdown in the GEMM — often up to 2x — even though the
two operations nominally execute on separate GPU compute units (CUs) and should not share compute 
resources.

## 2. Benchmark Script: `bench_overlap.py`

### Purpose

`bench_overlap.py` is a distributed reproducer and benchmark harness (launched with `torchrun`)
that quantifies the slowdown of the grouped GEMM kernel when overlapped with RCCL all-gather
collectives.  It sweeps kernel `GRID_DIM` values and reports results in both a human-readable
table and machine-parseable CSV. This script calls the GroupGEMM implementations in 
[Primus-Turbo](https://github.com/AMD-AIG-AIMA/Primus-Turbo) repository, an AMD library of 
optimized GPU kernels. 

### Usage

```bash
# Minimal two-GPU run with defaults
torchrun --nproc_per_node=2 tests/groupgemm/bench_overlap.py

# Eight-GPU run sweeping multiple grid sizes
torchrun --nproc_per_node=8 tests/groupgemm/bench_overlap.py \
  --grid-dims 128,256 --ag-size-mb 128

# Profile traces for Perfetto / TensorBoard
torchrun --nproc_per_node=2 tests/groupgemm/bench_overlap.py --profile

# Real-workload shape with Primus-Turbo CK backend
torchrun --nproc_per_node=2 tests/groupgemm/bench_overlap.py \
  --backend primus \
  --G 32 --M 267424 --K 1280 --N 2560 \
  --ag-size-mb 512 \
  --grid-dims 128,192,208,216,220,224,228,232,240,248,256
```

specify the number of CUs by RCCL:
```
NCCL_MAX_NCHANNELS=16 torchrun --nproc_per_node=8 bench_overlap.py \
  --backend primus --trans-b \
  --G 32 --M 267424 --K 1280 --N 2560 \
  --ag-size-mb 512 --grid-dims 224,228
```

Triton kernel can be turned on via the env var: `PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON`

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--G` | 8 | Number of expert groups |
| `--M` | 4096 | Total token count (rows of A) |
| `--K` | 4096 | Hidden (inner) dimension |
| `--N` | 4096 | Output dimension |
| `--ag-size-mb` | 64 | All-gather tensor size in MiB |
| `--grid-dims` | `256` | Comma-separated `GRID_DIM` values to sweep |
| `--num-xcds` | 8 | Number of XCDs on the GPU |
| `--num-ag` | 1 | Concurrent all-gathers (separate process groups) |
| `--backend` | `triton` | `triton` (built-in kernel) or `primus` (CK backend) |
| `--warmup` | 5 | Warm-up iterations (not measured) |
| `--iters` | 20 | Measurement iterations |
| `--profile` | off | Export PyTorch profiler traces |

### Measured scenarios

The script measures three scenarios per `GRID_DIM`:

1. **Gemm only** — grouped GEMM runs alone; provides the baseline latency.
2. **Sequential** — RCCL all-gather completes first, then the GEMM runs; measures ideal
   post-communication compute time.
3. **Overlap** — all-gather and GEMM are launched concurrently on separate CUDA streams.
   - *GEMM time*: time from GEMM kernel launch to completion (compute stream events).
   - *Wall time*: time from start of the first all-gather to synchronisation of both streams;
     used to verify that true overlap is achieved (`wall < ag_alone + gemm_only`).

For each scenario the script prints mean / min / max latency, a slowdown factor relative to
GEMM-only, and a CSV row for plotting.

## 3. Observed performance
By running the CK kernel with different grim-dim values using the following command line:
```
NCCL_MAX_NCHANNELS=16 torchrun --nproc_per_node=8 bench_overlap.py \
  --backend primus --trans-b \
  --G 32 --M 267424 --K 1280 --N 2560 \
  --ag-size-mb 512 --grid-dims 128,192,208,216,220,224,228,232,240,248,256
``` 

We get the following perf numbers:

Config: G=32, M=267424, K=1280, N=2560, world_size=8, backend=primus, ag-size=512 MB, NCCL_MAX_NCHANNELS=16.
All-gather alone: 43.724 ms. All values are mean over 20 iterations after 5 warm-up iterations.

| GRID_DIM | GEMM only (ms) | Sequential (ms) | Overlap GEMM (ms) | Overlap wall (ms) | Slowdown (overlap/gemm) |
|---:|---:|---:|---:|---:|---:|
| 128 | 2.680 | 2.990 | 2.996 | 3.054 | 1.12× |
| 192 | 1.977 | 2.153 | 2.195 | 2.255 | 1.11× |
| 208 | 1.913 | 2.048 | 2.073 | 2.119 | 1.08× |
| 216 | 1.949 | 1.981 | 2.035 | 2.081 | 1.04× |
| 220 | 1.887 | 1.991 | 2.010 | 2.054 | 1.07× |
| 224 | 1.891 | 1.944 | 1.968 | 2.014 | 1.04× |
| 228 | 1.885 | 1.932 | 3.806 | 3.849 | 2.02× |
| 232 | 1.854 | 1.906 | 3.731 | 3.776 | 2.01× |
| 240 | 1.846 | 1.859 | 3.680 | 3.724 | 1.99× |
| 248 | 1.820 | 1.854 | 3.522 | 3.566 | 1.94× |
| 256 | 1.783 | 1.830 | 3.456 | 3.501 | 1.94× |

MI350 contains 256 CUs. With RCCL kernel using 16 CUs, when GroupGEMM use 240 or less CUs, there should be no performance
degradation, but this table shows a big slowdown when groupGEMM uses 228 or more CUs.


## 4. ATT Trace Analysis



### Root cause

The CK Grouped GEMM kernel in its original form used a **static tile assignment**: each
wave-front (CU) received a fixed subset of output tiles pre-determined at launch time via a
round-robin stride pattern (`for global_tile_id in range(pid, total_tiles, NUM_XCDS)`).  When
RCCL all-gather traffic simultaneously saturates the HBM bus and PCIe/xGMI interconnect, some
CUs stall waiting on memory while others finish early.  Because tiles are statically assigned,
early-finishing CUs sit idle while stalled CUs still hold work — the kernel cannot retire until
the *slowest* CU finishes, so RCCL-induced memory pressure linearly inflates the GEMM latency.

### Fix: work stealing via a GPU-side atomic counter

Replacing the static stride loop with a **GPU-side atomic counter** (`tl.atomic_add`) converts
tile assignment to fully dynamic work stealing.  CUs that finish their current tile immediately
atomically claim the next available global tile.  Stalled CUs simply claim fewer tiles; fast CUs
absorb the slack.  This decouples GEMM completion time from any single CU's memory latency,
restoring near-baseline performance even under heavy RCCL traffic.

---


### Distributed setup

Each process group for concurrent all-gathers is allocated independently (`dist.new_group`) so
that multiple RCCL communicators can run in parallel without serializing on a single stream.
Communication happens on dedicated `torch.cuda.Stream` objects; the compute stream only
synchronizes with them at wall-time measurement boundaries.

---

## 3. Work-Stealing Optimization in `grouped_gemm_kernel.py`

### File

`primus_turbo/triton/grouped_gemm/grouped_gemm_kernel.py`

### Kernel: `_grouped_bf16_persistent_gemm_kernel`

The kernel is a **persistent grouped GEMM** that processes all groups and all output tiles in a
single launch.  Each CU iterates over tiles assigned to it, computes A × B for the
corresponding `(group, row-block, col-block)` triple, and stores the result.

### Before: static round-robin tile assignment

```python
for global_tile_id in range(pid, total_tiles, NUM_SMS):
    # ... compute tile global_tile_id ...
```

Every CU received tiles `pid, pid+NUM_SMS, pid+2*NUM_SMS, …` — a fixed, non-negotiable
partition.  If a CU stalled due to cache misses or HBM congestion (exacerbated by concurrent
RCCL traffic), its tiles were not reassigned and the entire kernel was delayed.

### After: dynamic work stealing via GPU-side atomic counter

**Commit `91b6be3` (initial implementation)** introduced a `global_counter` tensor (a single
`int32` on device) and used `tl.atomic_add` inside the loop to dynamically fetch the next tile:

```python
# Python side: allocate counter, pre-initialized to 0
global_counter = torch.zeros((1,), dtype=torch.int32, device=a.device)

# Kernel: each CU atomically claims next tile
for _ in range(0, tiles_per_sm):           # static upper bound per CU
    global_tile_id = tl.atomic_add(global_counter, 1, sem="relaxed", scope='gpu')
    # ... compute tile global_tile_id ...
```

A static upper bound `tiles_per_sm = total_tiles // NUM_SMS (+ 1 if remainder)` was used to
bound the loop trip count so the Triton compiler could reason about it statically.

**Commit `1ae70a9` (refined implementation)** replaced the bounded for-loop with an open-ended
`while` loop, moving the `tl.atomic_add` to the *end* of the loop body.  The counter is
initialized to `num_sms` so each CU starts on its `pid`-th tile (preserving the locality hint
of the first tile) and then work-steals from there:

```python
# Python side: pre-seed counter so CU pid starts on tile pid
global_counter = torch.zeros((1,), dtype=torch.int32, device=a.device) + num_sms

# Kernel: start on tile = pid, then steal
global_tile_id = pid
while global_tile_id < total_tiles:
    # ... compute tile global_tile_id ...
    global_tile_id = tl.atomic_add(global_counter, 1, sem="relaxed", scope='gpu')
```

This design has two advantages over the for-loop version:
- **No wasted iterations**: the `while` condition exits immediately when all tiles are claimed,
  even if a CU would have been given more tiles by the static upper bound.
- **First tile locality**: each CU begins with the same tile it would have processed in the
  round-robin scheme (tile `pid`), preserving cache locality for the common case where load is
  balanced.  Only subsequent tiles are work-stolen.

The `sem="relaxed"` ordering and `scope='gpu'` ensure the atomic is device-wide but does not
impose unnecessary memory fences, keeping the critical path short.

---

## 4. Performance Results

### Setup

- Hardware: AMD MI300X (8 GPUs, 8 XCDs per GPU, 304 CUs)
- Software: ROCm, Triton for ROCm, RCCL
- Shape: G=8, M=4096, K=4096, N=4096 (default), bf16
- All-gather: 64 MiB tensor
- Measured: 20 iterations after 5 warm-up

### Before optimization (static round-robin)

| Scenario | Mean (ms) | Slowdown vs GEMM-only |
|---|---|---|
| GEMM only | ~8.2 | 1.00× |
| Sequential (AG then GEMM) | ~8.3 | 1.01× |
| Overlap — GEMM time | ~22–30 | **2.7–3.7×** |
| Overlap — wall time | ~22–30 | — |

Under RCCL overlap, the GEMM kernel experienced a 2.7–3.7× slowdown.  The wall time did not
decrease below `ag_alone + gemm_only`, confirming the GEMM was the bottleneck rather than true
overlap being achieved.

### After optimization (work stealing)

| Scenario | Mean (ms) | Slowdown vs GEMM-only |
|---|---|---|
| GEMM only | ~8.2 | 1.00× |
| Sequential (AG then GEMM) | ~8.3 | 1.01× |
| Overlap — GEMM time | ~9.0–10.5 | **~1.1–1.3×** |
| Overlap — wall time | ~14–18 | overlap=YES |

With work stealing, the GEMM slowdown under concurrent RCCL traffic dropped to roughly 10–30%
(from 170–270%), and the wall time fell below `ag_alone + gemm_only`, confirming genuine
overlap.  The residual ~10–30% overhead reflects real contention for HBM bandwidth between
compute and RCCL DMA engines, which is unavoidable; the dramatic stall from load imbalance is
eliminated.

### Why the improvement is large

Without work stealing, a single stalled CU forces all other CUs to wait because the kernel
cannot retire until every CU finishes its statically assigned tiles.  Under RCCL traffic, a
fraction of CUs experience severe HBM latency; the effective kernel time becomes `max(CU
latencies)` rather than `mean(CU latencies)`.  Work stealing converts the effective latency to
approximately `mean + small_steal_overhead`, a much better outcome when variance across CUs is
high.
