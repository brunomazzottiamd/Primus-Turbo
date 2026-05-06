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

### Distributed setup

Each process group for concurrent all-gathers is allocated independently (`dist.new_group`) so
that multiple RCCL communicators can run in parallel without serializing on a single stream.
Communication happens on dedicated `torch.cuda.Stream` objects; the compute stream only
synchronizes with them at wall-time measurement boundaries.

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

We get the following perf numbers for both the CK and Triton backend:

### Triton performance numbers

| GRID_DIM | GEMM only (ms) | Sequential (ms) | Overlap GEMM (ms) | Overlap wall (ms) | Slowdown (overlap/gemm) |
|---:|---:|---:|---:|---:|---:|
| 128 | 3.603 | 3.811 | 3.863 | 3.914 | 1.07× |
| 192 | 2.522 | 2.778 | 2.823 | 2.872 | 1.12× |
| 208 | 2.411 | 2.607 | 2.650 | 2.697 | 1.10× |
| 216 | 2.357 | 2.553 | 2.604 | 2.654 | 1.11× |
| 220 | 2.368 | 2.534 | 2.584 | 2.632 | 1.09× |
| 224 | 2.294 | 2.493 | 2.521 | 2.571 | 1.10× |
| 228 | 2.240 | 2.438 | 4.868 | 4.916 | 2.17× |
| 232 | 2.249 | 2.419 | 4.751 | 4.800 | 2.11× |
| 240 | 2.194 | 2.366 | 4.634 | 4.681 | 2.11× |
| 248 | 2.150 | 2.305 | 4.456 | 4.505 | 2.07× |
| 256 | 2.150 | 2.275 | 4.365 | 4.413 | 2.03× |

### CK Performance numbers:

| GRID_DIM | GEMM only (ms) | Sequential (ms) | Overlap GEMM (ms) | Overlap wall (ms) | Slowdown (overlap/gemm) |
|---:|---:|---:|---:|---:|---:|
| 128 | 2.664 | 2.998 | 3.019 | 3.068 | 1.13× |
| 192 | 1.972 | 2.148 | 2.166 | 2.211 | 1.10× |
| 208 | 1.926 | 1.997 | 2.067 | 2.115 | 1.07× |
| 216 | 1.911 | 1.987 | 2.036 | 2.081 | 1.07× |
| 220 | 1.905 | 1.988 | 2.010 | 2.057 | 1.06× |
| 224 | 1.899 | 1.963 | 1.970 | 2.017 | 1.04× |
| 228 | 1.894 | 1.941 | 3.808 | 3.854 | 2.01× |
| 232 | 1.864 | 1.916 | 3.755 | 3.801 | 2.02× |
| 240 | 1.835 | 1.874 | 3.692 | 3.737 | 2.01× |
| 248 | 1.822 | 1.866 | 3.555 | 3.601 | 1.95× |
| 256 | 1.778 | 1.807 | 3.452 | 3.499 | 1.94× |

MI350 contains 256 CUs. With RCCL kernel using 16 CUs, when GroupGEMM use 240 or less CUs, there should be no performance
degradation, but this table shows a big slowdown when groupGEMM uses 228 or more CUs. Reason is due to the algorithm used to 
dispatch workgroups to CU. From the ATT trace in the next section, we can see that for the case with 228 CUs for groupgemm,
there are 13 CUs idle, at the same time, there are multiple works groups dispatched to the same CUs, which almost doubles
the groupgemm kernel time. A second note is that the CK kernel is faster then the Triton, and we will try to optimize the
Triton kernel.

With a newer version of fw, we can move the cliff from 240CUs for groupgemm, see the following table:

(add a table here for perf numbers)

## 4. ATT Trace Analysis

ATT trace indicates that with RCCL using 16 CUs, there are 240 CUs availabe for groupgemm, but the firmware algorithm stills
dispatches two workgroups to the same CUs, which doubles kernel time, and at the same time, there are 13CUs idle.

(Add a picture for the ATT trace)

## 5. Optimization

The Triton Grouped GEMM kernel in its original form used a **static tile assignment**: each
wave-front (CU) received a fixed subset of output tiles pre-determined at launch time via a
round-robin stride pattern (`for global_tile_id in range(pid, total_tiles, NUM_XCDS)`).  When
RCCL all-gather traffic simultaneously saturates the HBM bus and PCIe/xGMI interconnect, some
CUs stall waiting on memory while others finish early.  Because tiles are statically assigned,
early-finishing CUs sit idle while stalled CUs still hold work — the kernel cannot retire until
the *slowest* CU finishes, so RCCL-induced memory pressure linearly inflates the GEMM latency.

With the new firmware, we can configure at most 240 CUs to avoid the slowdown. But in the training scenario, there are also
scenarios that there is no overlap with RCCL, and we want to configure 256 workgroups for groupGEMM to fully utilize all 
hardware resources. We introduce the work stealing to dynamically run different tiles on CUs.

### Work Stealing

Work stealing is a dynamic tile-scheduling strategy that eliminates the load-imbalance problem
caused by RCCL memory pressure.

**Static assignment (the problem).**  In the original kernel each CU is pre-assigned a fixed
stripe of output tiles at launch time via a round-robin stride:

```python
for tile_id in range(pid, total_tiles, GRID_DIM):
    compute(tile_id)
```

Every CU must finish its entire stripe before the kernel can retire.  Under concurrent RCCL
traffic some CUs stall on HBM/interconnect latency; those CUs hold unfinished tiles while
fast CUs sit idle.  The effective kernel time becomes `max(per-CU latency)`, so a single
stalled CU stretches the whole kernel.

**Work stealing (the fix).**  A single 64-bit counter (`tile_counter`) lives in global memory,
initialised to zero.  Instead of striding through a pre-assigned range, each CU atomically
increments the counter to claim the next available tile:

```python
while True:
    tile_id = tl.atomic_add(tile_counter_ptr, 1)   # claim one tile
    if tile_id >= total_tiles:
        break
    compute(tile_id)
```

CUs that finish quickly loop back and steal more tiles; CUs stalled by memory pressure
naturally claim fewer.  The kernel retires as soon as the last tile is computed — no CU
waits for another.

**Why the overhead is small.**  Each `tl.atomic_add` touches one L2-cached cache line.  The
operation is fast (~10–20 cycles) relative to a full tile computation (hundreds of cycles of
matrix math), so the atomic is not a bottleneck even at 256 CUs all hammering the same
counter simultaneously.

**Boundary condition.**  The loop check `tile_id >= total_tiles` correctly handles the case
where more CUs are launched than there are tiles: excess CUs exit immediately without doing
any work.

## 6. Performance numbers 

With the work stealing optimization, we got the following performance numbers:

| GRID_DIM | GEMM only (ms) | Sequential (ms) | Overlap GEMM (ms) | Overlap wall (ms) | Slowdown (overlap/gemm) |
|---:|---:|---:|---:|---:|---:|
| 128 | 3.597 | 3.798 | 3.873 | 3.925 | 1.08× |
| 192 | 2.531 | 2.803 | 2.821 | 2.868 | 1.11× |
| 208 | 2.381 | 2.628 | 2.660 | 2.708 | 1.12× |
| 216 | 2.367 | 2.575 | 2.650 | 2.698 | 1.12× |
| 220 | 2.364 | 2.558 | 2.609 | 2.656 | 1.10× |
| 224 | 2.302 | 2.490 | 2.543 | 2.591 | 1.10× |
| 228 | 2.278 | 2.449 | 2.538 | 2.586 | 1.11× |
| 232 | 2.230 | 2.412 | 2.518 | 2.566 | 1.13× |
| 240 | 2.147 | 2.358 | 2.503 | 2.550 | 1.17× |
| 248 | 2.151 | 2.292 | 2.520 | 2.568 | 1.17× |
| 256 | 2.054 | 2.227 | 2.518 | 2.565 | 1.23× |

---

From the results, we can see that the biggest slowdown is 23% compared to running the GroupGEMM using all 256CUs.
There are two factor invovlved when overlapping with RCCL, 16 fewer CUs are used for GroupGEMM and there is a 
general 10% overhead.


### Why the improvement is large

Without work stealing, a single stalled CU forces all other CUs to wait because the kernel
cannot retire until every CU finishes its statically assigned tiles.  Under RCCL traffic, a
fraction of CUs experience severe HBM latency; the effective kernel time becomes `max(CU
latencies)` rather than `mean(CU latencies)`.  Work stealing converts the effective latency to
approximately `mean + small_steal_overhead`, a much better outcome when variance across CUs is
high.
