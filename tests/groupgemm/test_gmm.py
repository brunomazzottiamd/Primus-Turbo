import torch

from primus_turbo.pytorch.ops import grouped_gemm as primus_turbo_gmm
from aiter.ops.triton.gmm import gmm as aiter_gmm

DEFAULT_GRID_DIM = 256


def gen_tensors():
    device = "cuda"
    dtype = torch.bfloat16
    gs_dtype = torch.int64
    M = 267424
    K = 1280
    N = 2560
    G = 32

    torch.manual_seed(42)
    lhs = torch.randn(M, K, dtype=dtype, device=device)
    rhs = torch.randn(G, N, K, dtype=dtype, device=device)
    gs_list = [M // G] * G
    gs_list[-1] += M % G
    group_sizes = torch.tensor(gs_list, dtype=gs_dtype, device=device)
    group_offs = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(
        group_sizes
    )

    return lhs, rhs, group_sizes, group_offs


def run_primus_turbo(lhs, rhs, group_sizes, group_offs, grid_dim=DEFAULT_GRID_DIM):
    return primus_turbo_gmm(
        lhs,
        rhs,
        group_sizes,
        group_offs=group_offs,
        trans_b=True,
        num_cu=grid_dim,
        work_stealing=True,
    )


def run_aiter(lhs, rhs, group_sizes, grid_dim=DEFAULT_GRID_DIM):
    # TODO: update AITER to accept int64 group_sizes.
    return aiter_gmm(
        lhs,
        rhs,
        group_sizes.to(torch.int32),
        preferred_element_type=lhs.dtype,
        grid_dim=grid_dim,
    )


def test_gmm(grid_dim=DEFAULT_GRID_DIM):
    lhs, rhs, group_sizes, group_offs = gen_tensors()
    out_primus_turbo = run_primus_turbo(
        lhs, rhs, group_sizes, group_offs, grid_dim=grid_dim
    )
    out_aiter = run_aiter(lhs, rhs, group_sizes, grid_dim=grid_dim)
    torch.testing.assert_close(out_primus_turbo, out_aiter, atol=1e-3, rtol=1e-3)


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
        torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    min_time = min(times)
    avg_time = sum(times) / iters
    max_time = max(times)

    return min_time, avg_time, max_time


def bench_gmm(grid_dim=DEFAULT_GRID_DIM):
    lhs, rhs, group_sizes, group_offs = gen_tensors()

    def primus_turbo_fn():
        return run_primus_turbo(lhs, rhs, group_sizes, group_offs, grid_dim=grid_dim)

    min_pt, avg_pt, max_pt = bench(primus_turbo_fn)

    def aiter_fn():
        return run_aiter(lhs, rhs, group_sizes, grid_dim=grid_dim)

    min_a, avg_a, max_a = bench(aiter_fn)

    print(f"Primus-Turbo: min={min_pt:.4f}, avg={avg_pt:.4f}, max={max_pt:.4f}")
    print(f"       AITER: min={min_a:.4f}, avg={avg_a:.4f}, max={max_a:.4f}")


if __name__ == "__main__":
    # TODO: add CLI parser that accepts --grid-dim and --work-stealing
    grid_dim = 240
    print("Testing...")
    test_gmm(grid_dim=grid_dim)
    print("Benchmarking...")
    bench_gmm(grid_dim=grid_dim)
