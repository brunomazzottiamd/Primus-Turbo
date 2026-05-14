import torch

from primus_turbo.pytorch.ops import grouped_gemm as primus_turbo_gmm
from aiter.ops.triton.gmm import gmm as aiter_gmm


def test_gmm():
    device = "cuda"
    dtype = torch.bfloat16
    gs_dtype = torch.int64
    M = 267424
    K = 1280
    N = 2560
    G = 32
    grid_dim = 240

    torch.manual_seed(42)
    lhs = torch.randn(M, K, dtype=dtype, device=device)
    rhs = torch.randn(G, N, K, dtype=dtype, device=device)
    gs_list = [M // G] * G
    gs_list[-1] += M % G
    group_sizes = torch.tensor(gs_list, dtype=gs_dtype, device=device)

    out_primus_turbo = primus_turbo_gmm(
        lhs, rhs, group_sizes, trans_b=True, num_cu=grid_dim, work_stealing=True
    )
    out_aiter = aiter_gmm(
        lhs, rhs, group_sizes.to(torch.int32), preferred_element_type=dtype
    )

    torch.testing.assert_close(out_primus_turbo, out_aiter, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_gmm()
