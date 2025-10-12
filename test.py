import torch
import torch.nn as nn
import torch.nn.functional as F 

import triton
import triton.language as tl

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=8, num_stages=2),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _swiglu_backward(
    do_ptr, x_ptr, g1_ptr, g2_ptr,
    dw1_ptr, dw2_ptr,
    M, N, K,
    stride_do_m, stride_do_n,
    stride_x_m,  stride_x_k,
    stride_g_m,  stride_g_n,
    stride_w_n,  stride_w_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute:
      dw1 += (g1 * do)^T @ x   -> (N, K)
      dw2 += (g2 * do)^T @ x   -> (N, K)

    Tiling: program_id(0) -> N tiles, program_id(1) -> K tiles.
    Reduce over M in inner loop.
    """

    pid_n = tl.program_id(axis=0)   # tile index along N
    pid_k = tl.program_id(axis=1)   # tile index along K

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)   # (BLOCK_N,)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)   # (BLOCK_K,)
    offs_m = tl.arange(0, BLOCK_SIZE_M)                          # (BLOCK_M,)

    # accumulators in (BLOCK_N, BLOCK_K), use float32
    dw1_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    dw2_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)

    for mt in range(num_m_tiles):
        m_start = mt * BLOCK_SIZE_M
        # validity masks
        valid_m = (m_start + offs_m) < M        # (BM,)
        valid_n = offs_n < N                    # (BN,)
        valid_k = offs_k < K                    # (BK,)

        # build pointer arrays for this m tile:
        # x[(m_start + offs_m), offs_k] -> (BM, BK)
        x_ptrs = x_ptr + (m_start + offs_m)[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        # g and do [(m_start + offs_m), offs_n] -> (BM, BN)
        g1_ptrs = g1_ptr + (m_start + offs_m)[:, None] * stride_g_m + offs_n[None, :] * stride_g_n
        g2_ptrs = g2_ptr + (m_start + offs_m)[:, None] * stride_g_m + offs_n[None, :] * stride_g_n
        do_ptrs = do_ptr  + (m_start + offs_m)[:, None] * stride_do_m + offs_n[None, :] * stride_do_n

        # masks for loads
        mask_x = valid_m[:, None] & valid_k[None, :]
        mask_g = valid_m[:, None] & valid_n[None, :]
        mask_do = mask_g

        # load tiles (cast to float32)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)    # (BM, BK)
        g1_tile = tl.load(g1_ptrs, mask=mask_g, other=0.0).to(tl.float32)  # (BM, BN)
        g2_tile = tl.load(g2_ptrs, mask=mask_g, other=0.0).to(tl.float32)  # (BM, BN)
        do_tile = tl.load(do_ptrs, mask=mask_do, other=0.0).to(tl.float32) # (BM, BN)

        # IMPORTANT: transpose (g * do) -> shape (BN, BM) so dot works:
        # (BN, BM) x (BM, BK) -> (BN, BK)
        mul1_T = (g1_tile * do_tile).trans()   # (BN, BM)
        mul2_T = (g2_tile * do_tile).trans()   # (BN, BM)

        # accumulate
        dw1_acc = tl.dot(mul1_T, x_tile, dw1_acc)  # (BN, BK)
        dw2_acc = tl.dot(mul2_T, x_tile, dw2_acc)

    # store accumulators into dw1/dw2 tiles (casting handled by tl.store)
    store_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    dw1_ptrs = dw1_ptr + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
    dw2_ptrs = dw2_ptr + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k

    tl.store(dw1_ptrs, dw1_acc, mask=store_mask)
    tl.store(dw2_ptrs, dw2_acc, mask=store_mask)
    return


def swiglu_backward_wrapper(x, do, dA, dB):
    # x: (M, K), do/dA/dB: (M, N)
    M, K = x.shape
    M2, N = do.shape
    assert M == M2 and dB.shape == do.shape
    
    # w: (N,K)
    ddw1 = torch.zeros((N, K), device=x.device, dtype=x.dtype)
    ddw2 = torch.zeros((N, K), device=x.device, dtype=x.dtype)

    x_c = x.contiguous()
    do_c = do.contiguous()
    dA_c = dA.contiguous()
    dB_c = dB.contiguous()
    ddw1_c = ddw1.contiguous()
    ddw2_c = ddw2.contiguous()

    s_do_m, s_do_n = do_c.stride()
    s_x_m, s_x_k = x_c.stride()
    s_g_m, s_g_n = dA_c.stride()
    s_w_n, s_w_k = ddw1_c.stride()
    
    
    grid=lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(K, META['BLOCK_SIZE_K']))
    _swiglu_backward[grid](
        do_c, x_c, dA_c, dB_c,
        ddw1_c, ddw2_c,
        M, N, K,
        s_do_m, s_do_n,
        s_x_m, s_x_k,
        s_g_m, s_g_n,
        s_w_n, s_w_k
    )
    return ddw1_c, ddw2_c


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W1, W2):
        Z1 = X @ W1.T
        Z2 = X @ W2.T
        S = torch.sigmoid(Z2)
        g1=S * Z2
        g2=Z1 * S * (1 + Z2 * (1 - S))
        
        Y = Z1 * g1
        ctx.save_for_backward(X, W1, W2, Z1, Z2, S, g1, g2)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, W1, W2, Z1, Z2, S, g1, g2 = ctx.saved_tensors

        dZ1 = grad_output * g1
        dZ2 = grad_output * g2

        dW1 = dZ1.T@X
        dW2 = dZ2.T@X

        return None, dW1, dW2


class SwiGLU_Layer_Triton(nn.Module):
  def __init__(self, dim, hidden_dim):
    super().__init__()
    self.w1=nn.Linear(dim, hidden_dim, bias=False)
    self.w2=nn.Linear(dim, hidden_dim, bias=False)
    self.swiglu=SwiGLUFunction.apply

  def forward(self,x):
    return self.swiglu(x, self.w1.weight, self.w2.weight)


class SwiGLU_Layer_Pytorch(nn.Module):
  def __init__(self, dim, hidden_dim):
    super().__init__()
    self.w1=nn.Linear(dim, hidden_dim, bias=False)
    self.w2=nn.Linear(dim, hidden_dim, bias=False)

  def forward(self,x):
    z1=self.w1(x)
    z2=self.w2(x)
    return z1*F.sigmoid(z2)*z2


# quick correctness test
if __name__ == "__main__":
    torch.manual_seed(0)
    m, hidden_dim, dim = 682, 1024, 128   # close to the sizes you printed
    device = 'cuda'
    dtype = torch.float16
    
    x=torch.randn(m, dim, device=device, dtype=torch.float32)
    do=torch.randn(m, hidden_dim, device=device, dtype=torch.float32)

    swiglu_pytorch=SwiGLU_Layer_Pytorch(dim, hidden_dim).to("cuda")
    swiglu_triton=SwiGLU_Layer_Triton(dim, hidden_dim).to("cuda")
    swiglu_triton.w1.weight.data.copy_(swiglu_pytorch.w1.weight.data)
    swiglu_triton.w2.weight.data.copy_(swiglu_pytorch.w2.weight.data)
    
    print("swiglu_triton shape", swiglu_triton.w1.weight.shape, swiglu_triton.w1.weight[:4,:4])
    print("swiglu_triton shape", swiglu_pytorch.w1.weight.shape, swiglu_pytorch.w1.weight[:4,:4]) 
    assert torch.allclose(swiglu_triton.w1.weight, swiglu_pytorch.w1.weight, rtol=1e-03, atol=1e-05, equal_nan=True), "forward w1 discripency"
    assert torch.allclose(swiglu_triton.w2.weight, swiglu_pytorch.w2.weight, rtol=1e-03, atol=1e-05, equal_nan=True), "forward w2 discripency"
    
    out_triton=swiglu_triton(x)
    out_torch=swiglu_pytorch(x)
    torch.cuda.synchronize()
    print("out_triton",out_triton.shape, out_triton[:4,:4])
    print("out_torch",out_torch.shape, out_torch[:4,:4])
    max_diff = (out_triton - out_torch).abs().max()
    mean_diff = (out_triton - out_torch).abs().mean()
    print("forward max_diff","forward mean_diff",max_diff, mean_diff)

    out_triton.backward(do, retain_graph=True)
    out_torch.backward(do, retain_graph=True)
    
    max_diff = (swiglu_triton.w1.weight.grad - swiglu_pytorch.w1.weight.grad).abs().max()
    mean_diff = (swiglu_triton.w1.weight.grad - swiglu_pytorch.w1.weight.grad).abs().mean()

    print("backward out_triton",swiglu_triton.w1.weight.grad.shape, swiglu_triton.w1.weight.grad[:4,:4])
    print("backward out_torch",swiglu_pytorch.w1.weight.grad.shape, swiglu_pytorch.w1.weight.grad[:4,:4])
    print("backward max_diff","backward mean_diff",max_diff, mean_diff)