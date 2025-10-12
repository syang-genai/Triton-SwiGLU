import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

DEVICE = triton.runtime.driver.active.get_active_torch_device()
print("device",DEVICE)

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64})
    ]



def get_hip_autotune_config():
    sizes = [{'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}]
    return [triton.Config(s) for s in sizes]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _swiglu_forward(
    x_ptr, w1_ptr, w2_ptr, o_ptr, g1_ptr, g2_ptr,
    M, N, K,
    stridexm, stridexk,
    stridewk, stridewn,
    strideom, strideon,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # tile indices (each program_id axis corresponds to a tile)
    pid_m = tl.program_id(axis=0)   # tile row index
    pid_n = tl.program_id(axis=1)   # tile col index

    # element offsets for this tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)   # shape (BLOCK_SIZE_M,)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)   # shape (BLOCK_SIZE_N,)

    # k-dimension local indices
    offs_k = tl.arange(0, BLOCK_SIZE_K)                          # shape (BLOCK_SIZE_K,)

    # create pointer matrices for loads (x: M x K, w: N x K-->K,N)
    x_ptrs = x_ptr + offs_m[:, None] * stridexm + offs_k[None, :] * stridexk   # (M_tile, K_tile)
    w1_ptrs = w1_ptr + offs_k[:,None] * stridewk + offs_n[None,:] * stridewn # (K_tile, N_tile)
    w2_ptrs = w2_ptr + offs_k[:,None] * stridewk + offs_n[None,:] * stridewn # (K_tile, N_tile)

    # accumulators
    z1_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    z2_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # loop over K in tiles
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for kt in range(num_k_tiles):
        k_start = kt * BLOCK_SIZE_K
        # masks: valid rows/cols in this tile
        mask_x = (offs_m[:, None] < M) & ((offs_k[None, :] + k_start) < K)
        mask_w = (offs_n[None,:] < N) & ((offs_k[:,None] + k_start) < K)

        x = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)
        w1 = tl.load(w1_ptrs, mask=mask_w, other=0.0).to(tl.float32) 
        w2 = tl.load(w2_ptrs, mask=mask_w, other=0.0).to(tl.float32)
        
        # accumulate matmuls
        z1_acc = tl.dot(x, w1, z1_acc)
        z2_acc = tl.dot(x, w2, z2_acc)

        # advance pointers by BLOCK_SIZE_K in K dimension
        x_ptrs = x_ptrs + BLOCK_SIZE_K * stridexk
        w1_ptrs = w1_ptrs + BLOCK_SIZE_K * stridewk
        w2_ptrs = w2_ptrs + BLOCK_SIZE_K * stridewk


    # compute outputs and saved intermediates
    z1 = z1_acc
    z2 = z2_acc

    s = tl.sigmoid(z2)
    g1 = s*z2
    o = z1 * g1
    
    # optionally compute values useful for backward
    g2=z1 * s * (1 + z2 * (1 - s))

    # compute output pointers
    o_ptrs  = o_ptr  + offs_m[:, None] * strideom + offs_n[None, :] * strideon
    g1_ptrs = g1_ptr + offs_m[:, None] * strideom + offs_n[None, :] * strideon
    g2_ptrs = g2_ptr + offs_m[:, None] * strideom + offs_n[None, :] * strideon

    mask_store = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs,  o,  mask=mask_store)
    tl.store(g1_ptrs, g1, mask=mask_store)
    tl.store(g2_ptrs, g2, mask=mask_store)
    return 


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _swiglu_backward(
    x_ptr, dw1_ptr, dw2_ptr,
    g1_do_ptr, g2_do_ptr,
    M, N, K,
    stridexm, stridexk,
    stridewn, stridewk,       # dw layout (N, K)
    stridegm, stridegn,       # g layout (M, N)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)   # (BLOCK_N,)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)   # (BLOCK_K,)
    offs_m = tl.arange(0, BLOCK_SIZE_M)                          # (BLOCK_M,)

    # compute pointers per tile
    x_ptrs = x_ptr + offs_m[:, None] * stridexm + offs_k[None, :] * stridexk 
    g1_do_ptrs = g1_do_ptr + offs_m[None,:] * stridegm + offs_n[:,None] * stridegn
    g2_do_ptrs = g2_do_ptr + offs_m[None,:] * stridegm + offs_n[:,None] * stridegn

    
    # accumulators
    dw1_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    dw2_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    
    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    for mt in range(num_m_tiles):
        m_start = mt * BLOCK_SIZE_M
        # masks
        mask_x = (offs_m[:, None] < M-m_start) & (offs_k[None, :] < K)
        mask_g_o = (offs_m[None,:] < M-m_start) & (offs_n[:,None] < N)
        
        # load tiles
        x = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)
        g1_do_t = tl.load(g1_do_ptrs, mask=mask_g_o, other=0.0).to(tl.float32)
        g2_do_t = tl.load(g2_do_ptrs, mask=mask_g_o, other=0.0).to(tl.float32)
        
        dw1_acc = tl.dot(g1_do_t, x, dw1_acc)
        dw2_acc = tl.dot(g2_do_t, x, dw2_acc)

        x_ptrs=x_ptrs+BLOCK_SIZE_M*stridexm
        g1_do_ptrs=g1_do_ptrs+BLOCK_SIZE_M*stridegm
        g2_do_ptrs=g2_do_ptrs+BLOCK_SIZE_M*stridegm
    

    # store results
    dw1_ptrs = dw1_ptr + offs_n[:, None] * stridewn + offs_k[None, :] * stridewk
    dw2_ptrs = dw2_ptr + offs_n[:, None] * stridewn + offs_k[None, :] * stridewk

    mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    tl.store(dw1_ptrs, dw1_acc, mask=mask_w)
    tl.store(dw2_ptrs, dw2_acc, mask=mask_w)
    return 

class SwiGLUFUNC(torch.autograd.Function):
  @staticmethod
  def forward(ctx,x,w1,w2):
    M, K=x.shape 
    assert w1.shape==w2.shape
    N, _=w1.shape 
    
    o=torch.zeros([M,N], device=x.device, dtype=x.dtype)
    g1=torch.zeros_like(o, device=o.device, dtype=o.dtype)
    g2=torch.zeros_like(o, device=o.device, dtype=o.dtype)

    grid=lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    _swiglu_forward[grid](x,w1,w2,o,g1,g2,
                          M,N,K,
                          stridexm=x.stride(0), stridexk=x.stride(1),
                          stridewn=w1.stride(0), stridewk=w1.stride(1),  
                          strideom=o.stride(0), strideon=o.stride(1))
    
    ctx.save_for_backward(x,g1,g2)
    ctx.M, ctx.N, ctx.K=M, N, K 
    return o


  @staticmethod
  def backward(ctx, do):
    x,g1,g2=ctx.saved_tensors

    dw1=torch.empty((ctx.N, ctx.K), device=x.device, dtype=x.dtype) 
    dw2=torch.empty((ctx.N, ctx.K), device=x.device, dtype=x.dtype) 
    
    g1_do=(do*g1).contiguous() 
    g2_do=(do*g2).contiguous()
    
    grid=lambda META: (triton.cdiv(ctx.N, META['BLOCK_SIZE_N']), triton.cdiv(ctx.K, META['BLOCK_SIZE_K']))
    _swiglu_backward[grid](x,dw1,dw2,g1_do,g2_do,
                           ctx.M,ctx.N,ctx.K,
                           stridexm=x.stride(0), stridexk=x.stride(1),
                           stridewn=dw1.stride(0), stridewk=dw1.stride(1),
                           stridegm=g1_do.stride(0),stridegn=g1_do.stride(1)
                           )
    return None, dw1, dw2


class SwiGLU_Layer_Triton(nn.Module):
  def __init__(self, dim, hidden_dim):
    super().__init__()
    self.w1=nn.Linear(dim, hidden_dim, bias=False)
    self.w2=nn.Linear(dim, hidden_dim, bias=False)
    self.swiglufunc=SwiGLUFUNC.apply

  def forward(self,x):
    return self.swiglufunc(x, self.w1.weight, self.w2.weight)


class SwiGLU_Layer_Pytorch(nn.Module):
  def __init__(self, dim, hidden_dim):
    super().__init__()
    self.w1=nn.Linear(dim, hidden_dim, bias=False)
    self.w2=nn.Linear(dim, hidden_dim, bias=False)

  def forward(self,x):
    z1=self.w1(x)
    z2=self.w2(x)
    return z1*F.sigmoid(z2)*z2



def test(x, do, dim, hidden_dim):
    swiglu_triton=SwiGLU_Layer_Triton(dim, hidden_dim).to(DEVICE)
    swiglu_pytorch=SwiGLU_Layer_Pytorch(dim, hidden_dim).to(DEVICE)
    swiglu_triton.w1.weight.data.copy_(swiglu_pytorch.w1.weight.data)
    swiglu_triton.w2.weight.data.copy_(swiglu_pytorch.w2.weight.data)
    
    assert torch.allclose(swiglu_triton.w1.weight, swiglu_pytorch.w1.weight, rtol=1e-02, atol=1e-02, equal_nan=True), "forward w1 discripency"
    assert torch.allclose(swiglu_triton.w2.weight, swiglu_pytorch.w2.weight, rtol=1e-02, atol=1e-02, equal_nan=True), "forward w2 discripency"
    
    out_triton=swiglu_triton(x)
    out_torch=swiglu_pytorch(x)
    torch.cuda.synchronize()

    max_diff = (out_triton - out_torch).abs().max()
    mean_diff = (out_triton - out_torch).abs().mean()
    assert torch.allclose(out_triton,out_torch, rtol=1e-02, atol=1e-02, equal_nan=True), "output discripency"
    
    out_triton.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    out_torch.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    
    max_diff = (swiglu_triton.w1.weight.grad - swiglu_pytorch.w1.weight.grad).abs().max()
    mean_diff = (swiglu_triton.w1.weight.grad - swiglu_pytorch.w1.weight.grad).abs().mean()

    assert torch.allclose(swiglu_triton.w1.weight.grad, swiglu_pytorch.w1.weight.grad, rtol=1e-01, atol=1e-01, equal_nan=True), "backward w1.weight.grad discripency"
    assert torch.allclose(swiglu_triton.w2.weight.grad, swiglu_pytorch.w2.weight.grad, rtol=1e-01, atol=1e-01, equal_nan=True), "backward w2.weight.grad discripency"
    return 

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['dim', 'hidden_dim'],
        x_vals=[
            (d, int(d*2/3))
            for d in [512, 1024, 2048, 4096, 4096, 8192, 16384]
        ],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='SwiGLU',
        args={'batch_size': 16,
              "seq_len": 1024,
              "device": DEVICE,
              "quantiles":[0.25, 0.5, 0.75]
            }
    ))

def bench_swiglu(batch_size, seq_len, dim, hidden_dim, provider, device, quantiles):
    x=torch.randn((batch_size*seq_len, dim),device=device, dtype=torch.float32)
    do=torch.randn(batch_size*seq_len, hidden_dim, device=device, dtype=torch.float32)
    def fwd_bwd():
        if provider == "triton":
            swiglu_triton=SwiGLU_Layer_Triton(dim, hidden_dim).to(device)
            out_triton=swiglu_triton(x)
            out_triton.backward(do, retain_graph=True)
            torch.cuda.synchronize()


        if provider == "torch":
            swiglu_pytorch=SwiGLU_Layer_Pytorch(dim, hidden_dim).to(device)
            out_torch=swiglu_pytorch(x)
            out_torch.backward(do, retain_graph=True)
            torch.cuda.synchronize()
        return

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    ms, min_ms, max_ms = triton.testing.do_bench(fwd_bwd, quantiles=quantiles, rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
    
    # batch_size=16
    # seq_len=512
    # dim=1024
    # hidden_dim=512
    # x=torch.randn(batch_size*seq_len, dim, device=DEVICE, dtype=torch.float32)
    # do=torch.randn(batch_size*seq_len, hidden_dim, device=DEVICE, dtype=torch.float32)
    # test(x, do, dim, hidden_dim)
    
    bench_swiglu.run(save_path=".", show_plots=True, print_data=True)