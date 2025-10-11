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
    return [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8})]


def get_hip_autotune_config():
    sizes = [{'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}]
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
def _swiglu_forward(x_ptr,w1_ptr,w2_ptr,o_ptr,z1_ptr,z2_ptr,g1_ptr,g2_ptr,
                    M,N,K,
                    stridexm, stridexk,
                    stridewk, stridewn,
                    strideom, strideon,
                    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE: tl.constexpr
                    ):
    
    # tl.device_print("start forward")
    # tl.device_print("M",M)
    # tl.device_print("N",N)
    # tl.device_print("K",K)
    
    pid_m=tl.program_id(axis=0)
    pid_n=tl.program_id(axis=1)
    pid=pid_m*BLOCK_SIZE_M+pid_n*BLOCK_SIZE_N

    # pid=tl.program_id(axis=0)

    num_pid_m=tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n=tl.cdiv(N, BLOCK_SIZE_N)

    # group of rows: height=group size m, width = num pid n
    num_pid_in_group=GROUP_SIZE*num_pid_n
    group_id=pid//num_pid_in_group
    first_pid_m=group_id*GROUP_SIZE
    group_size_m=min(GROUP_SIZE, M-first_pid_m)

    pid_m=first_pid_m+((pid%num_pid_in_group)%group_size_m)
    pid_n=(pid%num_pid_in_group)//group_size_m

    offs_xm=(pid_m*BLOCK_SIZE_M+tl.arange(0,BLOCK_SIZE_M))%M
    offs_wn=(pid_n*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N))%N
    offs_k=tl.arange(0,BLOCK_SIZE_K)

    x_ptrs=x_ptr+offs_xm[:,None]*stridexm+offs_k[None,:]*stridexk # x:(m,k)
    
    w1_ptrs=w1_ptr+offs_k[:,None]*stridewk+offs_wn[None,:]*stridewn # w1:(k,n)
    w2_ptrs=w2_ptr+offs_k[:,None]*stridewk+offs_wn[None,:]*stridewn # w2:(k,n)
    

    o_accumulator=tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    z1_accumulator=tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    z2_accumulator=tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    g1_accumulator=tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    g2_accumulator=tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x=tl.load(x_ptrs, mask=(offs_xm[:,None]<M)&(offs_k[None,:]<K-k*BLOCK_SIZE_K), other=0.0) 
        w1=tl.load(w1_ptrs, mask=(offs_wn[None,:]<N)&(offs_k[:,None]<K-k*BLOCK_SIZE_K), other=0.0)
        w2=tl.load(w2_ptrs, mask=(offs_wn[None,:]<N)&(offs_k[:,None]<K-k*BLOCK_SIZE_K), other=0.0)


        z1=tl.dot(x,w1) # x:(m,k)*w1:(k,n)=z:(m,n)
        z2=tl.dot(x,w2) # x:(m,k)*w1:(k,n)=z:(m,n)
        g1=z2*tl.sigmoid(z2) # g1:(m,n)
        g2=z1*tl.sigmoid(z2)*(1+z2*(1-tl.sigmoid(z2))) #g2:(m,n)
    
        o_accumulator+=z1*g1
        z1_accumulator+=z1
        z2_accumulator+=z2
        g1_accumulator+=g1
        g2_accumulator+=g2
        
        # update x, w1 and w2 pointers
        x_ptrs+=BLOCK_SIZE_K*stridexk
        w1_ptrs+=BLOCK_SIZE_K*stridewk
        w2_ptrs+=BLOCK_SIZE_K*stridewk

    o=o_accumulator.to(tl.float16)
    z1=z1_accumulator.to(tl.float16)
    z2=z2_accumulator.to(tl.float16)
    g1=g1_accumulator.to(tl.float16)
    g2=g2_accumulator.to(tl.float16)

    offs_om=pid_m*BLOCK_SIZE_M+tl.arange(0,BLOCK_SIZE_M)
    offs_on=pid_n*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N)
    
    o_ptrs=o_ptr+offs_om[:,None]*strideom+offs_on[None,:]*strideon
    z1_ptrs=z1_ptr+offs_om[:,None]*strideom+offs_on[None,:]*strideon
    z2_ptrs=z2_ptr+offs_om[:,None]*strideom+offs_on[None,:]*strideon
    g1_ptrs=z1_ptr+offs_om[:,None]*strideom+offs_on[None,:]*strideon
    g2_ptrs=z2_ptr+offs_om[:,None]*strideom+offs_on[None,:]*strideon


    tl.store(o_ptrs, o, mask=(offs_om[:,None]<M)&(offs_on[None,:]<N))
    tl.store(z1_ptrs, z1, mask=(offs_om[:,None]<M)&(offs_on[None,:]<N))
    tl.store(z2_ptrs, z2, mask=(offs_om[:,None]<M)&(offs_on[None,:]<N))
    tl.store(g1_ptrs, g1, mask=(offs_om[:,None]<M)&(offs_on[None,:]<N))
    tl.store(g2_ptrs, g2, mask=(offs_om[:,None]<M)&(offs_on[None,:]<N))

    # tl.device_print("complete forward")
    return


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _swiglu_backward(x_ptr,dw1_ptr,dw2_ptr,
                    z1_ptr,z2_ptr,g1_ptr,g2_ptr,
                    M,N,K,
                    stridexm, stridexk,
                    stridewk, stridewn,
                    stridezm, stridezn,
                    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE: tl.constexpr
                    ): 

    # tl.device_print("start backward")
    # X:(M,K), G:(M,N)
    pid_k=tl.program_id(axis=0)
    pid_n=tl.program_id(axis=1)

    pid=pid_k*BLOCK_SIZE_K+pid_n*BLOCK_SIZE_N

    num_pid_k=tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n=tl.cdiv(N, BLOCK_SIZE_N)

    # group of rows: height=group_size_m, width=num_pid_n
    num_pid_in_group=GROUP_SIZE*num_pid_n
    group_id=pid//num_pid_in_group
    first_pid_k=group_id*GROUP_SIZE
    group_size_k=min(GROUP_SIZE, K-first_pid_k)

    # reindex
    pid_k=first_pid_k+((pid%num_pid_in_group)%group_size_k)
    pid_n=(pid%num_pid_in_group)//group_size_k

    offs_k=(pid_k*BLOCK_SIZE_K+tl.arange(0,BLOCK_SIZE_K))%K
    offs_n=(pid_n*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N))%N
    offs_m=tl.arange(0,BLOCK_SIZE_M)

    x_ptrs=x_ptr+offs_m[None,:]*stridexm+offs_k[:,None]*stridexk


    # Z, G: (M,N)
    z1_ptrs=z1_ptr+offs_m[:,None]*stridezm+offs_n[None,:]*stridezn
    z2_ptrs=z2_ptr+offs_m[:,None]*stridezm+offs_n[None,:]*stridezn

    g1_ptrs=g1_ptr+offs_m[:,None]*stridezm+offs_n[None,:]*stridezn
    g2_ptrs=g2_ptr+offs_m[:,None]*stridezm+offs_n[None,:]*stridezn


    # dw: (K,N)
    dw1_ptrs=dw1_ptr+offs_k[:,None]*stridewk+offs_n[None,:]*stridewn
    dw2_ptrs=dw2_ptr+offs_k[:,None]*stridewk+offs_n[None,:]*stridewn


    dw1_accumulator=tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N),dtype=tl.float32)
    dw2_accumulator=tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N),dtype=tl.float32)


    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        x=tl.load(x_ptrs, mask=(offs_m[None,:]<M-m*BLOCK_SIZE_M)&(offs_k[:,None]<K), other=0.0)
        z1=tl.load(z1_ptrs, mask=(offs_m[:,None]<M-m*BLOCK_SIZE_M)&(offs_n[None,:]<N), other=0.0)
        z2=tl.load(z2_ptrs, mask=(offs_m[:,None]<M-m*BLOCK_SIZE_M)&(offs_n[None,:]<N), other=0.0)
        g1=tl.load(g1_ptrs, mask=(offs_m[:,None]<M-m*BLOCK_SIZE_M)&(offs_n[None,:]<N), other=0.0)
        g2=tl.load(g2_ptrs, mask=(offs_m[:,None]<M-m*BLOCK_SIZE_M)&(offs_n[None,:]<N), other=0.0)
        
        
        dw1_accumulator+=tl.dot(x,g1) # x:(m,k), g:(m,n), dw:(k,n)
        dw2_accumulator+=tl.dot(x,g2) # x:(m,k), g:(m,n), dw:(k,n)

        x_ptrs+=offs_m[None,:]*stridexm
        z1_ptrs+=offs_m[:,None]*stridezm
        z2_ptrs+=offs_m[:,None]*stridezm
        g1_ptrs+=offs_m[:,None]*stridezm
        g2_ptrs+=offs_m[:,None]*stridezm

    dw1=dw1_accumulator.to(tl.float16)
    dw2=dw2_accumulator.to(tl.float16)

    # dw: (K,N)
    dw1_ptrs=dw1_ptr+offs_n[None,:]*stridewn+offs_k[:,None]*stridewk
    dw2_ptrs=dw2_ptr+offs_n[None,:]*stridewn+offs_k[:,None]*stridewk
    
    tl.store(dw1_ptrs, dw1, mask=(offs_n[None,:]<N)&(offs_k[:,None])<K)
    tl.store(dw2_ptrs, dw2, mask=(offs_n[None,:]<N)&(offs_k[:,None])<K)
    # tl.device_print("complete backward")
    return


class SwiGLU(torch.autograd.Function):
  @staticmethod
  def forward(ctx,x,w1,w2):
    M, K=x.shape # x: batch_size*nhead*seq_len, dim (m,k)
    assert w1.shape==w2.shape
    N, _=w1.shape # w: hidden_dim, dim (n,k)
    # print("x shape", x.shape)
    # print("w1 shape", w1.shape)
    # print("w2 shape", w2.shape)
    
    o=torch.zeros([M,N], device=x.device, dtype=x.dtype)# o: batch_size*nhead*seq_len, hidden_dim
    # print("o shape",o.shape)
    
    z1=torch.zeros_like(o, device=o.device, dtype=o.dtype)
    z2=torch.zeros_like(o, device=o.device, dtype=o.dtype)
    g1=torch.zeros_like(o, device=o.device, dtype=o.dtype)
    g2=torch.zeros_like(o, device=o.device, dtype=o.dtype)

    grid=lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    _swiglu_forward[grid](x,w1.transpose(1,0).contiguous(),w2.transpose(1,0).contiguous(),o,z1,z2,g1,g2,
                          M,N,K,
                          stridexm=x.stride(0), stridexk=x.stride(1),
                          stridewk=w1.stride(1), stridewn=w1.stride(0), #wk transposed
                          strideom=o.stride(0), strideon=o.stride(1))

    ctx.save_for_backward(x,z1,z2,g1,g2)
    ctx.M, ctx.N, ctx.K=M, N, K 
    # print("M","N","K",M,N,K)
    return o


  @staticmethod
  def backward(ctx, do):
    # print("backward start")
    x,z1,z2,g1,g2=ctx.saved_tensors # X: (M,K), Z:(M,N), G:(M,N)

    dw1=torch.empty((ctx.N, ctx.K), device=x.device, dtype=x.dtype) # W: K,N
    dw2=torch.empty((ctx.N, ctx.K), device=x.device, dtype=x.dtype) 
    # print("dw1 shape", dw1.shape)
    # print("dw2 shape", dw2.shape)


    # print("M","N","K",ctx.M,ctx.N,ctx.K)
    # (m,k) (m,n)-->(k,n)
    grid=lambda META: (triton.cdiv(ctx.K, META['BLOCK_SIZE_K']), triton.cdiv(ctx.N, META['BLOCK_SIZE_N']))
    _swiglu_backward[grid](x,dw1.transpose(1,0).contiguous(),dw2.transpose(1,0).contiguous(),
                           z1,z2,g1,g2,
                           ctx.M,ctx.N,ctx.K,
                           stridexm=x.stride(0), stridexk=x.stride(1),
                           stridewk=dw1.stride(1), stridewn=dw1.stride(0), # stride transposed 
                           stridezm=z1.stride(0),stridezn=z1.stride(1))
    # print("complete backward")
    return None, dw1, dw2



class SwiGLU_Layer_Triton(nn.Module):
  def __init__(self, dim, hidden_dim):
    super().__init__()
    self.w1=nn.Linear(dim, hidden_dim, bias=False)
    self.w2=nn.Linear(dim, hidden_dim, bias=False)
    self.swiglu=SwiGLU.apply

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



def test(x, do, dim, hidden_dim):
    # print("start test")
    swiglu_triton=SwiGLU_Layer_Triton(dim, hidden_dim).to(DEVICE)
    swiglu_pytorch=SwiGLU_Layer_Pytorch(dim, hidden_dim).to(DEVICE)
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
    print(max_diff, mean_diff)
    # assert torch.allclose(out_triton,out_torch, rtol=1e-05, atol=1e-08, equal_nan=True), "output discripency"
    
    out_triton.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    out_torch.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    # assert torch.allclose(swiglu_triton.w1.weight.grad, swiglu_pytorch.w1.weight.grad, rtol=1e-03, atol=1e-05, equal_nan=True), "backward w1.weight.grad discripency"
    # assert torch.allclose(swiglu_triton.w2.weight.grad, swiglu_pytorch.w2.weight.grad, rtol=1e-03, atol=1e-05, equal_nan=True), "backward w2.weight.grad discripency"
    return


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['dim', 'hidden_dim'],
        x_vals=[
            (d, int(d*2/3))
            for d in [512, 1024, 2048, 4096]
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
              "quantiles":[0.5, 0.25, 0.75]
            }
    ))

def bench_swiglu(batch_size, seq_len, dim, hidden_dim, provider, device, quantiles):
    def fwd_bwd():
        if provider == "triton":
            x=torch.randn(batch_size*seq_len, dim, device=device)
            do=torch.randn(batch_size*seq_len, hidden_dim, device=device)
            swiglu_triton=SwiGLU_Layer_Triton(dim, hidden_dim).to(device)
            out_triton=swiglu_triton(x)
            out_triton.backward(do, retain_graph=True)
            torch.cuda.synchronize()


        if provider == "torch":
            x=torch.randn((batch_size*seq_len, dim),device=device)
            do=torch.randn((batch_size*seq_len, hidden_dim),device=device)
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
    torch.manual_seed(seed)  # seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # seed for current GPU
        torch.cuda.manual_seed_all(seed)
    
    batch_size=16
    seq_len=1024
    dim=512
    hidden_dim=int(dim*2/3)
    # print("hidden_dim", hidden_dim)
    
    x=torch.randn(batch_size*seq_len, dim, device=DEVICE)
    do=torch.randn(batch_size*seq_len, hidden_dim, device=DEVICE)
    test(x, do, dim, hidden_dim)
    
    # bench_swiglu.run(save_path=".", show_plots=True, print_data=True)