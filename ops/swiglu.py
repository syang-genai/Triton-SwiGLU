import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import is_hip
from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}),
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}),
    ],
    key=["gate_size_0", "gate_size_1", "up_size_0", "up_size_1"]
)
@triton.jit
def _swiglu_forward_kernel(
        o_ptr, o_size_0, o_size_1, o_stride_0, o_stride_1,
        gate_ptr, gate_size_0, gate_size_1, gate_stride_0, gate_stride_1, 
        up_ptr, up_size_0, up_size_1, up_stride_0, up_stride_1, 
        BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N:tl.constexpr, BLOCK_SIZE_K:tl.constexpr):
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    gate_block_ptr = tl.make_block_ptr(
        gate_ptr,
        shape=(gate_size_0, gate_size_1),
        strides=(gate_stride_0, gate_stride_1),
        offsets=(
            pid_m * BLOCK_SIZE_M,
            pid_n * BLOCK_SIZE_N,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    up_block_ptr = tl.make_block_ptr(
        up_ptr,
        shape=(up_size_0, up_size_1),
        strides=(up_stride_0, up_stride_1),
        offsets=(
            pid_m * BLOCK_SIZE_M,
            pid_n * BLOCK_SIZE_N,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    gate=tl.load(gate_block_ptr, boundary_check=(0,1)).to(tl.float32)
    up=tl.load(up_block_ptr, boundary_check=(0,1))
    
    # sigmoid requires type float32
    gate_sigmoid=tl.sigmoid(gate)
    gate_silu=silu(gate)
    
    l1_out=(gate_sigmoid+gate_silu*(1-gate_sigmoid)).to(up.dtype)*up
    l2_out=gate_silu

    out=gate_silu*up
    
    o_block_ptr = tl.make_block_ptr(
        o_ptr,
        shape=(o_size_0, o_size_1),
        strides=(o_stride_0, o_stride_1),
        offsets=(
            pid_m * BLOCK_SIZE_M,
            pid_n * BLOCK_SIZE_N,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    tl.store(o_block_ptr, out, boundary_check=(0, 1))
    tl.store(gate_block_ptr, l1_out, boundary_check=(0, 1))
    tl.store(up_block_ptr, l2_out, boundary_check=(0, 1))
    return

def swiglu_forward(x, w1, w2):
    batch_size, seq_len, hidden_dim=x.shape
    x=x.view(batch_size*seq_len, hidden_dim).contiguous()
    intermediate_dim, hidden_dim=w1.shape
    
    M, N, K=batch_size*seq_len, intermediate_dim, hidden_dim
    
    gate=torch.matmul(x, w1.transpose(1,0))
    up=torch.matmul(x, w2.transpose(1,0))
    o=torch.zeros(batch_size*seq_len, intermediate_dim, device=x.device)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))   
    _swiglu_forward_kernel[grid](
        o, o.size()[0], o.size()[1], o.stride()[0], o.stride()[1],
        gate, gate.size()[0], gate.size()[1], gate.stride()[0], gate.stride()[1],
        up, up.size()[0], up.size()[1], up.stride()[0], up.stride()[1],
    )

    return o.view(batch_size, seq_len, intermediate_dim).contiguous(), gate, up


def swiglu_backward(dc, x, gate, up, w1, w2):
    batch_size, seq_len, intermediate_dim=dc.shape
    dc=dc.view(batch_size*seq_len, intermediate_dim).contiguous()
    _,_,hidden_dim=x.shape
    M, N, K=batch_size*seq_len, intermediate_dim, hidden_dim
    
    x=x.view(batch_size*seq_len, hidden_dim).contiguous()
    dl1=(dc*gate) # batch_size*seq_len, intermediate_dim
    dl2=(dc*up) # batch_size*seq_len, intermediate_dim
    dw1=torch.matmul(dl1.transpose(0,1),x)
    dw2=torch.matmul(dl2.transpose(0,1),x)
    dx=(torch.matmul(dl1,w1)+torch.matmul(dl2,w2)).view(batch_size, seq_len, hidden_dim)
    return dx, dw1, dw2


class LigerSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, w1, w2):
        o, gate, up = swiglu_forward(x, w1, w2)
        ctx.save_for_backward(x, gate, up, w1, w2)
        return o
    
    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        gate, up = ctx.saved_tensors
        dx, dw1, dw2 = swiglu_backward(dc, x, gate, up, w1, w2)
        return dx, dw1, dw2
