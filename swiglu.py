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
    key=["x_size_0","x_size_1","w1_size_0","w1_size_1","w2_size_0","w2_size_1"]
)
@triton.jit
def _swiglu_forward_kernel(x_ptr, x_size_0, x_size_1, x_stride_0, x_stride_1, 
        w1_ptr, w1_size_0, w1_size_1, w1_stride_0, w1_stride_1, 
        w2_ptr, w2_size_0, w2_size_1, w2_stride_0, w2_stride_1,
        o_ptr, o_size_0, o_size_1, o_stride_0, o_stride_1,
        l1_ptr, l1_size_0, l1_size_1, l1_stride_0, l1_stride_1, 
        l2_ptr, l2_size_0, l2_size_1, l2_stride_0, l2_stride_1, 
        BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N:tl.constexpr, BLOCK_SIZE_K:tl.constexpr):
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    K_dim = x_size_1
    
    x_block_ptr = tl.make_block_ptr(
        x_ptr, 
        shape=(x_size_0, x_size_1),
        strides=(x_stride_0, x_stride_1),
        offsets=(pid_m * BLOCK_SIZE_M,0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    # hidden_dim, intermediate_dim
    w1_block_ptr = tl.make_block_ptr(
        w1_ptr, 
        shape=(w1_size_0, w1_size_1),
        strides=(w1_stride_0, w1_stride_1),
        offsets=(0,pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    # hidden_dim, intermediate_dim
    w2_block_ptr = tl.make_block_ptr(
        w2_ptr, 
        shape=(w2_size_0, w2_size_1),
        strides=(w2_stride_0, w2_stride_1),
        offsets=(0,pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )

    
    acc_gate = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    for _ in range(0, tl.cdiv(K_dim, BLOCK_SIZE_K)):
        x_block = tl.load(x_block_ptr, boundary_check=(0, 1))
        w1_block = tl.load(w1_block_ptr, boundary_check=(0, 1))
        w2_block = tl.load(w2_block_ptr, boundary_check=(0, 1))
        
        acc_gate += tl.dot(x_block, w1_block)
        acc_up += tl.dot(x_block, w2_block)
        
        x_block_ptr = tl.advance(x_block_ptr, offsets=(0, BLOCK_SIZE_K))
        w1_block_ptr = tl.advance(w1_block_ptr, offsets=(BLOCK_SIZE_K, 0))
        w2_block_ptr = tl.advance(w2_block_ptr, offsets=(BLOCK_SIZE_K, 0))
    

    # sigmoid requires type float32
    acc_gate_sigmoid=tl.sigmoid(acc_gate)
    acc_gate_silu=silu(acc_gate)
    
    l1_out=(acc_gate_sigmoid+acc_gate_silu*(1-acc_gate_sigmoid))*acc_up
    l2_out=acc_gate_silu

    out=acc_gate_silu*acc_up
    
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


    l1_block_ptr = tl.make_block_ptr(
        l1_ptr,
        shape=(l1_size_0, l1_size_1),
        strides=(l1_stride_0, l1_stride_1),
        offsets=(
            pid_m * BLOCK_SIZE_M,
            pid_n * BLOCK_SIZE_N,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    l2_block_ptr = tl.make_block_ptr(
        l2_ptr,
        shape=(l2_size_0, l2_size_1),
        strides=(l2_stride_0, l2_stride_1),
        offsets=(
            pid_m * BLOCK_SIZE_M,
            pid_n * BLOCK_SIZE_N,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    tl.store(o_block_ptr, out, boundary_check=(0, 1))
    tl.store(l1_block_ptr, l1_out, boundary_check=(0, 1))
    tl.store(l2_block_ptr, l2_out, boundary_check=(0, 1))
    return


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
    key=["x_size_0","x_size_1","dl1_size_0","dl1_size_1","dl2_size_0","dl2_size_1"]
)
@triton.jit
def _swiglu_backward_kernel(
        x_ptr, x_size_0, x_size_1, x_stride_0, x_stride_1,
        dl1_ptr, dl1_size_0, dl1_size_1, dl1_stride_0, dl1_stride_1,
        dl2_ptr, dl2_size_0, dl2_size_1, dl2_stride_0, dl2_stride_1,
        dw1_ptr, dw1_size_0, dw1_size_1, dw1_stride_0, dw1_stride_1,
        dw2_ptr, dw2_size_0, dw2_size_1, dw2_stride_0, dw2_stride_1,
        BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N:tl.constexpr, BLOCK_SIZE_K:tl.constexpr,
):

    pid_n = tl.program_id(0) # intermediate_dim
    pid_k = tl.program_id(1) # hidden_dim
    
    M_dim = x_size_0
    
    x_block_ptr = tl.make_block_ptr(
        x_ptr, 
        shape=(x_size_0, x_size_1),
        strides=(x_stride_0, x_stride_1),
        offsets=(0, pid_k * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    # intermediate_dim, batch_size*seq_len
    dl1_block_ptr = tl.make_block_ptr(
        dl1_ptr, 
        shape=(dl1_size_0, dl1_size_1),
        strides=(dl1_stride_0, dl1_stride_1),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_M),
        order=(1, 0),
    )
    
    
    # intermediate_dim, batch_size*seq_len
    dl2_block_ptr = tl.make_block_ptr(
        dl2_ptr, 
        shape=(dl2_size_0, dl2_size_1),
        strides=(dl2_stride_0, dl2_stride_1),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_M),
        order=(1, 0),
    )
    

    dw1_acc = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=tl.float32)
    dw2_acc = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=tl.float32)
    

    for _ in range(0, tl.cdiv(M_dim, BLOCK_SIZE_M)):
        x_block = tl.load(x_block_ptr, boundary_check=(0, 1))
        dl1_block = tl.load(dl1_block_ptr, boundary_check=(0, 1))
        dl2_block = tl.load(dl2_block_ptr, boundary_check=(0, 1))
        
        # intermediate_dim, hidden_dim
        dw1_acc += tl.dot(dl1_block, x_block)
        dw2_acc += tl.dot(dl2_block, x_block)
        
        x_block_ptr = tl.advance(x_block_ptr, offsets=(BLOCK_SIZE_M, 0))
        dl1_block_ptr = tl.advance(dl1_block_ptr, offsets=(0, BLOCK_SIZE_M))
        dl2_block_ptr = tl.advance(dl2_block_ptr, offsets=(0, BLOCK_SIZE_M))
    
    
    dw1_block_ptr = tl.make_block_ptr(
        dw1_ptr,
        shape=(dw1_size_0, dw1_size_1),
        strides=(dw1_stride_0, dw1_stride_1),
        offsets=(
            pid_n * BLOCK_SIZE_N,
            pid_k * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )
    

    dw2_block_ptr = tl.make_block_ptr(
        dw2_ptr,
        shape=(dw2_size_0, dw2_size_1),
        strides=(dw2_stride_0, dw2_stride_1),
        offsets=(
            pid_n * BLOCK_SIZE_N,
            pid_k * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )

    tl.store(dw1_block_ptr, dw1_acc, boundary_check=(0, 1))
    tl.store(dw2_block_ptr, dw2_acc, boundary_check=(0, 1))
    return 

def swiglu_forward(x, w1, w2):
    batch_size, seq_len, hidden_dim=x.shape
    x=x.view(batch_size*seq_len, hidden_dim).contiguous()
    intermediate_dim, hidden_dim=w1.shape
    
    M, N, K=batch_size*seq_len, intermediate_dim, hidden_dim

    # hidden_dim, intermediate_dim
    w1=w1.transpose(1,0).contiguous()
    w2=w2.transpose(1,0).contiguous()
    
    o=torch.zeros(batch_size*seq_len, intermediate_dim, device=x.device)
    l1=torch.zeros(batch_size*seq_len, intermediate_dim, device=x.device)
    l2=torch.zeros(batch_size*seq_len, intermediate_dim, device=x.device)

    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))   
    _swiglu_forward_kernel[grid](
        x, x.size()[0], x.size()[1], x.stride()[0], x.stride()[1],
        w1, w1.size()[0], w1.size()[1], w1.stride()[0], w1.stride()[1],
        w2, w2.size()[0], w2.size()[1], w2.stride()[0], w2.stride()[1],
        o, o.size()[0], o.size()[1], o.stride()[0], o.stride()[1],
        l1, l1.size()[0], l1.size()[1], l1.stride()[0], l1.stride()[1],
        l2, l2.size()[0], l2.size()[1], l2.stride()[0], l2.stride()[1],
    )

    return x.view(batch_size, seq_len, hidden_dim).contiguous(), o.view(batch_size, seq_len, intermediate_dim).contiguous(), l1, l2, w1.transpose(1,0).contiguous(), w2.transpose(1,0).contiguous()


def swiglu_backward(dc, x, l1, l2, w1, w2):
    batch_size, seq_len, intermediate_dim=dc.shape
    dc=dc.view(batch_size*seq_len, intermediate_dim).contiguous()
    _,_,hidden_dim=x.shape
    
    x=x.view(batch_size*seq_len, hidden_dim).contiguous()
    
    M, N, K=batch_size*seq_len, intermediate_dim, hidden_dim
    
    dl1=(dc*l1) # batch_size*seq_len, intermediate_dim
    dl2=(dc*l2) # batch_size*seq_len, intermediate_dim
    
    dw1=torch.zeros(intermediate_dim, hidden_dim, device=x.device)
    dw2=torch.zeros(intermediate_dim, hidden_dim, device=x.device)

    # dx=(torch.matmul(dl1,w1)+torch.matmul(dl2,w2)).view(batch_size, seq_len, hidden_dim)

    dl1=dl1.transpose(0,1).contiguous() # intermediate_dim, batch_size*seq_len
    dl2=dl2.transpose(0,1).contiguous() # intermediate_dim, batch_size*seq_len
    # dw1=torch.matmul(dl1,x)
    # dw2=torch.matmul(dl2,x)
    
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]), triton.cdiv(K, META["BLOCK_SIZE_K"]))
    _swiglu_backward_kernel[grid](
        x, x.size()[0], x.size()[1], x.stride()[0], x.stride()[1],
        dl1, dl1.size()[0], dl1.size()[1], dl1.stride()[0], dl1.stride()[1],
        dl2, dl2.size()[0], dl2.size()[1], dl2.stride()[0], dl2.stride()[1],
        dw1, dw1.size()[0], dw1.size()[1], dw1.stride()[0], dw1.stride()[1],
        dw2, dw2.size()[0], dw2.size()[1], dw2.stride()[0], dw2.stride()[1],
    ) 
    return None, dw1, dw2

class LigerSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, w1, w2):
        x, o, l1, l2, w1, w2 = swiglu_forward(x, w1, w2)
        ctx.save_for_backward(x, l1, l2, w1, w2)
        return o
    
    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        x, l1, l2, w1, w2 = ctx.saved_tensors
        _, dw1, dw2 = swiglu_backward(dc, x, l1, l2, w1, w2)
        return _, dw1, dw2