import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    x_ptr,
    pos_ptr,
    inv_freq_ptr,
    x_batch_stride,
    x_seq_stride,
    x_head_stride,
    x_dim_stride,
    pos_stride,
    inv_freq_stride,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_head = tl.program_id(2)

    x_base = x_ptr + pid_batch * x_batch_stride + pid_seq * x_seq_stride + pid_head * x_head_stride
    
    m = tl.load(pos_ptr + pid_seq * pos_stride)
    
    num_pairs = D // 2
    for i in tl.range(0, num_pairs, BLOCK_SIZE):
        pair_indices = i + tl.arange(0, BLOCK_SIZE)
        mask = pair_indices < num_pairs
        
        inv_freq = tl.load(inv_freq_ptr + pair_indices * inv_freq_stride, mask=mask)
        
        theta = m * inv_freq
        cos_theta = tl.cos(theta)
        sin_theta = tl.sin(theta)
        
        pair_offsets = pair_indices * 2 * x_dim_stride
        x0_ptrs = x_base + pair_offsets
        x1_ptrs = x_base + pair_offsets + x_dim_stride
        
        x0 = tl.load(x0_ptrs, mask=mask)
        x1 = tl.load(x1_ptrs, mask=mask)
        
        x0_new = x0 * cos_theta - x1 * sin_theta
        x1_new = x0 * sin_theta + x1 * cos_theta
        
        # Store results
        tl.store(x0_ptrs, x0_new, mask=mask)
        tl.store(x1_ptrs, x1_new, mask=mask)

def rotary_embedding(x: torch.Tensor, pos: torch.Tensor):
    assert x.dim() == 4, "Input must be 4D (batch, seq, heads, dim)"
    assert x.size(-1) % 2 == 0, "Feature dimension must be even"
    
    batch_size, seq_len, num_heads, dim = x.shape
    device = x.device
    
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim//2, device=device, dtype=torch.float32) / (dim//2)))
    
    x = x.contiguous()
    pos = pos.contiguous().to(torch.float32)
    inv_freq = inv_freq.contiguous()

    grid = (batch_size, seq_len, num_heads)
    
    BLOCK_SIZE = 128  
    rotary_kernel[grid](
        x,
        pos,
        inv_freq,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        pos.stride(0),
        inv_freq.stride(0),
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x

if __name__ == "__main__":
    batch_size = 2
    seq_len = 1024
    num_heads = 12
    dim = 128

    x = torch.randn(batch_size, seq_len, num_heads, dim, device='cuda')
    pos = torch.arange(seq_len, device='cuda')

    x_rotary = rotary_embedding(x, pos)
    print("Rotary embedding applied:", x_rotary.shape)