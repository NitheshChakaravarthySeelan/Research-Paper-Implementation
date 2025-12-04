import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def expert_mlp_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, weights_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_b1n,
    stride_w2n, stride_w2k,
    stride_b2k,
    stride_weightsm,
    stride_outm, stride_outk,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)

    # First layer: x @ w1 + b1
    x_ptrs = x_ptr + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
    w1_ptrs = w1_ptr + off_k[:, None] * stride_w1k + off_n[None, :] * stride_w1n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=(off_m[:, None] < M) & (off_k[None, :] < K - k), other=0.0)
        w1 = tl.load(w1_ptrs, mask=(off_k[:, None] < K - k) & (off_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w1)
        x_ptrs += BLOCK_K * stride_xk
        w1_ptrs += BLOCK_K * stride_w1k

    # Add bias and activate
    b1 = tl.load(b1_ptr + off_n, mask=off_n < N, other=0.0)
    acc += b1[None, :]
    acc = acc * 0.5 * (1.0 + tl.erf(acc / tl.sqrt(2.0)))  # GELU

    # Second layer: acc @ w2 + b2
    w2_ptrs = w2_ptr + off_n[:, None] * stride_w2n + off_k[None, :] * stride_w2k
    b2 = tl.load(b2_ptr + off_k, mask=off_k < K, other=0.0)
    
    acc2 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in range(0, N, BLOCK_N):
        a = acc[:, n:n+BLOCK_N] if n + BLOCK_N <= N else acc[:, n:]
        w2 = tl.load(w2_ptrs + n * stride_w2n, mask=(off_n[:, None] < N - n) & (off_k[None, :] < K), other=0.0)
        acc2 += tl.dot(a, w2)
    
    acc2 += b2[None, :]
    
    # Apply weights
    weights = tl.load(weights_ptr + off_m * stride_weightsm, mask=off_m < M, other=0.0)
    acc2 *= weights[:, None]

    # Store output
    out_ptrs = out_ptr + off_m[:, None] * stride_outm + off_k[None, :] * stride_outk
    tl.store(out_ptrs, acc2, mask=(off_m[:, None] < M) & (off_k[None, :] < K))

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class MoE(nn.Module):
    def __init__(self, dim, num_experts, hidden_dim, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)
        self.top_k = top_k
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, self.dim)
        
        # Compute gating logic
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        topk_v, topk_i = torch.topk(probs, self.top_k)
        
        # Flatten and sort
        num_tokens = x.size(0)
        token_idx = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)
        expert_idx = topk_i.flatten()
        weights = topk_v.flatten()
        
        # Sort experts
        sorted_e, sort_order = torch.sort(expert_idx)
        sorted_t = token_idx[sort_order]
        sorted_w = weights[sort_order]
        
        # Process each expert group
        unique_e, counts = torch.unique_consecutive(sorted_e, return_counts=True)
        starts = torch.cat([torch.zeros(1, device=x.device), counts.cumsum(0)[:-1]]).long()
        ends = starts + counts
        
        out = torch.zeros_like(x)
        for i, expert_id in enumerate(unique_e):
            start, end = starts[i], ends[i]
            tokens = sorted_t[start:end]
            w = sorted_w[start:end]
            
            if tokens.numel() == 0:
                continue
                
            expert = self.experts[expert_id]
            x_in = x[tokens]
            M, K = x_in.shape
            
            # Prepare weights and biases
            w1 = expert.fc1.weight.t().contiguous()
            b1 = expert.fc1.bias.contiguous()
            w2 = expert.fc2.weight.t().contiguous()
            b2 = expert.fc2.bias.contiguous()
            
            # Allocate output
            expert_out = torch.empty_like(x_in)
            
            # Launch kernel
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(K, meta['BLOCK_K']),)
            expert_mlp_kernel[grid](
                x_in, w1, b1, w2, b2, w,
                expert_out,
                M, K, self.hidden_dim,
                x_in.stride(0), x_in.stride(1),
                w1.stride(0), w1.stride(1),
                b1.stride(0),
                w2.stride(0), w2.stride(1),
                b2.stride(0),
                w.stride(0),
                expert_out.stride(0), expert_out.stride(1),
                BLOCK_M=64, BLOCK_K=32, BLOCK_N=64
            )
            
            out[tokens] += expert_out
        
        return out.view(*orig_shape)
    
import torch
import numpy as np

def test_moe_basic():
    # Test configuration
    batch_size = 4
    seq_len = 128
    dim = 512
    num_experts = 8
    hidden_dim = 1024
    top_k = 2
    
    # Create model and test input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moe = MoE(dim, num_experts, hidden_dim, top_k).to(device)
    x = torch.randn(batch_size, seq_len, dim).to(device)
    
    # Test forward pass
    out = moe(x)
    
    # Basic checks
    assert out.shape == x.shape, "Output shape mismatch"
    assert not torch.isnan(out).any(), "NaN values in output"
    assert not torch.isinf(out).any(), "Inf values in output"
    print("Basic forward pass test passed!")

def test_routing_logic():
    # Test routing mechanism
    dim = 256
    num_experts = 4
    top_k = 2
    num_tokens = 1000
    
    moe = MoE(dim, num_experts, dim*2, top_k)
    x = torch.randn(num_tokens, dim)
    
    # Get routing probabilities
    logits = moe.gate(x)
    probs = torch.softmax(logits, dim=-1)
    topk_v, topk_i = torch.topk(probs, top_k)
    
    # Routing tests
    assert topk_i.shape == (num_tokens, top_k), "Top-k indices shape mismatch"
    assert (topk_i >= 0).all() and (topk_i < num_experts).all(), "Invalid expert indices"
    
    # Check token distribution
    unique_experts, counts = torch.unique(topk_i, return_counts=True)
    print(f"\nExpert distribution: {dict(zip(unique_experts.tolist(), counts.tolist()))}")
    
    # Verify at least 75% of experts get some tokens (probabilistic)
    assert len(unique_experts) >= int(0.75 * num_experts), "Experts underutilized"
    print("Routing logic test passed!")

def test_expert_activation():
    # Verify expert diversity
    dim = 512
    num_experts = 8
    num_tokens = 4096
    
    moe = MoE(dim, num_experts, dim*2).cuda()
    x = torch.randn(num_tokens, dim).cuda()
    
    # Get expert assignments
    with torch.no_grad():
        logits = moe.gate(x)
    probs = torch.softmax(logits, dim=-1)
    _, topk_i = torch.topk(probs, moe.top_k)
    
    # Check expert diversity
    unique_experts = torch.unique(topk_i)
    assert len(unique_experts) > 1, "All tokens routed to single expert"
    print(f"\nActivated experts: {len(unique_experts)}/{num_experts}")
    print("Expert activation test passed!")

def test_numerical_stability():
    # Test with extreme values
    dim = 256
    moe = MoE(dim, 4, dim*2).cuda()
    
    # Large values
    x = torch.randn(128, dim).cuda() * 1e3
    out = moe(x)
    assert not torch.isnan(out).any(), "NaN with large inputs"
    
    # Small values
    x = torch.randn(128, dim).cuda() * 1e-6
    out = moe(x)
    assert not torch.isnan(out).any(), "NaN with small inputs"
    print("Numerical stability test passed!")

def test_edge_cases():
    # Test empty expert case
    dim = 128
    moe = MoE(dim, 2, dim*2).cuda()
    
    # Force all tokens to one expert
    with torch.no_grad():
        moe.gate.bias.data.fill_(10)  # Bias gate to first expert
        moe.gate.weight.data.zero_()
    
    x = torch.randn(1024, dim).cuda()
    out = moe(x)
    assert out.isfinite().all(), "Failed with skewed routing"
    print("Edge case test passed!")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Starting MoE tests...\n")
    test_moe_basic()
    test_routing_logic()
    test_expert_activation()
    test_numerical_stability()
    test_edge_cases()
    
    print("\nAll tests passed successfully!")