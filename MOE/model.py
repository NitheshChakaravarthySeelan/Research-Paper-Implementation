import torch
import torch.nn as nn
import triton
from typing import Optional
    
from flash_attn import TritonAttention


class TritonSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.softmax_scale = 1.0 / (self.head_dim ** 0.5)


    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ):
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Project inputs
        query = self.q_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Use Triton Attention
        output = TritonAttention.apply(query, key, value, causal, self.softmax_scale)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, hidden_size
        )
        return self.out_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TritonSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size)
        )
        
    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TritonTransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        
        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and positions
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm and head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits