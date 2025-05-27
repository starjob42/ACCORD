import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Combined projections for Q, K, V
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape for multi-head attention
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch, heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
            
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, seq_len, seq_len]
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        
        # Reshape output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        context = self.out_proj(context)
        
        return context, attn_weights


class EnhancedTextClassifier(nn.Module):
    """Improved text classifier with multi-head attention."""
    
    def __init__(self, tokenizer_name, num_classes=6, embedding_dim=768, hidden_dim=256, max_position=40000):
        super().__init__()
        # Get tokenizer configuration to determine vocabulary size
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(tokenizer_name)
        vocab_size = config.vocab_size
        
        # Embedding layers
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_position, embedding_dim)
        self.max_position = max_position
        
        # Layer normalization and dropout for embeddings
        self.emb_layer_norm = nn.LayerNorm(embedding_dim)
        self.emb_dropout = nn.Dropout(0.1)
        
        # Project embeddings to hidden dimension
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Multi-head attention layers
        self.attention1 = MultiHeadAttention(hidden_dim, num_heads=8)
        self.attention2 = MultiHeadAttention(hidden_dim, num_heads=8)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Attention-based pooling
        self.pool_attn = nn.Linear(hidden_dim, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids, attention_mask=None):
        # Get embeddings from tokenized input_ids
        batch_size, seq_length = input_ids.shape
        
        # Generate position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = torch.clamp(position_ids, 0, self.max_position - 1)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get token and position embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Add them up and apply normalization and dropout
        embeddings = token_embeds + position_embeds
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        
        # Project to hidden dimension
        hidden_states = self.projection(embeddings)
        
        # First multi-head attention block with residual connection
        attn_output, _ = self.attention1(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + ffn_output)
        
        # Second multi-head attention block with residual connection
        attn_output, _ = self.attention2(hidden_states, attention_mask)
        hidden_states = self.layer_norm3(hidden_states + attn_output)
        
        # Attention-based pooling for classification
        if attention_mask is not None:
            # Calculate attention weights for pooling
            attn_scores = self.pool_attn(hidden_states).squeeze(-1)
            # Mask out padding tokens
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)
            # Weighted sum
            pooled = torch.sum(hidden_states * attn_weights, dim=1)
        else:
            # If no mask, use mean pooling
            pooled = hidden_states.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        # Important: Return just the logits tensor, not a dictionary
        return logits
