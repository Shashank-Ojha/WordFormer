# Shashank Ojha
#
# Implements a Standard Self-Attention Block.
#
# The input to this is a sequence of tokens. It's assumed the tokens
# are embedded so they all have dimensions embed_dim. Thus the input
# becomes a tensor of shape (B, seq_len, embed_dim)
#
# The output of this is the same sequence of tokens but embedded in a
# new space that encodes context. This new context matrix has shape
# (B, seq_len, v_dim). 
#
# Reference: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
  def __init__(self, embed_dim, kq_dim, v_dim) -> None:
    '''
    Initializes Block.

    embed_dim: the number of dimensions used to embed a token.
    kq_dim: desired dimensions of the queries and keys.
    v_dim: desired dimensions of the values.
    '''
    super().__init__() 

    self.embed_dim = embed_dim
    self.kq_dim = kq_dim
    self.v_dim = v_dim

    self.query_embed = nn.Linear(embed_dim, kq_dim, bias=False)
    self.key_embed = nn.Linear(embed_dim, kq_dim, bias=False)
    self.value_embed = nn.Linear(embed_dim, v_dim, bias=False)

  def forward(self, X):
    '''
    Computes self-attention on each sequence of tokens in the batch.

    X: tensor with shape (B, seq_len, embed_dim)
    '''
    # Shape: (B, seq_len, kq_dim)
    Q = self.query_embed(X)
    K = self.key_embed(X)

    # Shape: (B, seq_len, v_dim)
    V = self.value_embed(X)


    # TODO: Add masking so we can't look at future tokens!

    # This does batch matrix multiplication, so it becomes:
    #      (B, seq_len, kq_dim) x (B, kq_dim, seq_len) 
    # which has the final shape:
    #      (B, seq_len, seq_len)
    attention_matrix = torch.matmul(Q, K.transpose(-1, -2))
    scaled_attention_matrix = attention_matrix / self.kq_dim ** -0.5
    normalized_attention_matrix = F.softmax(scaled_attention_matrix, dim=-1)
  
    # This does batch matrix multiplication, so it becomes:
    #      (B, seq_len, seq_len) x (B, seq_len, v_dim) 
    # which has the final shape:
    #      (B, seq_len, v_dim)
    context = torch.matmul(normalized_attention_matrix, V)

    return context

