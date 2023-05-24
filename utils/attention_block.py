# Shashank Ojha
#
# Implements a Standard Self-Attention Block.
#
# The input to this is a sequence of tokens. It's assumed the tokens
# are embedded so they all have dimensions embed_dim. Thus the input
# becomes a tensor of shape (num_tokens, embed_dim)
#
# The output of this is the same sequence of tokens but embedded in a
# new space that encodes context. This new context matrix has shape
# (num_tokens, v_dim). 
#
# Reference: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

import torch

class SelfAttention(torch.nn.Module):
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

    self.Wq = torch.nn.Parameter(torch.rand(kq_dim, embed_dim))
    self.Wk = torch.nn.Parameter(torch.rand(kq_dim, embed_dim))
    self.Wv = torch.nn.Parameter(torch.rand(v_dim, embed_dim))

  def forward(self, X):
    '''
    Computes f(x) on input.

    X: tensor with shape (num_tokens, embed_dim)
    '''
    # Shape: (num_tokens, kq_dim)
    queries = torch.matmul(self.Wq, X.T).T
    keys = torch.matmul(self.Wk, X.T).T

    # Shape: (num_tokens, v_dim)
    values = torch.matmul(self.Wv, X.T).T

    # Shape = (num_tokens, num_tokens)
    attention_matrix = torch.matmul(queries, keys.T)
    scaled_attention_matrix = attention_matrix / self.kq_dim ** 0.5
    normalized_attention_matrix = torch.nn.functional.softmax(scaled_attention_matrix, dim=-1)
    
    # Shape = (num_tokens, v_dim)
    context = torch.matmul(normalized_attention_matrix, values)

    return context



