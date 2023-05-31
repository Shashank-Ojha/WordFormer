# Shashank Ojha
#
# Implements the generative pre-trained transformer (GPT) model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

# Implements a Standard Self-Attention Block.
#
# The input to this is a sequence of tokens. It's assumed the tokens
# are embedded so they all have dimensions embed_dim. Thus the input
# becomes a tensor of shape (B, seq_len, embed_dim)
#
# The output of this is the sames sequence of tokens but embedded in a
# new space that encodes context. This new context matrix has shape
# (B, seq_len, v_dim). 
#
# References: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
#             https://www.youtube.com/watch?v=kCc8FmEb1nY&t=51s
class SelfAttention(nn.Module):
  def __init__(self, embed_dim, kq_dim, v_dim, max_seq_length) -> None:
    '''
    Initializes Block.

    embed_dim: the number of dimensions used to embed a token.
    kq_dim: desired dimensions of the queries and keys.
    v_dim: desired dimensions of the values.
    '''
    super().__init__() 

    self.kq_dim = kq_dim

    self.query_embed = nn.Linear(embed_dim, kq_dim, bias=False)
    self.key_embed = nn.Linear(embed_dim, kq_dim, bias=False)
    self.value_embed = nn.Linear(embed_dim, v_dim, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(max_seq_length, max_seq_length)))

  def forward(self, X):
    '''
    Computes self-attention on each sequence of tokens in the batch.

    X: tensor with shape (B, seq_len, embed_dim)
    '''
    (_, seq_len, _) = X.shape

    # Shape: (B, seq_len, kq_dim)
    Q = self.query_embed(X)
    K = self.key_embed(X)

    # Shape: (B, seq_len, v_dim)
    V = self.value_embed(X)

    # This does batch matrix multiplication, so it becomes:
    #      (B, seq_len, kq_dim) x (B, kq_dim, seq_len) 
    # which has the final shape:
    #      (B, seq_len, seq_len)
    attention_matrix = torch.matmul(Q, K.transpose(-1, -2))
    # Observe that that attention_matrix[i, j] is the dot product between
    # Q[i] and K[j]. For self attention that means the ith token is estimating
    # how related the jth token is to itself. However, when i<j, that means
    # the ith token is looking at a token from the future that it should not
    # have access to. Thus we want to negate all values j > i. These are all 
    # the values that are 0 in the lower triangle matrix of shape 
    # seq_len x seq_len.s
    attention_matrix = attention_matrix.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
    scaled_attention_matrix = attention_matrix / self.kq_dim ** -0.5
    normalized_attention_matrix = F.softmax(scaled_attention_matrix, dim=-1)
  
    # This does batch matrix multiplication, so it becomes:
    #      (B, seq_len, seq_len) x (B, seq_len, v_dim) 
    # which has the final shape:
    #      (B, seq_len, v_dim)
    context = torch.matmul(normalized_attention_matrix, V)

    return context


class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Shape = (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)
        # Shape = (ceil(d_model/2),)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Shape = (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # Shape = (max_len, ceil(d_model/2))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len]``

        Returns:
            Tensor with shape [seq_len, embedding_dim]
        """
        return self.pe[:x.size(1)]

# Config parameters for the GPT class below.
@dataclass
class GPTConfig:
  # Data Info
  vocab_size: int
  max_seq_length: int

  # Embedding Layer
  embed_dim: int

  # Attention Layer
  kq_dim: int
  v_dim: int

# GPT model.
class GPT(torch.nn.Module):
  def __init__(self, config) -> None:
    '''
    Initializes Model.
    '''
    super().__init__()

    self.config = config

    self.token_embedding_table = nn.Embedding(config.vocab_size, config.embed_dim)
    self.positional_encoder = PositionalEncoder(config.embed_dim, config.max_seq_length)

    # Note that while block_size is passed to the Self Attention module, it really is
    # just an upper bound on the max block size. Any input shape of (batch_size, t) where
    # t <= block_size is valid.
    self.model = nn.Sequential(
        # Transforms shape to (batch_size, block_size, v_dim)
        SelfAttention(config.embed_dim, config.kq_dim, config.v_dim, config.max_seq_length),
        # Transforms shape to (batch_size, block_size, vocab_size)
        nn.Linear(config.v_dim, config.vocab_size)
    )

  def forward(self, x):
    '''
    X: tensor with shape (batch_size, seq_len)
    '''
    # Shape = (batch_size, seq_len, embed_dim)
    token_embed = self.token_embedding_table(x)
    # Shape = (seq_len, embed_dim)
    position_embed = self.positional_encoder(x)
    # Shape = (batch_size, seq_len, embed_dim)
    x = token_embed + position_embed
    return self.model(x)
    
  def generate(self, context, max_new_tokens):
    '''
    context: array tokens with shape (B, t), where t can be any length.
    max_new_tokens: max number of tokens to generate.
    '''
    for _ in range(max_new_tokens):
        sub_context = context if context.shape[1] < self.config.max_seq_length else context[:, -self.config.max_seq_length:]
        # Shape = (batch_size, block_size, vocab_size)
        logits = self(sub_context)
        # Get the last character's predictions. Shape = (B, vocab_size)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1) 
        
        # Sample from the distribution. Shape = (B, 1)
        preds = torch.multinomial(probs, num_samples=1)   
        
        # Add to context
        context = torch.cat((context, preds), dim=1)

    return context