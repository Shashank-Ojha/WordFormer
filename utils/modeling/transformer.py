# Shashank Ojha
#
# Implements the generative pre-trained transformer (GPT) model

import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(torch.nn.Module):
  def __init__(self, embed_dim, kq_dim, v_dim) -> None:
    '''
    Initializes Model.
    '''
    super().__init__()


  def forward(self, X):
    '''
    Computes f(x) on input.
    '''
    

def generate(model, context, max_context_size, max_new_tokens):
    """
    model: Pytorch model.
    context: array tokens with shape (B, t), where t can be any length.
    max_context_size: max number of tokens in context to keep when invoking model.
    max_new_tokens: max number of tokens to generate.
    """
    for _ in range(max_new_tokens):
        sub_context = context if context.shape[1] < max_context_size else context[:, -max_context_size:]
        # Shape = (batch_size, block_size, vocab_size)
        logits = model(sub_context)
        # Get the last character's predictions. Shape = (B, vocab_size)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1) 
        
        # Sample from the distribution. Shape = (B, 1)
        preds = torch.multinomial(probs, num_samples=1)   
        
        # Add to context
        context = torch.cat((context, preds), dim=1)

    return context

