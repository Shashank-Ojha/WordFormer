# Utils to facilitate training and evaluation

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_pipelines import get_batch


def loss_fn(logits, targets):
    """
    logits: Shape (B, T, C)
    targets: Shape (B, T)
    """
    (B, T, C) = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)    
    # This wants logits to have have (N, C) and targets to have
    # shape (N,), where N is the batch size and C is the number of
    # classes.
    return F.cross_entropy(logits, targets)
    
@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size):
    out = {}
    model.eval()
    for dataset_type, dataset in {'train': train_data, 'val': val_data}.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(dataset, batch_size, block_size)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            losses[k] = loss.item()
        out[dataset_type] = losses.mean()
    model.train()
    return out