"""Gradient caching without retaining encoder graphs across chunks.

Algorithm: Gao et al. (2021), https://arxiv.org/abs/2101.06983.
AMP scales the representation loss ONCE; cached derivatives already carry
that scale, so replay uses autograd.backward, never a second scaler.scale.
The caller still owns clipping, optimizer.step and scaler.update.
"""
from contextlib import contextmanager, nullcontext

import torch
from torch.nn import functional as F


def masked_inbatch_loss(vectors, temperature, allowed, targets):
    """First B vectors are queries; remaining B*16 are grouped passages."""
    batch = targets.numel()
    if allowed.shape != (batch, batch * 16) or vectors.shape[0] != batch * 17:
        raise ValueError("Expected B queries and B*16 grouped passages")
    if allowed.dtype != torch.bool or not allowed[torch.arange(batch, device=targets.device), targets].all():
        raise ValueError("Every designated positive must be allowed")
    if (allowed.sum(1) < 2).any():
        raise ValueError("Each query needs a positive and at least one legal negative")
    # Float32 loss and matrix product even inside an enclosing AMP context.
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        queries = F.normalize(vectors[:batch].float(), dim=-1)
        passages = F.normalize(vectors[batch:].float(), dim=-1)
        scores = queries @ passages.T / temperature.float()
        logits = scores.masked_fill(~allowed, -torch.inf)
        loss = F.cross_entropy(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite masked InfoNCE loss")
    return loss, logits


class TorchRNGSnapshot:
    """Replay dropout while leaving the caller's RNG at its post-cache state."""
    def __init__(self, device):
        self.device = torch.device(device)
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None

    @contextmanager
    def replay(self):
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.cuda is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu)
            if self.cuda is not None:
                torch.cuda.set_rng_state(self.cuda, self.device)
            yield


def cached_backward(encode, chunks, loss_fn, backward, autocast_context=nullcontext, checkpointing_model=None):
    """Exact chain-rule replay; includes loss-owned parameters (e.g. T) once.

    ``backward`` may be Accelerator.backward or loss.backward. Its scaling
    must stay constant until all replay chunks finish. No optimizer step here.
    """
    representations, states, sizes = [], [], []
    was_checkpointing = getattr(checkpointing_model, "gradient_checkpointing", False)
    try:
        if was_checkpointing:
            # There is no graph to checkpoint in the cache pass. Reentrant
            # checkpoint would otherwise warn about intentionally absent grads.
            checkpointing_model.gradient_checkpointing = False
        with torch.no_grad():
            for inputs in chunks:
                device = next(iter(inputs.values())).device
                states.append(TorchRNGSnapshot(device))
                with autocast_context():
                    output = encode(inputs)
                if not torch.isfinite(output).all():
                    raise FloatingPointError("Non-finite GradCache embeddings")
                representations.append(output.float())
                sizes.append(len(output))
    finally:
        if was_checkpointing:
            checkpointing_model.gradient_checkpointing = True
    leaf = torch.cat(representations).detach().requires_grad_(True)
    del representations, output
    loss = loss_fn(leaf)
    backward(loss)
    if leaf.grad is None:
        raise RuntimeError("Representation loss was detached; no gradient cache")
    cache = leaf.grad.detach().split(sizes)
    value = loss.detach()
    del loss, leaf
    # Non-finite SCALED derivatives are allowed through GradScaler's normal
    # overflow path; forward embeddings/loss and weights are still checked.
    for inputs, state, derivative in zip(chunks, states, cache):
        with state.replay(), autocast_context():
            recomputed = encode(inputs)
        torch.autograd.backward(recomputed.float(), derivative)
        del recomputed
    return value
