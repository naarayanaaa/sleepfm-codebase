"""Utility functions for computing gradient-based attribution maps.

This module implements Saliency, Integrated Gradients, and SmoothGrad for
1-D time-series inputs used by SleepFM.  Each helper accepts a callable model
and returns tensors that can be post-processed or saved by the caller.

The functions are intentionally framework-agnostic beyond PyTorch so that they
can be re-used in research notebooks and the new `xai_head.py` CLI without
introducing a hard dependency on Captum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def _gather_targets(logits: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
    """Selects the relevant logit for attribution backpropagation.

    Args:
        logits: Output of the model with shape ``(batch, classes)``.
        target: Optional tensor of shape ``(batch,)`` containing class indices.

    Returns:
        Tensor of shape ``(batch,)`` with the logit for each requested target.
    """

    if target is None:
        target = logits.argmax(dim=1)
    if target.dim() == 1:
        target = target.view(-1, 1)
    gathered = logits.gather(1, target.long())
    return gathered.squeeze(1)


def compute_saliency(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes vanilla saliency maps.

    Args:
        model: Callable returning logits.  The model is evaluated in ``eval`` mode
            for deterministic gradients.
        inputs: Input tensor with ``requires_grad`` enabled.
        target: Optional class indices for which to compute gradients.  When
            ``None`` the predicted class is used.

    Returns:
        Gradient tensor with the same shape as ``inputs``.
    """

    model.eval()
    inputs = inputs.clone().detach().requires_grad_(True)
    logits = model(inputs)
    selected = _gather_targets(logits, target)
    grads = torch.autograd.grad(selected.sum(), inputs, retain_graph=False)[0]
    return grads.detach()


def compute_integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 50,
) -> torch.Tensor:
    """Computes Integrated Gradients for a batch of samples.

    Args:
        model: Callable returning logits.
        inputs: Tensor of shape ``(batch, channels, time)``.
        target: Optional class indices.  Defaults to model predictions.
        baseline: Reference input.  Defaults to zeros matching ``inputs``.
        steps: Number of integration steps (including the endpoint).

    Returns:
        Attribution tensor with the same shape as ``inputs``.
    """

    if steps < 1:
        raise ValueError("steps must be a positive integer")

    model.eval()
    device = inputs.device
    inputs = inputs.clone().detach()
    baseline = baseline if baseline is not None else torch.zeros_like(inputs)
    baseline = baseline.to(device)

    scaled_inputs = [
        baseline + (float(i) / steps) * (inputs - baseline) for i in range(1, steps + 1)
    ]
    grads = []
    for scaled in scaled_inputs:
        scaled.requires_grad_(True)
        logits = model(scaled)
        selected = _gather_targets(logits, target)
        grad = torch.autograd.grad(selected.sum(), scaled, retain_graph=False)[0]
        grads.append(grad)

    total_grad = torch.stack(grads).mean(dim=0)
    return (inputs - baseline) * total_grad


def compute_smoothgrad(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    noise_level: float = 0.1,
    samples: int = 25,
) -> torch.Tensor:
    """Computes SmoothGrad attributions.

    Args:
        model: Callable returning logits.
        inputs: Tensor of shape ``(batch, channels, time)``.
        target: Optional class indices.  Defaults to predictions per sample.
        noise_level: Relative noise standard deviation (percentage of input
            standard deviation).
        samples: Number of noisy samples to average over.

    Returns:
        Attribution tensor with the same shape as ``inputs``.
    """

    if samples < 1:
        raise ValueError("samples must be at least 1")

    model.eval()
    inputs = inputs.clone().detach()
    std = inputs.std(dim=-1, keepdim=True, unbiased=False)
    noise_sigma = noise_level * (std + 1e-6)

    saliency_accumulator = torch.zeros_like(inputs)
    for _ in range(samples):
        noise = torch.normal(mean=0.0, std=noise_sigma)
        noisy_input = (inputs + noise).requires_grad_(True)
        logits = model(noisy_input)
        selected = _gather_targets(logits, target)
        grads = torch.autograd.grad(selected.sum(), noisy_input, retain_graph=False)[0]
        saliency_accumulator += grads.detach()

    return saliency_accumulator / samples


@dataclass
class SaliencyOutputs:
    """Container aggregating the standard attribution methods."""

    saliency: torch.Tensor
    integrated_gradients: torch.Tensor
    smoothgrad: torch.Tensor


def compute_all_attributions(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    baseline: Optional[torch.Tensor] = None,
    ig_steps: int = 50,
    smoothgrad_samples: int = 25,
    smoothgrad_noise: float = 0.1,
) -> SaliencyOutputs:
    """Convenience wrapper to compute the three default attribution maps."""

    return SaliencyOutputs(
        saliency=compute_saliency(model, inputs, target=target),
        integrated_gradients=compute_integrated_gradients(
            model, inputs, target=target, baseline=baseline, steps=ig_steps
        ),
        smoothgrad=compute_smoothgrad(
            model,
            inputs,
            target=target,
            noise_level=smoothgrad_noise,
            samples=smoothgrad_samples,
        ),
    )

