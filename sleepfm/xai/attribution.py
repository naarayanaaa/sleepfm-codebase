from __future__ import annotations

from typing import Callable, Optional

import torch


def saliency_map(forward_fn: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor, target_index: int) -> torch.Tensor:
    """Compute vanilla saliency maps for a batch of inputs."""

    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = forward_fn(inputs)
    scores = outputs[:, target_index]
    grad = torch.autograd.grad(scores, inputs, torch.ones_like(scores), retain_graph=False)[0]
    return grad.detach()


def integrated_gradients(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    target_index: int,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 50,
) -> torch.Tensor:
    """Approximate Integrated Gradients for a batch of inputs."""

    if baseline is None:
        baseline = torch.zeros_like(inputs)

    alphas = torch.linspace(0.0, 1.0, steps=steps, device=inputs.device, dtype=inputs.dtype)
    total_grad = torch.zeros_like(inputs)

    for alpha in alphas:
        interpolated = baseline + alpha * (inputs - baseline)
        interpolated.requires_grad_(True)
        outputs = forward_fn(interpolated)
        scores = outputs[:, target_index]
        grad = torch.autograd.grad(scores, interpolated, torch.ones_like(scores), retain_graph=False)[0]
        total_grad += grad.detach()

    avg_grad = total_grad / steps
    return (inputs - baseline) * avg_grad


def smooth_grad(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    target_index: int,
    noise_std: float = 0.1,
    num_samples: int = 25,
) -> torch.Tensor:
    """Compute SmoothGrad saliency maps."""

    device = inputs.device
    total_grad = torch.zeros_like(inputs)

    for _ in range(num_samples):
        noise = torch.normal(mean=0.0, std=noise_std, size=inputs.shape, device=device, dtype=inputs.dtype)
        noisy_inputs = (inputs + noise).detach().requires_grad_(True)
        outputs = forward_fn(noisy_inputs)
        scores = outputs[:, target_index]
        grad = torch.autograd.grad(scores, noisy_inputs, torch.ones_like(scores), retain_graph=False)[0]
        total_grad += grad.detach()

    return total_grad / num_samples


def grad_cam_1d(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_index: int,
    layer: torch.nn.Module,
) -> torch.Tensor:
    """Generate 1D Grad-CAM activations for the provided layer."""

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_full_backward_hook(backward_hook)

    try:
        inputs = inputs.clone().detach().requires_grad_(True)
        outputs = model(inputs)
        scores = outputs[:, target_index]
        model.zero_grad()
        scores.backward(torch.ones_like(scores))
    finally:
        handle_f.remove()
        handle_b.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture any data")

    activation = activations[0]
    gradient = gradients[0]
    weights = gradient.mean(dim=-1, keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=inputs.shape[-1], mode="linear", align_corners=False)
    cam = cam.squeeze(1)
    return cam.detach()


__all__ = ["saliency_map", "integrated_gradients", "smooth_grad", "grad_cam_1d"]
