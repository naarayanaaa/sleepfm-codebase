"""1-D Grad-CAM utilities for SleepFM encoders."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class GradCAM1D:
    """Minimal Grad-CAM implementation for 1-D convolutional models."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self.handles.append(
            self.target_layer.register_forward_hook(self._forward_hook)
        )
        self.handles.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(
        self,
        inputs: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        upsample_to: Optional[int] = None,
    ) -> torch.Tensor:
        """Generates Grad-CAM maps.

        Args:
            inputs: Batch tensor of shape ``(batch, channels, time)``.
            target: Optional target indices.  Uses argmax when omitted.
            upsample_to: Optional output length.  Defaults to the input length.
        """

        self.model.eval()
        inputs = inputs.clone().detach().requires_grad_(True)
        logits = self.model(inputs)
        if target is None:
            target = logits.argmax(dim=1)
        if target.dim() == 1:
            target = target.view(-1, 1)

        scores = logits.gather(1, target)
        self.model.zero_grad(set_to_none=True)
        scores.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients")

        weights = self.gradients.mean(dim=-1, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        upsample_to = upsample_to or inputs.size(-1)
        cam = cam.unsqueeze(1)
        cam = F.interpolate(cam, size=upsample_to, mode="linear", align_corners=False)
        cam = cam.squeeze(1)
        cam_min, cam_max = cam.min(dim=-1, keepdim=True).values, cam.max(
            dim=-1, keepdim=True
        ).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach()


def compute_gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    inputs: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    upsample_to: Optional[int] = None,
) -> torch.Tensor:
    """Functional wrapper around :class:`GradCAM1D`."""

    gradcam = GradCAM1D(model, target_layer)
    try:
        return gradcam(inputs, target=target, upsample_to=upsample_to)
    finally:
        gradcam.remove_hooks()

