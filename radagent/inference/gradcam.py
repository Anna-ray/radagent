"""
radagent.inference.gradcam
--------------------------
Grad-CAM++ for the specialist CXR model.

We hook the last spatial feature map of the ConvNeXt-V2 backbone (already
exposed by SpecialistCXR.forward via self._last_feat) and the gradients
flowing back to it.

Output: a [H, W] heatmap in [0, 1] you can overlay on the original image.
This is what populates the "click a finding to see its visual region"
panel in the demo.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAMpp:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):
        self.model = model
        self.target_module = target_module
        self._features: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._h_fwd = target_module.register_forward_hook(self._fwd_hook)
        self._h_bwd = target_module.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _m, _inp, out):
        self._features = out

    def _bwd_hook(self, _m, _grad_in, grad_out):
        self._grads = grad_out[0]

    def remove(self):
        self._h_fwd.remove()
        self._h_bwd.remove()

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        """
        x: [1, 3, H, W] preprocessed input
        class_idx: which finding to explain
        returns: [H, W] heatmap in [0,1] at the input resolution
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)               # [1, C]
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        feats = self._features              # [1, C, h, w]
        grads = self._grads                 # [1, C, h, w]

        # Grad-CAM++ weights
        grads2 = grads ** 2
        grads3 = grads ** 3
        sum_feats = feats.sum(dim=(2, 3), keepdim=True)
        denom = 2 * grads2 + sum_feats * grads3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alphas = grads2 / denom
        weights = (alphas * F.relu(grads)).sum(dim=(2, 3), keepdim=True)  # [1,C,1,1]

        cam = (weights * feats).sum(dim=1, keepdim=True)                  # [1,1,h,w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()

        # Normalize to [0,1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_heatmap(
    image_rgb_u8: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Compose a colored heatmap on top of the original image."""
    cam_u8 = np.uint8(cam * 255)
    color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(color, alpha, image_rgb_u8, 1 - alpha, 0)
