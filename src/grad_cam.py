import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: torch model in eval mode
        target_layer: the layer module to hook (e.g., model.features[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            # out shape: (B, C, H, W)
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] shape: (B, C, H, W)
            self.gradients = grad_out[0].detach()

        # clear previous hooks if any by reassigning (simple approach)
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        """
        input_tensor: torch tensor (1, C, H, W) on same device as model
        class_idx: int or None (if None, uses predicted class)
        returns: cam (H, W) numpy normalized 0-1
        """
        self.model.zero_grad()
        output = self.model(input_tensor)             # (1, num_classes)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        loss = output[0, class_idx]
        loss.backward(retain_graph=True)

        # get gradients and activations
        grads = self.gradients[0].cpu().numpy()       # (C, H, W)
        acts = self.activations[0].cpu().numpy()      # (C, H, W)

        # global average pooling of gradients over spatial dims
        weights = np.mean(grads, axis=(1, 2))         # (C,)

        cam = np.zeros(acts.shape[1:], dtype=np.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]

        cam = np.maximum(cam, 0)
        if cam.max() == 0:
            return np.zeros_like(cam)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
