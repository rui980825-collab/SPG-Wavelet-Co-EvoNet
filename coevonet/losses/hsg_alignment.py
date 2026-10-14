import torch
import torch.nn as nn
import torch.nn.functional as F


class HSGLoss(nn.Module):
    """
    HSG: Hyper-Symmetric Gradient alignment loss (Sobel gradient L1).
    Notes:
      - Under AMP, inputs may be fp16. We compute Sobel gradients in fp32 for stability.
      - This avoids dtype mismatch in conv2d and improves numerical robustness.
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    @staticmethod
    def _to_gray(x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W); if RGB, convert to gray; else keep as is
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.shape}")
        if x.size(1) == 1:
            return x
        # assume RGB in [0,1]
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def _grad(self, x: torch.Tensor):
        """
        Compute Sobel gradients in fp32 (even if x is fp16 under AMP).
        """
        x = self._to_gray(x)

        # Force fp32 for stable sobel and to match kernel dtype
        x32 = x.float()
        kx = self.kx.to(device=x32.device, dtype=x32.dtype)
        ky = self.ky.to(device=x32.device, dtype=x32.dtype)

        gx = F.conv2d(x32, kx, padding=1)
        gy = F.conv2d(x32, ky, padding=1)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Loss in fp32 for stability; return fp32 scalar
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pgx, pgy = self._grad(pred)
            tgx, tgy = self._grad(target)
            return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)

    @torch.no_grad()
    def grad_alignment_score(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        A scalar score (higher is better alignment). Implemented as 1/(1+L1(grad)).
        Computed in fp32; safe under AMP.
        """
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pgx, pgy = self._grad(pred)
            tgx, tgy = self._grad(target)
            l = F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)
            score = 1.0 / (1.0 + l)
            return score
