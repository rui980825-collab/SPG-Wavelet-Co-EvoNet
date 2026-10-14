import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_filters(device, dtype):
    ll = torch.tensor([[0.5, 0.5],
                       [0.5, 0.5]], device=device, dtype=dtype)
    lh = torch.tensor([[-0.5, -0.5],
                       [ 0.5,  0.5]], device=device, dtype=dtype)
    hl = torch.tensor([[-0.5,  0.5],
                       [-0.5,  0.5]], device=device, dtype=dtype)
    hh = torch.tensor([[ 0.5, -0.5],
                       [-0.5,  0.5]], device=device, dtype=dtype)
    return ll, lh, hl, hh


class HaarDWT(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        if (h % 2) != 0 or (w % 2) != 0:
            raise ValueError("H and W must be even for Haar DWT (choose img_size even, e.g., 64/128).")

        ll, lh, hl, hh = haar_filters(x.device, x.dtype)

        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4,1,2,2)
        weight = weight.repeat(c, 1, 1, 1)                          # (4C,1,2,2)

        y = F.conv2d(x, weight, stride=2, padding=0, groups=c)      # (B,4C,H/2,W/2)
        y = y.view(b, c, 4, h // 2, w // 2)
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


class HaarIDWT(nn.Module):
    def forward(self, LL, LH, HL, HH):
        b, c, h, w = LL.shape
        ll, lh, hl, hh = haar_filters(LL.device, LL.dtype)

        y = torch.stack([LL, LH, HL, HH], dim=2).view(b, 4 * c, h, w)

        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4,1,2,2)
        weight = weight.repeat(c, 1, 1, 1)                          # (4C,1,2,2)

        x = F.conv_transpose2d(y, weight, stride=2, padding=0, groups=c)
        return x


class KANLikeHF(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.pw1 = nn.Conv2d(ch, ch, 1, bias=True)
        self.pw2 = nn.Conv2d(ch, ch, 1, bias=True)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + 0.1 * self.pw2(self.act(self.pw1(x)))


class WKANBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        self.hf = KANLikeHF(ch)

    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        LH2 = self.hf(LH)
        HL2 = self.hf(HL)
        HH2 = self.hf(HH)
        return self.idwt(LL, LH2, HL2, HH2)
