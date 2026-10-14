import torch
import torch.nn as nn

from coevonet.models.wkan import WKANBlock


class _StemRGB(nn.Module):
    def __init__(self, base_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class _StemTIR(nn.Module):
    def __init__(self, base_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class _SharedEncoder(nn.Module):
    def __init__(self, base_ch: int, latent_ch: int):
        super().__init__()
        ch = base_ch
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1), nn.SiLU(),  # 64->32
            nn.Conv2d(ch * 2, ch * 4, 4, stride=2, padding=1), nn.SiLU(),  # 32->16
            nn.Conv2d(ch * 4, ch * 8, 4, stride=2, padding=1), nn.SiLU(),  # 16->8
            nn.Flatten(),
            nn.Linear(ch * 8 * 8 * 8, latent_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class _SharedDecoder(nn.Module):
    def __init__(self, base_ch: int, latent_ch: int):
        super().__init__()
        ch = base_ch
        self.fc = nn.Sequential(nn.Linear(latent_ch, ch * 8 * 8 * 8), nn.SiLU())
        self.de1 = nn.Sequential(nn.ConvTranspose2d(ch * 8, ch * 4, 4, stride=2, padding=1), nn.SiLU())  # 8->16
        self.de2 = nn.Sequential(nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1), nn.SiLU())  # 16->32
        self.de3 = nn.Sequential(nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1), nn.SiLU())      # 32->64
        self.wkan = WKANBlock(ch)

    def forward(self, z):
        x = self.fc(z).view(z.shape[0], -1, 8, 8)
        x = self.de1(x)
        x = self.de2(x)
        x = self.de3(x)
        x = self.wkan(x)
        return x


class SiameseWKANAE(nn.Module):
    """
    输入：rgb(3,H,W), tir(1,H,W)
    输出：rgb_hat(3,H,W), tir_hat(1,H,W)
    """
    def __init__(self, base_ch: int = 32, latent_ch: int = 256):
        super().__init__()
        self.stem_rgb = _StemRGB(base_ch)
        self.stem_tir = _StemTIR(base_ch)

        self.encoder = _SharedEncoder(base_ch, latent_ch)
        self.decoder = _SharedDecoder(base_ch, latent_ch)

        self.head_rgb = nn.Sequential(nn.Conv2d(base_ch, 3, 3, padding=1), nn.Sigmoid())
        self.head_tir = nn.Sequential(nn.Conv2d(base_ch, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, rgb, tir):
        fr = self.stem_rgb(rgb)
        ft = self.stem_tir(tir)

        zr = self.encoder(fr)
        zt = self.encoder(ft)

        z = 0.5 * (zr + zt)  # shared latent

        feat = self.decoder(z)
        rgb_hat = self.head_rgb(feat)
        tir_hat = self.head_tir(feat)
        return rgb_hat, tir_hat
