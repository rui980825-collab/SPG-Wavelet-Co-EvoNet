import torch
import torch.nn as nn
import torch.nn.functional as F

from coevonet.models.wkan import HaarDWT


@torch.no_grad()
def wavelet_energy_stats(pred, target):
    dwt = HaarDWT()
    LLp, LHp, HLp, HHp = dwt(pred)
    E_ll = (LLp ** 2).mean().item()
    E_hf = ((LHp ** 2).mean() + (HLp ** 2).mean() + (HHp ** 2).mean()).item() / 3.0
    return {"E_ll": E_ll, "E_hf": E_hf}


class FIBWaveletLoss(nn.Module):
    """
    FIB：小波子带一致性（LL + HF）
    """
    def __init__(self):
        super().__init__()
        self.dwt = HaarDWT()

    def forward(self, pred, target):
        LLp, LHp, HLp, HHp = self.dwt(pred)
        LLt, LHt, HLt, HHt = self.dwt(target)
        loss_ll = F.l1_loss(LLp, LLt)
        loss_hf = (F.l1_loss(LHp, LHt) + F.l1_loss(HLp, HLt) + F.l1_loss(HHp, HHt)) / 3.0
        return loss_ll + loss_hf
