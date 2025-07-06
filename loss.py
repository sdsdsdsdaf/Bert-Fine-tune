import torch.nn.functional as F
from typing import Union, Optional
import torch

def quantile_loss(y_pred, y_true, q=0.9):
    e = y_true - y_pred
    return torch.mean(torch.max(q * e, (q - 1) * e))

def combined_top10_loss(y_pred, y_true,
                        alpha_mse=0.4,
                        alpha_wmse=0.3,
                        alpha_quant=0.3):
  
    loss_mse   = F.mse_loss(y_pred, y_true)

    th         = torch.quantile(y_true, 0.9)
    w          = torch.where(y_true > th, 2.0, 1.0).to(y_true.device)
    loss_wmse  = torch.mean(w * (y_pred - y_true).pow(2))

    loss_quant = quantile_loss(y_pred, y_true, q=0.9)
    # 4) 가중합
    return alpha_mse * loss_mse + alpha_wmse * loss_wmse + alpha_quant * loss_quant

class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean"
    ):
        """
        gamma: focusing 파라미터
        alpha: 클래스 가중치 (scalar or tensor of shape [C])
        label_smoothing: [0.0, 1.0) 사이 값
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C), target: (N,) int
        # 1) label smoothing
        C = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.scatter_(1, target.unsqueeze(1), 1)
            true_dist = true_dist * (1 - self.smoothing) + self.smoothing / C

        # 2) log-probabilities
        log_p = F.log_softmax(logits, dim=1)             # (N, C)
        pt = torch.exp((log_p * true_dist).sum(dim=1))   # (N,)

        # 3) focal term
        focal_term = (1 - pt) ** self.gamma              # (N,)

        # 4) cross-entropy part
        ce = -(true_dist * log_p).sum(dim=1)             # (N,)

        # 5) alpha weighting
        if self.alpha is not None:
            # alpha can be tensor of length C or scalar
            if isinstance(self.alpha, torch.Tensor):
                at = self.alpha[target]                  # (N,)
            else:
                at = self.alpha
            ce = at * ce

        loss = focal_term * ce                           # (N,)

        # reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss