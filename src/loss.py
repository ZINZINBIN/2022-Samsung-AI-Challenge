import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction = 'mean')
    def forward(self, pred : torch.Tensor, target : torch.Tensor)->torch.Tensor:
        return torch.sqrt(self.mse(pred, target))