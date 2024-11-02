import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yhat = torch.sigmoid(self.linear(x))
        return yhat
