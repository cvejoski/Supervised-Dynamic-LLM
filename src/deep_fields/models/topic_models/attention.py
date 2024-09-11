import torch
import torch.nn.functional as F
from torch import nn


class AttentionLayer(nn.Module):
    """
    Attention mechanism from
    "Hierarchical Attention Networks for Document Classification" (Z Yang et al)
    """

    def __init__(self, in_dim, attention_dim, delta: float):
        super(AttentionLayer, self).__init__()
        self.m_context = nn.Parameter(torch.Tensor(attention_dim, in_dim))
        self.b_context = nn.Parameter(torch.Tensor(attention_dim))
        self.delta = delta
        self.param_init()

    def forward(self, x, v, t):
        """_summary_

        Args:
            x (tensor[float]): input [B, L, H]
            v (tensor[float]): topic embeddings [T, D]
            t (tensor[float]): topic proportion [B, T]

        Returns:
            _type_: _description_
        """

        u = torch.tensordot(x, self.m_context, dims=[[2], [1]]) + self.b_context
        u = torch.tanh(u)  # [B, L, D]
        if v.dim() == 2:
            u = torch.tensordot(u, v, dims=[[2], [-1]])  # [B, L, T]
        else:
            u = torch.bmm(u, v.transpose(2, 1))
        a = F.softmax(u, dim=1)
        theta = (t - self.delta).unsqueeze(1)
        alpha = torch.bmm(a, theta.transpose(2, 1))  # [B, L, 1]

        out = torch.bmm(alpha.transpose(2, 1), x).squeeze()  # [B, H]
        return out, alpha.squeeze()

    def param_init(self):
        """
        Parameters initialization.
        """
        torch.nn.init.normal_(self.m_context)
        torch.nn.init.zeros_(self.b_context)