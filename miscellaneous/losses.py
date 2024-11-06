import torch


class TikMSE(torch.nn.Module):
    def __init__(self, lmbda):
        super(TikMSE, self).__init__()

        self.lmbda = lmbda

    def forward(self, x_pred, x_true, z):
        return torch.sum(torch.square(x_pred - x_true)) + (
            self.lmbda * torch.sum(torch.square(z))
        )
