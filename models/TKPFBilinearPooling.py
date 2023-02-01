import tensorly as tl
import torch
import torch.nn.functional as F
from torch import nn


class TKPFBilinearPooling(nn.Module):
    """
    Compute efficient bilinear pooling over input feature map using Two-Level Kronecker Product Facotrization.

    Args:
        input_dim: input dimension or channels of feature map.
        a: dimension of A matrix, default = 64
        b: dimension of B matrix, default = 64
        r: reduction factor, default = 16
        q: number of duplicated networks, default = 2
        cuda: cuda-enabled.


    Shape:
        - Input: (batch_size,channels, width, height)
        - Output: (batch_size,ab)
    References:
        Tan Yu et al. "Efficient Compact Bilinear Pooling via Kronecker Product" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).

    """

    def __init__(self, input_dim, a=64, b=64, r=16, q=2):
        super(TKPFBilinearPooling, self).__init__()

        self.a = a
        self.b = b
        self.r = r
        self.q = q
        self.d = input_dim
        tl.set_backend('pytorch')

        # Create A matrix and B matrix parameters
        self.A_hat_list = nn.ParameterList()
        self.B_hat_list = nn.ParameterList()
        for i in range(self.q):
            A_hat = torch.zeros(int(self.a / self.r), int(self.d / self.r))
            B_hat = torch.zeros(int(self.b / self.r), int(self.d / self.r))
            A_hat = nn.Parameter(A_hat, requires_grad=True)
            B_hat = nn.Parameter(B_hat, requires_grad=True)
            nn.init.xavier_normal_(A_hat.data)
            nn.init.xavier_normal_(B_hat.data)
            self.A_hat_list.append(A_hat)
            self.B_hat_list.append(B_hat)


    def forward(self, x):
        batch, _, height, width = x.size()
        N = width * height
        b_hats = []

        # Reshape
        x = x.permute(0, 2, 3, 1).view(-1, N, self.d)

        for i in range(self.q):
            A_hat = self.A_hat_list[i]
            B_hat = self.B_hat_list[i]
            # First level Kronecker product factorization
            X_a = tl.fold(x, mode=0, shape=(batch, N, int(self.d / self.r), self.r))
            X_b = tl.fold(x, mode=0, shape=(batch, N, self.r, int(self.d / self.r)))
            # print(f"X_a {X_a.shape}, X_b {X_b.shape}")

            # mode-2 X x2 A_hat
            T = tl.tenalg.mode_dot(X_a, A_hat, 2)
            T = tl.unfold(T, 0).view(-1, self.a, N)
            # print(f"T {T.shape}")

            # mode-3 product X x3 B_hat
            S = tl.tenalg.mode_dot(X_b, B_hat, 3)
            S = tl.unfold(S, 0).view(-1, N, self.b)
            # print(f"S {S.shape}")

            # Second level Kronecker product factorization
            b_hat = torch.matmul(T, S)
            b_hat = b_hat.view(-1, self.a * self.b)

            # Element-wise signed square root and L2 normalisation on b_hat
            b_hat = b_hat.sign().mul(torch.sqrt(b_hat.abs() + 1e-5))
            b_hat = F.normalize(b_hat)
            b_hats.append(b_hat)

        b_hat = torch.stack(b_hats, dim=0).sum(dim=0)


        return b_hat
