import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import Tensor


def compute_cov_loss(z1: Tensor, z2: Tensor) -> Tensor:
    z1 = (z1 - z1.mean(dim=0))
    z2 = (z2 - z2.mean(dim=0))

    z1 = F.normalize(z1, p=2, dim=0)  # (B x D); l2-norm
    z2 = F.normalize(z2, p=2, dim=0)

    fxf_cov_z1 = torch.mm(z1.T, z1)  # (feature * feature)
    fxf_cov_z2 = torch.mm(z2.T, z2)

    ind = np.diag_indices(fxf_cov_z1.shape[0])
    fxf_cov_z1[ind[0], ind[1]] = torch.zeros(fxf_cov_z1.shape[0]).to(z1.device)
    fxf_cov_z2[ind[0], ind[1]] = torch.zeros(fxf_cov_z2.shape[0]).to(z1.device)
    return (fxf_cov_z1 ** 2).mean() + (fxf_cov_z2 ** 2).mean()


def compute_sloped_diff_scores(diff_scores: Tensor) -> Tensor:
    """
    :param diff_scores: (B, `n_enc_h`)
    :return:
    """
    L = 0.
    for i in range(len(diff_scores) - 1):
        y = diff_scores[i]
        dydx = diff_scores[i + 1] - diff_scores[i]
        dydx = - torch.abs(dydx)
        # if dydx > 0:
        #     y = y * torch.Tensor([10.]).float().to(y.device)[0]  # to penalize positive grad
        L += y * dydx
    return L


class Reshape(nn.Module):
    def __init__(self, target_shape: tuple):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, input: Tensor) -> Tensor:
        B = input.shape[0]
        return input.view(B, *self.target_shape)


class PosEmb(nn.Module):
    def __init__(self, seq_len: int, dim: int):
        super(PosEmb, self).__init__()
        self.pos_emb = self.sinusoidal_embedding(seq_len, dim)
        self.pos_emb = torch.transpose(self.pos_emb, 1, 2)

    @staticmethod
    def sinusoidal_embedding(seq_len, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(seq_len)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pos_emb.to(x.device)


class LHFAE(nn.Module):
    def __init__(self,
                 L: int,
                 embL_l: int,
                 embL_h: int,
                 pos_embL: int,
                 emb_depth: int,
                 in_channels: int = 1,
                 n_enc_h: int = 3,
                 hid_dims_l: tuple = (64, 128, 256),
                 hid_dims_h: tuple = (32, 64, 128),
                 *args,
                 **kwargs
                 ):
        """
        :param L: length of input timeseries
        :param embL: length in the embedding (latent) space
        :param in_channels:
        :param n_enc_h: number of high-freq encoder(s)
        :param hid_dims: hidden channel sizes
        :param latent_dim:
        """
        super().__init__()
        self.L = L
        self.embL_l = embL_l
        self.embL_h = embL_h
        self.pos_embL = pos_embL
        self.emb_depth = emb_depth
        self.in_channels = in_channels
        self.n_enc_h = n_enc_h

        self.enc_l = self.build_enc(hid_dims_l, add_pos_emb=True)
        self.encs_h = nn.ModuleList([self.build_enc(hid_dims_h, add_pos_emb=True) for _ in range(n_enc_h)])

        rev_hid_dims_l = tuple(list(hid_dims_l)[::-1])
        rev_hid_dims_h = tuple(list(hid_dims_h)[::-1])
        self.dec_l = self.build_dec(rev_hid_dims_l)
        self.decs_h = nn.ModuleList([self.build_dec(rev_hid_dims_h) for _ in range(n_enc_h)])

    def build_enc(self, hid_dims, flatten: bool = True, add_pos_emb: bool = False):
        in_channels = self.in_channels
        modules = nn.ModuleList()
        for i, h_dim in enumerate(hid_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

            if add_pos_emb and (i == 0):
                modules.append(PosEmb(seq_len=self.pos_embL, dim=h_dim))

        if flatten:
            modules.append(nn.Sequential(nn.Conv1d(in_channels, self.emb_depth, kernel_size=1, stride=1),)
                           )
        return nn.Sequential(*modules)

    def build_dec(self, rev_hid_dims):
        modules = nn.ModuleList()

        modules.append(nn.ConvTranspose1d(self.emb_depth, rev_hid_dims[0], kernel_size=1, stride=1, ))

        for i in range(len(rev_hid_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(rev_hid_dims[i],
                                       rev_hid_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm1d(rev_hid_dims[i + 1]),
                    nn.ReLU()
                )
            )

        final_layer = nn.Sequential(nn.ConvTranspose1d(rev_hid_dims[-1],
                                                       rev_hid_dims[-1],
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       output_padding=1),
                                    nn.BatchNorm1d(rev_hid_dims[-1]),
                                    nn.ReLU(),
                                    nn.Conv1d(rev_hid_dims[-1], out_channels=1, kernel_size=3, padding=1)
                                    )
        modules.append(final_layer)
        modules.append(nn.Upsample(size=(self.L,)))
        return nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        z_l = self.enc_l(x)
        recons_l = self.dec_l(z_l)

        residual = (x - recons_l) #.detach()  # so that minimizing(diff) only affects high-freq-models.
        residuals = torch.zeros((1+self.n_enc_h, *residual.shape)).float().to(x.device)
        residuals[0] = residual
        recons_hs = torch.zeros((self.n_enc_h, *x.shape)).float()
        B = x.shape[0]
        z_hs = torch.zeros(self.n_enc_h, B, self.emb_depth, self.embL_h).float().to(x.device)
        for i in range(self.n_enc_h):
            z_h = self.encs_h[i](residual.detach())
            z_hs[i] = z_h
            recons_h_ = self.decs_h[i](z_h)
            recons_hs[i] = recons_h_.cpu()
            residual = residual - recons_h_
            # if (i+1) != self.n_enc_h:
            #     residual = residual.detach()
            residuals[i+1] = residual

        return (recons_l, z_l.squeeze()), (recons_hs, z_hs), residuals

    def _compute_cov_loss(self, z: Tensor):
        """
        :param z:  (B, C, L); temporal representation with the embedding depth
        """
        norm_z = (z - z.mean(dim=0))
        norm_z = F.normalize(norm_z, p=2, dim=2)
        corr_mat_z = torch.bmm(norm_z, norm_z.transpose(1, 2))  # (B, C, L) x (B, L, C) -> (B, C, C)
        torch.diagonal(corr_mat_z, 0, 1, 2).zero_()  # inplace func
        return (corr_mat_z ** 2).mean()

    def _compute_var_loss(self, z: Tensor):
        """
        :param z:  (B, C, L); temporal representation with the embedding depth
        """
        # return torch.mean(torch.relu(1. - torch.sqrt(z_l.var(dim=0) + 1e-4)))
        return torch.mean((1. - torch.sqrt(z.var(dim=0) + 1e-4)) ** 2)  # to make the feature space "compact"

    def loss_function(self,
                      x: Tensor,
                      recons_l: Tensor,
                      z_l: Tensor,
                      z_hs: Tensor,
                      residuals: Tensor,
                      config: dict) -> Tensor:
        params = config['model']['LHFAE']

        # recons loss
        loss_l = F.l1_loss(input=recons_l, target=x)
        loss_h = torch.FloatTensor([0.]).to(x.device)
        for residual in residuals:
            loss_h += F.mse_loss(input=residual, target=torch.zeros(residual.shape).to(residual.device))

        # var loss (from vibcreg)
        var_loss_l = self._compute_var_loss(z_l)
        var_loss_h = torch.FloatTensor([0.]).to(x.device)
        for z_h in z_hs:
            var_loss_h += self._compute_var_loss(z_h)
        var_loss = var_loss_l + var_loss_h
        # var_loss = -1

        # cov loss (from vibcreg)
        if config['model']['LHFAE']['emb_depth'] > 1:
            # z_l: (B, C, L)
            # z_hs: (n_enc_h, B, C, L)
            cov_loss_l = self._compute_cov_loss(z_l)
            cov_loss_h = torch.sum(torch.Tensor([self._compute_cov_loss(z_hs[i]) for i in range(self.n_enc_h)]))
            cov_loss = params['xi_l'] * cov_loss_l + params['xi_h'] * cov_loss_h
        else:
            cov_loss = -1

        loss = params['lambda_'] * loss_l \
               + params['mu'] * loss_h \
               + params['nu'] * var_loss \
               + cov_loss

        return {'loss': loss,
                'loss_l': loss_l,
                'loss_h': loss_h,
                'var_loss': var_loss,
                'cov_loss': cov_loss,
                }


if __name__ == '__main__':
    # toy dataset
    B, C, L = 4, 1, 300
    x = torch.rand(B, C, L)
    print('input.shape:', x.shape)

    # check `embL`
    arbitrary_embL = 10
    emb_depth = 4
    hid_dims = (64, 128)
    encoder = LHFAE(L, arbitrary_embL, arbitrary_embL, emb_depth).build_enc(hid_dims, flatten=False)
    last_actmap = encoder(x)
    print('last activation map.shape:', last_actmap.shape)
    print('embL:', last_actmap.shape[-1])

    # model
    # model = LHFAE(L, embL_l=38, embL_h=75, )
