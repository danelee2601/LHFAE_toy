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


class LHFAE(nn.Module):
    def __init__(self,
                 L: int,
                 embL_l: int,
                 embL_h: int,
                 in_channels: int = 1,
                 n_enc_h: int = 3,
                 hid_dims_l: tuple = (64, 128, 256),
                 hid_dims_h: tuple = (32, 64, 128),
                 latent_dim_l: int = 16,
                 latent_dim_h: int = 16,
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
        self.in_channels = in_channels
        self.n_enc_h = n_enc_h
        self.latent_dim_l = latent_dim_l
        self.latent_dim_h = latent_dim_h

        self.enc_l = self.build_enc(hid_dims_l)
        self.encs_h = nn.ModuleList([self.build_enc(hid_dims_h) for _ in range(n_enc_h)])

        self.enc_l_output = nn.Linear(hid_dims_l[-1] * embL_l, latent_dim_l)
        self.encs_h_output = nn.ModuleList([nn.Linear(hid_dims_h[-1] * embL_h, latent_dim_h) for _ in range(n_enc_h)])

        self.dec_l_input = nn.Sequential(nn.Linear(latent_dim_l, hid_dims_l[-1] * embL_l), Reshape((hid_dims_l[-1], embL_l)))
        self.decs_h_input = nn.ModuleList(
            [nn.Sequential(nn.Linear(latent_dim_h, hid_dims_h[-1] * embL_h), Reshape((hid_dims_h[-1], embL_h)))
             for _ in range(n_enc_h)])

        rev_hid_dims_l = tuple(list(hid_dims_l)[::-1])
        rev_hid_dims_h = tuple(list(hid_dims_h)[::-1])
        self.dec_l = self.build_dec(rev_hid_dims_l)
        self.decs_h = nn.ModuleList([self.build_dec(rev_hid_dims_h) for _ in range(n_enc_h)])

    def build_enc(self, hid_dims, flatten: bool = True):
        in_channels = self.in_channels
        modules = nn.ModuleList()
        for h_dim in hid_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU())
            )
            in_channels = h_dim
        if flatten:
            modules.append(nn.Flatten(start_dim=1))
        return nn.Sequential(*modules)

    def build_dec(self, rev_hid_dims):
        modules = nn.ModuleList()
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
                    nn.GELU()
                )
            )

        final_layer = nn.Sequential(nn.ConvTranspose1d(rev_hid_dims[-1],
                                                       rev_hid_dims[-1],
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       output_padding=1),
                                    nn.BatchNorm1d(rev_hid_dims[-1]),
                                    nn.GELU(),
                                    nn.Conv1d(rev_hid_dims[-1], out_channels=1, kernel_size=3, padding=1)
                                    )
        modules.append(final_layer)
        modules.append(nn.Upsample(size=(self.L,)))
        return nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        z_l = self.enc_l_output(self.enc_l(x))
        recons_l = self.dec_l(self.dec_l_input(z_l))

        residual = (x - recons_l).clone().detach()  # so that minimizing(diff) only affects high-freq-models.
        recons_h = torch.zeros(*x.shape).float().to(x.device)
        for i in range(self.n_enc_h):
            z_h = self.encs_h_output[i](self.encs_h[i](residual))
            recons_h_ = self.decs_h[i](self.decs_h_input[i](z_h))
            recons_h += recons_h_
            residual = residual - recons_h_

        return (recons_l, z_l), (recons_h, ), residual

    def loss_function(self,
                      x: Tensor,
                      recons_l,
                      z_l,
                      residual: Tensor,
                      config: dict) -> Tensor:
        params = config['model']['LHFAE']

        # recons loss
        loss_l = F.l1_loss(input=recons_l, target=x)
        loss_h = F.mse_loss(input=residual, target=torch.zeros(residual.shape).to(residual.device))

        # var loss (from vibcreg)
        var_loss = torch.mean(torch.relu(1. - torch.sqrt(z_l.var(dim=0) + 1e-4)))

        # cov loss (from vibcreg)
        norm_z_l = (z_l - z_l.mean(dim=0))
        norm_z_l = F.normalize(norm_z_l, p=2, dim=0)  # (B x D); l2-norm
        corr_mat_z_l = torch.mm(norm_z_l.T, norm_z_l)  # (D x D)
        ind = np.diag_indices(corr_mat_z_l.shape[0])
        corr_mat_z_l[ind[0], ind[1]] = torch.zeros(corr_mat_z_l.shape[0]).to(x.device)
        cov_loss = (corr_mat_z_l ** 2).mean()

        loss = params['lambda_'] * loss_l \
               + params['mu'] * loss_h \
               + params['nu'] * var_loss \
               + params['xi'] * cov_loss

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
    hid_dims = (64, 128)
    encoder = LHFAE(L, arbitrary_embL, arbitrary_embL).build_enc(hid_dims, flatten=False)
    last_actmap = encoder(x)
    print('last activation map.shape:', last_actmap.shape)
    print('embL:', last_actmap.shape[-1])

    # model
    # model = LHFAE(L, embL_l=38, embL_h=75, )
