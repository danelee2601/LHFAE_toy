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

        self.linear_sigma_l = nn.Sequential(nn.Linear(embL_l, embL_l*3),
                                            nn.GELU(),
                                            nn.Linear(embL_l*3, embL_l))
        self.linear_sigma_h = nn.Sequential(nn.Linear(embL_h, embL_h * 3),
                                            nn.GELU(),
                                            nn.Linear(embL_h * 3, embL_h))

        # self.enc_l_output = nn.Linear(hid_dims_l[-1] * embL_l, latent_dim_l)
        # self.encs_h_output = nn.ModuleList([nn.Linear(hid_dims_h[-1] * embL_h, latent_dim_h * 2) for _ in range(n_enc_h)])

        # self.dec_l_input = nn.Sequential(nn.Linear(latent_dim_l, hid_dims_l[-1] * embL_l), Reshape((hid_dims_l[-1], embL_l)))
        # self.decs_h_input = nn.ModuleList(
        #     [nn.Sequential(nn.Linear(latent_dim_h, hid_dims_h[-1] * embL_h), Reshape((hid_dims_h[-1], embL_h)))
        #      for _ in range(n_enc_h)])

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
            #modules.append(nn.Flatten(start_dim=1))
            modules.append(nn.Conv1d(in_channels, 2, kernel_size=1, stride=1, ))
        return nn.Sequential(*modules)

    def build_dec(self, rev_hid_dims):
        modules = nn.ModuleList()

        modules.append(nn.ConvTranspose1d(1, rev_hid_dims[0], kernel_size=1, stride=1, ))

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

    def forward(self, x: Tensor, sigma_weight: float = 1., n_samples: int = 5) -> Tensor:
        # z_l = self.enc_l_output(self.enc_l(x))
        # z_l_mu, z_l_sigma = z_l[:, self.latent_dim_l:], z_l[:, :self.latent_dim_l]
        enc_l_out = self.enc_l(x) #.squeeze()
        z_l_mu, z_l_sigma = enc_l_out[:, 0, :], enc_l_out[:, 1, :]
        # z_l_sigma = self.linear_sigma_l(z_l_mu)
        # z_l_sigma = torch.log(1. + torch.exp(z_l_sigma))
        z_l_sigma = torch.abs(z_l_sigma)

        # avg_z_l = 0.
        recons_l_min = 0.
        recons_l_max = 0.
        recons_l_avg = 0.
        for i in range(n_samples):
            z_l = z_l_mu + sigma_weight * z_l_sigma * torch.randn(z_l_sigma.shape).to(z_l_mu.device) if self.training else z_l_mu
            recons_l = self.dec_l(z_l[:, None, :])

            # avg_z_l = avg_z_l + z_l
            recons_l_avg = recons_l_avg + recons_l
            if i == 0:
                recons_l_min = recons_l.clone()
                recons_l_max = recons_l.clone()
            else:
                recons_l_min = torch.min(recons_l_min, recons_l)
                recons_l_max = torch.max(recons_l_max, recons_l)
        # avg_z_l /= n_samples
        recons_l_avg /= n_samples
        recons_l = recons_l_avg

        diff = (x - recons_l).clone().detach()  # so that minimizing(diff) only affects high-freq-models.

        z_hs = []
        recons_h = torch.zeros(*x.shape).float().to(x.device)
        diff_scores = torch.zeros(self.n_enc_h, ).float().to(x.device)
        for i in range(self.n_enc_h):
            # z_h_ = self.encs_h_output[i](self.encs_h[i](diff))
            # z_h_ = self.encs_h[i](diff).squeeze()
            # z_h_mu, z_h_sigma = z_h_[:, self.latent_dim_h:], z_h_[:, :self.latent_dim_h]
            enc_h_out = self.encs_h[i](diff).squeeze()
            z_h_mu, z_h_sigma = enc_h_out[:, 0, :], enc_h_out[:, 1, :]
            # z_h_sigma = self.linear_sigma_h(z_h_mu)
            # z_h_sigma = torch.log(1. + torch.exp(z_h_sigma))
            # z_h_ = z_h_mu + sigma_weight * z_h_sigma * torch.randn(z_h_sigma.shape).to(z_h_mu.device) \
            #     if self.training else z_h_mu
            z_h_ = z_h_mu

            # recons_h_ = self.decs_h[i](self.decs_h_input[i](z_h_))
            recons_h_ = self.decs_h[i](z_h_[:, None, :])
            diff = diff - recons_h_

            z_hs.append(z_h_)
            recons_h += recons_h_
            diff_scores[i] = torch.sum(diff ** 2).mean()
        diff_scores = F.normalize(diff_scores.view(1, -1), p=1)  # `p=1` is used to make sum 1.
        diff_scores = torch.flatten(diff_scores, 0, 1)

        if self.n_enc_h == 1:
            z_h = z_hs[0]
        else:
            z_h = torch.zeros(*z_h_.shape).float().to(x.device)
            for i, p in enumerate(diff_scores):
                z_h += p * z_hs[i]

        return recons_l, recons_l_min, recons_l_max, z_l_mu, z_l_sigma, recons_h

    def loss_function(self,
                      x: Tensor,
                      recons_l,
                      recons_l_min,
                      recons_l_max,
                      z_l_mu,
                      z_l_sigma,
                      recons_h: Tensor,
                      config: dict) -> Tensor:
        params = config['model']['LHFAE']

        # recons loss
        # loss_l = F.l1_loss(input=recons_l, target=x)
        # loss_h = F.l1_loss(input=recons_h, target=x - recons_l.detach())
        # loss_l = (recons_l_max - x).abs().mean() + (x - recons_l_min).abs().mean()
        loss_l = torch.relu(x - recons_l_max).mean() + torch.relu(recons_l_min - x).mean()
        x_ = x - recons_l.detach()
        loss_h = torch.relu(recons_h - x_).mean() + torch.relu(x_ - recons_h).mean()

        # cov_loss = -1.
        sloped_diff_scores_loss = -1

        # sigma loss
        # tau = 0.0
        # z_sigma_loss = torch.abs(tau - torch.mean(z_l_sigma))
        # z_sigma_loss = (0.1 - torch.topk(z_l_sigma, k=1, dim=-1).values.mean()) ** 2
        z_sigma_loss = -1 #(0.1 - torch.mean(z_l_sigma)) ** 2

        # var loss
        var_loss = torch.mean(torch.relu(1. - torch.sqrt(z_l_mu.var(dim=0) + 1e-4)))

        # cov loss
        norm_z_l_mu = (z_l_mu - z_l_mu.mean(dim=0))
        norm_z_l_mu = F.normalize(norm_z_l_mu, p=2, dim=0)  # (B x D); l2-norm
        corr_mat_z_l_mu = torch.mm(norm_z_l_mu.T, norm_z_l_mu)  # (D x D)
        ind = np.diag_indices(corr_mat_z_l_mu.shape[0])
        corr_mat_z_l_mu[ind[0], ind[1]] = torch.zeros(corr_mat_z_l_mu.shape[0]).to(x.device)
        cov_loss = (corr_mat_z_l_mu ** 2).mean()

        loss = params['lambda_'] * loss_l \
               + params['mu'] * loss_h \
               + params['nu'] * cov_loss \
               + params['xi'] * sloped_diff_scores_loss \
               + 0.01 * z_sigma_loss \
               + 1. * var_loss \
               + 1. * cov_loss

        return {'loss': loss,
                'loss_l': loss_l,
                'loss_h': loss_h,
                'sloped_diff_scores_loss': sloped_diff_scores_loss,
                'z_sigma_loss': z_sigma_loss,
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
    encoder = LHFAE(L, arbitrary_embL, arbitrary_embL).build_enc(hid_dims, flatten=True)
    last_actmap = encoder(x)
    print('last activation map.shape:', last_actmap.shape)
    print('embL:', last_actmap.shape[-1])

    # model
    model = LHFAE(L, embL_l=38, embL_h=75, )

    # forward
    recons_l, recons_l_min, recons_l_max, z_l_sigma, recons_h = model(x)
    print('recons_l_min:', recons_l_min)
    print('recons_l_max:', recons_l_max)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 3))
    plt.plot(recons_l[0, 0, :].detach().numpy())
    plt.plot(recons_l_min[0, 0, :].detach().numpy(), alpha=0.7)
    plt.plot(recons_l_max[0, 0, :].detach().numpy(), alpha=0.7)
    plt.show()
