import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from pl_modules.loggers.base_logger import BaseLogger
from utils import *


class LHFAELogger(BaseLogger):
    def __init__(self):
        super().__init__()

    def log(self,
            kind: str,
            outs: Tensor,
            current_epoch: int,
            config: dict,
            track_vars: dict = None,
            ) -> None:
        assert kind in ['train', 'valid'], 'Type correct `kind`.'

        params = config['exp_params']

        mean_outs = {}
        for k in outs[0].keys():
            mean_outs.setdefault(k, 0.)
            for i in range(len(outs)):
                mean_outs[k] += outs[i][k]
            mean_outs[k] /= len(outs)

        # log numerical status
        log_items = {'epoch': current_epoch,
                     f'{kind}/loss': mean_outs['loss'],
                     f'{kind}/loss_l': mean_outs['loss_l'],
                     f'{kind}/loss_h': mean_outs['loss_h'],
                     f'{kind}/var_loss': mean_outs['var_loss'],
                     f'{kind}/cov_loss': mean_outs['cov_loss'],
                     f'{kind}/sloped_diff_scores_loss': mean_outs['sloped_diff_scores_loss'],
                     f'{kind}/z_sigma_loss': mean_outs['z_sigma_loss'],
                     }
        wandb.log(log_items)

        # log recons img
        log_recons_term = 10  # [epoch]
        if (kind == 'valid') and ((current_epoch+1) % log_recons_term == 0):
            x, recons_l, recons_h = track_vars['x'].numpy(), track_vars['recons_l'].numpy(), track_vars['recons_h'].numpy()
            recons_l_min, recons_l_max = track_vars['recons_l_min'].numpy(), track_vars['recons_l_max'].numpy()
            channel_idx = 0
            sample_idx = 0
            alpha = 0.7

            plt.figure(figsize=(12, 2*3))
            x_ = x[sample_idx, channel_idx, :]
            recons_l_ = recons_l[sample_idx, channel_idx, :]
            recons_l_min_ = recons_l_min[sample_idx, channel_idx,:]
            recons_l_max_ = recons_l_max[sample_idx, channel_idx, :]
            recons_h_ = recons_h[sample_idx, channel_idx, :]
            ymin, ymax = np.min(x_), np.max(x_)
            eps = 0.2

            plt.subplot(3, 1, 1)
            plt.plot(x_, label='GT')
            plt.plot(recons_l_, label='LF_recons', alpha=alpha)
            plt.plot(recons_l_min_, color='grey', linestyle='--')
            plt.plot(recons_l_max_, color='grey', linestyle='--')
            plt.ylim(ymin - eps, ymax + eps)
            plt.legend()

            plt.subplot(3, 1, 2)
            plt.plot(x_ - recons_l_, label='GT - LF_recons', alpha=alpha)
            plt.plot(recons_h_, label='HF recons')
            plt.ylim(ymin - eps, ymax + eps)
            plt.legend()
            plt.grid()

            plt.subplot(3, 1, 3)
            plt.plot((x_ - recons_l_) - recons_h_, label='(GT - LF_recons) - HF_recons', alpha=alpha)
            plt.ylim(ymin - eps, ymax + eps)
            plt.legend()
            plt.grid()

            plt.suptitle(f'epoch: {current_epoch}')
            plt.tight_layout()

            wandb.log({'recons': wandb.Image(plt)})
            plt.close()

