from abc import ABC, abstractmethod
import os

import torch

from utils import root_dir


class BaseLogger(object):
    def __init__(self):
        pass

    @abstractmethod
    def log(self, *args, **kwargs):
        pass

    def save_model(self, current_epoch: int, config: dict, model):
        """
        Saves a model checkpoint.
        :param current_epoch: `self.current_epoch` in `pl.LightningModule`
        :param model
        :return:
        """
        current_epoch = current_epoch + 1  # make `epoch` starts from 1 instead of 0
        if current_epoch % config['exp_params']['model_save_ep_period'] == 0:
            if not os.path.isdir(root_dir.joinpath('checkpoints')):
                os.mkdir(root_dir.joinpath('checkpoints'))

            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
            }, root_dir.joinpath(os.path.join('checkpoints', f'checkpoint-ep_{current_epoch}.ckpt')))
