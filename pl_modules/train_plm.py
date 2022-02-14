"""
Pytorch lightning module (plm) for training
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl


class TrainPLM(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 config: dict,
                 n_train_samples: int,
                 logger_,
                 ):
        super().__init__()
        self.model = model
        self.config = config
        self.logger_ = logger_

        self.T_max = config['trainer_params']['max_epochs'] * \
                     np.ceil(n_train_samples / config['dataset']['batch_size'])  # Maximum number of iterations

        self.track_vars = {}

    def forward(self, *args):
        return self.model(*args)

    def _store_tracking_data(self, batch_idx, x, recons_l, recons_h, ):
        if batch_idx in [0, 1, 2, 3]:
            if batch_idx == 0:
                self.track_vars['x'] = x.cpu()
                self.track_vars['recons_l'] = recons_l.detach().cpu()
                self.track_vars['recons_h'] = recons_h.detach().cpu()
            else:
                self.track_vars['x'] = torch.cat((self.track_vars['x'], x.cpu()), dim=0)
                self.track_vars['recons_l'] = torch.cat((self.track_vars['recons_l'], recons_l.detach().cpu()), dim=0)
                self.track_vars['recons_h'] = torch.cat((self.track_vars['recons_h'], recons_h.detach().cpu()), dim=0)

    def _detach_the_unnecessary(self, loss_hist: dict):
        """
        apply `.detach()` on Tensors that do not need back-prop computation.
        :return:
        """
        for k in loss_hist.keys():
            if k not in ['loss']:
                try:
                    loss_hist[k] = loss_hist[k].detach()
                except AttributeError:
                    pass

    def training_step(self, batch, batch_idx):
        self.model.train()

        x, (lano_loc, sano_loc, seqano_locs) = batch
        (recons_l, z_l), (recons_h,), residual = self.model(x)
        train_loss = self.model.loss_function(x,
                                              recons_l,
                                              z_l,
                                              residual,
                                              self.config)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        self._detach_the_unnecessary(train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        x, (lano_loc, sano_loc, seqano_locs) = batch
        (recons_l, z_l), (recons_h,), residual = self.model(x)
        val_loss = self.model.loss_function(x,
                                            recons_l,
                                            z_l,
                                            residual,
                                            self.config)

        # get some data for tracking training status
        self._store_tracking_data(batch_idx, x, recons_l, recons_h)

        self._detach_the_unnecessary(val_loss)
        return val_loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.model.parameters(), 'lr': self.config['exp_params']['LR']},
                                 ],
                                weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt,
                'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}

    def training_epoch_end(self, outs) -> None:
        self.logger_.log('train', outs, self.current_epoch, self.config)
        self.logger_.save_model(self.current_epoch, self.config, self.model)

    def validation_epoch_end(self, outs) -> None:
        self.logger_.log('valid', outs, self.current_epoch, self.config, self.track_vars)
