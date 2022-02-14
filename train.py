from argparse import ArgumentParser

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from utils import root_dir, load_yaml_param_settings
from preprocessing.build_datapipeline import build_datapipeline
from models import models
from pl_modules.train_plm import TrainPLM
from pl_modules.loggers import loggers


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=root_dir.joinpath('configs', 'config.yaml'))
    return parser.parse_args()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # data pipeline
    train_dataloader, test_dataloader = build_datapipeline(config)

    # build model
    model_name = config['model']['name']
    model = models[model_name](**config['model'][model_name])

    # fit
    logger_ = loggers[model_name]()
    train_plm = TrainPLM(model, config, len(train_dataloader.dataset), logger_)
    wandb_logger = WandbLogger(project='LHFAE_toy', name=None, config=config)
    trainer = pl.Trainer(logger=wandb_logger,
                         checkpoint_callback=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         **config['trainer_params'])
    trainer.fit(train_plm, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    wandb.finish()
