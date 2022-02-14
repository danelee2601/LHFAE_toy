from torch.utils.data import DataLoader
from preprocessing.preprocess import ToyDataset


def build_datapipeline(config: dict) -> (DataLoader, DataLoader):
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']

    train_dataset = ToyDataset('train', config)
    test_dataset = ToyDataset('valid', config)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers,
                                  pin_memory=True if num_workers > 0 else False
                                  )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=num_workers,
                                 pin_memory=True if num_workers > 0 else False
                                 )
    return train_dataloader, test_dataloader
