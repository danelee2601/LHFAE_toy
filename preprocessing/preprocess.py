import numpy as np
import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self,
                 kind: str,
                 configs: dict,
                 n_sines: int = 2,
                 n_anomalies: int = 1,
                 ts_len: int = 300,
                 n_samples_train: int = 3000,
                 n_samples_test: int = 100,
                 ):
        super().__init__()
        assert kind in ['train', 'valid']

        if kind == 'train':
            np.random.seed(configs['dataset']['rand_seed']['train'])
            n_samples = n_samples_train
        elif kind == 'valid':
            np.random.seed(configs['dataset']['rand_seed']['test'])
            n_samples = n_samples_test
        else:
            raise ValueError

        # sample sine frequencies, amplitudes, phases
        sine_freqs = np.random.uniform(0.01, 0.2, size=(n_samples, n_sines))
        sine_amps = np.random.uniform(0.1, 1., size=(n_samples, n_sines))
        sine_eps = np.random.uniform(0, 2*np.pi, size=(n_samples, n_sines))

        # sample sequential anomalies
        seqano_freq = 1.0
        seqano_amp = 0.2
        seqano_len = int(np.floor(ts_len * 0.05))
        vshifts = np.random.choice([-1., 1.], size=(n_samples,))
        sample_prob = 0.01 if kind == 'train' else 1.
        self.seqano_locs1 = np.random.randint(0, ts_len - seqano_len, size=(n_samples,))
        self.seqano_locs2 = np.random.randint(0, ts_len - seqano_len, size=(n_samples,))

        # sample large anomaly loc, amp
        self.lano_locs = np.random.randint(0, ts_len, size=(n_samples, n_anomalies))  # lano: large anomaly
        lano_amp = np.random.choice([-1, 1], size=(n_samples,))

        self.sano_locs = np.random.randint(0, ts_len, size=(n_samples, n_anomalies))  # sano: small anomaly
        sano_amp = np.random.choice([-0.5, 0.5], size=(n_samples,))

        # generate a timeseries dataset
        self.t = np.arange(0, ts_len)
        self.sines = np.zeros((n_samples, ts_len))
        for i in range(n_sines):
            self.sines += sine_amps[:, [i]] * np.sin(sine_freqs[:, [i]] * self.t + sine_eps[:, [i]])

        # add anomalies
        B = self.sines.shape[0]
        for j in range(n_anomalies):
            large_anomalies, small_anomalies = np.zeros(self.sines.shape), np.zeros(self.sines.shape)
            large_anomalies[range(B), self.lano_locs[:, j]] = lano_amp
            small_anomalies[range(B), self.sano_locs[:, j]] = sano_amp
            self.sines += large_anomalies
            self.sines += small_anomalies

        # add predictable high-freq noise
        for i in range(B):
            t_ = np.arange(self.seqano_locs1[i], self.seqano_locs1[i] + seqano_len)
            if (kind == 'train' and (np.random.rand() <= sample_prob)) or (kind == 'valid'):
                self.sines[i, self.seqano_locs1[i]:self.seqano_locs1[i] + seqano_len] = vshifts[i]

            t_ = np.arange(self.seqano_locs2[i], self.seqano_locs2[i] + seqano_len)
            # if (kind == 'train' and (np.random.rand() <= sample_prob)) or (kind == 'valid'):
            self.sines[i, self.seqano_locs2[i]:self.seqano_locs2[i] + seqano_len] += seqano_amp * np.sin(seqano_freq * t_)

        # add channel dim
        self.sines = self.sines[:, None, :]  # (1, ts_len)

        self._len = n_samples

    def __getitem__(self, idx):
        # fetch a sample
        x = self.sines[idx]
        lano_loc = self.lano_locs[idx]
        sano_loc = self.sano_locs[idx]
        seqano_loc = self.seqano_locs1[idx]

        # scale
        pass

        # convert to FloatTensor
        x = torch.from_numpy(x).float()

        return x, (lano_loc, sano_loc, seqano_loc)

    def __len__(self):
        return self._len


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    configs = {'dataset': {'rand_seed': {'train': 0, 'test': 1}}}
    dataset = ToyDataset('valid', configs)
    # print(dataset.sines)
    print(dataset.sines.shape)
    print(dataset.lano_locs.shape)
    print(dataset.sano_locs.shape)
    print(dataset.t.shape)

    # plot
    n_plots = 5
    channel_idx = 0
    plt.figure(figsize=(10, 1.2*n_plots))
    for i in range(n_plots):
        plt.subplot(n_plots, 1, i+1)
        plt.plot(dataset.t, dataset.sines[i, channel_idx, :])

        for j in range(dataset.lano_locs.shape[-1]):
            k = dataset.lano_locs[i, j]
            plt.scatter(dataset.t[k], dataset.sines[i, channel_idx, :][k], color='red', s=20, alpha=0.5)

        for j in range(dataset.sano_locs.shape[-1]):
            k = dataset.sano_locs[i, j]
            plt.scatter(dataset.t[k], dataset.sines[i, channel_idx, :][k], color='green', s=20, alpha=0.5)

    plt.tight_layout()
    plt.show()

