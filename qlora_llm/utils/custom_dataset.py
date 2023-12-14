# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Iterable, List
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class FineTuneDataset(Dataset):
    def __init__(self, data_sources: Iterable[str], max_seq_len: int = 2048) -> None:
        """
        Args:
            data_sources: a list of string path to where to load the dataset.
            max_seq_len: prompt_tokens + completion_tokens length greater than this will be discarded.
        """

        assert len(data_sources) > 0
        assert max_seq_len > 128

        self.data_sources = data_sources
        self.max_seq_len = max_seq_len

        self.data = []

        seq_length_stats = []  # track statistics

        # Load datasets
        for source in data_sources:
            samples = pickle.load(open(source, 'rb'))
            for sample in samples:
                x, y = sample['prompt_tokens'], sample['completion_tokens']
                seq_length = len(x) + len(y)
                if seq_length <= self.max_seq_len:
                    self.data.append((x, y))
                    seq_length_stats.append(seq_length)

        self.total_num_tokens = sum(seq_length_stats)
        self.seq_length_stats = {
            'min': int(np.min(seq_length_stats)),
            'max': int(np.max(seq_length_stats)),
            'mean': int(np.mean(seq_length_stats)),
            'std': int(np.std(seq_length_stats)),
        }

        self.shuffle()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def shuffle(self):
        random.shuffle(self.data)

    def get_metadata(self):
        return {
            'num_samples': len(self),
            'num_tokens': self.total_num_tokens,
            'sequence_length_stats': self.seq_length_stats,
            'data_sources': self.data_sources,
        }
