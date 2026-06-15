"""
Dataset and collation for material-pair Hamaker meta-training.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from lmdb_dataset import collate_pool


class HamakerPairDataset(Dataset):
    """Each item is a crystal pair (graph_a, graph_b) and target A_12 in zJ."""

    def __init__(self, crystal_dataset, pairs_npz_path, pair_indices):
        data = np.load(pairs_npz_path)
        self.crystal = crystal_dataset
        pair_indices = np.asarray(pair_indices, dtype=np.int64)
        self.idx_a = data["idx_a"][pair_indices]
        self.idx_b = data["idx_b"][pair_indices]
        self.a_true_zj = data["a_true_zj"][pair_indices]

    def __len__(self):
        return len(self.idx_a)

    def __getitem__(self, i):
        ia = int(self.idx_a[i])
        ib = int(self.idx_b[i])
        inputs_a, _, _ = self.crystal[ia]
        inputs_b, _, _ = self.crystal[ib]
        return inputs_a, inputs_b, float(self.a_true_zj[i])


def collate_hamaker_pairs(batch):
    """Collate P pairs into one batched forward pass (2P crystals)."""
    batch_a = [item[0] for item in batch]
    batch_b = [item[1] for item in batch]
    a_true = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    inputs_a, _, _ = collate_pool(batch_a)
    inputs_b, _, _ = collate_pool(batch_b)
    return inputs_a, inputs_b, a_true


def load_pair_splits(pairs_dir):
    path = os.path.join(pairs_dir, "splits.json")
    with open(path) as f:
        return json.load(f)
