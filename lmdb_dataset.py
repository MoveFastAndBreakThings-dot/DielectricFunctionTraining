"""
LMDB dataset for crystal structures and KKR dielectric targets ε(iξ).

Supports two graph backends:
  - CGCNN-style (default): atom_fea, nbr_fea, nbr_fea_idx
  - Official MACE: 3D positions + MACE radius graph (requires geometry in LMDB)
"""

import pickle

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from mace_data import structure_arrays_to_atomic_data, z_table_from_atom_init


class LMDBCrystalDataset(Dataset):
    """Load pre-computed crystal graphs and KKR data from LMDB."""

    def __init__(
        self,
        lmdb_path,
        target_components=("xx",),
        use_kkr=True,
        metal_gap_threshold=None,
        use_mace=False,
        mace_r_max=None,
        atom_init_path="atom_init.json",
        z_table=None,
    ):
        self.lmdb_path = lmdb_path
        self.target_components = target_components
        self.use_kkr = use_kkr
        self.use_mace = use_mace
        self.mace_r_max = mace_r_max

        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            subdir=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin() as txn:
            metadata = pickle.loads(txn.get(b"__metadata__"))

        self.total_samples = metadata["total_samples"]
        self.kkr_xi_array = metadata.get("kkr_xi_array")
        self.energy_array = metadata.get("energy_array")
        self.radius = metadata["radius"]
        self.max_num_nbr = metadata["max_num_nbr"]
        self.atom_fea_len = metadata["atom_fea_len"]
        self.nbr_fea_len = metadata["nbr_fea_len"]

        if use_kkr:
            self.n_frequencies = metadata.get("n_xi_points", 0)
            self.frequency_array = self.kkr_xi_array
        else:
            self.n_frequencies = metadata.get("n_frequencies", 0)
            self.frequency_array = self.energy_array

        self.metal_gap_threshold = metal_gap_threshold
        self._txn = self.env.begin()

        if use_mace:
            self.z_table = z_table if z_table is not None else z_table_from_atom_init(atom_init_path)
            self.mace_r_max = float(mace_r_max or self.radius)
            self._check_mace_geometry(0)

    def set_mace_graph_config(self, z_table, mace_r_max):
        """Override element table / cutoff (e.g. after loading MACE-MP-0)."""
        self.z_table = z_table
        self.mace_r_max = float(mace_r_max)

    def _check_mace_geometry(self, idx):
        key = f"{idx}".encode("utf-8")
        data = pickle.loads(self._txn.get(key))
        if "positions" not in data:
            raise RuntimeError(
                "LMDB lacks 3D geometry (positions/cell/atomic_numbers). "
                "Re-run preprocess_dataset.py to enable official MACE."
            )

    def __len__(self):
        return self.total_samples

    def _target_from_data(self, data):
        if self.use_kkr:
            target_parts = []
            for comp in self.target_components:
                target_parts.append(data["kkr_targets"][f"kkr_{comp}"])
            return np.stack(target_parts, axis=0)
        real_parts = [data["targets"][f"real_{comp}"] for comp in self.target_components]
        imag_parts = [data["targets"][f"imag_{comp}"] for comp in self.target_components]
        if len(self.target_components) == 1:
            return np.stack([real_parts[0], imag_parts[0]], axis=0)
        return np.concatenate([np.stack(real_parts), np.stack(imag_parts)], axis=0)

    def __getitem__(self, idx):
        key = f"{idx}".encode("utf-8")
        data = pickle.loads(self._txn.get(key))
        target = self._target_from_data(data)

        is_metal = False
        if self.metal_gap_threshold is not None:
            is_metal = float(data.get("opt_gap", 1.0)) < self.metal_gap_threshold

        if self.use_mace:
            graph = structure_arrays_to_atomic_data(
                data["positions"],
                data["cell"],
                data["atomic_numbers"],
                data["pbc"],
                self.z_table,
                self.mace_r_max,
            )
            return graph, target, is_metal

        atom_fea = data["atom_fea"]
        nbr_fea = data["nbr_fea"]
        nbr_fea_idx = data["nbr_fea_idx"]
        return (atom_fea, nbr_fea, nbr_fea_idx), target, is_metal

    def get_sample_info(self, idx):
        key = f"{idx}".encode("utf-8")
        data = pickle.loads(self._txn.get(key))
        return {
            "mpid": data["mpid"],
            "jid": data["jid"],
            "formula": data["formula"],
            "n_atoms": data["n_atoms"],
            "opt_gap": data["opt_gap"],
            "mbj_gap": data["mbj_gap"],
        }

    def get_output_channels(self):
        if self.use_kkr:
            return len(self.target_components)
        return 2 * len(self.target_components)

    def close(self):
        self._txn.abort()
        self.env.close()


def collate_crystal_graphs(graph_list):
    """Collate CGCNN-style graphs."""
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx = []
    base_idx = 0
    for atom_fea, nbr_fea, nbr_fea_idx in graph_list:
        n_atoms = atom_fea.shape[0]
        batch_atom_fea.append(torch.FloatTensor(atom_fea))
        batch_nbr_fea.append(torch.FloatTensor(nbr_fea))
        batch_nbr_fea_idx.append(torch.LongTensor(nbr_fea_idx) + base_idx)
        crystal_atom_idx.append(torch.LongTensor(list(range(base_idx, base_idx + n_atoms))))
        base_idx += n_atoms
    return (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_fea_idx, dim=0),
        crystal_atom_idx,
    )


def collate_pool(dataset_list):
    """Collate CGCNN-style batches."""
    graphs = [item[0] for item in dataset_list]
    inputs = collate_crystal_graphs(graphs)
    batch_target = [torch.FloatTensor(item[1]) for item in dataset_list]
    batch_is_metal = [item[2] for item in dataset_list]
    is_metal_tensor = (
        torch.tensor(batch_is_metal, dtype=torch.bool)
        if any(isinstance(m, bool) for m in batch_is_metal)
        else None
    )
    return inputs, torch.stack(batch_target, dim=0), is_metal_tensor


def get_collate_fn(use_mace=False):
    if use_mace:
        from mace_data import collate_mace_pool
        return collate_mace_pool
    return collate_pool
