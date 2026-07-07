"""
Build official MACE AtomicData graphs from LMDB geometry fields.

Requires ``mace-torch`` (not installed in-repo — install in your training env).
"""

import json

import numpy as np
import torch

try:
    from mace.data.neighborhood import get_neighborhood
    from mace.tools import AtomicNumberTable, atomic_numbers_to_indices, to_one_hot
    from mace.tools.torch_geometric.batch import Batch
    from mace.tools.torch_geometric.data import Data

    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    Batch = None
    Data = None


def require_mace():
    if not MACE_AVAILABLE:
        raise ImportError(
            "Official MACE requires mace-torch. Install in your environment:\n"
            "  pip install mace-torch"
        )


def z_table_from_atom_init(atom_init_path="atom_init.json"):
    """Element table covering all keys in atom_init.json (atomic numbers 1–100)."""
    require_mace()
    with open(atom_init_path) as f:
        data = json.load(f)
    zs = sorted(int(k) for k in data.keys())
    return AtomicNumberTable(zs=zs)


def structure_arrays_to_atomic_data(
    positions,
    cell,
    atomic_numbers,
    pbc,
    z_table,
    r_max,
    dtype=torch.float32,
):
    """One crystal → MACE AtomicData dict (unbatched)."""
    require_mace()
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    pbc = tuple(bool(x) for x in np.asarray(pbc).tolist())

    edge_index, shifts, unit_shifts, cell_out = get_neighborhood(
        positions=positions,
        cutoff=float(r_max),
        pbc=pbc,
        cell=cell,
    )
    indices = atomic_numbers_to_indices(atomic_numbers.tolist(), z_table=z_table)
    node_attrs = to_one_hot(
        torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(z_table),
    )

    return Data(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        positions=torch.tensor(positions, dtype=dtype),
        shifts=torch.tensor(shifts, dtype=dtype),
        unit_shifts=torch.tensor(unit_shifts, dtype=dtype),
        cell=torch.tensor(cell_out, dtype=dtype),
        node_attrs=node_attrs.to(dtype),
        head=torch.zeros((), dtype=torch.long),
    )


def collate_mace_pool(dataset_list):
    """Collate LMDB samples into a MACE Batch + targets."""
    require_mace()
    graphs = [item[0] for item in dataset_list]
    batch_target = [torch.as_tensor(item[1], dtype=torch.float32) for item in dataset_list]
    batch_is_metal = [item[2] for item in dataset_list]
    batch = Batch.from_data_list(graphs)
    is_metal_tensor = (
        torch.tensor(batch_is_metal, dtype=torch.bool)
        if any(isinstance(m, bool) for m in batch_is_metal)
        else None
    )
    return batch, torch.stack(batch_target, dim=0), is_metal_tensor


def collate_mace_hamaker_pairs(batch):
    """Collate crystal pairs for meta-training with official MACE."""
    require_mace()
    batch_a = [item[0][0] for item in batch]
    batch_b = [item[0][1] for item in batch]
    a_true = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    inputs_a = Batch.from_data_list(batch_a)
    inputs_b = Batch.from_data_list(batch_b)
    return inputs_a, inputs_b, a_true
