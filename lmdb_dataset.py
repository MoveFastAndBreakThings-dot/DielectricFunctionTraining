"""
LMDB Dataset for CGCNN - KKR dielectric function ε(iξ)
"""

import pickle
import lmdb
import torch
from torch.utils.data import Dataset
import numpy as np


class LMDBCrystalDataset(Dataset):
    """
    Dataset that loads pre-computed crystal graphs and KKR data from LMDB.
    """
    
    def __init__(self, lmdb_path, target_components=('xx',), use_kkr=True, metal_gap_threshold=None):
        """
        Parameters
        ----------
        lmdb_path : str
            Path to LMDB database file
        target_components : tuple
            Which dielectric tensor components to use as targets.
            Options: 'xx', 'yy', 'zz', 'xy', 'yz', 'zx'
        use_kkr : bool
            Use KKR targets (default True)
        """
        self.lmdb_path = lmdb_path
        self.target_components = target_components
        self.use_kkr = use_kkr
        
        # Open LMDB in read-only mode (subdir=False since it's a single file)
        self.env = lmdb.open(
            lmdb_path, 
            readonly=True, 
            lock=False,
            subdir=False,
            readahead=False,
            meminit=False
        )
        
        # Load metadata
        with self.env.begin() as txn:
            metadata = pickle.loads(txn.get(b'__metadata__'))
        
        self.total_samples = metadata['total_samples']
        self.kkr_xi_array = metadata.get('kkr_xi_array')
        self.energy_array = metadata.get('energy_array')
        self.radius = metadata['radius']
        self.max_num_nbr = metadata['max_num_nbr']
        self.atom_fea_len = metadata['atom_fea_len']
        self.nbr_fea_len = metadata['nbr_fea_len']
        
        # Set frequency array based on mode
        if use_kkr:
            self.n_frequencies = metadata.get('n_xi_points', 0)
            self.frequency_array = self.kkr_xi_array
        else:
            self.n_frequencies = metadata.get('n_frequencies', 0)
            self.frequency_array = self.energy_array
        
        self.metal_gap_threshold = metal_gap_threshold
        self._txn = self.env.begin()
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        key = f'{idx}'.encode('utf-8')
        data = pickle.loads(self._txn.get(key))
        
        atom_fea = data['atom_fea']
        nbr_fea = data['nbr_fea']
        nbr_fea_idx = data['nbr_fea_idx']
        
        if self.use_kkr:
            # KKR targets: ε(iξ) for each component
            target_parts = []
            for comp in self.target_components:
                target_parts.append(data['kkr_targets'][f'kkr_{comp}'])
            target = np.stack(target_parts, axis=0)
        else:
            # Original targets: real and imag parts
            real_parts = [data['targets'][f'real_{comp}'] for comp in self.target_components]
            imag_parts = [data['targets'][f'imag_{comp}'] for comp in self.target_components]
            if len(self.target_components) == 1:
                target = np.stack([real_parts[0], imag_parts[0]], axis=0)
            else:
                target = np.concatenate([np.stack(real_parts), np.stack(imag_parts)], axis=0)
        
        is_metal = False
        if self.metal_gap_threshold is not None:
            is_metal = float(data.get('opt_gap', 1.0)) < self.metal_gap_threshold

        return (atom_fea, nbr_fea, nbr_fea_idx), target, is_metal
    
    def get_sample_info(self, idx):
        """Get metadata for a specific sample."""
        key = f'{idx}'.encode('utf-8')
        data = pickle.loads(self._txn.get(key))
        return {
            'mpid': data['mpid'],
            'jid': data['jid'],
            'formula': data['formula'],
            'n_atoms': data['n_atoms'],
            'opt_gap': data['opt_gap'],
            'mbj_gap': data['mbj_gap'],
        }
    
    def get_output_channels(self):
        """Get number of output channels."""
        if self.use_kkr:
            return len(self.target_components)
        else:
            return 2 * len(self.target_components)
    
    def close(self):
        """Close LMDB environment."""
        self._txn.abort()
        self.env.close()


def collate_pool(dataset_list):
    """Collate function for batching crystal graph data."""
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    
    batch_is_metal = []
    base_idx = 0
    for (atom_fea, nbr_fea, nbr_fea_idx), target, is_metal in dataset_list:
        n_atoms = atom_fea.shape[0]
        batch_atom_fea.append(torch.FloatTensor(atom_fea))
        batch_nbr_fea.append(torch.FloatTensor(nbr_fea))
        batch_nbr_fea_idx.append(torch.LongTensor(nbr_fea_idx) + base_idx)
        crystal_atom_idx.append(torch.LongTensor(list(range(base_idx, base_idx + n_atoms))))
        batch_target.append(torch.FloatTensor(target))
        batch_is_metal.append(is_metal)
        base_idx += n_atoms

    is_metal_tensor = torch.tensor(batch_is_metal, dtype=torch.bool) if any(isinstance(m, bool) for m in batch_is_metal) else None
    return (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_fea_idx, dim=0),
        crystal_atom_idx
    ), torch.stack(batch_target, dim=0), is_metal_tensor
