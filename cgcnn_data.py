"""
Dataset class for CGCNN with Materials Project API support
Loads CIF files from Materials Project API with local caching
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.core import Element
from mp_api.client import MPRester
import json
import os
from pathlib import Path
import hashlib


class RemoteCIFLoader:
    """Handles loading CIF files from Materials Project API with caching."""
    
    def __init__(self, cache_dir=".cif_cache", use_cache=True):
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        if use_cache:
            self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, mpid):
        cache_key = hashlib.md5(mpid.encode()).hexdigest()
        return self.cache_dir / f"{cache_key}.cif"
    
    def load_from_materials_project(self, mpid, api_key):
        cache_path = self._get_cache_path(mpid) if self.use_cache else None
        
        if cache_path and cache_path.exists():
            with open(cache_path, 'r') as f:
                return f.read()
        
        with MPRester(api_key) as mpr:
            structure = mpr.get_structure_by_material_id(mpid)
            cif_string = structure.to(fmt="cif")
        
        if cache_path:
            with open(cache_path, 'w') as f:
                f.write(cif_string)
        
        return cif_string


class GaussianDistance:
    """Expands the distance by Gaussian basis."""
    
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        self.var = var if var is not None else step

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)


class RemoteCIFData(Dataset):
    """
    Dataset class that loads CIF files from Materials Project API.
    Works with JSON format (e.g., ENdiele20.json).
    
    - Uses 'cif' field if present, otherwise fetches from MP API using 'mpid'
    - Always extracts opt_realxx and opt_imagxx as targets for frequency-dependent properties
    """
    
    def __init__(self, json_file, atom_init, max_num_nbr=12, 
                 radius=8, dmin=0, step=0.2, cache_dir=".cif_cache", use_cache=True,
                 api_key=None):
        """
        Parameters
        ----------
        json_file: str
            Path to JSON file (e.g., ENdiele20.json)
        atom_init: str
            Path to atom_init.json file
        max_num_nbr: int
            Maximum number of neighbors
        radius: float
            Cutoff radius for neighbor search
        dmin: float
            Minimum distance for GaussianDistance
        step: float
            Step size for GaussianDistance
        cache_dir: str
            Directory for caching downloaded CIF files
        use_cache: bool
            Whether to cache downloaded files
        api_key: str, optional
            Materials Project API key (or set MP_API_KEY env var)
        """
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.dmin = dmin
        self.step = step
        
        # Load data
        with open(json_file, 'r') as f:
            self.data_list = json.load(f)
        
        # Setup remote loader
        api_key = api_key or os.getenv('MP_API_KEY')
        if not api_key:
            raise ValueError("Materials Project API key required. Set MP_API_KEY env var or pass api_key parameter.")
        self.remote_loader = RemoteCIFLoader(cache_dir=cache_dir, use_cache=use_cache)
        self.api_key = api_key
        
        # Load atom_init and convert from Z->features format
        with open(atom_init, 'r') as f:
            atom_init_data = json.load(f)
        self.atom_init = {Element.from_Z(int(k)).symbol: v for k, v in atom_init_data.items()}
        
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
        
        # Extract energy array from dielectric data
        if len(self.data_list) > 0 and 'opt_en' in self.data_list[0]:
            self.energy_array = np.array(self.data_list[0]['opt_en'])
    
    def __len__(self):
        return len(self.data_list)
    
    def _get_cif_string(self, sample):
        """Get CIF string from sample - use embedded CIF or fetch from MP API"""
        cif_string = sample.get('cif', '')
        if cif_string:
            return cif_string
        
        mpid = sample.get('mpid', '')
        if not mpid:
            raise ValueError(f"Sample missing both 'cif' and 'mpid' fields")
        return self.remote_loader.load_from_materials_project(mpid, self.api_key)
    
    @staticmethod
    def get_atom_fea(structure, atom_init):
        atom_fea = []
        for site in structure:
            element = str(site.specie)
            atom_fea.append(atom_init[element])
        return np.array(atom_fea, dtype=np.float32)
    
    def get_nbr_fea(self, structure):
        all_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append([i[2] for i in nbr] +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append([i[1] for i in nbr] +
                               [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append([i[2] for i in nbr[:self.max_num_nbr]])
                nbr_fea.append([i[1] for i in nbr[:self.max_num_nbr]])
        
        nbr_fea_idx = np.array(nbr_fea_idx, dtype=np.int64)
        nbr_fea = np.array(nbr_fea, dtype=np.float32)
        nbr_fea = self.gdf.expand(nbr_fea)
        
        return nbr_fea_idx, nbr_fea
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        cif_string = self._get_cif_string(sample)
        parser = CifParser.from_str(cif_string)
        structure = parser.parse_structures(primitive=True)[0]
        
        atom_fea = self.get_atom_fea(structure, self.atom_init)
        nbr_fea_idx, nbr_fea = self.get_nbr_fea(structure)
        
        # Always extract dielectric targets
        real_part = np.array(sample['opt_realxx'], dtype=np.float32)
        imag_part = np.array(sample['opt_imagxx'], dtype=np.float32)
        target = np.stack([real_part, imag_part], axis=0)
        return (atom_fea, nbr_fea, nbr_fea_idx), target, idx


def collate_pool(dataset_list):
    """
    Collate function for batching crystal graph data.
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    
    base_idx = 0
    for (atom_fea, nbr_fea, nbr_fea_idx), target, _ in dataset_list:
        n_atoms = atom_fea.shape[0]
        batch_atom_fea.append(torch.FloatTensor(atom_fea))
        batch_nbr_fea.append(torch.FloatTensor(nbr_fea))
        batch_nbr_fea_idx.append(torch.LongTensor(nbr_fea_idx) + base_idx)
        new_idx = torch.LongTensor(list(range(base_idx, base_idx + n_atoms)))
        crystal_atom_idx.append(new_idx)
        batch_target.append(torch.FloatTensor(target))
        base_idx += n_atoms
    
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), torch.stack(batch_target, dim=0), None