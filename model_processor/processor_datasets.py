import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch

from config.config import BaseConfig
from pdb_datasets.dataloader_utils import SafeDataLoader
from pdb_datasets.geometric_features import ProteinFeatures
from pdb_datasets.pdb_utils import PDBBackbone, Samples
from utils.utils import get_files


class DSFromFile(Dataset):
    def __init__(self, file, chain):
        assert os.path.exists(file)
        self.file = file
        self.chain = chain
        self.protein_features = ProteinFeatures()

    def __getitem__(self, index):
        samples = Samples()
        target_path = samples.extract_chains(self.file, self.chain)
        pdb_extractor = PDBBackbone(target_path)
        try:
            features = pdb_extractor.extract_features()
            data = self.protein_features.build_features(features)
            assert features is not None
            assert features[0] is not None
            assert features[1] is not None
        except AssertionError:
            return None
        finally:
            samples.clear(target_path)
        return data

    def __len__(self):
        return 1


class DSFromFolder(Dataset):
    def __init__(self, folder):
        assert os.path.exists(folder)
        self.filelist = get_files(folder, ext='*.*')
        self.protein_features = ProteinFeatures()

    def __getitem__(self, index):
        file = self.filelist[index]
        pdb_extractor = PDBBackbone(file)
        try:
            features = pdb_extractor.extract_features()
            data = self.protein_features.build_features(features)
            assert features is not None
            assert features[0] is not None
            assert features[1] is not None
        except AssertionError:
            return None
        data.file_idx = int(os.path.basename(file).replace('.pdb', ''))
        return data

    def __len__(self):
        return len(self.filelist)


def get_batch_from_file(file, chain):
    assert os.path.exists(file)
    file_ds = DSFromFile(file, chain)
    loader = SafeDataLoader(file_ds,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)
    batch = next(iter(loader))
    if type(batch) == list:
        return None
    return batch


def get_loader_from_folder(folder):
    config = BaseConfig()
    assert os.path.exists(folder)
    folder_ds = DSFromFolder(folder)
    loader = SafeDataLoader(folder_ds,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=1)
    return loader
