import os.path
import pickle

import numpy as np

from utils.utils import get_files
from config.config import BaseConfig
from pdb_utils import PDBBackbone, PDBFeaturesExtractor


class Preprocessor(object):
    def __init__(self):
        self.config = BaseConfig()
        self.pdb_files = get_files(self.config.get_pdb_dir())
        self.total_processed = 0
        self.logging = self.config.logging

    def make_datasets(self):
        np.random.shuffle(self.pdb_files)
        feature_size = self.config.buffer_size
        train_size = int(feature_size * self.config.split_factor) + 1
        val_size = feature_size - train_size
        assert len(self.pdb_files) > feature_size
        train_set = self.pdb_files[:train_size]
        val_set = self.pdb_files[train_size:train_size + val_size]
        if self.logging:
            print(f'Generate train set: {train_size} files')

        train_dir, val_dir = self.config.make_working_dir()
        self.processing_set(train_set, out_dir=train_dir)
        self.processing_set(val_set, out_dir=val_dir)

        if self.logging:
            print(f'Generate val set: {val_size} files')

    def processing_set(self, files, out_dir):
        n_total = len(files)
        n_processed = 0

        for file in files:
            features = self.extract_features(file)
            if not features:
                continue
            n_processed += 1
            filename = os.path.basename(file)
            if self.logging:
                print(f'Extracted {filename} {n_processed} of {n_total}')
            outfile = out_dir / filename.replace('pdb', 'pkl')
            with open(outfile, 'wb') as f:
                pickle.dump(features, f)

    @staticmethod
    def extract_features(pdb_file):
        pdb = PDBBackbone(pdb_file)
        positions, sequence, res_positions = pdb.get_pdb_features()
        if positions is None:
            return None
        extractor = PDBFeaturesExtractor(positions, sequence, res_positions)
        features = extractor.calculate()

        return features


if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.make_datasets()
