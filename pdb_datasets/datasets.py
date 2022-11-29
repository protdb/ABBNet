import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

from config.config import BaseConfig
from pdb_datasets.dataloader_utils import SafeDataLoader
from pdb_datasets.pdb_utils import PDBBackbone, PDBFeaturesExtractor, Samples
from pdb_datasets.geometric_features import ProteinFeatures
from sklearn.model_selection import train_test_split

from utils.utils import get_files

PAD_SEQ = 'Z'
SAML = ['A', 'Y', 'B', 'C', 'D', 'G', 'I', 'L', 'E', 'F', 'H', 'K', 'N', 'S', 'T', 'V', 'W', 'X', 'M', 'P', 'Q', 'R',
        PAD_SEQ]


def init_data():
    config = BaseConfig()
    config.make_working_dir()
    struct_data_file = config.get_train_file_path()
    pdb_dir = config.get_pdb_dir()
    with open(struct_data_file) as file:
        lines = file.readlines()
    is_append = False
    items = {}
    c_count = 0
    max_sample_size = config.get_sample_size() * 2
    for idx, line in enumerate(lines):
        if idx > max_sample_size:
            break
        if idx % 2 == 0:
            pdb_id = line.split(" ")[0]
            chain = pdb_id[-1]
            pdb_id = pdb_id[1:-1]
            filename = pdb_dir / f"{pdb_id}.pdb"
            if not os.path.exists(filename):
                is_append = False
                continue
            items.update({c_count: [filename, chain]})
            is_append = True
            c_count += 1
        else:
            if not is_append:
                continue
            record = items.get(c_count - 1)
            record.append(line)

    train_set, test_set = train_test_split(items, test_size=0.2)
    train_items = {}
    test_items = {}
    for i, rec in enumerate(train_set):
        train_items.update({i: rec})
    for i, rec in enumerate(test_set):
        test_items.update({i: rec})

    train_dir, val_dir, _ = config.get_samples_dir()
    print('Building train samples...')
    build_samples(train_items, train_dir)
    print('Building test samples...')
    build_samples(test_items, val_dir)


def build_samples(items, out_dir):
    assert os.path.exists(out_dir)
    samples = Samples()
    for idx in tqdm(items, total=len(items)):
        record = items[idx]
        try:
            filename, chain, alphabet = record
            target_path = samples.extract_chains(filename, chain)
            pdb_extractor = PDBBackbone(target_path)
            features = pdb_extractor.extract_features()
            assert features is not None
            assert features[0] is not None
            assert features[1] is not None
            alphabet_data = process_alphabet(alphabet, pad_size=len(features[1]))
            target_file = out_dir / f'{idx}.pkl'
            record = {'features': features, 'alphabet': alphabet_data}
            with open(target_file, 'wb') as file:
                pickle.dump(record, file)
        except:
            continue


def process_alphabet(alphabet_record, pad_size):
    alphabet_record = alphabet_record.strip("\n")
    pad_idx = SAML.index(PAD_SEQ)
    alphabet_idx = torch.as_tensor([SAML.index(s) for s in alphabet_record], dtype=torch.long)
    pad = torch.full(size=(pad_size,), fill_value=pad_idx, dtype=torch.long)
    left_x = (pad_size - len(alphabet_idx)) // 2
    right_x = (pad_size - len(alphabet_idx)) - left_x
    pad[left_x:-right_x] = alphabet_idx
    return pad


class FSSSDataset(Dataset):
    def __init__(self, sample_set='train'):
        config = BaseConfig()
        train_dir, val_dir, _ = config.get_samples_dir()
        current_dir = train_dir if sample_set == 'train' else val_dir
        self.sample_files = get_files(current_dir)
        self.protein_feature = ProteinFeatures()

    def __getitem__(self, index):
        filename = self.sample_files[index]
        assert os.path.exists(filename)
        with open(filename, 'rb') as file:
            record = pickle.load(file)
        features = record['features']
        alphabet_data = record['alphabet']
        data = self.protein_feature.build_features(features)
        data.alphabet = alphabet_data
        return data

    def __len__(self):
        return len(self.sample_files)


def get_loaders():
    config = BaseConfig()
    train_dir, val_dir, _ = config.get_samples_dir()
    train_dataset = FSSSDataset(sample_set='train')
    train_loader = SafeDataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)

    val_dataset = FSSSDataset(sample_set='val')
    val_loader = SafeDataLoader(val_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=config.num_workers)
    return train_loader, val_loader


if __name__ == "__main__":
    train_, val_ = get_loaders()
    while True:
        for item in train_:
            pass
