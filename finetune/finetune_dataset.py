import os.path
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from config.config import BaseConfig, DBConfig
from pdb_datasets.dataloader_utils import SafeDataLoader
from pdb_datasets.datasets import process_alphabet
from pdb_datasets.geometric_features import ProteinFeatures
from utils.utils import get_files

STRIDE_LETTERS =   ['H', 'G', 'I', 'E', 'B', 'b', 'T', 'C', ' ']

def stride_to_idx(stride):
    stride_idx = [STRIDE_LETTERS.index(s) for s in stride]
    return stride_idx


class FinetuneSamples(object):
    def __init__(self):
        config = BaseConfig()
        config.make_working_dir()
        stride_folder = config.stride_folder
        alphabet_records_file = config.get_train_file_path()
        self.samples_file = config.get_samples_dir()[0].parent / config.finetune_samples_file
        assert os.path.exists(stride_folder)
        config = DBConfig()
        self.preprocessed_folder = config.preprocessed_dir
        assert os.path.exists(self.preprocessed_folder)
        self.stride_files = get_files(stride_folder)
        pdb_files = get_files(self.preprocessed_folder)
        pdb_keys = [str(Path(file).stem) for file in pdb_files]
        self.alphabet_records = self.make_alphabet_set(alphabet_records_file, pdb_keys)
        pdb_keys = list(self.alphabet_records.keys())
        self.pdb_items = {k.split('_')[0]:[] for k in pdb_keys}
        for k in pdb_keys:
            pdb_id, chain = k.split('_')
            self.pdb_items[pdb_id].append(chain)
        self.stride_files = [f for f in self.stride_files if Path(f).stem in self.pdb_items]
        self.samples = {}


    def build_samples(self):
        for file in self.stride_files:
            stride_file_key = Path(file).stem
            stride_record = self.get_parsed_stride(file)
            if not stride_record:
                continue
            self.extract_stride_seq(stride_record, stride_file_key)
        self.save_samples()

    def save_samples(self):
        print(f'Total samples: {len(self.samples)} saved to {self.samples_file}')
        with open(self.samples_file, 'wb') as fh:
            pickle.dump(self.samples, fh)


    def extract_stride_seq(self, record, pdb_id):
        actual_chains = self.pdb_items[pdb_id]
        for chain in actual_chains:
            chain_seq = [r for r in record if r['chain'] == chain]
            if not chain_seq:
                continue
            stride_sequence = [r['sec_letter'] for r in chain_seq]
            stride_sequence = ''.join(stride_sequence)
            self.verify_record(pdb_id, chain, stride_sequence)


    def verify_record(self, pdb_id, chain, stride_sequence):
        try:
            record_key = f'{pdb_id}_{chain}'
            features_file = self.preprocessed_folder / f'{record_key}.pkl'
            assert os.path.exists(features_file)
            with open(features_file, 'rb') as file:
                record = pickle.load(file)
            _, sequence, _ = record
            assert len(sequence) == len(stride_sequence)
            alphabet = self.alphabet_records.get(record_key)
            assert alphabet is not None
            assert len(alphabet) == len(sequence) - 4
            self.samples.update({record_key: (alphabet, stride_sequence)})
        except AssertionError:
            pass



    @staticmethod
    def get_parsed_stride(filename):
        with open(filename, 'r') as fh:
            raw = fh.read()
        if raw is None:
            return None
        try:
            asg = filter(lambda x: x[:3] == "ASG", raw.splitlines(keepends=False))
        except AttributeError:
            return None
        parsed = []
        try:
            for line in asg:
                parsed.append({
                    "amino": line[5:8],
                    "total_no": int(line[10:15]),
                    "file_no": int(line[16:20]),
                    "chain": line[9],
                    "sec_letter": line[24],
                    "sec_name": line[25:39].strip()
                })
        except ValueError:
            return None
        return parsed

    @staticmethod
    def make_alphabet_set(filepath, pdb_keys):
        with open(filepath) as file:
            lines = file.readlines()
        c_count = 0
        items = {}
        is_append = False

        for idx, line in enumerate(lines):
            if idx % 2 == 0:
                pdb_id = line.split(" ")[0]
                chain = pdb_id[-1]
                pdb_id = pdb_id[1:-1]
                items.update({c_count: [pdb_id, chain]})
                is_append = True
                c_count += 1
            else:
                if not is_append:
                    continue
                record = items.get(c_count - 1)
                record.append(line)

        out = {}

        for cnt in items:
            record = items.get(cnt)
            pdb_id, chain, alphabet = record
            alphabet = alphabet.rstrip(os.linesep)
            out.update({f'{pdb_id}_{chain}': alphabet})

        source_set = set(pdb_keys)
        out_set = set(out.keys())
        t_set = source_set.intersection(out_set)
        out = {k:out[k] for k in out if k in t_set}

        return out

    def split_samples(self):
        config = BaseConfig()
        with open(self.samples_file, 'rb') as fh:
            data = pickle.load(fh)
            keys = list(data.keys())
        keys = np.random.permutation(keys)
        train_size = int(len(keys) * config.finetune_split_factor)
        train_set_idx = keys[:train_size]
        test_set_idx = keys[len(train_set_idx):]
        assert len(train_set_idx) + len(test_set_idx) == len(keys)
        train_dir, test_dir, _ = config.get_samples_dir()
        train_file_idx = train_dir / config.finetune_set_names[0]
        test_file_idx = test_dir / config.finetune_set_names[1]
        with open(train_file_idx, 'wb') as fh:
            pickle.dump(train_set_idx, fh)

        with open(test_file_idx, 'wb') as fh:
            pickle.dump(test_set_idx, fh)

def get_samples_set(mode):
    config = BaseConfig()
    assert mode in ['train', 'test']
    train_dir, test_dir, _ = config.get_samples_dir()
    sample_folder = train_dir if mode == 'train' else test_dir
    sample_file = config.finetune_set_names[0] if mode == 'train' else config.finetune_set_names[1]
    sample_path = sample_folder / sample_file
    assert os.path.exists(sample_path)
    with open(sample_path, 'rb') as fh:
        keys = pickle.load(fh)

    samples_file = config.get_samples_dir()[0].parent / config.finetune_samples_file
    with open(samples_file, 'rb') as fh:
        samples = pickle.load(fh)

    to_remove =  set(samples.keys()).difference(keys)
    for d in to_remove:
        del samples[d]
    return samples


class FinetuneDataset(Dataset):
    def __init__(self, sample_set='train'):
        self.samples = get_samples_set(mode=sample_set)
        self.protein_feature = ProteinFeatures()
        config = DBConfig()
        self.preprocessed_dir = config.preprocessed_dir

    def __getitem__(self, index):
        key = list(self.samples.keys())[index]
        datafile = self.preprocessed_dir / f'{key}.pkl'
        assert os.path.exists(datafile)
        with open(datafile, 'rb') as file:
            backbone_coo, sequence, _  = pickle.load(file)
        features = (backbone_coo, sequence)
        data = self.protein_feature.build_features(features)
        alphabet_data, stride = self.samples[key]
        alphabet_idx = process_alphabet(alphabet_data, len(sequence))
        stride_idx = stride_to_idx(stride)
        data.alphabet = alphabet_idx
        data.stride = torch.as_tensor(stride_idx, dtype=torch.long)
        return data

    def __len__(self):
        return len(self.samples)


def get_loaders():
    config = BaseConfig()
    train_dataset = FinetuneDataset(sample_set='train')
    train_loader = SafeDataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)

    val_dataset = FinetuneDataset(sample_set='test')
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


