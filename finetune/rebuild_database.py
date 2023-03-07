import os
import pickle
import subprocess
from pathlib import Path

import torch
from torch.utils.data import Dataset
from config.config import BaseConfig, DBConfig
from model_processor.data_processor import ModelProcessor
from model_processor.model_initialization import load_shared_model
from pdb_datasets.dataloader_utils import SafeDataLoader
from pdb_datasets.geometric_features import ProteinFeatures
from utils.utils import get_files
from tqdm import tqdm


class FullDBDataset(Dataset):
    def __init__(self):
        config = DBConfig()
        self.protein_feature = ProteinFeatures()
        self.files = get_files(config.preprocessed_dir)

    def __getitem__(self, index):
        datafile = self.files[index]
        assert os.path.exists(datafile)
        with open(datafile, 'rb') as fh:
            backbone_coo, sequence, _  = pickle.load(fh)
        features = (backbone_coo, sequence)
        data = self.protein_feature.build_features(features)
        data.file_idx = Path(datafile).stem
        return data

    def __len__(self):
        return len(self.files)


def get_db_loader():
    config = BaseConfig()
    dataset = FullDBDataset()
    loader = SafeDataLoader(dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)

    return loader


class FinetuneDBBuilder(object):
    def __init__(self):
        self.config = DBConfig()
        model = load_shared_model()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model_processor = ModelProcessor(model)
        model.to(self.device)
        self.db_data = {}

    def build_database(self):
        loader = get_db_loader()
        for batch in  tqdm(loader, total=len(loader)):
            results = self.model_processor.process_batch(batch)
            for item in results:
                pdb_id = item['pdb_id']
                chain = item['chain']
                key = f'{pdb_id}_{chain}'
                self.db_data.update({key: item['alphabet']})

        self.save_database()

    def save_database(self):
        db_file = self.config.get_db_path()
        print(f'PDB indexing completed, {len(self.db_data)} sequences processed')
        with open(db_file, 'wb') as fp:
            pickle.dump(self.db_data, fp)

    def create_fa(self):
        fa_path = self.config.get_fa_path()
        db_file = self.config.get_db_path()

        with open(db_file, 'rb') as fp:
            data = pickle.load(fp, fix_imports=True)

        with open(fa_path, 'w') as fa_out:
            for key in data:
                saml = data[key]
                fa_str = f">{key}\n{saml}\n"
                fa_out.write(fa_str)

        print(f'FASTA file completed')

    def build_blast_db(self):
        fa_path = self.config.get_fa_path()
        blast_db = self.config.get_blast_db_name()
        blast_db_folder = self.config.get_blast_folder()
        [f.unlink() for f in Path(blast_db_folder).glob("*") if f.is_file() and str(f) != str(fa_path)] # remove old db

        cmd = f"makeblastdb  -dbtype prot  -in {fa_path} -out {blast_db} -parse_seqids"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout.readlines():
            print(line.decode("utf-8"))
        retval = p.wait()
        print(f'Database builder complete. Return code: {retval}')

def rebuild_database():
    dbBuilder = FinetuneDBBuilder()
   # dbBuilder.build_database()
    dbBuilder.create_fa()
    dbBuilder.build_blast_db()


if __name__ == '__main__':
    rebuild_database()