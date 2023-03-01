import os
import pickle
from pathlib import Path
import queue
from torch.utils.data import Dataset
from config.config import DBConfig, BaseConfig
from pdb_datasets.dataloader_utils import SafeDataLoader
from pdb_datasets.geometric_features import ProteinFeatures
from torch.multiprocessing import Queue



class FastsearchDataset(Dataset):
    def __init__(self,
                 pdb_ids,
                 data_queue,
                 is_extract=False
                 ):
        config = DBConfig()
        name_ls = [f"{s.replace(':', '_')}.pkl" for s in pdb_ids]
        self.files = [os.path.join(config.preprocessed_dir, f) for f in name_ls]
        self.files = [f for f in self.files if os.path.exists(f)]
        self.protein_feature_extractor = ProteinFeatures()
        self.data_queue = data_queue
        self.is_extract = is_extract
    def __getitem__(self, index):
        filename = self.files[index]
        with open(filename, 'rb') as file:
            record = pickle.load(file)
        backbone_coo, sequence, aa_positions = record
        data = self.protein_feature_extractor.build_features([backbone_coo, sequence])
        key = Path(filename).stem.replace('_', ':')
        coo_rec = None if not self.is_extract else backbone_coo
        self.data_queue.put((key, index, sequence, aa_positions, coo_rec), block=False)
        data.file_idx = index
        return data



    def __len__(self):
        return len(self.files)


def get_fs_loaders(pdb_ids, extract=False):
    config = BaseConfig()
    data_queue = Queue(maxsize=-1)
    dataset = FastsearchDataset(pdb_ids,
                                data_queue,
                                is_extract=extract
                                )
    total_items = len(dataset)
    loader = SafeDataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers)

    return loader, data_queue, total_items

