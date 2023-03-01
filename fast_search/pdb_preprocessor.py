import os.path
import pickle
import time
import warnings
from datetime import datetime

import numpy as np

from model_processor.model_initialization import load_shared_model
from search.blast import SearchBlast
from Bio.PDB import PDBParser, PDBExceptions
from config.config import  DBConfig
from pdb_datasets.pdb_utils import  Samples

warnings.filterwarnings("ignore", category=PDBExceptions.PDBConstructionWarning)


class FastSearchBackboneFeatures(object):
    BACKBONE_MASK = ['N', 'CA', 'C', 'O']

    def __init__(self, pdb_path, pdb_id=0):
        self.filename = pdb_path
        self.structure = PDBParser().get_structure(pdb_id, self.filename)
        self.model = None

    def extract_features(self):
        sequence = []
        backbone_coo = []
        positions = []
        for res in self.structure.get_residues():
            res_name = res.get_resname()
            full_name = res.get_full_id()
            res_id = full_name[2]
            res_pos = full_name[3][1]
            positions_record = f'{res_id}{res_pos}'
            try:
                res_coo = []
                for el in self.BACKBONE_MASK:
                    coo = res[el].get_coord()
                    res_coo.append(coo)
            except KeyError:
                continue
            sequence.append(res_name)
            backbone_coo.append(res_coo)
            positions.append(positions_record)

        assert sequence
        assert backbone_coo
        assert len(sequence) == len(backbone_coo) == len(positions)
        backbone_coo = np.array(backbone_coo)
        return backbone_coo, sequence, positions

class PDBPreprocessor(object):
    def __init__(self):
        self.config = DBConfig()
        self.total_processed = 0
        self.saml_data = None
        self.__load_saml_files()
        self.output_dir = self.config.preprocessed_dir
        assert os.path.exists(self.output_dir)

    def __load_saml_files(self):
        data_file = self.config.saml_db_file_path
        assert os.path.exists(data_file)
        with open(data_file, 'rb') as fp:
            self.saml_data = pickle.load(fp)

    def build_preprocessed_db(self):
        assert self.saml_data is not None


        for i, idx in enumerate(self.saml_data):
            try:
                saml_rec = self.saml_data[idx]
                pdb_key = saml_rec['file']
                chain = saml_rec['chain']
                self.__create_record(pdb_id=pdb_key, chain=chain)

                if i % 1000 == 0:
                    self.log_message(f'Preprocessed {i} of {len(self.saml_data)}')

            except (KeyError, AssertionError, Exception):
                continue

    def __create_record(self, pdb_id, chain):
        sampler = Samples()
        pdb_file = self.config.pdb_dir / f'{pdb_id}.pdb'
        assert os.path.exists(pdb_file)
        pdb_data = sampler.extract_chains(pdb_file, [chain])
        backbone_features = FastSearchBackboneFeatures(pdb_data)
        features = backbone_features.extract_features()

        assert features[0] is not None
        assert len(features[0]) > 0xf

        output_file = self.output_dir / f'{pdb_id}_{chain}.pkl'
        with open(output_file, 'wb') as fh:
            pickle.dump(features, fh)

    def log_message(self, message):
        current_datetime = datetime.now()
        current_time = current_datetime.strftime("%H:%M:%S")

        with open(self.config.logging_file, 'a') as fh:
            fh.write(f"{current_time}:{message} \n")

    def create_test_dataset(self, test_ids_list):
        model = load_shared_model()
        for rec in  test_ids_list:
            source_pdb_id, source_chain = rec
            task_id = source_pdb_id
            search = SearchBlast(task_id)
            search.processor.model = model
            source_pdb_path = self.config.pdb_dir / f'{source_pdb_id}.pdb'
            b_result = search.search_blast_hits(source_pdb_path, source_chain)
            keys = list(b_result.keys())
            keys.remove('source')
            for result_rec in keys:
                pdb_id, chain = result_rec.split(':')
                self.__create_record(pdb_id, chain)


def build_preprocessed_db():
    preprocessor = PDBPreprocessor()
    preprocessor.build_preprocessed_db()

def build_test_db(pdb_list):
    preprocessor = PDBPreprocessor()
    preprocessor.create_test_dataset(pdb_list)

test_db_ids = [('2ko3', 'A'), ('2ocs', 'A'), ('1tit', 'A')]
if __name__ == '__main__':
    build_test_db(test_db_ids)
