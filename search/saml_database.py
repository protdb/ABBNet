import os.path
import pickle
import shutil
import subprocess
import uuid
from pathlib import Path

import Bio.PDB.PDBExceptions
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from config.config import DBConfig
from model_processor.data_processor import ModelProcessor
from pdb_datasets.pdb_utils import Samples, ChainSelect
from utils.utils import get_files


class DatabaseBuilder(object):
    def __init__(self):
        self.config = DBConfig()
        self.pdb_dir = self.config.pdb_dir
        assert os.path.exists(self.pdb_dir)
        self.processor = ModelProcessor()
        self.db_data = {}
        self.db_idx = 0

    def build_database(self):
        pdb_files = get_files(self.pdb_dir, ext='*.pdb')
        for file in pdb_files:
            batch_dir = self.config.tmp_dir / str(uuid.uuid4())
            batch_dir.mkdir(exist_ok=True)
            try:
                files_map = self.parse_pdb(file, out_dir=batch_dir)
                if not files_map:
                    continue
                results = self.processor.process_folder(batch_dir)
                for idx in results:
                    file_record = files_map[idx]
                    file_id = file_record['file']
                    chain = file_record['chain']
                    saml = results[idx]
                    db_record = {'file': file_id, 'chain': chain, 'saml': saml}
                    self.db_data.update({self.db_idx: db_record})
                    self.db_idx += 1
            except:
                continue

            finally:
                shutil.rmtree(str(batch_dir))

    @staticmethod
    def parse_pdb(path, out_dir):
        files_map = {}
        parser = PDBParser(PERMISSIVE=1)
        try:
            structure = parser.get_structure(id=0, file=str(path))
        except Bio.PDB.PDBExceptions.PDBException:
            return {}
        chains = structure.get_chains()
        for idx, chain in enumerate(chains):
            try:
                io_w_no_h = PDBIO()
                io_w_no_h.set_structure(structure)
                out_path = out_dir / f'{idx}.pdb'
                io_w_no_h.save(str(out_path), ChainSelect(chain.id))
                record = {'chain': chain.id, 'file': Path(path).stem}
                files_map.update({idx: record})
            except Bio.PDB.PDBExceptions.PDBException:
                continue
        return files_map

    def save_database(self):
        db_file = self.config.get_db_path()
        with open(db_file, 'wb') as fp:
            pickle.dump(self.db_data, fp)

    def build_blast_db(self):
        fa_path = self.config.get_fa_path()
        blast_db = self.config.get_blast_db_name()
        cmd = f"makeblastdb  -dbtype prot  -in {fa_path} -out {blast_db} -parse_seqids"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout.readlines():
            print(line.decode("utf-8"))
        retval = p.wait()
        print(f'Return code: {retval}')

    def create_fa(self):
        fa_path = self.config.get_fa_path()
        db_file = self.config.get_db_path()
        assert os.path.exists(db_file)
        with open(db_file, 'rb') as fp:
            data = pickle.load(fp, fix_imports=True)
        keys = set()
        with open(fa_path, 'w') as fa_out:
            for idx in data:
                record = data[idx]
                pdb_id = record['file']
                chain = record['chain']
                saml = record['saml']
                key = f'{pdb_id}_{chain}'
                if len(saml) < self.config.min_saml_thresh_:
                    continue
                if key in keys:
                    continue
                fa_str = f">{key}\n{saml}\n"
                fa_out.write(fa_str)
                keys.add(key)


if __name__ == "__main__":
    db = DatabaseBuilder()
    db.create_fa()
    db.build_blast_db()

