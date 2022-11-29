import io
import os
import shutil
import warnings
import numpy as np

from Bio.PDB import PDBExceptions, PDBIO, Select
from Bio.PDB import PDBParser
from config.config import BaseConfig

warnings.filterwarnings("ignore", category=PDBExceptions.PDBConstructionWarning)

BACKBONE_MASK = ['N', 'CA', 'C', 'O']


class PDBFeaturesExtractor(object):
    def __init__(self, atom_pos, sequence, res_pos):
        self.config = BaseConfig()
        self.atom_pos = atom_pos
        self.sequence = sequence
        self.res_pos = res_pos
        res_count, res_desc_num, dim = atom_pos.shape
        assert res_count == len(sequence) == len(res_pos)
        self.symbols = ['N', 'Ca', 'C', 'O']

    def calculate(self):
        groups = self.split_by_chains()
        features = {}
        for chain in groups:
            positions, sequence, res_pos = groups[chain]

            assert len(sequence) == len(res_pos) == len(positions)
            features.update({chain: (positions, sequence, res_pos)})
        return features

    def split_by_chains(self):
        chain_groups = {}
        chains = np.array([s.split('_')[0] for s in self.res_pos])
        chains_set = sorted(set(chains))
        for chain in chains_set:
            mask = np.zeros(len(chains), dtype=np.bool)
            mask[chains == chain] = True
            res_pos = np.array(self.res_pos)[mask]
            sequence = np.array(self.sequence)[mask]
            positions = self.atom_pos[mask]
            assert len(res_pos) == len(sequence) == len(positions)
            if len(sequence) < self.config.min_chain_len:
                continue
            chain_groups.update({chain: (positions, sequence, res_pos)})

        return chain_groups


class PDBBackbone(object):
    def __init__(self, pdb_path, pdb_id=0):
        self.filename = pdb_path
        self.structure = PDBParser().get_structure(pdb_id, self.filename)
        self.model = None

    def extract_features(self):
        sequence = []
        backbone_coo = []
        for res in self.structure.get_residues():
            res_name = res.get_resname()
            try:
                res_coo = []
                for el in BACKBONE_MASK:
                    coo = res[el].get_coord()
                    res_coo.append(coo)
            except KeyError:
                continue
            sequence.append(res_name)
            backbone_coo.append(res_coo)

        assert sequence
        assert backbone_coo
        backbone_coo = np.array(backbone_coo)
        return backbone_coo, sequence


class ChainSelect(Select):
    dx = 2

    def __init__(self, chains, position=None):
        self.chains = chains
        self.current_chain = None
        self.position = position

    def accept_model(self, model):
        if model.id == 0:
            return 1

    def accept_chain(self, chain):
        if chain.get_id() in self.chains:
            self.current_chain = chain.get_id()
            return 1
        else:
            return 0

    def accept_residue(self, residue):
        if residue.id[0] != " ":
            return False
        if self.position is None:
            return True
        _, res_pos, _ = residue.get_id()
        if self.position[0] - self.dx <= res_pos <= self.position[1] + self.dx:
            return True
        return False

    def accept_atom(self, atom):
        return True if atom.get_name() in BACKBONE_MASK else False


class Samples(object):
    def __init__(self):
        self.parser = PDBParser(PERMISSIVE=1)

    def extract_chains(self, path, chain_id):
        assert os.path.exists(path)
        structure = self.parser.get_structure(id=0, file=str(path))
        output = io.StringIO()
        io_w_no_h = PDBIO()
        io_w_no_h.set_structure(structure)
        io_w_no_h.save(output, ChainSelect(chain_id))
        output.seek(0)
        return output

    @staticmethod
    def clear(tmp_file):
        tmp_file.close()
