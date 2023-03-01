import os.path
from pathlib import Path

import numpy as np

from config.config import SearchConfig
from pdb_datasets.pdb_utils import Samples
from Bio.PDB import PDBParser, PDBIO, PDBExceptions
from Bio.PDB.StructureBuilder import StructureBuilder


def is_extraction():
    config = SearchConfig()
    return config.upload_result


class PDBExtractor(object):
    BACKBONE_MASK = ['N', 'CA', 'C', 'O']

    def __init__(self,
                 task_id,
                 source_path,
                 chain,
                 ):
        assert os.path.exists(source_path), f'PDB file not found {source_path}'
        selection = Samples()
        pdb_data = selection.extract_chains(source_path, [chain])
        self.source_structure = PDBParser().get_structure('source', pdb_data)
        self.source_chain = chain
        self.source_pdb_id = Path(source_path).stem

        config = SearchConfig()
        self.task_folder = config.get_upload_dir() / str(task_id)
        self.task_folder.mkdir(exist_ok=True)
        self.extracted_copy_folder = config.get_extracted_copy_folder()
        if self.extracted_copy_folder:
            self.extracted_copy_folder = self.task_folder / self.extracted_copy_folder
            self.extracted_copy_folder.mkdir(exist_ok=True)

        self.current_folder_id = 0


    def extract(self,
                msg,
                file_data,
                ):
        self.current_folder_id += 1
        pdb_id = msg['pdb_id']
        subj_filename, _, subj_sequence, subj_positions, backbone_coo = file_data
        sup_matrix = msg['sup_matrix']
        subj_chain = msg['chain']
        positions = [s.replace(subj_chain, '') for s in subj_positions]
        subj_structure = self.build_structure(chain=subj_chain,
                                              sequence=subj_sequence,
                                              positions=positions,
                                              coo=backbone_coo
                                        )



        apply_to = sup_matrix['apply_to']
        rotation_mx = np.array(sup_matrix['rotation'])
        translation_mx = np.array(sup_matrix['translation'])

        source_structure = self.source_structure

        if apply_to == 'source':
            source_structure = self.apply_impose(source_structure.copy(), rotation_mx, translation_mx)

        else:
            subj_structure = self.apply_impose(subj_structure, rotation_mx, translation_mx)

        self.save_structure(pdb_id, chain=subj_chain, structure=subj_structure)
        self.save_structure(self.source_pdb_id, chain=self.source_chain, structure=source_structure)



    @staticmethod
    def apply_impose(structure, rotation_mx, translation_mx):
        for atom in structure.get_atoms():
            atom.coord = rotation_mx.dot(atom.coord.T).T + translation_mx
        return structure

    def save_structure(self, pdb_id, chain, structure):
        output_folder = self.task_folder / str(self.current_folder_id)
        output_folder.mkdir(exist_ok=True)
        outfile = output_folder / f'{pdb_id}_{chain}.pdb'
        io_w_no_h = PDBIO()
        io_w_no_h.set_structure(structure)
        io_w_no_h.save(str(outfile))


    def build_structure(self,
                        chain,
                        sequence,
                        coo,
                        positions,
                        ):
        structure = StructureBuilder()
        structure.init_structure(structure_id='subj')
        structure.init_model(model_id=0)
        structure.init_chain(chain)
        structure.init_seg(1)

        for i in range(len(coo)):
            try:
                xyz = coo[i, :]
                structure.init_residue(
                    resname=sequence[i],
                    field=' ',
                    resseq=int(positions[i]),
                    icode=' '
                )

                for j, atom_name in enumerate(self.BACKBONE_MASK):
                    atom_xyz = xyz[j, :]
                    structure.init_atom(
                        name=atom_name,
                        coord=atom_xyz,
                        element=atom_name[0],
                        occupancy=0,
                        b_factor=1.0,
                        fullname=f' {atom_name} ',
                        altloc=' '
                    )
            except PDBExceptions.PDBConstructionException:
                continue


        return structure.get_structure()

