import os.path
import shutil
from pathlib import Path

import torch
from config.config import SearchConfig
from model_processor.data_processor import ModelProcessor
from Bio.SVDSuperimposer import SVDSuperimposer
from pdb_datasets.pdb_utils import Samples, PDBBackbone, ChainSelect
from Bio.PDB import PDBIO


class SearchProcessor(ModelProcessor):
    def __init__(self, task_id, search_mode):
        super().__init__()
        self.config = SearchConfig()
        self.pdb_path = self.config.get_pdb_dir()
        assert os.path.exists(self.pdb_path)
        self.is_upload = self.config.upload_result or search_mode == 1
        self.is_upload_extract = self.config.upload_extract or search_mode == 1
        self.is_upload_source = self.config.upload_source
        self.imposer = None
        self.task_folder = None
        if self.is_upload:
            self.task_folder = self.config.get_upload_dir() / str(task_id)
            self.task_folder.mkdir(exist_ok=True)
            self.extracted_copy_folder = self.config.get_extracted_copy_folder()
            if self.extracted_copy_folder:
                self.extracted_copy_folder = self.task_folder / self.extracted_copy_folder
                self.extracted_copy_folder.mkdir(exist_ok=True)

        self.results_enum = 1

    def inference(self, result_records):
        source_rec = result_records['source']
        source_emb = source_rec['embedding']
        source_file = source_rec['file']
        source_chain = source_rec['chain']
        source_id = Path(source_file).stem.split('_')[0]
        imposer = Imposer(source_file=source_file, source_chain=source_chain)
        subjects = result_records.copy()
        del subjects['source']
        results_records = {
            'source': {'key': f'{source_id}:{source_chain}, file:{source_file} embedding:{source_emb}'}
        }
        results_item = {}
        for key in subjects:
            pdb_id, chain = key.split(':')
            pdb_file_path = self.pdb_path / f'{pdb_id}.pdb'
            if not os.path.exists(pdb_file_path):
                continue
            subj_result = self.process_file(pdb_file_path, chain)
            subj_emb = subj_result['embedding']

            with torch.no_grad():
                align_record = self.find_align_position(source_emb, subj_emb)
                imposes_rec = imposer.impose(pdb_file_path, chain, align_record)

            subjects[key].update({'sequence': subj_result['sequence']})
            upload_rec = self.upload_results(imposes_rec, source_id=source_id) if self.is_upload else {}
            result = {
                'blast': subjects[key],
                'impose': imposes_rec,
                'upload': upload_rec,
            }
            results_item.update({key: result})

        results_records.update({'results': results_item})
        return results_records

    @staticmethod
    def find_align_position(source_emb, subj_emb):
        source_size = source_emb.size(0)
        direction = 'source_2_target'
        subj_size = subj_emb.size(0)
        if source_size >= subj_size:
            align_base = source_emb
            align_subj = subj_emb
        else:
            align_base = subj_emb
            align_subj = source_emb
            direction = 'target_2_source'
        idx = 0
        min_distance = 1e8
        min_idx = -1
        while idx + len(align_subj) <= len(align_base):
            region = align_base[idx:idx + len(align_subj)]
            distance = torch.pairwise_distance(region, align_subj).sum()
            if min_distance > distance:
                min_distance = distance
                min_idx = idx
            idx += 1
        assert min_idx >= 0
        return {'idx': min_idx, 'direction': direction}

    def upload_results(self, impose_record, source_id):
        reference_struct = impose_record['reference_struct']
        target_struct = impose_record['target_struct']
        ref_id = reference_struct.get_id()
        target_id = target_struct.get_id()
        ref_chain = next(reference_struct.get_chains()).get_id()
        target_chain = next(target_struct.get_chains()).get_id()
        out_dir = f'{self.results_enum}'
        out_dir = self.task_folder / out_dir
        out_dir.mkdir(exist_ok=True)
        if source_id == target_id:
            upload_struct = target_struct
            upload_chain = target_chain
            extract_struct = reference_struct
            extract_chain = ref_chain
        elif source_id == ref_id:
            upload_struct = reference_struct
            upload_chain = ref_chain
            extract_struct = target_struct
            extract_chain = target_chain
        else:
            shutil.rmtree(out_dir)
            return {}
        if self.is_upload_source:
            upload_file = out_dir / f'{upload_struct.get_id()}{upload_chain}.pdb'
            self.upload_pdb(upload_struct, upload_file)
        else:
            upload_file = None
        result = {'ref_file': str(upload_file),
                  'substructure_data': None
                  }

        if self.is_upload_extract:
            position = impose_record['position']
            direction = impose_record['direction']
            if direction == 'target_2_source':
                subj_file = self.extract_substructure(extract_struct,
                                                      extract_chain,
                                                      position,
                                                      out_dir,
                                                      self.extracted_copy_folder
                                                      )
            else:
                subj_file = out_dir / f'{extract_struct.get_id()}{extract_chain}.pdb'
                self.upload_pdb(extract_struct, subj_file)
                if self.extracted_copy_folder:
                    extracted_path = self.extracted_copy_folder / os.path.basename(subj_file)
                    shutil.copy(subj_file, extracted_path)
        else:
            subj_file = out_dir / f'{extract_struct.get_id()}{extract_chain}.pdb'
            self.upload_pdb(extract_struct, subj_file)

        result.update({'subj_data': (str(subj_file), extract_chain)})
        self.results_enum += 1

        return result

    @staticmethod
    def extract_substructure(structure, chain, position, out_dir, extracted_dir=None):
        filename = f'{structure.get_id()}_{chain}{position[0]}_{chain}{position[1]}.pdb'
        filepath = out_dir / filename
        io_w_no_h = PDBIO()
        io_w_no_h.set_structure(structure)
        io_w_no_h.save(str(filepath), ChainSelect(chains=[chain], position=position))
        if extracted_dir:
            extracted_path = extracted_dir / filename
            shutil.copy(filepath, extracted_path)
        return filepath

    @staticmethod
    def upload_pdb(structure, file):
        io_w_no_h = PDBIO()
        io_w_no_h.set_structure(structure)
        io_w_no_h.save(str(file))

    def clear_task_ws(self):
        if not self.task_folder:
            return
        if os.path.exists(self.task_folder):
            shutil.rmtree(self.task_folder)


class Imposer(object):
    def __init__(self, source_file, source_chain):
        assert os.path.exists(source_file)
        self.source_features, self.source_structure = self.__load_pdb(source_file, source_chain)

    @staticmethod
    def __load_pdb(file, chain):
        assert os.path.exists(file)
        samples = Samples()
        stream = samples.extract_chains(file, chain)
        pdb_id = str(Path(file).stem)[:4]
        pdb_extractor = PDBBackbone(stream, pdb_id=pdb_id)
        features = pdb_extractor.extract_features()
        return features, pdb_extractor.structure

    def impose(self, target_file, chain, align_record):
        direction = align_record['direction']
        align_idx = align_record['idx']
        target_features, target_structure = self.__load_pdb(target_file, chain)
        reference_structure = self.source_structure.copy()
        target_structure = target_structure.copy()
        ref_atoms, _ = self.source_features
        alt_atoms, _ = target_features
        if direction == 'target_2_source':
            ref_atoms, alt_atoms = alt_atoms, ref_atoms
            reference_structure, target_structure = target_structure, reference_structure
        align_size = (align_idx, align_idx + alt_atoms.shape[0])
        ref_atoms = ref_atoms[align_size[0]:align_size[1], :, :]
        ref_atoms = ref_atoms.reshape(ref_atoms.shape[0] * ref_atoms.shape[1], 3)
        alt_atoms = alt_atoms.reshape(alt_atoms.shape[0] * alt_atoms.shape[1], 3)
        sup = SVDSuperimposer()
        sup.set(ref_atoms, alt_atoms)
        sup.run()
        rms = sup.get_rms()
        rot_mat, tran_mat = sup.get_rotran()
        for atom in target_structure.get_atoms():
            atom.transform(rot_mat, tran_mat)

        residues = []
        for res in reference_structure.get_residues():
            _, res_id, _ = res.get_id()
            residues.append(res_id)

        start_res_pos = residues[align_size[0]]
        end_res_pos = residues[align_size[1] - 1]
        position = (start_res_pos, end_res_pos) if direction == 'target_2_source' else (-1, -1)
        result_record = {
            'direction': direction,
            'reference_struct': reference_structure,
            'target_struct': target_structure,
            'rms': rms,
            'position': position,
            'abs_position': align_size
        }
        return result_record
