import os.path
import uuid
from itertools import islice

import numpy as np
from Bio import pairwise2

from config.config import SearchConfig
from search.blast import SearchBlast


class SearchTask(object):
    def __init__(self,
                 task_id,
                 pdb_file,
                 chain,
                 e_value_trash,
                 page_size=20,
                 search_mode=0,  # 0-simple search, 1- hierarchy search
                 hierarchy_max_depth=2,
                 limit=None
                 ):
        self.pdb_file = pdb_file
        if not os.path.exists(self.pdb_file):
            raise FileNotFoundError("PDB file not found")
        self.chain = chain
        self.e_value_trash = e_value_trash
        self.task_id = task_id
        self.search_processor = SearchBlast(task_id=self.task_id,
                                            search_mode=search_mode)
        self.page_size = page_size
        self.mode = search_mode
        self.max_depth = hierarchy_max_depth
        self.processed_keys = set()
        self.base_groups = None
        self.limit = limit if limit else 1e7
        self.base_score = 0
        self.config = SearchConfig()
        self.total_items = 0
        self.search_processor.set_e_value_trash(self.e_value_trash)
        self.source_sequence = None
        self.output_queue = None

    def set_model(self, model):
        self.search_processor.processor.model = model

    def set_output_queue(self, queue):
        self.output_queue = queue

    def search(self):
        if self.search_processor.processor.model is None:
            raise EnvironmentError('Model not initialization')

        if self.mode == 0:  # simple search
            self.basic_search(pdb_file=self.pdb_file, chain=self.chain)
        elif self.mode == 1:  # 1- hierarchy search
            self.hierarchy_search(pdb_file=self.pdb_file,
                                  chain=self.chain,
                                  depth=0
                                  )
        else:
            raise NotImplementedError(f'Invalid mode: {self.mode}')

    def basic_search(self, pdb_file, chain):
        blast_results = self.search_processor.search_blast_hits(pdb_file, chain)
        self.__init_sequence_align(blast_results)
        all_records = self.process_records(blast_results)
        return all_records

    def process_records(self, blast_records):
        total = len(blast_records)
        source_rec = blast_records['source']
        all_records = {}
        for result_data in self.split_by_page(blast_records):
            result_data.update({'source': source_rec})
            inference_data = self.search_processor.inference_results(result_data)
            search_data = self.extract_data(inference_data)
            self.total_items += len(search_data)
            queue_chunk = {
                'task_id': self.task_id,
                'n_count': self.total_items,
                'total': total if self.mode == 0 else -1,
                'data': search_data
            }
            if self.output_queue is not None:
                self.output_queue.put(queue_chunk)

            all_records.update({k: search_data[k] for k in search_data})
            if self.total_items >= self.limit:
                break

        return all_records

    def hierarchy_search(self, pdb_file, chain, depth):
        if depth >= self.max_depth or self.total_items >= self.limit:
            return
        blast_records = self.search_processor.search_blast_hits(pdb_file, chain)
        self.__init_sequence_align(blast_records)

        if not self.base_groups:
            self.__init_groups_align(blast_records)
        else:
            is_match = self.__matching_groups(blast_records)
            if not is_match:
                return
        for key in list(blast_records.keys()):
            if key in self.processed_keys:
                del blast_records[key]
        if not blast_records:
            return
        inference_records = self.process_records(blast_records)
        for key in list(inference_records.keys()):
            if key in self.processed_keys:
                del inference_records[key]

        for key in inference_records:
            self.processed_keys.add(key)
        for key in inference_records:
            file = inference_records[key]['aligned_subj_file']
            chain = inference_records[key]['chain']
            self.hierarchy_search(file, chain, depth + 1)

    def split_by_page(self, result_rec):
        if self.page_size == -1:
            return [result_rec]

        it = iter(result_rec)
        for i in range(0, len(result_rec), self.page_size):
            yield {k: result_rec[k] for k in islice(it, self.page_size)}

    def extract_data(self, records):
        search_results_data = {}
        result_rec = records['results']

        for ids in result_rec:
            pdb_id, chain = ids.split(':')
            e_value = result_rec[ids]['blast']['e_value']
            position = result_rec[ids]['impose']['position']
            rmsd = result_rec[ids]['impose']['rms']
            sequence = result_rec[ids]['blast']['sequence']
            abs_position = result_rec[ids]['impose']['abs_position']
            sequence = sequence if position[0] == -1 else sequence[abs_position[0]:abs_position[1]]
            fasta_identity_score = self.__get_sequence_identity(sequence)
            uploaded_result = result_rec[ids]['upload']
            if uploaded_result:
                aligned_source_file = uploaded_result['ref_file']
                aligned_subj_file, subj_chain = uploaded_result['subj_data']
                assert subj_chain == chain
            else:
                aligned_source_file = ''
                aligned_subj_file = ''
            search_result_record = {
                'e_value': e_value,
                'pdb_id': pdb_id,
                'chain': chain,
                'position': position,  # -1, -1 for full chain
                'fasta': sequence,
                'rmsd': rmsd,
                'fasta_identity_score': fasta_identity_score,
                'aligned_source_file': aligned_source_file,
                'aligned_subj_file': aligned_subj_file,
            }
            search_results_data.update({ids: search_result_record})
        return search_results_data

    @staticmethod
    def __select_main_component(group):
        main_group = ''.join(ch if ch.isupper() and ch != 'O' else '' for ch in group)
        return main_group

    def __init_sequence_align(self, b_result):
        if self.source_sequence is None:
            self.source_sequence = b_result['source']['sequence']

    def __init_groups_align(self, b_result):
        groups = b_result['source']['groups']
        self.base_groups = self.__select_main_component(groups)

    def __matching_groups(self, b_result):
        groups = b_result['source']['groups']
        groups = self.__select_main_component(groups)
        align_score = pairwise2.align.globalxx(self.base_groups, groups, score_only=True)
        if self.base_score == 0:
            self.base_score = align_score
        scale = (align_score / self.base_score)
        d_factor = round(scale, 1)
        factor = self.config.e_scale_factor
        if d_factor not in factor:
            return False
        if self.e_value_trash == 'auto':
            self.e_value_trash = self.search_processor.e_value_trash
        self.search_processor.set_e_value_trash(self.e_value_trash * factor[d_factor])
        return True

    def __get_sequence_identity(self, sequence):
        align_score = pairwise2.align.globalxx(self.source_sequence, sequence, score_only=True)
        identity_score = round(align_score / len(sequence), 2)
        return identity_score

    def clear_task_ws(self):
        self.search_processor.clear_task_ws()
