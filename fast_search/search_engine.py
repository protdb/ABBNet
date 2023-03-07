import os
import queue
import sys

from fast_search.fs_datasets import get_fs_loaders
from fast_search.fs_model import FastSearchModel
from fast_search.pdb_extractor import is_extraction, PDBExtractor
from finetune.finetune_dataset import STRIDE_LETTERS
from finetune.rebuild_database import rebuild_database
from logger_utils.logger_utils import setup_logger, log_message, log_error
from model_processor.model_initialization import load_shared_model
from search.blast import SearchBlast
from Bio.PDB.Polypeptide import three_to_one
from Bio import pairwise2

shared_model = None


def get_model_instance():
    global shared_model
    if shared_model is None:
            shared_model = load_shared_model()
    return shared_model


class FastSearch(object):
    def __init__(self,
                 task_id,
                 pdb_file,
                 chain,
                 e_value_trash,
                 page_size=32,
                 ):
        setup_logger()
        self.source_pdb_file = pdb_file
        if not os.path.exists(self.source_pdb_file):
            log_error(f"PDB file not found: {pdb_file}")
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        self.source_chain = chain
        self.e_value_trash = e_value_trash
        self.task_id = task_id
        try:
            model = get_model_instance()
        except (RuntimeError, Exception, FileNotFoundError) as e:
            log_error(f"Unable to load model: {e}")
            raise (e, f"Unable to load model: {e}")

        self.search_processor = SearchBlast(model)
        self.search_processor.set_e_value_trash(e_value_trash)
        self.page_size = page_size


    def run_search(self, callback_fn=None):
        log_message(message=f"Start task: {self.task_id}")

        blast_records = self.search_processor.search_blast_hits(self.source_pdb_file, self.source_chain)
        source_record = blast_records['source']
        source_coo = source_record['source_coo']
        source_sequence = source_record['sequence']
        source_embedding = source_record['embedding']
        source_stride = source_record['source_stride']

        del blast_records['source']
        subj_keys = list(blast_records.keys())

        is_extract = is_extraction()
        loader, processed_queue, n_total_items = get_fs_loaders(subj_keys, is_extract)

        log_message(message=f"Task: {self.task_id} Find {n_total_items} candidates")

        e_value_map = {k:blast_records[k]['e_value'] for k in blast_records}
        search_model = FastSearchModel(
            encoder_model=get_model_instance(),
            source_coo=source_coo,
            source_embedding=source_embedding
        )

        pdb_extractor = PDBExtractor(self.task_id,
                                     self.source_pdb_file,
                                     self.source_chain) if is_extract else None


        output_mgr = OutputMgr(
            task_id=self.task_id,
            e_value_map=e_value_map,
            source_sequence=source_sequence,
            source_stride=source_stride,
            processed_queue=processed_queue,
            n_total_items=n_total_items,
            page_size=self.page_size,
            callback_fn=callback_fn,
            pdb_extractor=pdb_extractor
        )

        total_processed = 0

        for idx, batch in enumerate(loader):
            try:
                results = search_model(batch)
                batch_size = batch.batch.max() + 1
                batch_size = batch_size.item()
                total_processed += batch_size
                output_mgr.put_results(results, total_processed==n_total_items)
            except (RuntimeError, Exception) as e:
                log_error(message=f"Task: {self.task_id} An error occurred while processing candidates: {e}")
                raise e

        log_message(message=f"Task: {self.task_id} completed")

class OutputMgr(object):
    def __init__(self,
                 task_id,
                 e_value_map,
                 source_sequence,
                 source_stride,
                 processed_queue,
                 n_total_items,
                 page_size,
                 callback_fn,
                 pdb_extractor=None
                 ):
        self.task_id = task_id
        self.source_sequence = source_sequence
        self.source_stride = source_stride
        self.e_value_map = e_value_map
        self.processed_items = {}
        self.processed_queue = processed_queue
        self.page_size = page_size
        self.callback_fn = callback_fn
        self.message_buffer = []
        self.local_buffer = []
        self.total_items = n_total_items
        self.total_processed = 0
        self.pdb_extractor = pdb_extractor

    def __read_all_processed_data(self):
        q_size = self.processed_queue.qsize()
        for _ in range(q_size + 1):
            try:
                data = self.processed_queue.get(block=False)
                file_idx = data[1]
                self.processed_items.update({file_idx: data})
            except queue.Empty:
                break

    def __read_item_data(self, subj_file_idx):
        while not subj_file_idx in self.processed_items:
            try:
                data = self.processed_queue.get(block=False)
                file_idx = data[1]
                self.processed_items.update({file_idx: data})
            except queue.Empty:
               continue


    def put_results(self, batch_results, last_batch=False):
        self.__read_all_processed_data()
        for record in batch_results:
            batch_msg = self.__format_message(record)
            self.local_buffer.append(batch_msg)
            self.message_buffer.append(batch_msg)

        if len(self.local_buffer) >= self.page_size or last_batch:
            self.total_processed += len(self.local_buffer)
            if self.callback_fn is not None:
                out_msg = {
                    'task_id': self.task_id,
                    'is_last_msg': last_batch,
                    'n_count': self.total_processed,
                    'total': self.total_items,
                    'data': self.local_buffer.copy()
                }
                self.callback_fn(out_msg)
                self.local_buffer.clear()

    def __format_message(self, record):
        file_idx = record['file_idx']
        select_idx_start = record['select_idx_start']
        select_idx_end = record['select_idx_end']
        apply_to = record['apply_to']
        rotation_mx = record['rotation_mx'].tolist()
        translation_mx = record['translation_mx'].tolist()
        rmsd = record['rmsd']
        subj_stride = self.__convert_to_stride(record['stride'])
        subj_stride = subj_stride[select_idx_start:select_idx_end]

        try:
            file_data = self.processed_items[file_idx]
        except KeyError:
            self.__read_item_data(file_idx)
            file_data = self.processed_items[file_idx]

        subj_filename, _, subj_sequence, subj_position, _ = file_data
        e_value = self.e_value_map[subj_filename]
        pdb_id, chain = subj_filename.split(':')
        positions_idx = [s.replace(chain, '') for s in subj_position]
        position = (-1, -1) if apply_to == 'subj' else (positions_idx[select_idx_start],
                                                        positions_idx[select_idx_end -1])
        subj_fasta = self.__convert_fasta(subj_sequence)
        fasta_identity_score = self.__fasta_identity(apply_to,
                                                    select_ids=(select_idx_start, select_idx_end),
                                                    subj_sequence=subj_fasta)

        out_msg = {
            'pdb_id': pdb_id,
            'chain': chain,
            'position': position,  # -1, -1 for full chain
            'e_value': e_value,
            'fasta': {'source': self.source_sequence, 'subj':subj_fasta[select_idx_start:select_idx_end]},
            'stride': {'source': self.source_stride, 'subj':subj_stride},
            'rmsd': rmsd,
            'fasta_identity_score': fasta_identity_score,
            'sup_matrix': {'apply_to': apply_to, 'rotation': rotation_mx, 'translation': translation_mx},
        }

        if self.pdb_extractor:
            self.extract_msg(out_msg, file_idx)


        del self.processed_items[file_idx] # release memory
        return out_msg

    def __fasta_identity(self,
                         apply_to,
                         select_ids,
                         subj_sequence,
                         ):


        reference_sequence = self.source_sequence
        target_sequence = subj_sequence

        if apply_to == 'subj':
            reference_sequence, target_sequence = target_sequence, reference_sequence

        target_sequence = target_sequence[select_ids[0]:select_ids[1]]
        assert len(reference_sequence) == len(target_sequence)
        try:
            align_score = pairwise2.align.globalxx(reference_sequence, target_sequence, score_only=True)
            identity_score = round(align_score / len(target_sequence), 2)
        except:
            identity_score = 0.6
        return identity_score

    @staticmethod
    def __convert_fasta(sequence):
        s = ''
        try:
            for i in range(len(sequence)):
                s += three_to_one(sequence[i])
        except IndexError:
            s += 'X'
        return s

    @staticmethod
    def __convert_to_stride(arr):
        s = ''.join(STRIDE_LETTERS[i] for i in arr)
        return s

    def extract_msg(self, msg, file_idx):
        try:
            file_data = self.processed_items[file_idx]
        except KeyError:
            self.__read_item_data(file_idx)
            file_data = self.processed_items[file_idx]

        self.pdb_extractor.extract(msg, file_data)







test_file = '/home/dp/Data/PDB/2ko3.pdb'

def test_callback(msg):
    print(msg)


def test_search():
    search_engine = FastSearch(task_id='2ocs',
                               pdb_file=test_file,
                               chain='A',
                               e_value_trash='auto')
    search_engine.run_search(callback_fn=test_callback)


if __name__ == '__main__':
    rebuild_database()
