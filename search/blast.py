import os
from io import StringIO

import numpy as np
from Bio.Blast import NCBIXML
from Bio.Blast.Applications import NcbiblastpCommandline
from config.config import DBConfig
from search.search_processor import SearchProcessor


class SearchBlast(object):
    def __init__(self, task_id, search_mode=0):
        self.config = DBConfig()
        self.pdb_dir = self.config.pdb_dir
        assert os.path.exists(self.pdb_dir)
        self.processor = SearchProcessor(task_id, search_mode)
        self.db_data = {}
        self.db_idx = 0
        self.e_value_trash = self.config.e_value_trash

    def set_e_value_trash(self, value):
        self.e_value_trash = value

    def set_hierarchy_mode(self):
        self.processor.is_upload = True

    def search_blast_hits(self, pdb_file, chain):
        assert os.path.exists(pdb_file)
        result = self.processor.process_file(pdb_file, chain)
        try:
            saml = result['alphabet']
            embedding = result['embedding']
            groups = result['groups']
            sequence = result['sequence']
        except KeyError:
            return result
        if self.e_value_trash == 'auto':
            scale = np.round(np.sqrt(len(embedding) / 10)) - 1
            self.e_value_trash = 1.0 * 10 ** -scale
        result = self.process_blast(saml_string=saml)
        source_record = dict({'saml': saml,
                              'file': pdb_file,
                              'chain': chain,
                              'groups': groups,
                              'sequence': sequence,
                              'embedding': embedding})
        result.update({'source': source_record})
        return result

    @staticmethod
    def out_result(result_data):
        for key in result_data:
            record = result_data[key]
            e_value = record['e_value']
            print(f'{key} E-value: {e_value}')

    def clear_task_ws(self):
        self.processor.clear_task_ws()

    def process_blast(self, saml_string):
        blast_db = self.config.get_blast_db_name()
        blast_cmd = NcbiblastpCommandline(db=str(blast_db), outfmt=5, max_target_seqs=7000)
        out, err = blast_cmd(stdin=saml_string)
        query_result = {}
        for record in NCBIXML.parse(StringIO(out)):
            for align in record.alignments:
                for hsp in align.hsps:
                    e_value = hsp.expect
                    if e_value <= self.e_value_trash:
                        keys = align.hit_id.split('|')
                        key = f'{keys[1]}:{keys[2]}'
                        record = {'query': hsp.query,
                                  'match': hsp.match,
                                  'e_value': e_value,
                                  'sbj': hsp.sbjct}
                        query_result.update({key: record})
        return query_result

    def inference_results(self, results_record):
        inference_data = self.processor.inference(results_record)
        return inference_data


TEST_FILE1 = "/home/dp/Data/SAML/test/pdb/sample10/1hxd.pdb"
TEST_FILE2 = "/home/dp/Data/SAML/test/pdb/sample2/3qvo.pdb"
TEST_FILE3 = "/home/dp/Data/SAML/test/pdb/sample3/6x9z.pdb"
TEST_FILE4 = "/home/dp/Data/SAML/test/pdb/sample4/1A1B.pdb"

if __name__ == "__main__":
    search = SearchBlast()
    b_result = search.search_blast_hits(TEST_FILE1, 'A')
    search.inference_results(b_result)

    # vis.plot_blast_out(b_result)
