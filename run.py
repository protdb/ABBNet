from search.search_worker import SearchWorker

# Test worker

TEST_FILE1 = "/home/dp/Data/SAML/test/pdb/sample1/1dyw.pdb"  # Barrel
TEST_FILE2 = "/home/dp/Data/SAML/test/pdb/sample4/1A1B.pdb"  # Beta-hairpin

if __name__ == '__main__':

    if __name__ == "__main__":
        worker = SearchWorker()
        task_id1 = worker.create_task(  # Input params
            pdb_path=TEST_FILE1,  # Source PDB file
            chain='A',  # Chain
            search_mode=0,  # 0- basic search 1 - hierarchy search
            e_value_trash='auto',  # 'auto' or float
            # (maybe select of ['auto', 1.0, 0.1, 0.001, 0.0001, 0.00001, 0.000001]
            page_size=50,  # size of chunk in the output queue
            hierarchy_max_depth=2,  # max depth of hierarchy search
            limit=None  # Maximum size of found results
        )
        worker.start_task(task_id1)
        print('Process 1 started')
        task_id2 = worker.create_task(
            pdb_path=TEST_FILE2,
            chain='A',
            search_mode=1,
            page_size=50,
            limit=100

        )
        print('Process 2 started')

        worker.start_task(task_id2)
        try:
            while True:
                item = worker.read_queue()
                print(item)
                """
                    Queue message:
                    {'task_id' Task id
                    'n_count': processed items from results,
                    'total': total items in results  -1 for hierarchy search mode
                    'data': {key: # pdb_id:chain
                             {'e_value': # E-value score
                             'pdb_id': # pdb id
                             'chain' # chain
                             'position': # residue positions (start, end)  or (-1, -1) for full chain
                             'fasta' 
                             'rmsd': # rmsd - rmsd  source - current file
                             'fasta_identity_score': # aligned fasta identity source - current file 
                             'aligned_source_file': # path to aligned source file if config.upload_result = True
                             'aligned_subj_file': # path to aligned source file if config.upload_result = True
                            }
                    }
                """
        except KeyboardInterrupt:
            worker.stop_worker()
