import os.path
from search.search_worker import SearchWorker

def process_file(filename, chain):
    assert os.path.exists(filename)

    worker = SearchWorker()
    task_id1 = worker.create_task(
        pdb_path=filename,
        chain=chain,
        search_mode=0,
        e_value_trash='auto',
        page_size=50,
        hierarchy_max_depth=2,
        limit=None
        )
    worker.start_task(task_id1)
    print(f'Start {filename}:{chain} started')
    try:
        while True:
                item = worker.read_queue()
                print(f"Found {item['n_count']}' of {item['total']}")
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


if __name__ == '__main__':
    file = '/home/dp/Data/PDB/2spc.pdb'
    process_file(file, 'A')