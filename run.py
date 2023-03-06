from fast_search.pdb_preprocessor import build_test_db
from fast_search.search_engine import FastSearch
from finetune.rebuild_database import rebuild_database
from trainers.train_model import train_abb_model


def start_task(task_id,
               pdb_file,
               chain,
               callback_fn,
               e_value_trash='auto',
               page_size=32):


        search_engine=FastSearch(task_id=task_id,
                                 pdb_file=pdb_file,
                                 chain=chain,
                                 e_value_trash=e_value_trash,
                                 page_size=page_size
                                 )
        search_engine.run_search(callback_fn=callback_fn)


def test_callback(msg):
    print(msg)

""" message = {
                'task_id': input task id 
                'is_last_msg': True if last message of total
                'n_count': Total processed items
                'total':  Total items 
                'data': [List of results]
            }   
            
            results format:
            {
                pdb_id': PDB ID,
                'chain': Chain,
                'position':  (start, end)# (-1, -1) for full chain
                'e_value': Statistical significance
                'fasta': FASTA
                'rmsd': RMSD,
                'fasta_identity_score':  FASTA identity,
                'sup_matrix': {'apply_to':  # str 'source' or 'subj' -- structure to apply impose matrix
                                'rotation': Rotation matrix  
                                'translation': Translation matrix} 
                                
                                ## Formula:  coord = rotation_mx.dot(atom.coord.T).T + translation_mx
            }
"""

## To generate test preprocessed DB:

def create_test_db():
    test_db_ids = [('2ko3', 'A'), ('2ocs', 'A'), ('1tit', 'A')] # PDB ID - Chain list
                                                                # Path to PDB files storage  class BaseConfig(object):
                                                                #                    pdb_dir = '/home/dp/Data/PDB/'
    build_test_db(test_db_ids)




def test_search():
    test_file = '/home/dp/Data/PDB/1ihv.pdb'
    start_task(task_id='1ihv',
               pdb_file=test_file,
               chain='A',
               callback_fn=test_callback
               )


if __name__ == '__main__':
   test_search()



