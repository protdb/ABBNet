from abbconfig import ABBNetConfig
from worker_framework import TaskTransfer
from typing import Callable, Dict
import json
from Bio.Align import PairwiseAligner
import pika
import logging
logging.getLogger("pika").setLevel(logging.WARNING)


def get_align_params(sequences):
    aligner = PairwiseAligner()
    aligner.mode = 'global'

    try:
        alns = aligner.align(sequences['source'], sequences['subj'])
        align = alns.__next__()
        score = align.score / len(sequences['source'])
        aln = str(align).split('\n')
    except StopIteration:
        return {
            "score": 0,
            "source_line": sequences['source'],
            "compare_line": sequences['subj'],
            "highlights": []
        }
    print(aln)
    highlights = []
    for idx, letter in enumerate(aln[0]):
        try:
            if aln[2][idx] == letter and letter not in ('-', ' '):
                highlights.append(idx)
        except IndexError:
            pass
    return {
        "score": score,
        "source_line": aln[0],
        "compare_line": aln[2],
        "highlights": highlights
    }


def push_result(cfg: ABBNetConfig, message: Dict) -> None:
    params = pika.connection.URLParameters(cfg.rmq_upload_uri_string)
    with pika.BlockingConnection(params) as conn:
        chan = conn.channel()
        chan.basic_publish(
            exchange=cfg.rmq_upload_exchange,
            routing_key='',
            body=json.dumps(message).encode('utf8'),
            properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent)
        )


def get_result_callback(cfg: ABBNetConfig, task: TaskTransfer) -> Callable[[Dict], None]:
    closure_data = {
        'total_sent': False
    }

    def callback(msg):
        logging.debug(json.dumps(msg))
        if not closure_data['total_sent']:
            push_result(cfg, {
                'task_type': 'SEARCH_UPDATE_TOTAL',
                'task_id': task.id,
                'total': msg['total']
            })
            closure_data['total_sent'] = True
        for struct in msg['data']:
            if struct['position'][0] == struct['position'][1]:
                struct['position'] = (None, None)

            result = {
                    'task_type': 'FOUND_STRUCT',
                    'experiment_id': struct['pdb_id'],
                    'chain_id': struct['chain'],
                    'start': struct['position'][0],
                    'end': struct['position'][1],
                    'e_value': struct['e_value'],
                    'name': task.search_parameters.name,
                    'task_id': task.id,
                    'fasta_score': struct['fasta_identity_score'],
                    'rmsd': struct['rmsd'],
                    'alignments': {
                        "fasta": get_align_params(struct['fasta']),
                        "stride": get_align_params(struct['stride']),
                        "sup_matrix": struct['sup_matrix']
                    }
                }
            # pushing result to RMQ
            push_result(cfg, result)

    return callback
