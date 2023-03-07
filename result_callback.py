from abbconfig import ABBNetConfig
from worker_framework import TaskTransfer
from typing import Callable
import json
from Bio.pairwise2 import align
import pika
import logging

logging.getLogger("pika").setLevel(logging.WARNING)


def get_align_params(sequences):
    aln = align.globalxd(sequences['source'], sequences['subj'], -1, -1, -0.5, -0.5)[0]
    highlights = []
    for idx, letter in enumerate(aln.seqA):
        if aln.seqB[idx] == letter:
            highlights.append(idx)
    return {
        "score": aln.score / len(sequences['source']),
        "source_line": aln.seqA,
        "compare_line": aln.seqB,
        "highlights": highlights
    }


def get_result_callback(cfg: ABBNetConfig, task: TaskTransfer) -> Callable:

    def callback(msg):
        logging.debug(json.dumps(msg))
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
            params = pika.connection.URLParameters(cfg.rmq_upload_uri_string)
            with pika.BlockingConnection(params) as conn:
                chan = conn.channel()
                chan.basic_publish(
                    exchange=cfg.rmq_upload_exchange,
                    routing_key='',
                    body=json.dumps(result).encode('utf8'),
                    properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent)
                )

    return callback
