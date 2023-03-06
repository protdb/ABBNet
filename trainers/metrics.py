import sys
from pathlib import Path

import numpy as np

from config.config import BaseConfig
from pdb_datasets.datasets import SAML


class mmMetrics(object):
    def __init__(self,
                 epoch,
                 phase):
        self.epoch = epoch
        self.phase = phase
        self.total_loss = 0.0
        self.total_corrects_stride = 0.0
        self.total_corrects_alphabet = 0.0
        self.stride_loss = 0.0
        self.alphabet_loss = 0.0
        self.total_nodes = 0
        config = BaseConfig()
        self.print_to_file = config.log_result == 'file'
        if self.print_to_file:
            self.metric_file = config.get_metric_log_file()

    def push_step_metric(self, metric_rec):
        items_metric = metric_rec['items_metric']
        self.stride_loss += items_metric['stride']['loss'] * metric_rec['num_nodes']
        self.alphabet_loss += items_metric['alphabet']['loss'] * metric_rec['num_nodes']
        self.total_loss += self.stride_loss + self.alphabet_loss
        self.total_corrects_stride += items_metric['stride']['correct']
        self.total_corrects_alphabet += items_metric['alphabet']['correct']
        self.total_nodes += metric_rec['num_nodes']

    def print_metrics(self):
        print('*' * 32)
        print(f'Epoch: {self.epoch} Phase: {self.phase}')
        print(f'Total loss: {self.total_loss / self.total_nodes:4f}')
        print(f'Correct (stride): {self.total_corrects_stride:2f}')
        print(f'Correct (alphabet): {self.total_corrects_alphabet:2f}')
        print(f'Acc (stride): {self.total_corrects_stride / self.total_nodes:4f}')
        print(f'Acc (alphabet): {self.total_corrects_alphabet / self.total_nodes:4f}')
        print(f'Loss (stride): {self.stride_loss / self.total_nodes:4f}')
        print(f'Loss (alphabet): {self.alphabet_loss / self.total_nodes:4f}')
        print()
        print()
        return self.total_loss / self.total_nodes

    def log_results(self):
        original_stdout = sys.stdout
        if self.print_to_file:
            with open(str(self.metric_file), 'a') as f:
                sys.stdout = f
                result = self.print_metrics()
        else:
            result = self.print_metrics()
        sys.stdout = original_stdout
        return result
