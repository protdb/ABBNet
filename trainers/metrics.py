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
        self.total_correct = 0.0
        self.total_nodes = 0
        self.confusion = np.zeros((len(SAML), len(SAML)))
        self.batch_size = None
        config = BaseConfig()
        self.print_to_file = config.log_result == 'file'
        if self.print_to_file:
            self.metric_file = config.get_metric_log_file()

    def push_step_metric(self, metric_rec):
        self.total_loss += metric_rec['loss'] * metric_rec['num_nodes']
        self.total_correct += metric_rec['correct']
        self.total_nodes += metric_rec['num_nodes']
        self.confusion += metric_rec['confusion']
        self.batch_size = metric_rec['batch_size']

    def print_metrics(self):
        print('*' * 32)
        print(f'Epoch: {self.epoch} Phase: {self.phase}')
        print(f'Loss: {self.total_loss / self.total_nodes:4f}')
        print(f'Correct: {self.total_correct:2f}')
        print(f'Acc: {self.total_correct / self.total_nodes:4f}')
        print()
        #print(self.confusion)
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
