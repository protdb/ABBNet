import os
import time

import torch
import torch.optim as optim
from tqdm import tqdm
from model.base_model import ABBModel
from finetune.finetune_dataset import get_loaders, STRIDE_LETTERS
import torch.nn as nn
from config.config import BaseConfig
from sklearn.metrics import confusion_matrix

from pdb_datasets.datasets import SAML
from trainers.metrics import mmMetrics


class TrainerBase(object):
    def __init__(self):
        self.config = BaseConfig()
        self.batch_size = self.config.batch_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        loaders = get_loaders()
        self.loaders = {'train': loaders[0], 'val': loaders[1]}
        self.model = ABBModel().float()
        self.model.to(self.device)
        self.n_epochs = self.config.train_epochs
        self.model_path = self.config.get_model_path()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        self.loss_fn = nn.NLLLoss()

    def load_model(self):
        if os.path.exists(self.model_path):
            print(self.model_path)
            self.model.load_state_dict(torch.load(self.model_path), strict=False)
            print(f'Model loaded {self.model_path}')
        else:
            print(f'Model not found {self.model_path}')

    def save_model(self):
        state_dict = self.model.state_dict()
        torch.save(state_dict, self.model_path)

    def get_loss_metrics(self, logit, target, labels_size):
        loss = self.loss_fn(logit, target)
        predicted_ = torch.argmax(logit, dim=-1).detach().cpu().numpy()
        true_ = target.detach().cpu().numpy()
        correct = (predicted_ == true_).sum()
        metric = {
            'correct': correct,
            'loss': loss.cpu().item()
        }
        return metric, loss

    def one_step(self, item, phase):
        alphabet = item.alphabet
        stride = item.stride
        logit_alphabet, logit_stride = self.model(item)
        alphabet_metric, alphabet_loss = self.get_loss_metrics(logit_alphabet, alphabet, labels_size=len(SAML))
        stride_metric, stride_loss = self.get_loss_metrics(logit_stride, stride, labels_size=len(STRIDE_LETTERS))
        total_loss = stride_loss + alphabet_loss
        metric_rec = {
            'num_nodes': len(alphabet),
            'items_metric': {'alphabet':alphabet_metric, 'stride':stride_metric}
        }

        return total_loss, metric_rec

    def train_model(self):
        since = time.time()
        best_result = 1e4
        for epoch in range(self.n_epochs):
            for phase in ['train', 'val']:
                if epoch % self.config.eval_models_every != 0 and phase == 'val':
                    continue
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0

                metric = mmMetrics(phase=phase, epoch=epoch)

                for item in tqdm(self.loaders[phase], total=len(self.loaders[phase])):
                    batch_size = int(item.batch.max()) + 1
                    item = item.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        loss, metrics_rec = self.one_step(item, phase)
                        running_loss += loss.detach().item() * batch_size
                        metric.push_step_metric(metrics_rec)
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1, error_if_nonfinite=True)
                            self.optimizer.step()
                epoch_result = metric.log_results()
                if phase == 'val':
                    if epoch_result < best_result:
                        best_result = epoch_result
                        self.save_model()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val f1: {:4f}'.format(best_result))


def train_abb_model():
    trainer = TrainerBase()
    trainer.load_model()
    trainer.train_model()


if __name__ == '__main__':
    train_abb_model()
