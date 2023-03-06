import os.path
import torch

from finetune.finetune_dataset import STRIDE_LETTERS
from model_processor.processor_datasets import get_batch_from_file, get_loader_from_folder
from pdb_datasets.datasets import SAML, PAD_SEQ
from Bio.PDB.Polypeptide import aa1

SURF_GROUP = {
    'H': ['A', 'Y', 'B', 'C', 'D'],
    'h': ['G', 'I', 'L'],
    'S': ['E', 'F', 'H'],
    's': ['K', 'N'],
    'O': ['S', 'T', 'V', 'W', 'X', 'M', 'P', 'Q', 'R']

}


class ModelProcessor(object):
    def __init__(self, model):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.model.eval()

    def process_file(self, filepath, chain):
        results = {}
        assert os.path.exists(filepath)
        batch = get_batch_from_file(filepath, chain)
        batch.file_idx = ['file_chain']

        if batch is None:
            return results
        results = self.process_batch(batch)
        return results[0]

    def process_batch(self, batch):
        batch_size = batch.batch.max() + 1
        batch_size = batch_size.item()
        result = self.predict(batch)
        file_ids = batch.file_idx
        sequences = batch.sequence
        alphabet_, stride_, embeddings = result
        batch_results = []
        x = batch.x

        for batch_id in range(batch_size):
            mask = [batch.batch == batch_id]
            sequence = sequences[mask]
            file_idx = file_ids[batch_id]
            pdb_id, chain = file_idx.split('_')
            alphabet_logit = alphabet_[mask]
            stride_logit = stride_[mask]
            embedding = embeddings[mask]
            coo = x[mask]

            out_rec = {
                'pdb_id': pdb_id,
                'chain': chain,
                'alphabet': self.convert_to_alphabet(alphabet_logit),
                'stride': self.convert_to_stride(stride_logit),
                'sequence': self.convert_sequence(sequence.cpu().detach().numpy()),
                'source_coo': coo,
                'embedding': embedding,
            }
            batch_results.append(out_rec)
        return batch_results

    def predict(self, batch):
        with torch.no_grad():
            batch.to(self.device)
            out = self.model(batch)
        return out

    @staticmethod
    def convert_to_alphabet(logit):
        alphabet = torch.argmax(logit, dim=-1).detach().cpu().numpy()
        s = ''.join(SAML[i] for i in alphabet)
        s = s.replace(PAD_SEQ, '')
        return s

    @staticmethod
    def convert_to_stride(logit):
        stride = torch.argmax(logit, dim=-1).detach().cpu().numpy()
        s = ''.join(STRIDE_LETTERS[i] for i in stride)
        return s

    @staticmethod
    def convert_to_group(alphabet):
        surf_group = ''
        for s in alphabet:
            for gr in SURF_GROUP:
                if s in SURF_GROUP[gr]:
                    surf_group += gr
        return surf_group

    @staticmethod
    def convert_sequence(arr):
        s = ''
        try:
            for i in arr:
                s += aa1[i]
        except IndexError:
            s += 'X'
        return s
