import os.path
import torch
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
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def process_file(self, filepath, chain):
        results = {}
        assert os.path.exists(filepath)
        batch = get_batch_from_file(filepath, chain)
        if batch is None:
            return results
        logit, embedding = self.predict(batch)
        predicted_ = torch.argmax(logit, dim=-1).detach().cpu().numpy()
        alphabet = self.convert_to_alphabet(predicted_)
        surf_group = self.convert_to_group(alphabet)
        sequence = batch.sequence.detach().cpu().numpy()
        sequence = self.convert_sequence(sequence)
        results = {
            'file': os.path.basename(filepath),
            'chain': chain,
            'sequence': sequence,
            'alphabet': alphabet,
            'groups': surf_group,
            'embedding': embedding,
            'source_coo': batch.x
        }
        return results

    def process_folder(self, folder):
        prediction_results = {}
        loader = get_loader_from_folder(folder)
        for batch in loader:
            batch_id = batch.batch
            logit = self.predict(batch)
            predicted_ = torch.argmax(logit, dim=-1).detach().cpu().numpy()
            batch_size = batch_id.max() + 1
            for i in range(batch_size):
                mask = batch_id == i
                file_idx = batch.file_idx[i].cpu().detach().item()
                result = predicted_[mask]
                alphabet = self.convert_to_alphabet(result)
                prediction_results.update({file_idx: alphabet})
        return prediction_results

    def predict(self, batch):
        self.model.eval()
        batch.to(self.device)
        with torch.no_grad():
            logit, embedding = self.model(batch)
        return logit, embedding

    @staticmethod
    def convert_to_alphabet(arr):
        s = ''.join(SAML[i] for i in arr)
        s = s.replace(PAD_SEQ, '')
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
