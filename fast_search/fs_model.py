import torch
import torch.nn as nn

class FastSearchModel(nn.Module):
    def __init__(self,
                 encoder_model,
                 source_embedding,
                 source_coo
                 ):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.encoder = encoder_model
        self.source_embedding = source_embedding
        self.source_coo = source_coo
        self.encoder.to(self.device)
        self.encoder.eval()

    def forward(self, batch):
        results = []

        with torch.no_grad():
            batch.to(self.device)
            embeddings = self.encoder(batch, return_embedding=True)
            coo = batch.x
            batch_size = batch.batch.max() + 1
            for batch_id in range(batch_size):
                mask = [batch.batch == batch_id]
                subj_embedding = embeddings[mask]
                subj_coo = coo[mask]
                assert len(subj_embedding) == len(subj_coo)
                inference_record = self.inference(subj_embedding, subj_coo)
                file_idx = batch.file_idx[batch_id]
                inference_record.update({'file_idx': file_idx.detach().item()})
                results.append(inference_record)

        return results

    def inference(self, subj_embedding, subj_coo):
        select_idx, apply_to = self.__find_align_position(self.source_embedding, subj_embedding)
        reference_coo = self.source_coo
        target_coo = subj_coo

        if apply_to == 'subj':
            reference_coo, target_coo = target_coo, reference_coo

        target_coo = target_coo[select_idx:select_idx + len(reference_coo), :]
        assert len(reference_coo) == len(target_coo)

        rx, tx =self.__svd_impose(target_coo, reference_coo)
        rmsd = self.align(reference_coo, target_coo, rx, tx)
        inference_record = {
            'apply_to': apply_to,
            'select_idx_start': select_idx,
            'select_idx_end': select_idx + len(reference_coo),
            'rotation_mx': rx.detach().cpu().numpy(),
            'translation_mx': tx.detach().cpu().numpy(),
            'rmsd': rmsd.detach().cpu().item(),
            'file_idx': -1
        }
        return inference_record


    @staticmethod
    def __svd_impose(reference_coo, target_coo):
        a_mean = reference_coo.mean(axis=0)
        b_mean = target_coo.mean(axis=0)
        a_centered = reference_coo - a_mean
        b_centered = target_coo - b_mean
        covariance_mx = a_centered.T.mm(b_centered)
        u, s, v = torch.svd(covariance_mx)
        rotation_mx = v.mm(u.T)

        if torch.linalg.det(rotation_mx) < 0:
             v[2] = -v[2]
             rotation_mx = v.mm(u.T)

        translation_mx = b_mean[None, :] - rotation_mx.mm(a_mean[None, :].T).T
        translation_mx = translation_mx.T
        return rotation_mx, translation_mx.squeeze()

    @staticmethod
    def align(reference_coo, target_coo, rx, tx):
        target_aligned = (rx.mm(target_coo.T)).T + tx
        rmsd = torch.sqrt(((target_aligned - reference_coo) ** 2).sum(axis=1).mean())
        return rmsd

    @staticmethod
    def __find_align_position(source_emb, subj_emb):
        source_size = source_emb.size(0)
        apply_to = 'subj'
        subj_size = subj_emb.size(0)
        if source_size >= subj_size:
            align_base = source_emb
            align_subj = subj_emb
        else:
            align_base = subj_emb
            align_subj = source_emb
            apply_to = 'source'

        idx = 0
        min_distance = 1e8
        select_idx = -1
        while idx + len(align_subj) <= len(align_base):
            region = align_base[idx:idx + len(align_subj)]
            distance = torch.pairwise_distance(region, align_subj).sum()
            if min_distance > distance:
                min_distance = distance
                select_idx = idx
            idx += 1
        assert select_idx >= 0
        return select_idx, apply_to


