"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from itertools import combinations

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        a = anchor_feature.detach().cpu().numpy()
        b = contrast_feature.T.detach().cpu().numpy()
        c = anchor_dot_contrast.detach().cpu().numpy()
        d = np.matmul(a, b)


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def testNan(self, x):
        x = x.detach().cpu().numpy()
        return np.isnan(x).any()

# CLOCS 中用于对比学习的loss
def obtain_contrastive_loss(latent_embeddings, pids, trial):
    """ Calculate NCE Loss For Latent Embeddings in Batch
    Args:
        latent_embeddings (torch.Tensor): embeddings from model for different perturbations of same instance (BxHxN)
        pids (list): patient ids of instances in batch
    Outputs:
        loss (torch.Tensor): scalar NCE loss
    """
    if trial in ['CMSC', 'CMLC', 'CMSMLC']:
        pids = np.array(pids, dtype=np.object)
        pid1, pid2 = np.meshgrid(pids, pids)
        pid_matrix = pid1 + '-' + pid2
        pids_of_interest = np.unique(pids + '-' + pids)  # unique combinations of pids of interest i.e. matching
        bool_matrix_of_interest = np.zeros((len(pids), len(pids)))
        for pid in pids_of_interest:
            bool_matrix_of_interest += pid_matrix == pid
        rows1, cols1 = np.where(np.triu(bool_matrix_of_interest, 1))
        rows2, cols2 = np.where(np.tril(bool_matrix_of_interest, -1))

    nviews = set(range(latent_embeddings.shape[2]))
    view_combinations = combinations(nviews, 2)
    loss = 0
    ncombinations = 0
    loss_terms = 2
    # 如果报错误 UnboundLocalError: local variable 'loss_terms' referenced before assignment
    # 那就重启PyCharm吧！
    for combination in view_combinations:
        view1_array = latent_embeddings[:, :, combination[0]]  # (BxH)
        view2_array = latent_embeddings[:, :, combination[1]]  # (BxH)
        norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
        norm2_vector = view2_array.norm(dim=1).unsqueeze(0)
        sim_matrix = torch.mm(view1_array, view2_array.transpose(0, 1))
        norm_matrix = torch.mm(norm1_vector.transpose(0, 1), norm2_vector)
        temperature = 0.1
        argument = sim_matrix / (norm_matrix * temperature)
        sim_matrix_exp = torch.exp(argument)

        if trial == 'CMC':
            """ Obtain Off Diagonal Entries """
            # upper_triangle = torch.triu(sim_matrix_exp,1)
            # lower_triangle = torch.tril(sim_matrix_exp,-1)
            # off_diagonals = upper_triangle + lower_triangle
            diagonals = torch.diag(sim_matrix_exp)
            """ Obtain Loss Terms(s) """
            loss_term1 = -torch.mean(torch.log(diagonals / torch.sum(sim_matrix_exp, 1)))
            loss_term2 = -torch.mean(torch.log(diagonals / torch.sum(sim_matrix_exp, 0)))
            loss += loss_term1 + loss_term2
            loss_terms = 2
        elif trial == 'SimCLR':
            self_sim_matrix1 = torch.mm(view1_array, view1_array.transpose(0, 1))
            self_norm_matrix1 = torch.mm(norm1_vector.transpose(0, 1), norm1_vector)
            temperature = 0.1
            argument = self_sim_matrix1 / (self_norm_matrix1 * temperature)
            self_sim_matrix_exp1 = torch.exp(argument)
            self_sim_matrix_off_diagonals1 = torch.triu(self_sim_matrix_exp1, 1) + torch.tril(self_sim_matrix_exp1, -1)

            self_sim_matrix2 = torch.mm(view2_array, view2_array.transpose(0, 1))
            self_norm_matrix2 = torch.mm(norm2_vector.transpose(0, 1), norm2_vector)
            temperature = 0.1
            argument = self_sim_matrix2 / (self_norm_matrix2 * temperature)
            self_sim_matrix_exp2 = torch.exp(argument)
            self_sim_matrix_off_diagonals2 = torch.triu(self_sim_matrix_exp2, 1) + torch.tril(self_sim_matrix_exp2, -1)

            denominator_loss1 = torch.sum(sim_matrix_exp, 1) + torch.sum(self_sim_matrix_off_diagonals1, 1)
            denominator_loss2 = torch.sum(sim_matrix_exp, 0) + torch.sum(self_sim_matrix_off_diagonals2, 0)

            diagonals = torch.diag(sim_matrix_exp)
            loss_term1 = -torch.mean(torch.log(diagonals / denominator_loss1))
            loss_term2 = -torch.mean(torch.log(diagonals / denominator_loss2))
            loss += loss_term1 + loss_term2
            loss_terms = 2
        elif trial in ['CMSC', 'CMLC', 'CMSMLC']:  # ours #CMSMLC = positive examples are same instance and same patient
            triu_elements = sim_matrix_exp[rows1, cols1]
            tril_elements = sim_matrix_exp[rows2, cols2]
            diag_elements = torch.diag(sim_matrix_exp)

            triu_sum = torch.sum(sim_matrix_exp, 1)
            tril_sum = torch.sum(sim_matrix_exp, 0)

            loss_diag1 = -torch.mean(torch.log(diag_elements / triu_sum))
            loss_diag2 = -torch.mean(torch.log(diag_elements / tril_sum))

            loss_triu = -torch.mean(torch.log(triu_elements / triu_sum[rows1]))
            loss_tril = -torch.mean(torch.log(tril_elements / tril_sum[cols2]))

            loss = loss_diag1 + loss_diag2
            loss_terms = 2

            if len(rows1) > 0:
                loss += loss_triu  # technically need to add 1 more term for symmetry
                loss_terms += 1

            if len(rows2) > 0:
                loss += loss_tril  # technically need to add 1 more term for symmetry
                loss_terms += 1

            # print(loss,loss_triu,loss_tril)

        ncombinations += 1
    loss = loss / (loss_terms * ncombinations)
    return loss