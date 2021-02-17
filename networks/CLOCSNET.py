#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:14:18 2020

@author: Dani Kiyasseh
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

# %%
""" Functions in this scripts:
    1) cnn_network_contrastive 
    2) second_cnn_network
"""

# %%

c1 = 1  # b/c single time-series
c2 = 4  # 4
c3 = 16  # 16
c4 = 32  # 32
k = 7  # kernel size #7
s = 3  # stride #3


# num_classes = 3

class cnn_network_contrastive(nn.Module):
    """ CNN for Self-Supervision """

    def __init__(self, dropout_type, p1, p2, p3, nencoders=1, embedding_dim=256, trial='', device=''):
        super(cnn_network_contrastive, self).__init__()

        self.embedding_dim = embedding_dim

        if dropout_type == 'drop1d':
            self.dropout1 = nn.Dropout(p=p1)  # 0.2 drops pixels following a Bernoulli
            self.dropout2 = nn.Dropout(p=p2)  # 0.2
            self.dropout3 = nn.Dropout(p=p3)
        elif dropout_type == 'drop2d':
            self.dropout1 = nn.Dropout2d(p=p1)  # drops channels following a Bernoulli
            self.dropout2 = nn.Dropout2d(p=p2)
            self.dropout3 = nn.Dropout2d(p=p3)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.maxpool = nn.MaxPool1d(2)
        self.trial = trial
        self.device = device

        self.encoder = nn.ModuleList()
        self.view_linear_modules = nn.ModuleList()
        for n in range(nencoders):
            self.encoder.append(nn.Sequential(
                nn.Conv1d(c1, c2, k, s),
                nn.BatchNorm1d(c2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                self.dropout1,
                nn.Conv1d(c2, c3, k, s),
                nn.BatchNorm1d(c3),
                nn.ReLU(),
                nn.MaxPool1d(2),
                self.dropout2,
                nn.Conv1d(c3, c4, k, s),
                nn.BatchNorm1d(c4),
                nn.ReLU(),
                nn.MaxPool1d(2),
                self.dropout3,
                #nn.Flatten()
            ))
            self.view_linear_modules.append(nn.Linear(c4 * 10, self.embedding_dim))

    def forward(self, x):
        """ Forward Pass on Batch of Inputs
        Args:
            x (torch.Tensor): inputs with N views (BxSxN)
        Outputs:
            h (torch.Tensor): latent embedding for each of the N views (BxHxN)
        """
        batch_size = x.shape[0]
        # nsamples = x.shape[2]
        nviews = x.shape[3]
        latent_embeddings = torch.empty(batch_size, c4 * 10, nviews, device=self.device)
        proj_embeddings = torch.empty(batch_size, self.embedding_dim, nviews, device=self.device)
        for n in range(nviews):
            """ Obtain Inputs From Each View """
            h = x[:, :, :, n]
            temp = h
            if self.trial == 'CMC':
                h = self.encoder[n](h)  # nencoders = nviews
                # 因为前面加了nn.Flatten, 所以下面不需要了
                h = torch.reshape(h, (h.shape[0], h.shape[1] * h.shape[2]))
                h = self.view_linear_modules[n](h)
            else:
                h = self.encoder[0](h)  # nencoder = 1 (used for all views)
                # 因为前面加了nn.Flatten, 所以下面不需要了
                h = torch.reshape(h, (h.shape[0], h.shape[1] * h.shape[2]))
                temp = h
                h = self.view_linear_modules[0](h)

            latent_embeddings[:, :, n] = temp
            latent_embeddings = F.normalize(latent_embeddings, dim=1)

            proj_embeddings[:, :, n] = h
            proj_embeddings = F.normalize(proj_embeddings, dim=1)

        return latent_embeddings, proj_embeddings


class second_cnn_network(nn.Module):

    def __init__(self, first_model, noutputs, embedding_dim=256):
        super(second_cnn_network, self).__init__()
        self.first_model = first_model
        self.linear = nn.Linear(embedding_dim, noutputs)

    def forward(self, x):
        _, h = self.first_model(x)
        h = h.squeeze()  # to get rid of final dimension from torch.empty before
        output = self.linear(h)
        return output

class linear_classifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim=c4 * 10, num_classes=4):
        super(linear_classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        features = features.squeeze()
        return self.fc(features)
