#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datasets import Chapman
from main_linear_ecg import set_model
from util import plot_tsne

def plot_tsne_all():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--model', type=str, default='CLOCSNET')
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'CMSC'], help='choose method')
    parser.add_argument('--ckpt', type=str,
                        default='./save/SupCon/chapman_models/SupCon_chapman_CLOCSNET_lr_0.1_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last-0228-supcon.pth',
                        #default='./save/SupCon/chapman_models/SupCE_chapman_CLOCSNET_lr_0.1_decay_0.0001_bsz_1024_trial_0_warm/last-0228-ce.pth',
                        help='path to pre-trained model')
    opt = parser.parse_args()
    opt.n_cls = 4
    dataset = Chapman(opt=opt)
    model, classifier, criterion = set_model(opt)
    print('model is ok')
    features, _ = model(dataset.data)
    print('feature is ok')
    plot_tsne(features, dataset.label, title='supcontimewarp-tsne')
    print('plot is ok')


if __name__ == '__main__':
    plot_tsne_all()
    print('success')