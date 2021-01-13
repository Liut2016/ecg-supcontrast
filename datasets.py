#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def normalization1(array):
    maxcols=array.max(axis=1)
    mincols=array.min(axis=1)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t

def normalization(data, label):

    # 去掉数据中的NAN
    data[np.isnan(data)] = 0

    max_arr = data.max(axis=1).reshape(data.shape[0], 1)
    min_arr = data.min(axis=1).reshape(data.shape[0], 1)
    ranges = max_arr - min_arr

    # 去除数据没有变化的行
    line = np.where(ranges==0)[0]
    data = np.delete(data, line, 0)
    min_arr = np.delete(min_arr, line, 0)
    ranges = np.delete(ranges, line, 0)
    label = np.delete(label, line, 0)


    # 防止除数为0
    #x = np.where(ranges==0)
    #ranges[np.where(ranges==0)] = 1

    # 归一化
    #norDataSet = np.zeros(data.shape)
    m = data.shape[1]
    norDataSet = data - np.tile(min_arr, (m))
    norDataSet = norDataSet/np.tile(ranges,(m))
    return norDataSet, label

# chapman
class Chapman(Dataset):
    def __init__(self, opt,
                 path='./data/chapman_ecg/contrastive_ss/leads_[\'II\', \'V2\', \'aVL\', \'aVR\']',
                 train=True,
                 transform=None,
                 target_transform=None):
        with open(os.path.join(path, 'frames_phases_chapman.pkl'), 'rb') as f:
            data = pickle.load(f)
            data = data['ecg'][1]
        with open(os.path.join(path, 'labels_phases_chapman.pkl'), 'rb') as f:
            label = pickle.load(f)
            label = label['ecg'][1]

        if train:
            data = np.concatenate((data['train']['All Terms'], data['val']['All Terms']), axis=0)
            label = np.concatenate((label['train']['All Terms'], label['val']['All Terms']), axis=0)
        else:
            data = data['test']['All Terms']
            label = label['test']['All Terms']

        # 归一化
        data, label = normalization(data, label)
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).long()

        if opt.model == 'resnet50':
            #for resnet-50
            data = data.reshape(-1, 1, 1, 2500)
            data = data.permute((0, 2, 3, 1))   # HWC
        elif opt.model == 'CNN':
            # for CNN
            data = data.reshape(-1, 1, 2500)
            data = data.permute((0, 2, 1))
        elif opt.model == 'CLOCSNET':
            # for CLOCSNET
            data = data.reshape(-1, 1, 2500)
            data = data.unsqueeze(3)
        else:
            raise ValueError('model not supported: {}'.format(opt.model))

        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

        #print(data.shape)
        #print(label.shape)
        pass

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]
        
        if self.transform is not None:
            data = self.transform(data)
            
        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    dataset = Chapman()

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, shuffle=True,
        num_workers=16, pin_memory=True, sampler=None)
    for i, (data, label) in enumerate(train_loader):
        print(i)
        print(data.shape)
        print(label.shape)
        break

    print('success')