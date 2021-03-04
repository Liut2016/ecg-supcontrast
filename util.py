from __future__ import print_function

import math
import os
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import warnings
import ecg_plot
import matplotlib.pyplot as plt
from transforms_ecg import dataReshape, Jitter, Scaling, MagWarp, TimeWarp, Rotation, Permutation, RandSampling

warnings.filterwarnings('once')


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, method=''):
        self.transform = transform
        self.method = method

    def __call__(self, x):
        if self.method == 'CMSC':
            length = x.shape[1] // 2
            r1, r2 = torch.split(x, [length, length], dim=1)
            res = [self.transform(r1), self.transform(r2)]
        else:
            res = [self.transform(x), self.transform(x)]
        return res

class NCropTransform:
    def __init__(self, transform, method='', nviews=2):
        self.transform = transform
        self.method = method
        self.nviews = nviews
        pass

    def __call__(self, x):
        if self.method in ['CMSC', 'CMSC-P']:
            #temp = x.detach().cpu().numpy()
            length = x.shape[1] // self.nviews
            arr = [length for _ in range(self.nviews)]
            r = torch.split(x, arr, dim=1)
            res = [self.transform(i) for i in r]
        elif self.method in ['SimCLR', 'SupCon']:
            res = [self.transform(x), self.transform(x)]
        else:
            raise ValueError('method not supported: {}'.format(self.method))

        #plot_aug(res[0], sample_rate=250, aug='Permutation')
        #plot_ecg(x, sample_rate=250)
        #plot_ecg(res[0], sample_rate=250)
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# https://www.cnblogs.com/caiyishuai/p/9435945.html
# https://blog.csdn.net/u010505915/article/details/106450150
'''
def calculate_auc(n_class,outputs_list,labels_list,dataset=''):
    if torch.is_tensor(outputs_list):
        outputs_list = outputs_list.detach().cpu().numpy()
    if torch.is_tensor(labels_list):
        labels_list = labels_list.detach().cpu().numpy()
    ohe = LabelBinarizer()
    labels_ohe = ohe.fit_transform(labels_list)
    if n_class is not None:
        if n_class != '2':
            all_auc = []
            for i in range(labels_ohe.shape[1]):
                #auc = roc_auc_score(labels_ohe[:,i],outputs_list[:,i])ValueError: multi_class must be in ('ovo', 'ovr')
                auc = roc_auc_score(labels_list, outputs_list, multi_class='ovo')
                all_auc.append(auc)
            epoch_auroc = np.mean(all_auc)
        elif n_class == '2':
            if 'physionet2020' in dataset or 'ptbxl' in dataset:
                """ Use This for MultiLabel Process -- Only for Physionet2020 """
                all_auc = []
                for i in range(labels_ohe.shape[1]):
                    auc = roc_auc_score(labels_ohe[:,i],outputs_list[:,i])
                    all_auc.append(auc)
                epoch_auroc = np.mean(all_auc)
            else:
                epoch_auroc = roc_auc_score(labels_list,outputs_list)
    else:
        print('This is not a classification problem!')
    return epoch_auroc
'''
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
def calculate_auc(n_class,outputs_list,labels_list,dataset=''):
    if torch.is_tensor(outputs_list):
        outputs_list = outputs_list.detach().cpu().numpy()
    if torch.is_tensor(labels_list):
        labels_list = labels_list.detach().cpu().numpy()

    auc = roc_auc_score(labels_list, outputs_list, multi_class='ovo')
    return auc
    pass

def calculate_other_metrics(output, target, average='macro'):
    if torch.is_tensor(output):
        output = output.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    pred = [c.argmax() for c in output]
    precision = precision_score(target, pred, average=average, labels=np.unique(pred))
    recall = recall_score(target, pred, average=average, labels=np.unique(pred))
    f1 = f1_score(target, pred, average=average, labels=np.unique(pred))
    return precision, recall, f1




def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):

    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    '''
    optimizer = optim.Adam(model.parameters(),
                          lr=opt.learning_rate,
                          #momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    '''
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func('EarlyStopping counter: {counter} out of {patience}'.format(counter=self.counter, patience=self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func('Validation loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'.format(val_loss_min=self.val_loss_min, val_loss=val_loss))
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# https://github.com/dy1901/ecg_plot
# https://github.com/Noploop/ecg_plot
# https://www.cnpython.com/pypi/ecg-plot
# https://blog.csdn.net/oscar6280868/article/details/104962792

def plot_ecg(data, sample_rate=500):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if len(data.shape) > 2:
        data = data.squeeze(2)
    ecg_plot.plot_1(data[0], sample_rate=sample_rate, title='test')
    ecg_plot.show()
    #plt.plot(data[0])
    #plt.show()
    pass

def plot_aug(data, sample_rate=500, aug=''):
    #if torch.is_tensor(data):
    #    data = data.detach().cpu().numpy()
    if len(data.shape) > 2:
        data = data.squeeze(2)

    if aug == 'Timewarp':
        augmentation = TimeWarp()
    elif aug == 'Jitter':
        augmentation = Jitter()
    elif aug == 'Scaling':
        augmentation = Scaling()
    elif aug == 'MagWarp':
        augmentation = MagWarp()
    elif aug == 'Permutation':
        augmentation = Permutation(nPerm=10, minSegLength=100)
    elif aug == 'RandSampling':
        augmentation = RandSampling(nSample=2500)
    else:
        raise ValueError('augmentation not supported: {}'.format(aug))


    data_aug = augmentation(data)
    data_aug = data_aug[0]
    data = data[0]
    data = [data, data_aug]
    ecg_plot.plot(data, sample_rate=sample_rate, title=aug, columns=1, lead_index=['origin', aug])
    plt.show()