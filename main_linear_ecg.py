from __future__ import print_function

import sys
import argparse
import time
import math
import os

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

#from main_ce import set_loader
from main_ce_ecg import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, calculate_auc, calculate_other_metrics
from util import set_optimizer
from util import plot_ecg
from networks.resnet_big import SupConResNet, LinearClassifier
from networks.CLOCSNET import cnn_network_contrastive, linear_classifier
from networks.TCN import TCN

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'chapman'], help='dataset')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'CMSC', 'CMSC-P'], help='choose method')

    # leads of data
    parser.add_argument('--lead', type=int, default=1, help='choose method')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'chapman':
        opt.n_cls = 4
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    if opt.model == 'resnet50':
        model = SupConResNet(name='resnet50_ecg')
    elif opt.model == 'CLOCSNET':
        model = cnn_network_contrastive(
            dropout_type='drop1d',
            p1=0.1,
            p2=0.1,
            p3=0.1,
            embedding_dim=128,
            device=(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        )
    elif opt.model == 'TCN':
        model = TCN(
            num_channels=[32, 32],
            embedding_dim=128,
            kernel_size=7,
            dropout=0.2
        )
    else:
        raise ValueError('model not supported: {}'.format(opt.model))
    criterion = torch.nn.CrossEntropyLoss()

    if opt.model == 'resnet50':
        classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    elif opt.model in ['CLOCSNET', 'TCN']:
        classifier = linear_classifier(
            feat_dim=320, # 128 320
            num_classes=opt.n_cls
        )
    else:
        raise ValueError('model not supported: {}'.format(opt.model))

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            if opt.model == 'resnet50':
                model.encoder = torch.nn.DataParallel(model.encoder)
            elif opt.model in ['CLOCSNET', 'TCN']:
                model = torch.nn.DataParallel(model)
            else:
                raise ValueError('model not supported: {}'.format(opt.model))
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels, pids) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        #plot_ecg(images[0], sample_rate=500)

        # 如果使用CMSC，需要把5000的数据截成前后各2500的两段
        # TODO:待处理

        if opt.method in ['CMSC', 'CMSC-P']:
            length = images.shape[2] // 2
            images = torch.split(images, [length, length], dim=2)
            images = torch.cat([images[0], images[1]], dim=0)
            labels = torch.cat([labels, labels], dim=0)
        #elif opt.method == 'CMSC-P':
            #images = images.reshape(-1, 1, 2500, 2)
        #    images = torch.cat([images[0], images[1]], dim=3)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            #features = model.encoder(images)
            if opt.model == 'resnet50':
                features = model.encoder(images)
            elif opt.model == 'CLOCSNET':
                features, _ = model(images)
                #_, features = model(images)
            elif opt.model == 'TCN':
                features = model(images)
            else:
                raise ValueError('model not supported: {}'.format(opt.model))
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        #acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        #top1.update(acc1[0], bsz)
        acc1 = accuracy(output, labels)[0]
        top1.update(acc1[0], bsz)
        auc = calculate_auc(n_class=opt.n_cls,
                            outputs_list=output,
                            labels_list=labels)
        precision, recall, f1 = calculate_other_metrics(output, labels)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Auc {auc:.3f}\t'
                  'precision {precision:.3f}\t'
                  'recall {recall:.3f}\t'
                  'f1 {f1:.3f}'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, auc=auc, precision=precision,
                    recall=recall, f1=f1
            ))
            sys.stdout.flush()

    return losses.avg, top1.avg, auc, precision, recall, f1


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, pids) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # 如果使用CMSC，需要把5000的数据截成前后各2500的两段
            # TODO:待处理
            if opt.method in ['CMSC', 'CMSC-P']:
                length = images.shape[2] // 2
                images = torch.split(images, [length, length], dim=2)
                images = torch.cat([images[0], images[1]], dim=0)
                labels = torch.cat([labels, labels], dim=0)
            #elif opt.method == 'CMSC-P':
                # images = images.reshape(-1, 1, 2500, 2)
            #    images = torch.cat([images[0], images[1]], dim=3)


            # forward
            if opt.model == 'resnet50':
                output = classifier(model.encoder(images))
            elif opt.model == 'CLOCSNET':
                output = classifier(model(images)[0])
            elif opt.model == 'TCN':
                output = classifier(model(images))
            else:
                raise ValueError('model not supported: {}'.format(opt.model))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            #acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            #top1.update(acc1[0], bsz)

            acc1 = accuracy(output, labels)[0]
            top1.update(acc1[0], bsz)

            auc = calculate_auc(n_class=opt.n_cls,
                                outputs_list=output,
                                labels_list=labels)
            precision, recall, f1 = calculate_other_metrics(output, labels)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Auc {auc:.3f}\t'
                      'precision {precision:.3f}\t'
                      'recall {recall:.3f}\t'
                      'f1 {f1:.3f}'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, auc=auc,
                        precision=precision, recall=recall, f1=f1
                ))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Auc {auc:.3f}'.format(auc=auc))
    print(' * precision {precision:.3f}'.format(precision=precision))
    print(' * recall {recall:.3f}'.format(recall=recall))
    print(' * f1 {f1:.3f}'.format(f1=f1))
    return losses.avg, top1.avg, auc, precision, recall, f1


def main():
    best_acc = 0
    best_auc = 0
    best_loss = 1e5
    metrics = dict()
    best_acc_acc = 0
    best_acc_auc = 0
    best_auc_acc = 0
    best_auc_auc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # build tensorboardX
    writer = SummaryWriter(comment='linear')

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc, auc, precision, recall, f1 = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}, auc:{:.2f}'.format(
            epoch, time2 - time1, acc, auc))

        #writer.add_graph(model, input_to_model=None, verbose=False)
        #writer.add_graph(classifier, input_to_model=None, verbose=False)
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        writer.add_scalar('train_auc', auc, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # eval for one epoch
        loss, val_acc, val_auc, val_precision, val_recall, val_f1 = validate(val_loader, model, classifier, criterion, opt)
        writer.add_scalar('val_loss', loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('val_auc', val_auc, epoch)



        if loss < best_loss:
            metrics['acc'] = val_acc
            metrics['auc'] = val_auc
            metrics['precision'] = val_precision
            metrics['recall'] = val_recall
            metrics['f1'] = val_f1


        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_acc = val_acc
            best_acc_auc = val_auc
        if val_auc > best_auc:
            best_auc = val_auc
            best_auc_acc = val_acc
            best_auc_auc = val_auc


    #print('best accuracy: {:.2f}'.format(best_acc))
    #print('best auc: {:.2f}'.format(best_auc))
    print('accuracy: {:.4f}'.format(metrics['acc']))
    print('auc: {:.4f}'.format(metrics['auc']))
    print('precision: {:.4f}'.format(metrics['precision']))
    print('recall: {:.4f}'.format(metrics['recall']))
    print('f1: {:.4f}'.format(metrics['f1']))

    print('best_acc: acc {:.4f}, auc {:.4f}'.format(best_acc_acc, best_acc_auc))
    print('best_auc: acc {:.4f}, auc {:.4f}'.format(best_auc_acc, best_auc_auc))

if __name__ == '__main__':
    main()
