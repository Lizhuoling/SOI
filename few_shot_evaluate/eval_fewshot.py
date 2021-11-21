from __future__ import print_function

import os
import argparse
import socket
import time
import sys
import pdb

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import DataLoader

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.omniglot import MetaOmniglot
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_list

from models.resnet import resnet12
from models.IN_resnet import IN_resnet12, IN_resnet18, IN_resnet50

torchvision.models.__dict__['resnet12'] = resnet12 # Add resnet12 to model list
torchvision.models.__dict__['in_resnet12'] = IN_resnet12
torchvision.models.__dict__['in_resnet18'] = IN_resnet18
torchvision.models.__dict__['in_resnet50'] = IN_resnet50

from eval.meta_eval import meta_test


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet18', choices = ['resnet12', 'resnet18', 'resnet50', 'in_resnet12', 'in_resnet18', 'in_resnet50'])
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100', 'Omniglot'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--data_root', type=str, default='data', metavar='N',
                        help='Root dataset')
    parser.add_argument('--num_workers', type=int, default=3, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')
    parser.add_argument('--classifier', type = str, default = 'LR',
                        help = 'The selected classifier.')

    opt = parser.parse_args()

    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.data_root = '/data/vision/phillipi/rep-learn/{}'.format(opt.dataset)
        opt.data_aug = True
    elif hostname.startswith('instance'):
        opt.data_root = '/mnt/globalssd/fewshot/{}'.format(opt.dataset)
        opt.data_aug = True
    elif opt.data_root != 'data':
        opt.data_aug = True
    else:
        raise NotImplementedError('server invalid: {}'.format(hostname))

    return opt


def main():

    opt = parse_option()

    # test loader
    args = opt
    args.batch_size = args.test_batch_size
    # args.n_aug_support_samples = 1

    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans,
                                                        fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans,
                                                       fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    elif opt.dataset == 'Omniglot':
        meta_testloader = DataLoader(MetaOmniglot(args=opt, partition='test',
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = None
        n_cls = 30
    else:
        raise NotImplementedError(opt.dataset)
    n_cls = 64  # Fix it when testing the SOI model.
    
    # load model
    model = torchvision.models.__dict__[opt.model](num_classes = n_cls)
    ckpt = torch.load(opt.model_path)
    if 'model' in ckpt.keys():
        state_dict = ckpt['model']    # Be compatible with the few-shot trained model.
    else:
        state_dict = ckpt['state_dict']   # Be compatible with the model unsupervised trained model. 
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    msg = model.load_state_dict(state_dict, strict = False)

    # total = sum([param.nelement() for param in model.parameters()]) # Show the parameter amount
    # print("Number of parameter: %.2fM" % (total/1e6))
    # pdb.set_trace()

    if len(list(model.children())) > 7:
        #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"} 
        model = torch.nn.Sequential(*list(model.children())[:9])
    else:
        model = torch.nn.Sequential(*list(model.children())[:5])
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
 
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    start = time.time()
    test_acc_feat, test_std_feat = meta_test(model, meta_testloader, classifier = opt.classifier, opt = opt)
    test_time = time.time() - start
    print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat,
                                                                         test_std_feat,
                                                                         test_time))


if __name__ == '__main__':
    main()
