# -*- coding: utf-8 -*-
import os, time, sys, argparse
from pprint import pprint

import numpy as np
import pandas as pd

import torch, PIL
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

import models
from utils import my_dataset, Logger
from NDCC import NDCC


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def train_NDCC(model, optimizer, scheduler, num_epochs=20):
    for epoch in range(num_epochs):
        since2 = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
        # for phase in ['train']:
            
            model.eval()  # NDCC is alwayes set to evaluate mode

            cnt = 0

            epoch_loss = 0.
            epoch_acc = 0.

            for step, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = (inputs.cuda())
                labels = (labels.long().cuda())

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    if opt.strategy == 1:
                        sigma2 = model.sigma ** 2
                        means = sigma2 * model.classifier.weight

                    elif opt.strategy == 2:
                        sigma2 = (model.sigma + model.delta) ** 2
                        means = model.classifier.weight * sigma2

                    if opt.strategy == 1:
                        loss_MD = ((((outputs - means[labels]) ** 2).sum()) / (sigma2.detach())) / (2 * outputs.shape[0])
                        loss_NLL = (model.dim_embedding * torch.log(sigma2)) / 2 + ((((outputs.detach() - means[labels]) ** 2).sum()) / (sigma2)) / (2 * outputs.shape[0])

                    elif opt.strategy == 2:
                        loss_MD = (torch.div((outputs - means[labels]) ** 2, sigma2.detach())).sum() / (2 * outputs.shape[0])
                        loss_NLL = (torch.log(sigma2).sum()) / 2 + (torch.div((outputs.detach() - means[labels]) ** 2, sigma2).sum() / outputs.shape[0]) / 2

                    logits = nn.parallel.data_parallel(model.classifier, outputs)
                    loss_CE = F.cross_entropy(logits, labels)

                    loss = loss_CE + opt.lmd * (loss_MD + opt.gma * loss_MD)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if step % 100 == 0:
                    print('{} step: {} loss: {:.4f}, loss_CE: {:.4f}, loss_MD: {:.4f}, loss_NLL: {:.4f}'.format(
                        phase, step, loss.item(), loss_CE.item(), loss_MD.item(), loss_NLL.item()))

                # statistics
                _, preds = torch.max(logits, 1)

                epoch_loss = (loss.item() * inputs.size(0) + cnt * epoch_loss) / (cnt + inputs.size(0))
                epoch_acc = (torch.sum(preds == labels.data) + epoch_acc * cnt).double() / (cnt + inputs.size(0))

                cnt += inputs.size(0)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'train':
                scheduler.step()

        print('this epoch takes {} seconds.'.format(time.time() - since2))

    saved_model_path = os.path.join(output_folder, 'NDCC_state_dict.pth')
    torch.save(model.state_dict(), saved_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)

    parser.add_argument('--data_folder', type=str, default='/datasets')
    parser.add_argument('--dataset', type=str, default='FounderType200', choices=['CUB200', 'StanfordDogs', 'FounderType200'])
    parser.add_argument('--network', type=str, default='alexnet', choices=['alexnet', 'vgg16'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--random_seed', type=int, default=42, help='random seed for train/test split')
    parser.add_argument('--strategy', type=int, default=1, choices=[1, 2], help='strategy for the parameterization of $\Sigma$')

    parser.add_argument('--num_classes', type=int, default=100, help='the number of training classes')
    parser.add_argument('--num_epochs', type=int, default=10, help='the number of training epochs')

    parser.add_argument('--lr1', type=float, default=1e-3, help='learning rate for embedding v(x)')
    parser.add_argument('--lr2', type=float, default=1e-1, help='learning rate for linear classifier {w_y, b_y}')
    parser.add_argument('--lr3', type=float, default=1e-1, help='learning rate for \sigma')
    parser.add_argument('--lr4', type=float, default=1e-3, help='learning rate for \delta_j')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_milestones', default=[5, 10])

    parser.add_argument('--lmd', type=float, default=2e-1, help='\lambda in Eq. (23)')
    parser.add_argument('--gma', type=float, default=1/4096, help='\gamma in Eq. (22)')
    parser.add_argument('--r', type=float, default=16, help='\|v(x)\|=r')

    parser.add_argument('--exp_id', type=str, default='1')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    opt = parser.parse_args()

    # import socket
    # opt.exp_id = socket.gethostname()
    
    output_folder = os.path.join(opt.checkpoint_dir, opt.dataset, opt.network, opt.exp_id)
    os.makedirs(output_folder, exist_ok=True)

    log_file = os.path.join(output_folder, 'log.out')
    err_file = os.path.join(output_folder, 'err.out')

    sys.stdout = Logger(log_file)
    sys.stderr = Logger(err_file)  # redirect std err, if necessary


    
    assert torch.cuda.is_available()
    
    # recommended choice for hyperparameters (according to Table C.1. in our Supplementary Material)
    if opt.dataset == 'StanfordDogs':
        opt.num_classes = 60
        opt.lr1 = 1e-3
        opt.lr2 = 1e-1
        opt.lr3 = 1e-1
        opt.lr4 = 1e-3
        opt.r = 16
        opt.lmd = 2e-1
        
        opt.lr_milestones = [25, 28, 30]
        opt.num_epochs = 30
        
    elif opt.dataset == 'FounderType200':
        opt.num_classes = 100
        opt.lr1 = 1e-2
        opt.lr2 = 1e-1 
        opt.lr3 = 1e-1
        opt.lr4 = 1e-3
        opt.r = 32
        opt.lmd = 2e-1
        opt.lr_milestones = [5, 10]
        opt.num_epochs = 10
        

    elif opt.dataset == 'CUB200':
        opt.num_classes = 100
        opt.lr1 = 1e-2
        opt.lr2 = 1e-1            
        opt.lr3 = 1e-1
        opt.lr4 = 1e-3
        opt.r = 8
        opt.lmd = 1e-4

        opt.lr_milestones = [30, 40]
        opt.num_epochs = 40

    pprint(vars(opt))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if opt.dataset in ['StanfordDogs', 'FounderType200']:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
    elif opt.dataset in ['CUB200']:
        transform_train = transforms.Compose([
            transforms.Resize(350),
            transforms.RandomCrop(336),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Resize(350),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            normalize,
        ])

    images, labels = pd.read_csv(opt.dataset + '.csv', sep=',').values.transpose()

    # train/test split
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.5,
                                                                            random_state=opt.random_seed,
                                                                            stratify=labels)

    # known/unknown split
    known_train_idx = np.where(train_labels < opt.num_classes)[0]
    unknown_train_idx = np.where(train_labels >= opt.num_classes)[0]
    known_test_idx = np.where(test_labels < opt.num_classes)[0]
    unknown_test_idx = np.where(test_labels >= opt.num_classes)[0]

    known_train_images = train_images[known_train_idx]
    known_train_labels = train_labels[known_train_idx]
    known_test_images = test_images[known_test_idx]
    known_test_labels = test_labels[known_test_idx]

    train_dataset = my_dataset(known_train_images, known_train_labels, transform_train, opt.data_folder)
    val_dataset = my_dataset(known_test_images, known_test_labels, transform_test, opt.data_folder)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True)

    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader

    dataset_sizes = {}
    dataset_sizes['train'] = len(train_dataset)
    dataset_sizes['val'] = len(val_dataset)



    if opt.network == 'vgg16':
        embedding = models.vgg16(pretrained=True)
        embedding.classifier[6] = nn.Sequential()
    elif opt.network == 'alexnet':
        embedding = models.alexnet(pretrained=True)
        embedding.classifier[6] = nn.Sequential()

    classifier = nn.Linear(4096, opt.num_classes)
    model = NDCC(embedding=embedding, classifier=classifier, strategy=opt.strategy, l2_normalize=True, r=opt.r)

    if opt.strategy == 1:
        optimizer = torch.optim.SGD([{'params': model.embedding.parameters(), 'lr': opt.lr1},
                               {'params': model.classifier.parameters(), 'lr': opt.lr2, 'weight_decay': 0e-4},
                               {'params': [model.sigma], 'lr': opt.lr3, 'weight_decay': 0e-4},
                               ], momentum=opt.momentum, weight_decay=5e-4)
    elif opt.strategy == 2:
        optimizer = torch.optim.SGD([{'params': model.embedding.parameters(), 'lr': opt.lr1},
                               {'params': model.classifier.parameters(), 'lr': opt.lr2, 'weight_decay': 0e-4},
                               {'params': [model.sigma], 'lr': opt.lr3, 'weight_decay': 0e-4},
                               {'params': [model.delta], 'lr': opt.lr4, 'weight_decay': 0e-4},
                               ], momentum=opt.momentum, weight_decay=5e-4)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_milestones, gamma=0.1)

    model = model.cuda()

    #==================== training ====================

    train_NDCC(model, optimizer, scheduler, num_epochs=opt.num_epochs)

    #==================== evaluation ====================

    ND_images = np.hstack([test_images[known_test_idx], test_images[unknown_test_idx], train_images[unknown_train_idx]])

    # binary labels (1 for novel and 0 for seen)
    ND_labels = np.hstack([np.zeros(len(known_test_idx)), np.ones(len(unknown_test_idx)), np.ones(len(unknown_train_idx))])

    ND_dataset = my_dataset(ND_images, ND_labels, transform_test, opt.data_folder)
    ND_loader = DataLoader(dataset=ND_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers,
                           pin_memory=True)

    ND_scores = model.get_ND_scores(ND_loader)
    print('AUC ROC: %f' % roc_auc_score(ND_labels, ND_scores))
