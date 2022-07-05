# -*- coding: utf-8 -*-
import os, time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import mahalanobis_metric
from tqdm import tqdm

class NDCC(nn.Module):
    def __init__(self, embedding, classifier, opt, l2_normalize=True):
        super(NDCC, self).__init__()
        self.embedding = embedding
        self.classifier = classifier

        self.dim_embedding = self.classifier.in_features
        self.num_classes = self.classifier.out_features        


        self.l2_normalize = l2_normalize
        if self.l2_normalize:           
            self.r = opt.r

        self.strategy = opt.strategy    
        self.lmd = opt.lmd
        self.gma = opt.gma

        if self.strategy == 1:
            self.sigma = torch.tensor(((np.ones(1) )).astype('float32'), requires_grad=True, device="cuda")
            self.delta  = None
            
        elif self.strategy ==2:
            self.sigma = torch.tensor(((np.ones(1) )).astype('float32'), requires_grad=True, device="cuda")
            self.delta = torch.tensor((np.zeros(self.dim_embedding)).astype('float32'), requires_grad=True, device="cuda")

    def forward(self, x):
        x = self.embedding(x)
        # x = nn.parallel.data_parallel(self.embedding, x)
        
        x = x.view(x.size(0), -1)
        
        if self.l2_normalize:
            x = self.r * torch.nn.functional.normalize(x, p=2, dim=1)
            
        return x

    def fit(self, optimizer, scheduler, dataloaders, num_epochs=20):
        for epoch in range(num_epochs):
            since2 = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            # for phase in ['train', 'val']:

            for phase in ['train']:
                
                self.eval()  # NDCC is alwayes set to evaluate mode

                cnt = 0

                epoch_loss = 0.
                epoch_acc = 0.

                for step, (inputs, labels) in enumerate(dataloaders[phase]):

                    inputs = (inputs.cuda())
                    labels = (labels.long().cuda())

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # outputs = self(inputs)
                        outputs = nn.parallel.data_parallel(self, inputs)

                        if self.strategy == 1:
                            sigma2 = self.sigma ** 2
                            means = sigma2 * self.classifier.weight

                            loss_MD = ((((outputs - means[labels]) ** 2).sum()) / (sigma2.detach())) / (2 * outputs.shape[0])
                            loss_NLL = (self.dim_embedding * torch.log(sigma2)) / 2 + ((((outputs.detach() - means[labels]) ** 2).sum()) / (sigma2)) / (2 * outputs.shape[0])

                        elif self.strategy == 2:
                            sigma2 = (self.sigma + self.delta) ** 2
                            means = self.classifier.weight * sigma2

                            loss_MD = (torch.div((outputs - means[labels]) ** 2, sigma2.detach())).sum() / (2 * outputs.shape[0])
                            loss_NLL = (torch.log(sigma2).sum()) / 2 + (torch.div((outputs.detach() - means[labels]) ** 2, sigma2).sum() / outputs.shape[0]) / 2

                        logits = nn.parallel.data_parallel(self.classifier, outputs)
                        loss_CE = F.cross_entropy(logits, labels)

                        loss = loss_CE + self.lmd * (loss_MD + self.gma * loss_MD)

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


    @torch.no_grad()
    def get_ND_scores(self, loader):
        self.eval()  # Set model to evaluate mode

        weight = self.classifier.weight
        weight = weight.cpu().detach().numpy()

        if self.strategy == 1:
            sigma2 = (self.sigma.detach().cpu().numpy()) ** 2    
            Sigma =  sigma2 * np.eye(self.dim_embedding)
            inv_Sigma = (1/sigma2) * np.eye(self.dim_embedding)
        elif self.strategy == 2:
            sigma2 = ((self.delta + self.sigma).detach().cpu().numpy()) ** 2    
            Sigma = np.diag(sigma2) 
            inv_Sigma = np.diag(sigma2 ** -1)
            
                
        means = weight @ Sigma
    
        distances = np.zeros((len(loader.dataset), self.num_classes))

        idx = 0
        # Iterate over data.
        for step, (inputs, _) in enumerate(tqdm(loader)):
            # if step%100 == 0:
            #     print(step)

            inputs = inputs.cuda()

            # outputs = self(inputs)
            outputs = nn.parallel.data_parallel(self, inputs)
            
            outputs = outputs.detach().cpu().numpy().squeeze()
            batch_distance = mahalanobis_metric(outputs, means, inv_Sigma)

            assert batch_distance.shape == distances[idx:idx+len(outputs), :].shape
            distances[idx:idx+len(outputs), :] = batch_distance

            idx += len(outputs)
                

                
        ND_scores = np.min(distances, axis=1)

        return ND_scores
    
