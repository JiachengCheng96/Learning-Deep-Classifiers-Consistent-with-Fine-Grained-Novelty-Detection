# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from utils import mahalanobis_metric

class NDCC(nn.Module):
    def __init__(self, embedding, classifier, strategy, l2_normalize=True, r=1.0):
        super(NDCC, self).__init__()
        self.embedding = embedding
        self.classifier = classifier

        self.dim_embedding = self.classifier.in_features
        self.num_classes = self.classifier.out_features        


        self.l2_normalize = l2_normalize
        if self.l2_normalize:           
            self.r = r

        self.strategy = strategy    
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

        with torch.no_grad():
            idx = 0
            # Iterate over data.
            for step, (inputs, _) in enumerate(loader):
                if step%100 == 0:
                    print(step)

                inputs = inputs.cuda()

                # outputs = self(inputs)
                outputs = nn.parallel.data_parallel(self, inputs)
                
                outputs = outputs.detach().cpu().numpy().squeeze()
                
                distances[idx:idx+len(outputs), :] = mahalanobis_metric(outputs, means, inv_Sigma)

                idx += len(outputs)
                

                
        ND_scores = np.min(distances, axis=1)

        return ND_scores
    
