# -*- coding: utf-8 -*-
import os, sys
import numpy as np

from torch.utils.data import Dataset
from PIL import Image  
from sklearn.metrics.pairwise import pairwise_distances

class my_dataset(Dataset):

    def __init__(self, files_list, labels_list, transform, data_folder):
        self.files_list = files_list
        self.labels_list = labels_list
        self.transform = transform
        self.data_folder = data_folder

        assert len(self.files_list) == len(self.labels_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_folder, self.files_list[index])).convert("RGB")
        img = self.transform(img)
        return img, self.labels_list[index]

    def __len__(self):
        return len(self.labels_list)
    
def mahalanobis_metric(X, Y, inv_Sigma):
    # X: numpy array of shape (n_samples_X, n_features)
    # Y: numpy array of shape (n_samples_Y, n_features)
    # inv_Sigma: inverse of Sigma (diagonal covariance)

    # X_normalized = X @ (np.sqrt(inv_Sigma))
    # Y_normalized = Y @ (np.sqrt(inv_Sigma))
    
    X_normalized = X * np.sqrt(np.diag(inv_Sigma))
    Y_normalized = Y * np.sqrt(np.diag(inv_Sigma))

    distances = pairwise_distances(X_normalized, Y=Y_normalized, metric='sqeuclidean', n_jobs=1)

    return distances



class Logger(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=1)
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()
 
    def close(self):
        self.log.close()