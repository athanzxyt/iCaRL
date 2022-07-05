"""
Re-implementation of 
S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert.
iCaRL: Incremental classifier and representation learning.
CVPR, 2017.
in PyTorch.

Author: Athan Zhang @athanzxyt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from resnet import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class iCaRLNet(nn.Module):
    def __init__(self,
                 feature_size,
                 num_classes,
                 transform,
                 epochs=50,
                 lr=0.002
                 ):
        super(iCaRLNet,self).__init__()

        # Hyperparameters
        self.epochs = epochs
        self.lr = lr
        self.transform = transform

        # Network architecture
        self.feature_extractor = resnet18()
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, feature_size)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.9)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, num_classes, bias=False)

        # Learning method
        self.clsf_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(),lr=self.lr,weight_decay=1e-5)

        # Exemplar Set and Means
        self.num_classes = num_classes
        self.exemplar_sets = []
        self.exemplar_means = []
        self.compute_means = True

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x

    def classify(self,image):
        """
        Classify images using nearest-means-of-exemplars rule
        Args:
            image: Image to be classified
        Returns:
            prediction: Prediction index (which class it belongs to)
        """
        mean_of_exemplars = np.array([np.mean(P_y) for P_y in self.exemplar_sets])
        feature_extractor_output = F.normalize(self.model.feature_extractor(image).detach()).cpu().numpy()
        x = feature_extractor_output - mean_of_exemplars
        x = np.linalg.norm(x, axis=1)
        prediction = np.argmin(x)
        return prediction

    def construct_exemplar_set(self, images, m):
        """
        Construct the exemplar set
        Args:
            images: Image set of a class
            m: Target number of exemplars
        Returns:
            exemplar_set: Exemplar set of a class
        """
        feature_extractor_output = F.normalize(self.model.feature_extractor(images).detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        sum_of_exemplars = np.zeros(feature_extractor_output.shape)
        exemplar_set = []

        for k in range(m):
            x = class_mean - (feature_extractor_output + sum_of_exemplars) / k+1
            x = np.linalg.norm(x, axis=1)
            idx = np.argmin(x)
            sum_of_exemplars += feature_extractor_output[idx]
            exemplar_set.append(images[idx])

        return exemplar_set

    def reduce_exemplar_set(self, exemplar_set, m):
        """
        Reduce an exemplar set
        Args:
            exemplar_set: Exemplar set of a class
            m: Target number of exemplars
        Returns:
            new_exemplar_set: Reduced exemplar set
        """
        new_exemplar_set = exemplar_set[:m]
        return new_exemplar_set


