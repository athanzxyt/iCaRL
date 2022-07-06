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
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from resnet import resnet18


class iCaRLNet(nn.Module):
    def __init__(self,
                 feature_size,
                 num_classes,
                 batch_size,
                 device,
                 transform=None,
                 epochs=5,
                 lr=0.002
                 ):
        super(iCaRLNet,self).__init__()

        # Hyperparameters
        self.epochs = epochs
        self.lr = lr
        self.transform = transform
        self.batch_size = batch_size
        self.device = device

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
        feature_extractor_output = F.normalize(self.feature_extractor(image).detach()).cpu()
        x = feature_extractor_output - mean_of_exemplars
        x = torch.linalg.norm(x, axis=1)
        prediction = torch.argmin(x)
        return prediction

    def construct_exemplar_set(self, dataset, m):
        """
        Construct the exemplar set
        Args:
            dataset: Image set of a class
            m: Target number of exemplars
        Returns:
            exemplar_set: Exemplar set of a class
        """
        images = torch.stack([i[0] for i in dataset]).to(self.device)
        feature_extractor_output = F.normalize(self.feature_extractor(images).detach()).cpu()
        class_mean = torch.mean(feature_extractor_output, axis=0)
        sum_of_exemplars = torch.zeros(feature_extractor_output.shape)
        exemplar_set = []

        for k in range(m):
            if k >= len(dataset): continue
            x = class_mean - (feature_extractor_output + sum_of_exemplars) / k+1
            x = torch.linalg.norm(x, axis=1)
            i = torch.argmin(x)
            sum_of_exemplars += feature_extractor_output[i]
            exemplar_set.append(dataset[i])

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

    def incremental_train(self, images, K):
        """
        Incrementally train on a new class and 
        updates exemplar sets
        Args:
            images: Image set of a class
            K: Total number of images to memorize                
        """
        self.update_representation(images)
        t = len(self.exemplar_sets)
        m = K // (t+1)
        
        new_exemplar_sets = []
        for s in self.exemplar_sets:
            new_exemplar_sets.append(self.reduce_exemplar_set(s,m))

        new_exemplar_sets.append(self.construct_exemplar_set(images,m))
        self.exemplar_sets = new_exemplar_sets

    def update_representation(self, dataset):
        """
        Incrementally improve the feature representation
        Args:
            dataset: Image set of a class
        """

        """
        NOTE TO SELF: WE ARE CURRENTLY ONLY ADDING ONE CLASS (one disease)
        HOWEVER MAYBE WE SHOULD ADD ONE TASK? (two diseases)
        IDK CHECK THE VARIABLE n_classses and n_known.
        """
        # Form combined training set
        for exemplar_set in self.exemplar_sets: dataset += exemplar_set
        loader = data.DataLoader(dataset, batch_size=self.batch_size,shuffle=True, num_workers=2)

        # Store network outputs with pre-updated parameters
        q = {}
        for i, (images, labels) in enumerate(loader):
            images = Variable(images).to(self.device)
            f = torch.sigmoid(self.forward(images))
            for c,image in enumerate(images):
                q[hash(str(image))] = f[c].data

        # Train
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(loader):
                images = Variable(images).to(self.device)
                labels = Variable(labels.to(self.device))

                self.optimizer.zero_grad()
                f = self.forward(images)
                
                # Classification loss for new class
                labels = torch.flatten(labels)
                clsf_loss = self.clsf_loss(f, labels)

                # Distilation loss for old classes
                dist_loss = 0
                if self.exemplar_sets:
                    f = torch.sigmoid(f)
                    for c,image in enumerate(images):
                        dist_loss += self.dist_loss(f[c].data, q[hash(str(image))]) 
                dist_loss /= len(images)
                
                total_loss = clsf_loss + dist_loss
                total_loss.backward()
                self.optimizer.step()

                # Print iteration
                if (i+1) % 10 == 0:
                    print(f'Epoch: {epoch+1}/{self.epochs} \
                            Iter: {i+1}/{len(dataset)//self.batch_size} \
                            Loss: {total_loss.data}'
                    )
            







