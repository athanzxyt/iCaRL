"""
Main script for Evaluation of MedMNIST

Author: Athan Zhang @athanzxyt
"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from icarl import iCaRLNet
import time

import medmnist
from medmnist import INFO

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

# Select which dataset from MedMNIST
data_flag = 'bloodmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
num_channels = info['n_channels']
num_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_data = DataClass(split='train', transform=data_transform, download=download)
val_data = DataClass(split='val', transform=data_transform, download=download)
test_data = DataClass(split='test', transform=data_transform, download=download)

print('\n')

t = time.process_time()
train_sets = [ [] for _ in range(num_classes)]
for sample in train_data:
    label = sample[1].item(0)
    train_sets[label].append(sample)
train_loaders = [data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True) for dataset in train_sets]
print(f'Training dataset divided in\t:\t{round(time.process_time() - t,3)}s')

t = time.process_time()
val_sets = [[]]*num_classes
for sample in val_data:
    label = sample[1].item(0)
    val_sets[label].append(sample)
test_loaders = [data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False) for dataset in val_sets]
print(f'Validation dataset divided in\t:\t{round(time.process_time() - t,3)}s')

t = time.process_time()
test_sets = [[]]*num_classes
for sample in test_data:
    label = sample[1].item(0)
    test_sets[label].append(sample)
test_loaders = [data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False) for dataset in test_sets]
print(f'Testing dataset divided in\t:\t{round(time.process_time() - t,3)}s')

print('\n')

K = 2000
icarl = iCaRLNet(2352,num_classes,32,device)
icarl.to(device)
for c in range(num_classes):
    print(len(train_sets[c]))
    icarl.incremental_train(train_sets[c],K)
