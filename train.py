import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18
import json
# from importance.offline_profiler import offline_profiler
from compress.generate_jpegs import generate_jpegs
import sys


def main():
    CIFAR10(root='./data/CIFAR10/', train=True, download=True)
    CIFAR10(root='./data/CIFAR10', train=False, download=True)
    c10_route = "./data/CIFAR10/cifar-10-batches-py/"

    if sys.argv[1] == "profile":
        batches = []
        for i in range(1, 6):
            batches.append(unpickle(f"{c10_route}data_batch_{i}"))
        batches.append(unpickle(f"{c10_route}test_batch"))
        #run profiler

        with open('./importance/profile_data.json') as f:
            profiler_data = json.load(f)

        generate_jpegs(profiler_data, batches)
        #run compression

    #use custom dataset
    #use default dataloader


    # NVIDIA and mac metal API are the easiest devices to check for
    device = "cpu"
    if(torch.backends.mps.is_available()):
        device = "mps"
    elif(torch.cuda.is_available()):
        device = "cuda"

    #ideally the following would be in some other folder since the profiler needs to do all of this

    #set up resnet18 model like this
    # model = resnet18(weights=None)
    # #cifar10 only classifies 10 things, get rid of early downsampling
    # model.maxpool = nn.Identity()
    # model.fc = nn.Linear(model.fc.in_features, 10)
    # model = model.to(device)
    
    # good hyperparameters for resnet18 on Cifar10
    # criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
    # # doesn't return one loss per image (used for getting average loss per batch)
    # criterion = nn.CrossEntropyLoss()  
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # epochs = 200
    # losses = []
    # valid_losses = []
    # min_loss = float('inf')
    # best_epoch = 0

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


    
if __name__ == "__main__":
    main()