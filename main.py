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
from importance.offline_profiler import run_offline_profiler
from importance.train_runner import train_model
from compress.generate_jpegs import generateJPEGS
from torch.utils.data import DataLoader
import sys


def main():
	cifar_transform_train = transforms.Compose([
		# transforms.RandomCrop(32, padding=4),
		# transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
	])

	cifar_transform_test = transforms.Compose([
		transforms.ToTensor(),
		# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
	])


	cifar_train = CIFAR10(root='./data/CIFAR10/', train=True, download=True, transform=cifar_transform_train)
	cifar_test = CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=cifar_transform_test)
	c10_route = "./data/CIFAR10/cifar-10-batches-py/"

	profiler_train_loader = DataLoader(
		cifar_train,
		batch_size=128,
		shuffle=True,
		num_workers=4
	)

	profiler_test_loader = DataLoader(
		cifar_test,
		batch_size=10000,
		shuffle=True,
		num_workers=4
	)

	# NVIDIA and mac metal API are the easiest devices to check for
	device = "cpu"
	if(torch.backends.mps.is_available()):
		device = "mps"
	elif(torch.cuda.is_available()):
		device = "cuda"

	#set up resnet18 model like this
	model = resnet18(weights=None)
	#cifar10 only classifies 10 things, get rid of early downsampling
	model.maxpool = nn.Identity()
	model.fc = nn.Linear(model.fc.in_features, 10)
	model = model.to(device)

	# train model expects reduction = none
	criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
	optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
	epochs = 200

	if len(sys.argv) > 1 and sys.argv[1] == "profile":
		batches = []
		for i in range(1, 6):
			batches.append(unpickle(f"{c10_route}data_batch_{i}"))
		batches.append(unpickle(f"{c10_route}test_batch"))

		losses_e, loss_e, accuracy_e = train_model(
			model,
			profiler_train_loader,
			profiler_test_loader,
			criterion_no_reduction,
			optimizer,
			scheduler,
			device,
			num_epochs=10
		)

		shifts_e, thresholds_e, accuracy_e = run_offline_profiler(losses_e, accuracy_e)
		#generateJPEGS(profiler_data, batches)
		#run compression


def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


	
if __name__ == "__main__":
	main()