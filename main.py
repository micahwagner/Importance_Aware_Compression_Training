import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms
from torchvision.models import resnet18, resnet50
import json
from importance.offline_profiler import run_offline_profiler
from compress.generate_jpegs import generateCIFAR10_JPEGS, print_quality_sizes
from dataloader.dynamic_dataset import CompressionDataset
from utils.eval import test_model
from utils.train_runner import train_model
from utils.smooth_thresh import clean_and_smooth_thresholds
from torch.utils.data import DataLoader
import sys
import os
from contextlib import redirect_stdout
from enum import Enum

IMAGENET100_DIR = "/path/to/imagenet100"
CIFAR10_DIR = "./data/CIFAR10/"

class Dataset(Enum):
    CIFAR10 = 1
    IMAGENET100= 2

class Model(Enum):
    RESNET18 = 1
    RESNET50 = 2

def main():
	current_dataset = Dataset.CIFAR10
	current_model = Model.RESNET18

	precompress = True

	if precompress and current_dataset == Dataset.IMAGENET100:
		print("can't precompress imagenet100")
		exit(1)

	# CIFAR10 
	cifar_transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
	])

	cifar_transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
	])

	c10_route = f"{CIFAR10_DIR}/cifar-10-batches-py/"

	#IMAGENET
	imagenet_train_transform = transforms.Compose([
    	transforms.RandomResizedCrop(224),
    	transforms.RandomHorizontalFlip(),
    	transforms.ToTensor(),
    	transforms.Normalize(
    	    mean=[0.485, 0.456, 0.406],
    	    std=[0.229, 0.224, 0.225]
    	)
	])

	imagenet_test_transform = transforms.Compose([
    	transforms.Resize(256),
    	transforms.CenterCrop(224),
    	transforms.ToTensor(),
    	transforms.Normalize(
    	    mean=[0.485, 0.456, 0.406],
    	    std=[0.229, 0.224, 0.225]
    	)
	])

	batches = []

	# SELECT DATASET
	if current_dataset == Dataset.CIFAR10:
		print("Using Cifar10")
		train_dataset = CIFAR10(root=CIFAR10_DIR, train=True, download=True, transform=cifar_transform_train)
		test_dataset = CIFAR10(root=CIFAR10_DIR, train=False, download=True, transform=cifar_transform_test)
		num_classes = 10
		for i in range(1, 6):
			batches.append(unpickle(f"{c10_route}data_batch_{i}"))
		batches.append(unpickle(f"{c10_route}test_batch"))

	elif current_dataset == Dataset.IMAGENET100:
		print("Using ImageNet100")
		train_dataset = ImageFolder(os.path.join(IMAGENET100_DIR, "train"), transform=imagenet_train_transform)
		test_dataset = ImageFolder(os.path.join(IMAGENET100_DIR, "val"), transform=imagenet_test_transform)
		num_classes = 100
	
	profiler_train_loader = DataLoader(
		train_dataset,
		batch_size=128,
		shuffle=True,
		num_workers=4
	)

	profiler_test_loader = DataLoader(
		test_dataset,
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

	#SELECT MODEL
	if current_model == Model.RESNET18:
		print("Using resnet18")
		model = resnet18(weights=None)
	elif current_model == Model.RESNET50:
		print("Using resnet50")
		model = resnet50(weights=None)
	
	if current_dataset == Dataset.CIFAR10:
		#cifar10 images are very small, get rid of early downsampling
		model.maxpool = nn.Identity()
	model.fc = nn.Linear(model.fc.in_features, num_classes)
	model = model.to(device)

	# HYPERPARAMETERS
	num_epochs = 200
	# train model expects reduction = none to get losses per sample (no average)
	criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
	# for now we set them equal, dont use shift factors yet
	profile_epochs = num_epochs
	
	
	if len(sys.argv) > 1 and sys.argv[1] == "profile":
		losses_per_epoch = []
		loss_per_epoch = []
		accuracy_per_epoch = []
		print(f"Running profiler for {profile_epochs} epochs...")
		for epoch in range(1, profile_epochs + 1):
			losses, train_loss, test_loss, accuracy = train_model(
				model,
				profiler_train_loader,
				profiler_test_loader,
				criterion_no_reduction,
				optimizer,
				scheduler,
				device,
				profiler=True
			)

			losses_per_epoch.append(losses)
			loss_per_epoch.append(train_loss)
			accuracy_per_epoch.append(accuracy)

			if epoch % 5 == 0:
				print("Epoch " + str(epoch) + " training loss: " + str(train_loss))
				print("Epoch " + str(epoch) + " testing loss: " + str(test_loss))

		with open("profiler_output.log", "w") as f:
			with redirect_stdout(f):
				shifts_e, thresholds_e, accuracy_e = run_offline_profiler(losses_per_epoch, accuracy_per_epoch)
		
		profiler_output = {
			'thresholds': thresholds_e,
			'shift_factors': shifts_e,
			'baseline_test_acc': accuracy_e
		}

		with open("profiler_output.json", "w") as f:
			json.dump(profiler_output, f)

		# first value in thresholds array is too high, ignore it when smoothing
		smoothed_thresholds = clean_and_smooth_thresholds(profiler_output["thresholds"][1:], min_ratio=0.05, sigma=5)
		smoothed_thresholds.insert(0, profiler_output["thresholds"][0])
		with open("clean_smoothed.json", "w") as f:
		    json.dump({"thresholds": smoothed_thresholds}, f, indent=2)

		if precompress:
			generateCIFAR10_JPEGS({"thresholds": smoothed_thresholds}, batches)
		return


	train_indices = list(range(len(train_dataset)))
	test_indices = list(range(len(train_dataset), len(train_dataset) + len(test_dataset)))
	all_labels = train_dataset.targets + test_dataset.targets

	with open("clean_smoothed.json", "r") as f:
		p_data = json.load(f)

	mode = "cluster"
	fixed_quality=None
	manual_thresholds=None
	fixed_test_quality=100

	file_tag = f"{mode}"
	if mode == "fixed":
		file_tag += f"_q{fixed_quality}"
	elif mode == "manual":
		file_tag += "_" + "-".join(f"{t:.1f}" for t in manual_thresholds)

	out_dir = os.path.join("results", file_tag)
	os.makedirs(out_dir, exist_ok=True)
	train_compression_dataset = CompressionDataset(
		root_dir="./data",
		indices=train_indices,
		mode="train",
		transform=train_dataset.transform,
		base_dataset=train_dataset,
		thresholds_by_epoch=p_data["thresholds"],
		labels=all_labels,
		log_dir=out_dir,
		compression_mode=mode,
		manual_thresholds=manual_thresholds,
		fixed_quality=fixed_quality,
		precompressed=precompress
	)

	test_compression_dataset = CompressionDataset(
		root_dir="./data",
		indices=test_indices,
		mode="test",
		base_dataset=test_dataset,
		transform=test_dataset.transform,
		labels=all_labels,
		log_dir=out_dir,
		compression_mode=mode,
		manual_thresholds=manual_thresholds,
		fixed_quality=fixed_quality,
		fixed_test_quality=fixed_test_quality,
		precompressed=precompress
	)
	
	print(f"Training for {num_epochs} epochs...")
	# high enough base for running on high quality
	train_loss_per_epoch = []
	test_loss_per_epoch = []
	losses = [100.0] * len(train_compression_dataset)
	best_losses = []
	min_loss = float('inf')
	best_epoch = 0
	best_model_path = os.path.join(out_dir, './best_model.pth')
	
	for epoch in range(1, num_epochs + 1):

		train_compression_dataset.set_epoch(epoch, losses)
	
		train_loader = DataLoader(train_compression_dataset, batch_size=128, shuffle=True, num_workers=4)
		test_loader = DataLoader(test_compression_dataset, batch_size=10000, shuffle=True, num_workers=4)
	
		train_losses, train_loss_avg, test_loss_avg, _ = train_model(
			model,
			train_loader,
			test_loader,
			criterion_no_reduction,
			optimizer,
			scheduler,
			device,
		)

		train_loss_per_epoch.append(train_loss_avg)
		test_loss_per_epoch.append(test_loss_avg)
		
		if epoch % 5 == 0:
				print("Epoch " + str(epoch) + " training loss: " + str(train_loss_avg))
				print("Epoch " + str(epoch) + " testing loss: " + str(test_loss_avg))
		
		if test_loss_avg < min_loss:
				min_loss = test_loss_avg
				best_epoch = epoch
				best_losses = losses
				torch.save(model.state_dict(), best_model_path)

		losses = train_losses

	train_compression_dataset.set_epoch(best_epoch, losses)
	train_loader = DataLoader(train_compression_dataset, batch_size=128, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_compression_dataset, batch_size=10000, shuffle=True, num_workers=4)
	
	test_model(
		model_path=best_model_path,
		model=model,
		out_dir=out_dir,
		device=device,
		train_loader=train_loader,
		test_loader=test_loader,
		criterion=criterion,
		losses=train_loss_per_epoch,
		test_losses=test_loss_per_epoch,
		epochs=num_epochs,
		best_epoch=best_epoch
	)

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


	
if __name__ == "__main__":
	main()