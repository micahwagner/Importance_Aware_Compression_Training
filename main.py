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
from compress.generate_jpegs import generateJPEGS, print_quality_sizes
from dataloader.dynamic_dataset import CIFARCompressionDataset
from utils.eval import test_best_model
from utils.train_runner import train_model
from torch.utils.data import DataLoader
import sys
from contextlib import redirect_stdout

def main():
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


	cifar_train = CIFAR10(root='./data/CIFAR10/', train=True, download=True, transform=cifar_transform_train)
	train_labels = cifar_train.targets
	cifar_test = CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=cifar_transform_test)
	test_labels = cifar_test.targets
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

	# train model expects reduction = none to get losses per sample (no average)
	criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
	num_epochs = 200
	# for now we set them equal, dont use shift factors yet
	profile_epochs = num_epochs
	
	batches = []
	for i in range(1, 6):
		batches.append(unpickle(f"{c10_route}data_batch_{i}"))
	batches.append(unpickle(f"{c10_route}test_batch"))
	
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

		with open('profiler_output.json', 'w') as f:
			json.dump(profiler_output, f)

		generateJPEGS(profiler_output, batches)

	with open("./data/compression_sizes.log", "w") as f:
		with redirect_stdout(f):
			print_quality_sizes()

	train_indices = list(range(50000))
	test_indices = list(range(50000, 60000))
	all_labels = train_labels + test_labels

	with open("profiler_output.json", "r") as f:
		p_data = json.load(f)

	mode = "cluster"
	fixed_quality=None
	manual_thresholds=None
	train_dataset = CIFARCompressionDataset(
		root_dir="./data",
		indices=train_indices,
		mode="train",
		thresholds_by_epoch=p_data["thresholds"],
		labels=all_labels,
		compression_mode=compression_mode,
		manual_thresholds=manual_thresholds,
		fixed_quality=fixed_quality,
		log_dir=out_dir
	)

	test_dataset = CIFARCompressionDataset(
		root_dir="./data",
		indices=test_indices,
		mode="test",
		labels=all_labels,
		compression_mode=compression_mode,
		manual_thresholds=manual_thresholds,
		fixed_quality=fixed_quality,
		log_dir=out_dir
	)
	
	print(f"Training for {num_epochs} epochs...")
	# high enough base for running on high quality
	train_loss_per_epoch = []
	test_loss_per_epoch = []
	losses = [100.0] * 60000
	min_loss = float('inf')
	best_epoch = 0

    file_tag = f"{mode}"
    if mode == "fixed":
        file_tag += f"_q{fixed_quality}"
    elif mode == "manual":
        file_tag += "_" + "-".join(f"{t:.1f}" for t in manual_thresholds)

    out_dir = os.path.join("results", file_tag)
    os.makedirs(out_dir, exist_ok=True)
	
	for epoch in range(1, num_epochs + 1):

		train_dataset.set_epoch(epoch, losses)
	
		train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, transform=cifar_transform_train)
		test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=True, num_workers=4, transform=cifar_transform_test)
	
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
				torch.save(model.state_dict(), './best_model.pth')

		losses = train_losses

	train_dataset.set_epoch(best_epoch, losses)
	train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=True, num_workers=4)
	
	test_best_model(
		model_path="./best_model.pth",
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