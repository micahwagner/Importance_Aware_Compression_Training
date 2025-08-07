import torch

def train_model(
	model,
	train_loader,
	test_loader,
	criterion,
	optimizer,
	scheduler,
	device,
	profiler=False
):
	model.train()
	loss_map = {} if not profiler else []
	running_loss = 0
	train_steps = 0
	for batch in train_loader:
		if not profiler:
			x, y, global_indices = batch
		else:
			x, y = batch
			global_indices = None 
		x, y = x.to(device), y.to(device)
		optimizer.zero_grad()
		output = model(x)
		loss = criterion(output, y)
		running_loss += loss.mean().item()
		train_steps += 1
		loss.mean().backward()
		optimizer.step()

		if not profiler:
			for i, global_idx in enumerate(global_indices):
				loss_map[int(global_idx)] = loss[i].item()
		else:
			for l in loss:
				loss_map.append(l)

	train_loss_avg = running_loss / train_steps

	if scheduler:
		scheduler.step()

	correct = 0
	total = 0
	test_steps = 0
	running_loss = 0
	model.eval()
	with torch.no_grad():
		for batch in test_loader:
			if not profiler:
				x, y, _ = batch
			else:
				x, y = batch
			x, y = x.to(device), y.to(device)
			outputs = model(x)
			loss = criterion(outputs, y)
			running_loss += loss.mean().item()
			test_steps += 1
			preds = outputs.argmax(dim=1)
			correct += (preds == y).sum().item()
			total += y.size(0)
		
	accuracy = correct / total
	test_loss_avg = running_loss/test_steps
	
	if not profiler:
		losses = [loss_map.get(i, 100.0) for i in range(len(train_loader.dataset))]
	else:
		losses = [float(l) for l in loss_map]
	return losses, train_loss_avg, test_loss_avg, accuracy
