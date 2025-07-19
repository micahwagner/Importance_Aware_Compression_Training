import torch

def train_model(
	model,
	train_loader,
	test_loader,
	criterion,
	optimizer,
	scheduler=None,
	device="cpu",
	num_epochs=200,
):
	losses_per_epoch = []
	loss_per_epoch = []
	accuracy_per_epoch = []
	print(f"Training for {num_epochs} epochs")
	for epoch in range(1, num_epochs + 1):
		model.train()
		losses = []
		running_loss = 0
		train_steps = 0
		for x, y in train_loader:
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			output = model(x)
			loss = criterion(output, y)
			running_loss += loss.mean().item()
			train_steps += 1
			for l in loss:
				losses.append(l.item())
			loss.mean().backward()
			optimizer.step()
		losses_per_epoch.append(losses)
		loss_per_epoch.append(running_loss / train_steps)

		if scheduler:
			scheduler.step()

		correct = 0
		total = 0
		test_steps = 0
		test_loss = 0
		with torch.no_grad():
			for x, y in test_loader:
				x, y = x.to(device), y.to(device)
				outputs = model(x)
				loss = criterion(outputs, y)
				test_loss += loss.mean().item()
				test_steps += 1
				preds = outputs.argmax(dim=1)
				correct += (preds == y).sum().item()
				total += y.size(0)
		
		accuracy_per_epoch.append(correct / total)

		if epoch % 5 == 0:
			print("Epoch " + str(epoch) + " training loss: " + str(running_loss/train_steps))
			print("Epoch " + str(epoch) + " testing loss: " + str(test_loss/test_steps))

	return losses_per_epoch, loss_per_epoch, accuracy_per_epoch
