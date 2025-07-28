import torch

def train_model(
	model,
	train_loader,
	test_loader,
	criterion,
	optimizer,
	scheduler=None,
	device="cpu",
):
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
	train_loss = running_loss / train_steps

	if scheduler:
		scheduler.step()

	correct = 0
	total = 0
	test_steps = 0
	running_loss = 0
	model.eval()
	with torch.no_grad():
		for x, y in test_loader:
			x, y = x.to(device), y.to(device)
			outputs = model(x)
			loss = criterion(outputs, y)
			running_loss += loss.mean().item()
			test_steps += 1
			preds = outputs.argmax(dim=1)
			correct += (preds == y).sum().item()
			total += y.size(0)
		
	accuracy = correct / total
	test_loss = running_loss/test_steps

	return losses, train_loss, test_loss, accuracy
