import os
from PIL import Image
from torch.utils.data import Dataset
from compress.generate_jpegs import nonlinear_qualities

class CIFARCompressionDataset(Dataset):
	def __init__(self, root_dir, indices, labels, mode, thresholds_by_epoch):
		self.root_dir = root_dir
		self.indices = indices
		self.labels = labels
		self.mode = mode
		self.thresholds_by_epoch = thresholds_by_epoch
		self.losses_per_epoch = []
		self.current_epoch = 0

	def set_epoch(self, epoch, losses):
		self.current_epoch = epoch
		self.losses_per_epoch = losses

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		global_idx = self.indices[idx]
		loss = self.losses_per_epoch[global_idx]

		raw_thresh = self.thresholds_by_epoch[self.current_epoch]
		thresholds = [0.0] + raw_thresh
		cluster_count = len(thresholds)

		for c in range(cluster_count):
			if loss < thresholds[c]:
				break
		cluster = c

		quality_levels = nonlinear_qualities(cluster_count, gamma=2.0)
		quality = quality_levels[cluster]

		image_path = os.path.join(
			self.root_dir,
			f"jpeg_q{quality}",
			self.mode,
			f"{global_idx:05d}.jpg"
		)

		image = Image.open(image_path).convert("RGB")
		label = self.labels[global_idx]
		return image, label, global_idx
