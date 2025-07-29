import os
from PIL import Image
from torch.utils.data import Dataset
from compress.generate_jpegs import nonlinear_qualities
from torchvision import transforms
from collections import Counter

class CIFARCompressionDataset(Dataset):
	def __init__(
		self,
		root_dir,
		indices,
		labels,
		mode,
		thresholds_by_epoch=None,
		fixed_test_quality=100,
		compression_mode="cluster",    # "fixed", "manual", or "cluster"
		fixed_quality=None,            # used if compression_mode == "fixed"
		manual_thresholds=None,        # used if compression_mode == "manual"
		log_dir
	):
		self.root_dir = root_dir
		self.indices = indices
		self.labels = labels
		self.mode = mode
		self.thresholds_by_epoch = thresholds_by_epoch
		self.losses_per_epoch = []
		self.current_epoch = 0
		self.fixed_test_quality = fixed_test_quality

		self.compression_mode = compression_mode
		self.fixed_quality = fixed_quality
		self.manual_thresholds = manual_thresholds
		self.log_dir = log_dir 
		self.log_path = os.path.join(self.log_dir, "compression_distribution.txt")
		# dataset prints compression dist, it has to append
		# this clears the file before we start appending
		open(log_path, "w").close()

	def set_epoch(self, epoch, losses):
		self.current_epoch = epoch - 1
		self.losses_per_epoch = losses

		if self.mode == "train" and self.compression_mode != "fixed":
			assignments = [self._get_quality(i) for i in self.indices]
			counts = Counter(assignments)

			# must append since dataset is running in parallel
			with open(self.log_path, "a") as f:
				f.write(f"Epoch {epoch} compression distribution:\n")
				for q in sorted(counts):
					f.write(f"    q{q}: {counts[q]} samples\n")
				f.write("\n")

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		global_idx = self.indices[idx]

		if self.mode == "train":
			quality = self._get_quality(global_idx)
		else:
			quality = self.fixed_test_quality

		image_path = os.path.join(
			self.root_dir,
			f"jpeg_q{quality}",
			self.mode,
			f"{global_idx:05d}.jpg"
		)

		image = Image.open(image_path).convert("RGB")
		image = transforms.ToTensor()(image)
		label = self.labels[global_idx]
		return image, label

	def _get_quality(self, global_idx):
		if self.compression_mode == "fixed":
			return self.fixed_quality

		loss = self.losses_per_epoch[global_idx]

		if self.compression_mode == "manual":
			thresholds = self.manual_thresholds
		else:  # cluster mode (default)
			thresholds = self.thresholds_by_epoch[self.current_epoch]

		cluster_count = len(thresholds) + 1

		for c in range(cluster_count):
			if c == cluster_count - 1 or loss < thresholds[c]:
				break

		quality_levels = nonlinear_qualities(cluster_count, gamma=2.0)
		return quality_levels[c]
