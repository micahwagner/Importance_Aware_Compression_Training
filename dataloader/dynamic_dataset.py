import os
from PIL import Image
from torch.utils.data import Dataset
from compress.generate_jpegs import skew_low_qualities, skew_high_qualities
from torchvision import transforms
from collections import Counter

class CIFARCompressionDataset(Dataset):
	def __init__(
		self,
		root_dir,
		indices,
		labels,
		mode,
		log_dir,
		transform,
		thresholds_by_epoch=None,
		fixed_test_quality=100,
		compression_mode="cluster",    # "fixed", "manual", or "cluster"
		fixed_quality=None,            # used if compression_mode == "fixed"
		manual_thresholds=None         # used if compression_mode == "manual"
	):
		self.root_dir = root_dir
		self.indices = indices
		self.labels = labels
		self.mode = mode
		self.transform = transform
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
		open(self.log_path, "w").close()

	def set_epoch(self, epoch, losses):
		self.current_epoch = epoch - 1
		self.losses_per_epoch = losses

		if self.mode == "train" and self.compression_mode != "fixed":
			assignments = [self._get_quality(i) for i in self.indices]
			counts = Counter(assignments)

			if self.compression_mode == "cluster":
				self.clusters = list(counts.items())
				self.new_clusters, self.promotion_map = self._promote_clusters(self.clusters)
				counts = Counter(dict(self.new_clusters)) 
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
			if self.compression_mode == "cluster":
				quality = self.promotion_map[self._get_quality(global_idx)]
		else:
			quality = self.fixed_test_quality

		image_path = os.path.join(
			self.root_dir,
			f"jpeg_q{quality}",
			self.mode,
			f"{global_idx:05d}.jpg"
		)

		image = Image.open(image_path).convert("RGB")
		image = self.transform(image)
		label = self.labels[global_idx]
		return image, label, global_idx

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

		quality_levels = skew_high_qualities(cluster_count)

		return quality_levels[c]

	def _promote_clusters(self, cs, min_percent=0.01):
		cs = sorted(cs, key=lambda x: x[0], reverse=True)
		buckets = [[q,0] for q, _ in cs]
		buckets = sorted(buckets, key=lambda x: x[0])
		current_bucket = len(buckets) - 1
		total = sum(value for _, value in cs)
		min_cluster_size = total * min_percent
		promotion_map = {}

		for cluster in cs:
			if buckets[current_bucket][1] >= min_cluster_size:
				current_bucket -= 1

			buckets[current_bucket][1] += cluster[1]
			promotion_map[cluster[0]] = buckets[current_bucket][0]

		if current_bucket < len(buckets) - 1 and buckets[current_bucket][1] < min_cluster_size:
			buckets[current_bucket + 1][1] += buckets[current_bucket][1]
			buckets[current_bucket][1] = 0
			promotion_map[buckets[current_bucket][0]] = buckets[current_bucket + 1][0]

		return buckets, promotion_map


			

