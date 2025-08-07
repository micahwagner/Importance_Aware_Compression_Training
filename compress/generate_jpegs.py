from PIL import Image
import os
import numpy as np
from utils.q_mapings import skew_low_qualities, skew_high_qualities

jpeg_root = "./data"


def generateCIFAR10_JPEGS(profiler_data, batches, compression_array=None):
	thresholds = profiler_data["thresholds"]

	cluster_counts = set(len(thresh) + 1 for thresh in thresholds)
	print(f"Detected cluster counts: {sorted(cluster_counts)}")
	# for now quality will be a quadratic scale factor (prevents lower clusters from having subtle quality drop)
	# this will probably change since thresholds can contain unequal number of samples
	cluster_to_qualities = set()
	for k in cluster_counts:
		if compression_array == None:
			cluster_to_qualities.update(skew_high_qualities(k))
		else:
			cluster_to_qualities.update(compression_array)
		print(f"Cluster count {k} → JPEG qualities: {cluster_to_qualities}")

	cluster_to_qualities = list(cluster_to_qualities)

	img_idx = 0
	split_point = 50000 
	# each batch is 10000x3072 (firs dimension being an image)
	for batch in batches:
		data = batch[b"data"]

		for i in range(data.shape[0]):
			img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
			img = Image.fromarray(img)

			for q in cluster_to_qualities:
				split = "train" if img_idx < split_point else "test"
				out_dir = f"./data/jpeg_q{q}/{split}"
				os.makedirs(out_dir, exist_ok=True)
				out_path = os.path.join(out_dir, f"{img_idx:05d}.jpg")
				img.save(out_path, quality=q)
			img_idx += 1
	print(f"Done. Saved {img_idx} images × {len(cluster_to_qualities)} total variants.")

def get_folder_size(path):
	total = 0
	for dirpath, dirnames, filenames in os.walk(path):
		for f in filenames:
			fp = os.path.join(dirpath, f)
			if os.path.isfile(fp):
				total += os.path.getsize(fp)
	return total

def print_quality_sizes():
	jpeg_dirs = [
		name for name in os.listdir(jpeg_root)
		if name.startswith("jpeg_q") and os.path.isdir(os.path.join(jpeg_root, name))
	]

	jpeg_dirs = sorted(jpeg_dirs, key=lambda x: int(x.split("_q")[1]))
	for name in jpeg_dirs:
		q_path = os.path.join(jpeg_root, name)
		for split in ["train", "test"]:
			split_path = os.path.join(q_path, split)
			size = get_folder_size(split_path)
			print(f"{name}/{split}: {size / 1e6:.2f} MB")
