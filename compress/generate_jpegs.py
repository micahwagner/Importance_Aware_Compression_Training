from PIL import Image
import os
import numpy as np

jpeg_root = "./data"

def nonlinear_qualities(n_levels, gamma=2.0):
	return [round(100 * ((i + 1) / n_levels) ** gamma) for i in range(n_levels)]

def generateJPEGS(profiler_data, batches):
	thresholds = profiler_data["thresholds"]

	cluster_counts = set(len(thresh) + 1 for thresh in thresholds)
	print(f"Detected cluster counts: {sorted(cluster_counts)}")
	# for now quality will be a quadratic scale factor (prevents lower clusters from having subtle quality drop)
	# this will probably change since thresholds can contain unequal number of samples
	for k in cluster_counts:
		qualities = nonlinear_qualities(k, gamma=2.0)
		qualities[-1] = 100  # ensure top quality is 100
		cluster_to_qualities[k] = qualities
		print(f"Cluster count {k} → JPEG qualities: {qualities}")


	img_idx = 0
	split_point = 50000 
	compressed_100 = False
	# each batch is 10000x3072 (firs dimension being an image)
	for batch in batches:
		data = batch[b"data"]

		for i in range(data.shape[0]):
			img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
			img = Image.fromarray(img)

			for k, quality_levels in cluster_to_qualities.items():
				for q in quality_levels:
					if q == 100 and compressed_100:
						continue
					split = "train" if img_idx < split_point else "test"
					out_dir = f"./data/jpeg_q{q}/{split}"
					os.makedirs(out_dir, exist_ok=True)
					out_path = os.path.join(out_dir, f"{img_idx:05d}.jpg")
					img.save(out_path, quality=q)
				compressed_100 = True
			compressed_100 = False
			img_idx += 1
	print(f"Done. Saved {img_idx} images × {sum(len(v)-1 for v in cluster_to_qualities.values())} total variants.")

def get_folder_size(path):
	total = 0
	for dirpath, dirnames, filenames in os.walk(path):
		for f in filenames:
			fp = os.path.join(dirpath, f)
			if os.path.isfile(fp):
				total += os.path.getsize(fp)
	return total

def print_quality_sizes():
	for name in os.listdir(jpeg_root):
		if name.startswith("jpeg_q") and os.path.isdir(os.path.join(jpeg_root, name)):
			q_path = os.path.join(jpeg_root, name)
			for split in ["train", "test"]:
				split_path = os.path.join(q_path, split)
				size = get_folder_size(split_path)
				print(f"{name}/{split}: {size / 1e6:.2f} MB")
