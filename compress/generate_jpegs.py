from PIL import Image
import os
import numpy as np


def generate_jpegs(profiler_data, batches):
	thresholds = profiler_data["thresholds"]

	n_levels = len(thresholds) + 1

	# for now this will be the same for every level
	# this will probably change since thresholds can contain unequal ranges
	quality_steps = 100 // n_levels
	quality_levels = [(i + 1) * (quality_steps) for i in range(n_levels)]

	# top level should be full quality
	quality_levels[-1] = 100

	img_idx = 0
	# each batch is 10000x3072 (firs dimension being an image)
	for batch in batches:
		data = batch[b"data"]

		for i in range(data.shape[0]):
			img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
			img = Image.fromarray(img)

			for q in quality_levels:
				out_dir = f"./data/jpeg_q{q}"
				os.makedirs(out_dir, exist_ok=True)
				img.save(f"{out_dir}/{img_idx:05d}.jpg", quality=q)

			img_idx += 1

