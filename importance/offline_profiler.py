import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering


def run_offline_profiler(losses_per_epoch, accuracy_per_epoch):
	thresholds_by_epoch = []
	shift_factors = []

	for idx, epoch_losses in enumerate(losses_per_epoch):
		print(f"Clustering Epoch {idx}...")
		max_sample_size = 10000
		if len(epoch_losses) > max_sample_size:
			sample = np.random.choice(epoch_losses, size=max_sample_size, replace=False).tolist()
		else:
			sample = epoch_losses
		thresholds = cluster_losses(sample)
		thresholds_by_epoch.append(thresholds)

	natural_decrease = compute_natural_decrease(thresholds_by_epoch)
	
	for e in range(len(thresholds_by_epoch)):
		# not exactly sure why we get the second threshold
		T1 = thresholds_by_epoch[e][1]
		Fe = 1.0 + (natural_decrease / T1)
		shift_factors.append(Fe)

	return shift_factors, thresholds_by_epoch, accuracy_per_epoch


def cluster_losses(losses_per_epoch):
	# clustering expects shape (num_samples, num_features)
	X = np.array(losses_per_epoch).reshape(-1, 1)
	Z = linkage(X, method='ward')

	# second array in Z is merge distances, get optimal num of clusters
	merge_distances = Z[:, 2]
	jumps = np.diff(merge_distances)
	biggest_jump = np.argmax(jumps) 
	K_opt = len(losses_per_epoch) - biggest_jump
	print(f"optimal number of clusters: {K_opt}")

	clustering = AgglomerativeClustering(n_clusters=K_opt, metric='euclidean', linkage='ward')
	labels = clustering.fit_predict(X)

	optimal_unique_labels = np.unique(labels)
	cluster_ranges = []
	for cluster in optimal_unique_labels:
	    cluster_values = X[labels == cluster]
	    min_value = np.min(cluster_values)
	    max_value = np.max(cluster_values)
	    num_elements = cluster_values.shape[0]
	    cluster_ranges.append({
	        "cluster": cluster,
	        "min_value": min_value,
	        "max_value": max_value,
	        "num_elements": num_elements
	    })
	
	for idx, cluster in enumerate(cluster_ranges):
		print(f"Cluster {idx}: Range = [{cluster['min_value']:.4f}, {cluster['max_value']:.4f}], Size = {cluster['num_elements']}")
	
	cluster_means = sorted((r["min_value"] + r["max_value"]) / 2 for r in cluster_ranges)
	thresholds = []
	for i in range(len(cluster_means) - 1):
		mid = (cluster_means[i] + cluster_means[i+1]) / 2
		thresholds.append(mid)

	print(f" Thresholds: {['{:.4f}'.format(t) for t in thresholds]}")
	return thresholds

def compute_natural_decrease(thresholds_by_epoch):
	diffs = []
	for i in range(1, len(thresholds_by_epoch)):
		diff = thresholds_by_epoch[i-1][1] - thresholds_by_epoch[i][1]
		diffs.append(diff)
	return sum(diffs) / len(diffs)