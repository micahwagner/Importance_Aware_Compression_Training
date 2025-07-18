import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


def run_offline_profiler(losses_per_epoch, accuracy_per_epoch):
	thresholds_by_epoch = []
	shift_factors = []

	for epoch_losses in losses_per_epoch:
		thresholds = cluster_losses(epoch_losses)
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

	labels = fcluster(Z, t=K_opt, criterion='maxclust')
	clusters = {}
	for i, label in enumerate(labels):
		clusters.setdefault(label, []).append(losses_per_epoch[i])

	cluster_means = sorted(np.mean(v) for v in clusters.values())
	thresholds = []
	for i in range(len(cluster_means) - 1):
		mid = (cluster_means[i] + cluster_means[i+1]) / 2
		thresholds.append(mid)

	return thresholds

def compute_natural_decrease(thresholds_by_epoch):
	diffs = []
	for i in range(1, len(thresholds_by_epoch)):
		diff = thresholds_by_epoch[i-1][1] - thresholds_by_epoch[i][1]
		diffs.append(diff)
	return sum(diffs) / len(diffs)