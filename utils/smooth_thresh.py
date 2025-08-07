import matplotlib.pyplot as plt
import numpy as np
import json
from collections import Counter
from scipy.ndimage import gaussian_filter1d


def forward_fill(arr):
    filled = []
    last = np.nan
    for val in arr:
        if not np.isnan(val):
            last = val
        filled.append(last if not np.isnan(last) else 0)
    return filled

def smooth_array(array, sigma=2):
    return gaussian_filter1d(array, sigma=sigma)

def interpolate_row(prev, nxt, alpha):
    return [(p * (1 - alpha) + n * alpha) for p, n in zip(prev, nxt)]

def clean_and_smooth_thresholds(thresholds, min_ratio=0.05, sigma=2):
    counts = Counter(len(row) for row in thresholds)
    total = len(thresholds)
    common_count = counts.most_common(1)[0][0]

    cleaned = []
    for i, row in enumerate(thresholds):
        if len(row) == common_count:
            cleaned.append(row)
        elif counts[len(row)] / total < min_ratio:
            prev, nxt = None, None
            for j in range(i - 1, -1, -1):
                if len(thresholds[j]) == common_count:
                    prev = thresholds[j]
                    break
            for k in range(i + 1, len(thresholds)):
                if len(thresholds[k]) == common_count:
                    nxt = thresholds[k]
                    break
            if prev and nxt:
                alpha = (i - j) / (k - j)
                interp = interpolate_row(prev, nxt, alpha)
            elif prev:
                interp = prev.copy()
            elif nxt:
                interp = nxt.copy()
            else:
                avg = np.mean([r for r in thresholds if len(r) == common_count], axis=0)
                interp = list(avg)
            cleaned.append(interp)
        else:
            cleaned.append(row)

    transposed = list(zip(*cleaned))
    smoothed = []
    for col in transposed:
        filled = forward_fill(col)
        smoothed.append(smooth_array(filled, sigma=sigma))
    return [list(row) for row in zip(*smoothed)]
