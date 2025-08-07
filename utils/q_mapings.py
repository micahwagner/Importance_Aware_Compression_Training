import numpy as np

def skew_low_qualities(n_levels, gamma=2.0):
	qualities = [round(100 * ((i + 1) / n_levels) ** gamma) for i in range(n_levels)]
	qualities[-1] = 100
	return qualities

def skew_high_qualities(n_levels, min_quality=25, gamma=2.0):
	qualities =  [
        round(min_quality + (100 - min_quality) * (1 - ((n_levels - i - 1) / n_levels) ** gamma))
        for i in range(n_levels)
	]
	qualities[-1] = 100
	return qualities