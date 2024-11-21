import numpy as np
import os
import sys
sys.path.append('../../..')
from scripts import get_composition

# Specify folder
xi_folder = sys.argv[1]

fs_obj = np.array([[0.185, 0.000, 0.815, 0.000, 0.000], # Ag18Pd82
				   [0.000, 0.093, 0.641, 0.000, 0.265], # Ir9Pd64Ru27
				   [0.000, 0.485, 0.000, 0.515, 0.000], # Ir49Pt51
				   [0.779, 0.000, 0.000, 0.000, 0.221], # Ag78Ru22
				   [0.000, 0.465, 0.000, 0.000, 0.535], # Ir47Ru53
				   [0.000, 0.101, 0.000, 0.000, 0.899], # Ir10Ru90
				   [0.000, 0.000, 0.000, 0.000, 1.000], # Ru
				  ])

n_fs_obj = len(fs_obj)

fs_found = [[] for _ in range(n_fs_obj)]
acts_found = [[] for _ in range(n_fs_obj)]

largest_diff = 0.
counter = np.zeros(n_fs_obj, dtype=int)

# Iterate through elements in the current directory
for folder in os.listdir(xi_folder):

	# If a directory
	if os.path.isdir(xi_folder + '/' + folder):
			
		# Get header of file with sampled molar fractions
		with open(xi_folder + '/' + folder + '/surrogate_max.csv', 'r') as file_:
			header = file_.readline()
		
		# Get metals from header
		metals = [m.strip() for m in header.split(',')[1:-3]]
		
		# Get found surrogate optima and their activities
		data = np.loadtxt(xi_folder + '/' + folder + '/surrogate_max.csv', skiprows=1, delimiter=',')
		sample_ns = data[:, 0]
		activities = data[:, -3]
		
		# Get number of samples at the end of the optimization
		sample_no = sample_ns[-1]
		
		# Get the optimal molar fractions at this number of samples
		fs = data[sample_ns == sample_no, 1:-3]
		acts = activities[sample_ns == sample_no]
		
		for f, act in zip(fs, acts):
			
			found_match = False
			n_matches = 0
			for obj_idx, f_obj in enumerate(fs_obj):
				diff = np.sum(np.abs(f-f_obj))
				if diff < 0.10:
					
					if diff > largest_diff:
						largest_diff = diff
					
					fs_found[obj_idx].append(f)
					acts_found[obj_idx].append(act)
					
					counter[obj_idx] += 1
					found_match = True
					n_matches += 1
					
			if n_matches != 1:
				print(n_matches, f)
			
			if not found_match:
				composition = get_composition(f, metals)
				print(f'Found no match for {folder}: {composition}')

# Get mean and spread in optimal compositions for each group of alloys
means = [np.mean(fs_found[obj_idx], axis=0) for obj_idx in range(n_fs_obj)]
stds = [np.std(fs_found[obj_idx], ddof=1, axis=0) for obj_idx in range(n_fs_obj)]
mins = [np.min(fs_found[obj_idx], axis=0) for obj_idx in range(n_fs_obj)]
maxs = [np.max(fs_found[obj_idx], axis=0) for obj_idx in range(n_fs_obj)]

# Get mean and sample standard deviation of activities
means_act = [np.mean(acts_found[obj_idx]) for obj_idx in range(n_fs_obj)]
stds_act = [np.std(acts_found[obj_idx], ddof=1) for obj_idx in range(n_fs_obj)]

filename = 'maxima_of_surrogates.csv'

# Write header
with open(filename, 'w') as file_:
	metals_mean = ' mean_' + ', mean_'.join(metals)
	metals_std = ' std_' + ', std_'.join(metals)
	metals_min = ' min_' + ', min_'.join(metals)
	metals_max = ' max_' + ', max_'.join(metals)
	file_.write(f'     composition,{metals_mean},{metals_std},{metals_min},{metals_max},    act, std_act, n_runs\n')

	# Save output to file
	for f, mean, std, min_, max_, mean_act, std_act, n_run in zip(fs_obj, means, stds, mins, maxs, means_act, stds_act, counter):
		composition = get_composition(f, metals)
		metals_mean = ','.join(map('{:>8.3f}'.format, mean))
		metals_std = ','.join(map('{:>7.3f}'.format, std))
		metals_min = ','.join(map('{:>7.3f}'.format, min_))
		metals_max = ','.join(map('{:>7.3f}'.format, max_))
		file_.write(f'{composition:>16},{metals_mean},{metals_std},{metals_min},{metals_max},{mean_act:>7.4f},{std_act:>8.4f},{n_run:>7}\n')
