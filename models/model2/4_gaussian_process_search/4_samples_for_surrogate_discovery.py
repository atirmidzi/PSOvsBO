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

diff_lim = 0.1
sample_ns_found = [[] for _ in range(n_fs_obj)]
acts_found = [[] for _ in range(n_fs_obj)]

# Iterate through elements in the current directory
for folder in os.listdir(xi_folder):

	# If a directory
	if os.path.isdir(xi_folder + '/' + folder):
			
		# Get header of file with surrogate maxima
		with open(xi_folder + '/' + folder + '/surrogate_max.csv', 'r') as file_:
			header = file_.readline()
		
		# Get metals from header
		metals = [m.strip() for m in header.split(',')[1:-3]]
		
		# Get sampled molar fractions and their activities
		data = np.loadtxt(xi_folder + '/' + folder + '/surrogate_max.csv', skiprows=1, delimiter=',')
		sample_ns = data[:, 0].astype(int)
		fs = data[:, 1:-3]
		activities = data[:, -3]
		
		found_obj = [False]*n_fs_obj
		
		# Iterate through surrogate optimal molar fractions
		for f_idx, (f, act) in enumerate(zip(fs, activities)):
			
			# Iterate through molar fractions to match
			for obj_idx, f_obj in enumerate(fs_obj):
				
				if not found_obj[obj_idx]:
				
					# Get stoiciometric number difference
					diff = np.sum(np.abs(f-f_obj))
					if diff < diff_lim:
						
						# Append sample number for discovery of this
						# objective molar fraction
						sample_ns_found[obj_idx].append(sample_ns[f_idx])
						acts_found[obj_idx].append(activities[f_idx])
						found_obj[obj_idx] = True

# Get mean and sample standard deviation for each group of alloys
means = [np.mean(sample_ns_found[obj_idx]) for obj_idx in range(n_fs_obj)]
stds = [np.std(sample_ns_found[obj_idx], ddof=1) for obj_idx in range(n_fs_obj)]
mins = [np.min(sample_ns_found[obj_idx]) for obj_idx in range(n_fs_obj)]
maxs = [np.max(sample_ns_found[obj_idx]) for obj_idx in range(n_fs_obj)]
ns = [len(sample_ns_found[obj_idx]) for obj_idx in range(n_fs_obj)]

# Get mean and sample standard deviation of activities
means_act = [np.mean(acts_found[obj_idx]) for obj_idx in range(n_fs_obj)]
stds_act = [np.std(acts_found[obj_idx], ddof=1) for obj_idx in range(n_fs_obj)]

filename = 'samples_for_surrogate_discovery.csv'

# Write header
with open(filename, 'w') as file_:
	file_.write('     composition,   mean,    std,    min,    max, n_runs,    act, std_act\n')

	# Save output to file
	for f, mean, std, min_, max_, n, mean_act, std_act in zip(fs_obj, means, stds, mins, maxs, ns, means_act, stds_act):
		composition = get_composition(f, metals)
		file_.write(f'{composition:>16},{mean:>7.1f},{std:>7.1f},{min_:>7},{max_:>7},{n:>7},{mean_act:>7.4f},{std_act:>8.4f}\n')
