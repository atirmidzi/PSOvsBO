import numpy as np
import itertools as it
import sys
sys.path.append('../../..')
from scripts import get_molar_fractions_around, get_time_stamp
from time import time

# Get start time
t0 = time()

# Get header
filename = 'grid_activities.csv'
with open(filename, 'r') as file_:
	header = file_.readline().split(',')

# Get metals from header
metals = [m.strip() for m in header[:-1]]

# Load molar fractions and activities
data = np.loadtxt(filename, delimiter=',', skiprows=1)
fs = data[:, :-1]
activities = data[:, -1]
n_fs = fs.shape[0]

# Write header to output file
filename_out = 'local_optima.csv'

# Initiate container for optimal molar fractions
fs_max = []
activities_max = []

# Iterate through all molar fractions in the given step size
for f_idx, f in enumerate(fs):
	
	# Print completion update after every 1000 molar fractions
	if f_idx % 1000 == 0:
		t1 = time()
		print(f'[COMPLETION] {f_idx}/{n_fs} ({get_time_stamp(t1-t0)})')

	# Get activity of molar fraction
	act = activities[f_idx]
	
	# Get neighbor molar fractions one step size away
	fs_neigh = get_molar_fractions_around(f, step_size=0.05)

	# Get indices of the neighbor molar fractions
	# in the array of molar fractions
	indices = []
	for fn in fs_neigh:
		indices.append(np.nonzero(np.all(np.isclose(fn, fs), axis=1))[0][0])
	
	# Get activities of the neighbors
	act_neigh = activities[indices]
	
	# If the molar fraction has higher activity than all its neighbors,
	# then it is a local maximum
	if all(act > act_neigh):
		
		# Append to list of optimal molar fractions
		fs_max.append(f)
		activities_max.append(act)

# Sort activities in descending order
sorted_ids = np.argsort(activities_max)[::-1]
activities_sorted = np.asarray(activities_max)[sorted_ids]
fs_sorted = np.asarray(fs_max)[sorted_ids]	

with open(filename_out, 'w') as file_:
	
	# Make header
	file_.write(','.join(map('{:>7s}'.format, metals)) + ', ' + 'activity\n')
	
	# Write molar fractions and activities to file in order of descending activity
	for f, activity in zip(fs_sorted, activities_sorted):
		file_.write(','.join(map('{:.5f}'.format, f)) + ',' + '{:.7f}\n'.format(activity))

# Print duration
t1 = time()
print(f'[DURATION] {get_time_stamp(t1-t0)}')
