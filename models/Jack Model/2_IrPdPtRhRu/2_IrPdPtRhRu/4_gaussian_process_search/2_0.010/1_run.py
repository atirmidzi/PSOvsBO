'''
Get the number of samples needed for locating the various optimal compositions as well as the variation by running the Bayesian optimization for various initial molar fractions.
'''

# Number of processes
n_processes = 4

# Number of random initial conditions to run for
n_runs = 64
run_ids = range(n_runs)

import multiprocessing as mp
import sys
sys.path.append('../../../..')
from scripts import metals, n_metals, get_time_stamp, BruteForceSurface,\
					OntopRegressor, FccRegressor, get_activity,\
					get_random_molar_fractions, get_molar_fractions_around,\
					expected_improvement, gpr, optimize_molar_fraction,\
					get_local_maxima, get_molar_fractions
from time import time, localtime, strftime
import numpy as np
from copy import deepcopy
import os

# Define elements to use (i.e. allow for sub alloy searches too)
elements = ['Ir', 'Pd', 'Pt', 'Rh', 'Ru']
element_indices = [metals.index(elem) for elem in elements]

# Define exploration-exploitation trade-off
# The higher this number, the more exploration
xi = 0.010

# Set random seed
np.random.seed(7745)

# Make random seeds for each simulation for reproduction
random_seeds = np.random.randint(10000, size=n_runs)
random_seeds = [random_seeds[run_idx] for run_idx in run_ids]

# Number of steps to run the optimization for
n_steps = 150

# Define slab size (no. of atom x no. of atoms)
size = (100, 100)
n_surface_atoms = np.prod(size)

# Get alloy as string
alloy = ''.join([metals[idx] for idx in element_indices])

# Load linear regression parameters for OH and O
path = '../../2_machine_learning_model'
params_OH = [np.loadtxt(f'{path}/OH_{metal}.csv', delimiter=',', usecols=1) for metal in metals]
params_O = np.loadtxt(f'{path}/O.csv', delimiter=',', usecols=1)

# Load adsorbate-specific regressors
reg_OH = OntopRegressor(params_OH)
reg_O = FccRegressor(params_O)

# Define optimal OH adsorption energy (relative to Pt(111))
E_opt = 0.1 # eV	

# Define potential at which to evaluate the activity
eU = 0.820 # eV

def append_to_file(filename, f, activity):
	'Append molar fractions and its activity to file'
	with open(filename, 'a') as file_:
		file_.write(','.join(map('{:.5f}'.format, [f[m] for m in metals])) + ',' + '{:.7f}\n'.format(activity))

# Objective function
def fill_and_get_activity(f, filename=None):
	
	# Get start time
	t0 = time()
	
	# Convert into a molar fractions dictionary
	f = {m: f0 for m, f0 in zip(metals, f)}
	
	# Make Surface instance
	surface = BruteForceSurface(f, reg_OH, reg_O,
				size=size, displace_OH=0.,
				displace_O=0., scale_O=0.5)

	# Determine gross energies of surface sites
	surface.get_gross_energies()

	# Get net adsorption energy distributions upon filling the surface
	surface.get_net_energies()

	# Get activity of the net distribution of *OH adsorption energies
	energies_OH = surface.energy_grid_OH_orig[surface.ontop_ads_sites]
	activity = get_activity(energies_OH, E_opt, n_surface_atoms, eU, jD=1.)

	# Add activity of the scaled net distribution of O* adsorption energies
	energies_O = surface.energy_grid_O_orig[surface.fcc_ads_sites]
	activity += get_activity(energies_O, E_opt, n_surface_atoms, eU, jD=1.)
	
	if filename is not None:
		# Append molar fractions and activity to file
		append_to_file(filename, f, activity)

	# Get end time
	t1 = time()
	
	# Print time
	f_str = ', '.join(map('{:.3f}'.format, [f[m] for m in metals]))
	print(f'{f_str} (A = {activity:.3f}) ({get_time_stamp(t1-t0)})')
	
	return activity

# Optimize the acquisition function
def opt_acquisition(X_known, gpr, acq_func='EI', xi=0.01, n_iter_max=1000, element_indices=None,
					n_random=1000, step_size=0.005):
	
	# Define the acquisition function
	if callable(acq_func):
		acquisition = acq_func
	elif acq_func == 'EI':
		acquisition = expected_improvement
	else:
		raise NotImplementedError(f"The acquisition function '{acq_func}' has not been implemented")
	
	# Get ´n_random´ molar fractions
	if element_indices is None:	
		random_samples = get_random_molar_fractions(n_random, n_elems=n_metals)
	else:
		random_samples = np.zeros((n_random, n_metals))
		random_samples[:, element_indices] = get_random_molar_fractions(n_random, n_elems=len(element_indices))

	# Calculate the acquisition function for each sample
	acq_vals = acquisition(random_samples, X_known, gpr, xi)
	
	# Get the index of the largest acquisition function
	idx_max = np.argmax(acq_vals)
	
	# Get the molar fraction with the largest acquisition value
	f_max = random_samples[idx_max]
	acq_max = acquisition(random_samples[idx_max].reshape(1, -1), X_known, gpr, xi)
	
	# Optimize the aquisition function starting from this molar fraction
	n_iter = 0
	while True:
		
		if n_iter == n_iter_max:
			raise ValueError(f'No maximum has been found after {n_iter} iterations,\
							 so convergence is unlikely to happen.')
		
		# Get molar fractions around the found maximum in the given step size
		if element_indices is None:
			fs_around = get_molar_fractions_around(f_max, step_size=step_size)
		else:
			fs_around_ = get_molar_fractions_around(f_max[element_indices], step_size=step_size)
			fs_around = np.zeros((len(fs_around_), n_metals))
			fs_around[:, element_indices] = fs_around_
		 		
		# Get acquisition values
		acq_vals = acquisition(fs_around, X_known, gpr, xi)
		
		# Get the largest acquisition value
		acq_max_around = np.max(acq_vals)
		
		# If the new aquisition value is higher, then repeat for that molar fraction
		if acq_max_around > acq_max:
			
			# Get the index of largest acquisition value
			idx_max = np.argmax(acq_vals)
			
			# Get molar fraction
			f_max = fs_around[idx_max]
			
			# Set the new acquisition maximum
			acq_max = acq_max_around

		# If the acquisition function did now improve around the molar fraction,
		# then return the found molar fraction
		else:
			return f_max
		
		n_iter += 1

# Define file name where sampled molar fraction are written to
filename_samples = 'samples.csv'

# Define surrogate maxima filename
filename_max = 'surrogate_max.csv'

# Generate two molar fractions at random to begin the search
n_elems = len(element_indices)
n_random = 2

def run(run_idx, random_seed):
	'Run Bayesian optimization for the given random seed'
	
	# Set random seed for reproducibility
	np.random.seed(random_seed)
	
	# Make directory for this run, if not already existing
	folder = f'{run_idx}/'
	if not os.path.exists(folder):
		os.makedirs(folder)
	
	# Create file and write header
	with open(folder + filename_samples, 'w') as file_:
		file_.write(','.join(map('{:>7s}'.format, metals)) + ', ' + 'activity\n')

	# Make file and write header
	with open(folder + filename_max, 'w') as file_:
		file_.write('  n,' + ','.join(map('{:>7s}'.format, metals)) + ',    act,   C**2,      l\n')
	
	# Generate ´n_random´ molar fractions to begin the search
	fs = np.zeros((n_random, n_metals))
	fs[:, element_indices] = get_random_molar_fractions(n_random, n_elems)
	
	# Initiate activities list
	activities = []
	
	for f_idx, f in enumerate(fs):
		activities.append(fill_and_get_activity(f, folder + filename_samples))

	# Fit the regressor
	gpr.fit(fs, activities)

	# Iterate through the optimization steps
	for n_samples in range(len(fs), n_steps):

		# Get local optima on a coarse grid of 10% steps
		fs_grid = get_molar_fractions(step_size=0.1)
		fs_max, preds_max = get_local_maxima(fs_grid, gpr.predict, step_size=0.1)

		# Get Gaussian process kernel parameters
		params = gpr.kernel_.get_params()
		c = params['k1__constant_value']
		l = params['k2__length_scale']
		
		# Iterate through the coarse grid surrogate maxima found
		for f in fs_max:
		
			# Optimize the coarse maximum molar fraction to get the "true" local maximum
			f_max, pred_max = optimize_molar_fraction(f, gpr.predict, step_size=0.005)
			
			# Write the maximum to file along with the hyperparameters
			# of the Gaussian process regressor
			with open(folder + filename_max, 'a') as file_:
				file_.write(f'{n_samples:>3d},' + ','.join(map('{:.5f}'.format, f_max)) + f',{pred_max:.5f}'\
							f',{c:.5f},{l:.5f}\n')
		
		# Select the next best point to sample
		# by sampling from the acquisition function
		f = opt_acquisition(fs, gpr,
				acq_func=expected_improvement,
				xi=xi,
				element_indices=element_indices,
				step_size=0.005,
				n_random=1000)

		# Sample the point
		activity = fill_and_get_activity(f, folder + filename_samples)

		# Add the data to the dataset
		fs = np.vstack((fs, [f]))
		activities += [activity]

		# Update the regressor
		gpr.fit(fs, activities)

args = [(run_idx, seed) for run_idx, seed in zip(run_ids, random_seeds)]

with mp.Pool(processes=n_processes) as pool:
	pool.starmap(run, args)
