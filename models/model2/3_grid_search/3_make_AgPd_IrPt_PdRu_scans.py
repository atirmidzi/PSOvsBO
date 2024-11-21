import sys
sys.path.append('../../..')
from scripts import metals, n_metals, get_time_stamp, BruteForceSurface,\
					OntopRegressor, FccRegressor, get_activity,\
					get_molar_fractions
import numpy as np
from time import time

# Set random seed
np.random.seed(9435)

# Define alloys systems to traverse
alloys = [['Ir', 'Pt'], ['Ag', 'Pd'], ['Pd', 'Ru']]

# Define elements to use (i.e. allow for sub alloy searches too)
elements = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru']

# Define slab size (no. of atom x no. of atoms)
size = (100, 100)
n_surface_atoms = np.prod(size)

# Define optimal OH adsorption energy (relative to Pt(111))
E_opt = 0.1 # eV	

# Define potential at which to evaluate the activity
eU = 0.820 # eV

# Load linear regression parameters for OH and O
path = '../2_machine_learning_model'
params_OH = [np.loadtxt(f'{path}/OH_{metal}.csv', delimiter=',', usecols=1) for metal in metals]
params_O = np.loadtxt(f'{path}/O.csv', delimiter=',', usecols=1)

# Load adsorbate-specific regressors
reg_OH = OntopRegressor(params_OH)
reg_O = FccRegressor(params_O)

def append_to_file(filename, f, activity):
	'Append molar fractions and its activity to file'
	with open(filename, 'a') as file_:
		file_.write(','.join(map('{:.5f}'.format, [f[m] for m in metals])) + ',' + '{:.7f}\n'.format(activity))

# Objective function
def fill_and_get_activity(f, filename):
	
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
	
	# Append molar fractions and activity to file
	append_to_file(filename, f, activity)

	# Get end time
	t1 = time()
	
	# Print time
	f_str = ', '.join(map('{:.3f}'.format, [f[m] for m in metals]))
	print(f'{f_str} (A = {activity:.3f}) completed ({get_time_stamp(t1-t0)})')
	
	return activity

# Iterate through alloys
for elements in alloys:
	
	# Get indices of elements
	element_indices = [metals.index(elem) for elem in elements]	
	
	# Get molar fractions on a grid of 1% steps
	n_fs = get_molar_fractions(step_size=0.01, n_elems=len(elements),
				return_number_of_molar_fractions=True)
	fs = np.zeros((n_fs, n_metals))
	fs[:, element_indices] = get_molar_fractions(step_size=0.01, n_elems=len(elements))
	
	# Get alloy as string
	alloy = ''.join([metals[idx] for idx in element_indices])
	
	# Get filename to write output to
	filename = f'{alloy}_activities.csv'
	
	# Make header for file
	with open(filename, 'w') as file_:
		file_.write(','.join(map('{:>7s}'.format, metals)) + ', ' + 'activity\n')
	
	# Print metals to terminal
	print(', '.join(map('{:>5s}'.format, metals)))
	
	# Iterate through molar fractions
	for f in fs:
		
		# Make surface and write activity and molar fractions to file
		fill_and_get_activity(f, filename)
