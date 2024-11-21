__version__ = '2.0'

from system_importer import metals, gpr, metal_colors, alloy, n_metals, ontop_sites, hollow_sites, sites
from Slab import Slab
from regressor_jack import OntopRegressor, FccRegressor, NormalDistributionSampler
from Surface_jack import BruteForceSurface
from acquisition_functions import expected_improvement
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import itertools as it
import scipy
from parity_plot import parity_plot
import re
from copy import deepcopy

# Define Boltzmann's constant
kB = 1.380649e-4 / 1.602176634  # eV K-1 (exact)
kBT = kB*300 # eV

def get_activity(energies, E_opt, n_surface_atoms, eU, jD=1.):
    '''
    Return the activity per surface atom calculated using the
    Angewandte Chemie equations 2-4 (doi: 10.1002/anie.202014374)
    '''
    jki = np.exp((-np.abs(energies - E_opt) + 0.86 - eU) / kBT)
    return np.sum(1. / (1. / jki + 1./jD)) / n_surface_atoms


#def get_Joule_activity(energies, E_opt):
#	'Return the activity per surface atom as done in the Joule paper'
#	return np.sum(np.exp(-np.abs(energies - E_opt) / kBT)) / n_surface_atoms


def get_time_stamp(dt):
	'''
	Return the elapsed time in a nice format.
	
	Parameters
	----------
	dt: float
		Elapsed time in seconds.
		
	Return
	------
	string
		Elapsed time in a neat human-radable format.
	'''
	dt = int(dt)
	if dt < 60:
		return '{}s'.format(dt)
	elif 60 < dt < 3600:
		mins = dt//60
		secs = dt%60
		return '{:d}min{:d}s'.format(mins, secs)
	else:
		hs = dt//3600
		mins = dt//60 - hs*60
		secs = dt%60
		return '{:d}h{:d}min{:d}s'.format(hs, mins, secs)

def make_histogram(filename, energies, sites, ads, bins=20):
	
	# Make figure
	fig, ax = plt.subplots()
	
	if ads == 'OH':
		unique_sites = ontop_sites
	elif ads == 'O':
		unique_sites = hollow_sites
	
	# Plot individual distributions
	for site in unique_sites:
		
		# Split at uppercase letters
		site_list = re.findall('[A-Z][^A-Z]*', site)
		
		# Give pure metals their distinct color, else grey
		if len(set(site_list)) == 1:
			color = metal_colors[site_list[0]]
			zorder = 2
			lw = 1.5
		else:
			color = 'grey'
			zorder = 1
			lw = 0.5
		
		Es = energies[sites == site]		
		ax.hist(Es, bins=bins, color=color, histtype='step', zorder=zorder, lw=lw)

	# Plot total distribution
	ax.hist(energies, bins=bins, color='black', zorder=0, histtype='step')

	# Make legend with the colors of the metals
	marker_params = dict(color='white', marker='o', markersize=15, markeredgecolor='black')
	handles = [Line2D(range(1), range(1), **marker_params, markerfacecolor=metal_colors[metal]) for metal in metals]
	handles.append(Line2D(range(1), range(1), **marker_params, markerfacecolor='black'))
	labels = list(metals) + ['total']
	fig.legend(handles, labels, loc='upper center', ncol=len(metals)+1, fontsize=14,
			   handletextpad=0.1, frameon=True, bbox_to_anchor=(0.46, 1.02), shadow=True,
			   columnspacing=1.2)

	# Hide y-axis ticks and labels as the values are a bit arbitrary
	#ax.tick_params(left=False, labelleft=False)

	# Set axis labels
	ax.set_xlabel('$\mathrm{{\Delta}} E_{{\mathrm{{*{ads}}}}}$ (eV)'.format(ads=ads), fontsize=14)
	ax.set_ylabel('frequency', fontsize=14)

	# Save figure
	fig.savefig(filename, dpi=300, bbox_inches='tight')
	print(f'[SAVED] {filename}')
	
def count_elements(elements, n_elems):
	count = np.zeros(n_elems, dtype=int)
	for elem in elements:
	    count[elem] += 1
	return count

def get_molar_fractions(step_size, n_elems=n_metals, total=1., return_number_of_molar_fractions=False):
	'Get all molar fractions with the given step size'
	
	interval = int(total/step_size)
	n_combs = scipy.special.comb(n_elems+interval-1, interval, exact=True)
	
	if return_number_of_molar_fractions:
		return n_combs
		
	counts = np.zeros((n_combs, n_elems), dtype=int)

	for i, comb in enumerate(it.combinations_with_replacement(range(n_elems), interval)):
		counts[i] = count_elements(comb, n_elems)

	return counts*step_size

def get_random_molar_fractions(n_molar_fractions=1, n_elems=n_metals, random_state=None):
	'Get ´size´ random molar fractions of ´n_elems´ elements'
	if random_state is not None:
		np.random.seed(random_state)

	fs = np.random.rand(n_molar_fractions, n_elems)
	return fs / np.sum(fs, axis=1)[:, None]

def get_molar_fractions_around(f, step_size, total=1., eps=1e-10):
	'Get all molar fractions with the given step size around the given molar fraction'	
	fs = []	
	n_elems = len(f)
	for pair, ids in zip(it.permutations(f, 2), it.permutations(range(n_elems), 2)):
	
		# Get molar fractions and their ids
		f0, f1 = pair
		id0, id1 = ids
		
		# Increment one molar fraction and decrement the other
		f0_new = f0 + (step_size - eps)
		f1_new = f1 - (step_size - eps)
		
		# Ignore if the new molar fractions are not between 0 and 1
		if f0_new <= total and f1_new >= 0.:
			
			# Make new molar fraction
			f_new = deepcopy(f)
			f_new[id0] = f0_new + eps
			f_new[id1] = f1_new - eps
			
			# Append to the output
			assert np.isclose(sum(f_new), 1.), "Molar fractions do not sum to unity : {}. Sum : {:.4f}".format(f_new, sum(f_new))
			fs.append(f_new)
			
	return np.array(fs)

def optimize_molar_fraction(f, func, func_args=[], n_iter_max=1000, step_size=0.005):
	'''
	Return the molar fractions and their function value that locally
	maximizes the specified function starting from the molar fractions ´f´
	'''
	
	# Get the number of decimals to round molar fractions to
	# from the number of decimals of the molar fraction step size
	n_decimals = len(str(step_size).split('.')[1])
	
	# Get the function value of the specified molar fraction
	func_max = float(func(f.reshape(1, -1), *func_args))
	
	# Initiate the number of iterations to zero
	n_iter = 0
	
	while True:
		
		# Raise error if the number of iterations reaches the threshold
		if n_iter == n_iter_max:
			raise ValueError(f'No maximum has been found after {n_iter} iterations,\
							 so convergence is unlikely to happen.')
		
		# Get molar fractions around the current molar fraction in the given step size
		fs_around = get_molar_fractions_around(f, step_size=step_size)
		
		# Get function values
		func_vals = func(fs_around, *func_args)
		
		# Get the largest function value
		func_max_around = np.max(func_vals)
		
		# If the new function value is higher, then repeat for that molar fraction
		if func_max_around > func_max:
			
			# Get the index of largest function value
			idx_max = np.argmax(func_vals)
			
			# Get molar fraction of the maximum
			f = fs_around[idx_max]
			
			# Set the new function maximum
			func_max = func_max_around
			
		# If the function did now improve around the current molar fraction,
		# then the found molar fration is a maximum and is returned
		else:
			# Round the molar fractions to the given number of decimals and 
			# use a trick of adding 0. to make all -0. floats positive
			return np.around(f, decimals=n_decimals) + 0., func_max
		
		# Increment iteration count
		n_iter += 1

def get_local_maxima(fs, func, step_size=0.01, func_args=[]):
	
	# Initiate list containers
	fs_max = []
	funcs_max = []

	for f in fs:
		
		# Get the function value of the current molar fraction
		func_max = func(f.reshape(1, -1), *func_args)

		# Get molar fractions around the current molar fraction in the given step size
		fs_around = get_molar_fractions_around(f, step_size=step_size)

		# Get function values
		func_vals = func(fs_around, *func_args)

		# Get the largest function value
		func_max_around = np.max(func_vals)
		
		# If none of the neighbors have a higher function value,
		# then the current molar fractions is a local maximum
		if func_max_around < func_max:
			
			# Append the found maximum
			fs_max.append(f)
			funcs_max.append(func_max)
		
	return np.asarray(fs_max), np.asarray(funcs_max)

def get_composition(f, metals, return_latex=False, saferound=True):
	
	# Make into numpy and convert to atomic percent
	f = np.asarray(f)*100
	
	if saferound:
		# Round while maintaining the sum, the iteround module may need
		# to be installed manually from pypi: "pip3 install iteround"
		import iteround
		f = iteround.saferound(f, 0)
	
	if return_latex:
		# Return string in latex format with numbers as subscripts
		return ''.join(['$\\rm {0}_{{{1}}}$'.format(m,f0) for m,f0 in\
			zip(metals, map('{:.0f}'.format, f)) if float(f0) > 0.])
	else:
		# Return composition as plain text
		return ''.join([''.join([m, f0]) for m,f0 in\
			zip(metals, map('{:.0f}'.format, f)) if float(f0) > 0.])
		
def get_simplex_vertices(n_elems=n_metals):

	# Initiate array of vertice coordinates
	vertices = np.zeros((n_elems, n_elems-1))
	
	for idx in range(1, n_elems):
		
		# Get coordinate of the existing dimensions as the 
		# mean of the existing vertices
		vertices[idx] = np.mean(vertices[:idx], axis=0)
		
		# Get the coordinate of the new dimension by ensuring it has a unit 
		# distance to the first vertex at the origin 
		vertices[idx][idx-1] = (1 - np.sum(vertices[idx][:-1]**2))**0.5
		
	return vertices

def molar_fractions_to_cartesians(fs):
	
	# Make into numpy
	fs = np.asarray(fs)

	if fs.ndim == 1:
		fs = np.reshape(fs, (1, -1))

	# Get vertices of the multidimensional simplex
	n_elems = fs.shape[1]
	vertices = get_simplex_vertices(n_elems)	
	vertices_matrix = vertices.T
	
	# Get cartisian coordinates corresponding to the molar fractions
	return np.dot(vertices_matrix, fs.T)

def make_triangle_ticks(ax, start, stop, tick, n, offset=(0., 0.),
						fontsize=18, ha='center', tick_labels=True):
	r = np.linspace(0, 1, n+1)
	x = start[0] * (1 - r) + stop[0] * r
	x = np.vstack((x, x + tick[0]))
	y = start[1] * (1 - r) + stop[1] * r
	y = np.vstack((y, y + tick[1]))
	ax.plot(x, y, 'k', lw=1., zorder=0)
	
	if tick_labels:
	
		# Add tick labels
		for xx, yy, rr in zip(x[0], y[0], r):
			ax.text(xx+offset[0], yy+offset[1], f'{rr*100.:.0f}',
					fontsize=fontsize, ha=ha)

def make_ternary_contour_plot(fs, zs, ax, elems, cmap='viridis', levels=10,
							  color_norm=None, filled=False, axis_labels=False,
							  n_ticks=5, tick_labels=True, corner_labels=True):

	# Get cartesian coordinates corresponding to the molar fractions
	xs, ys = molar_fractions_to_cartesians(fs)
	
	# Make contour plot
	if filled:
		ax.tricontourf(xs, ys, zs, levels=levels, cmap=cmap, norm=color_norm, zorder=0)
	else:
		ax.tricontour(xs, ys, zs, levels=levels, cmap=cmap, norm=color_norm, zorder=0)
	
	# Specify vertices as molar fractions
	fs_vertices = [[1., 0., 0.],
				   [0., 1., 0.],
				   [0., 0., 1.]]
	
	# Get cartesian coordinates of vertices
	xs, ys = molar_fractions_to_cartesians(fs_vertices)
	
	# Make ticks and tick labels on the triangle axes
	left, right, top = np.concatenate((xs.reshape(-1,1), ys.reshape(-1,1)), axis=1)
	
	tick_size = 0.025
	bottom_ticks = 0.8264*tick_size * (right - top)
	right_ticks = 0.8264*tick_size * (top - left)
	left_ticks = 0.8264*tick_size * (left - right)
		
	make_triangle_ticks(ax, right, left, bottom_ticks, n_ticks, offset=(0.03, -0.08), ha='center', tick_labels=tick_labels)
	make_triangle_ticks(ax, left, top, left_ticks, n_ticks, offset=(-0.03, -0.015), ha='right', tick_labels=tick_labels)
	make_triangle_ticks(ax, top, right, right_ticks, n_ticks, offset=(0.015, 0.02), ha='left', tick_labels=tick_labels)

	if axis_labels:	
		# Show axis labels (i.e. atomic percentages)
		ax.text(0.5, -0.12, f'{elems[0]} content (%)', rotation=0., fontsize=20, ha='center', va='center')
		ax.text(0.88, 0.5, f'{elems[1]} content (%)', rotation=-55., fontsize=20, ha='center', va='center')
		ax.text(0.12, 0.5, f'{elems[2]} content (%)', rotation=55., fontsize=20, ha='center', va='center')
	
	if corner_labels:
		
		# Define padding to put the text neatly
		pad = [[-0.13, -0.09],
			   [ 0.07, -0.09],
			   [-0.04,  0.09]]
		
		# Show the chemical symbol as text at each vertex
		for idx, (x, y, (dx, dy)) in enumerate(zip(xs, ys, pad)):
			ax.text(x+dx, y+dy, s=elems[idx], fontsize=24)
