import numpy as np
from time import time
from math import sqrt
import scipy.special
import itertools as it
import iteround
from Fingerprint import OntopZoneCountingFingerprint, FccZoneCountingFingerprint, SiteFingerprint
from system_importer import hollow_sites

class BruteForceSurface():
	
	# Define relative ids of on-top neighbor atoms
	ontop_1a = np.array([(0,0,0)])
	ontop_1b = np.array([(0,-1,0), (-1,0,0), (-1,1,0), (0,1,0), (1,0,0), (1,-1,0)])
	ontop_2a = np.array([(-1,0,1), (0,-1,1), (0,0,1)])
	ontop_3a = np.array([(-1,-1,2), (-1,0,2), (0,-1,2)])
	
	# Define relative ids of fcc neighbor atoms
	fcc_1a = np.array([(0,1,0), (1,0,0), (1,1,0)]) 
	fcc_1b = np.array([(0,0,0), (0,2,0), (2,0,0)])
	fcc_1c = np.array([(-1,1,0), (-1,2,0), (1,-1,0), (2,-1,0), (1,2,0), (2,1,0)])
	fcc_2a = np.array([(0,0,1), (0,1,1), (1,0,1)])
	fcc_2b = np.array([(-1,1,1), (1,-1,1), (1,1,1)])
	
	# Define relative neighbor index coordinates of on-top and fcc sites
	ontop_neighbors = np.array([(1,0), (0,1), (1,1)])
	fcc_neighbors = -ontop_neighbors
	
	def __init__(self, f, reg_OH, reg_O, size=(100, 100),
				 fp_OH='ontop zone counting', fp_O='fcc zone counting',
				 displace_OH=-0., displace_O=0., 
				 scale_O=0.5):
		'''
		f	dict	molar fractions as a dictionary: {metal: molar fraction}
		'''
		# Metal parameters
		self.metals = sorted(list(f.keys()))
		self.n_metals = len(self.metals)
		self.alloy = ''.join(self.metals)
		
		# Size parameters
		self.size = size
		self.nrows, self.ncols = size
		n_atoms_surface = np.prod(size)
		
		if fp_OH == 'ontop zone counting' or fp_O == 'fcc zone counting':
			n_layers = 3
		else:
			n_layers = 1
		
		# Get the total number of atoms
		n_atoms = n_atoms_surface * n_layers
		
		# Make molar fractions into numpy array
		self.f = np.asarray([f[metal] for metal in self.metals])
		
		# Displace OH and scale O adsorption energies
		self.displace_OH = displace_OH
		self.displace_O = displace_O
		self.scale_O = scale_O
		
		# Get the number of occurence of each metal
		n_each_metal = {metal_idx: self.f[metal_idx]*n_atoms for metal_idx in range(self.n_metals)}
		
		# Round the number of metals to integer values while maintaining the sum
		n_each_metal = iteround.saferound(n_each_metal, 0)
		
		# Get metal indices corresponding best to the molar fractions
		metal_ids = list(it.chain.from_iterable([[metal_idx]*int(n) for metal_idx, n in n_each_metal.items()]))
		
		# Shuffle to get randomly distributed elements
		np.random.shuffle(metal_ids)
		
		# Make 3D grid of atoms
		self.grid = np.reshape(metal_ids, (*size, n_layers))

		# Initiate arrays of 2D site indices grids
		self.ontop_grid = self.grid[:, :, 0]		
		self.fcc_grid = np.zeros(size, dtype=np.uint8)	
		
		# Load adsorbate-specific fingerprint readers
		if fp_OH == 'ontop zone counting':
			self.fp_OH = OntopZoneCountingFingerprint(self.metals, self.grid)
		else:
			self.fp_OH = SiteFingerprint(self.metals, self.grid, n_atoms_site=1)
		
		if fp_O == 'fcc zone counting':
			self.fp_O = FccZoneCountingFingerprint(self.metals, self.grid)
		else:
			self.fp_O = SiteFingerprint(self.metals, self.grid, n_atoms_site=3)

		# Load adsorbate-specific regressors
		self.reg_OH = reg_OH
		self.reg_O = reg_O
		
		# Make zero-filled energy grid of adsorption energies for each adsorbate
		self.energy_grid_OH_orig = np.zeros(size)
		self.energy_grid_O_orig = np.zeros(size)
		
		# Initialize boolean containers for whether an adsorption has occured
		self.ontop_ads_sites = np.zeros(size, dtype=bool)
		self.fcc_ads_sites = np.zeros(size, dtype=bool)
		
		# Initialize boolean containers for whether a blocking has occured
		#self.ontop_blocked_sites = np.zeros(size, dtype=bool)
		#self.fcc_blocked_sites = np.zeros(size, dtype=bool)
		
		# Define the number of atoms in on-top and fcc hollow sites
		#n_atoms_site = dict(ontop=1, fcc=3)
		
		# Get the unique adsorption site labels
		#self.sites_ontop = [''.join(comb) for comb in it.combinations_with_replacement(metals, n_atoms_site['ontop'])]
		#self.sites_fcc = [''.join(comb) for comb in it.combinations_with_replacement(metals, n_atoms_site['fcc'])]
		#self.sites = self.sites_ontop + self.sites_fcc
		
		#ids = self.ontop_grid.ravel()
		
		# Initialize energy grids as None to tell that they have not yet been assigned
		# adsorption energies with 'get_gross_energies'
		self.energy_grid_OH = None
		self.energy_grid_O = None
		
	def get_gross_energies(self):
		'Determine adsorption energies of surface sites'
		
		# Get adsorption energies of grid points
		for row in range(self.nrows):
			for col in range(self.ncols):
	
				# Current site coordinates
				coords = (row, col, 0)

				# Get OH fingerprint
				fp_OH = self.fp_OH.get_fingerprint(coords, self.ontop_1a,
									self.ontop_1b, self.ontop_2a, self.ontop_3a)
				
				# Predict OH energy
				self.energy_grid_OH_orig[row, col] = self.reg_OH.predict(fp_OH) + self.displace_OH
				
				# Get O fingerprint and fcc hollow adsorption site index
				fp_O, site_idx = self.fp_O.get_fingerprint(coords, self.fcc_1a,
									self.fcc_1b, self.fcc_1c, self.fcc_2a, self.fcc_2b,
									return_site_idx=True)
				
				self.fcc_grid[row, col] = site_idx
				
				# Predict O energy
				self.energy_grid_O_orig[row, col] = self.scale_O * (self.reg_O.predict(fp_O) + self.displace_O)
		
		# Save original adsorption energy grids for later
		# while using a masked array for updates
		self.energy_grid_OH = np.ma.masked_array(self.energy_grid_OH_orig)
		self.energy_grid_O = np.ma.masked_array(self.energy_grid_O_orig)
		
		return self

	def get_net_energies(self):
		'Determine adsorption energies of surface sites after mutual blocking of sites'
		
		# If gross adsorption energies have not been asigned yet, then do it
		if (self.energy_grid_OH is None) or (self.energy_grid_O is None):
			self.get_gross_energies()
			
		while True:
	
			# If all sites are blocked or adsorbed on, then break the while loop
			if np.all(self.energy_grid_OH.mask) and np.all(self.energy_grid_O.mask):
				break
		
			# Get lowest available adsorption energy for both ontop and hollow sites
			if np.all(self.energy_grid_OH.mask):
				# If all ontop sites have been taken then set the energy to infinity
				ontop_energy_min = np.inf
			else:
				ontop_energy_min = np.min(self.energy_grid_OH)

			if np.all(self.energy_grid_O.mask):
				# If all hollow sites have been taken then set the energy to infinity
				hollow_energy_min = np.inf
			else:
				hollow_energy_min = np.min(self.energy_grid_O)
	
			# Find which of the ontop and hollow site energy minimum is smaller
			hollow_is_smallest = np.argmin([ontop_energy_min, hollow_energy_min])

			if hollow_is_smallest:

				# Get list index of minimum
				ids_min = np.argmin(self.energy_grid_O)

				# Get grid indices of minimum
				ids = np.unravel_index(ids_min, self.size)

				# Get minimum energy
				E_min = self.energy_grid_O[ids]

				# Block this hollow site because of adsorption
				self.energy_grid_O[ids] = np.ma.masked
		
				# Mark this hollow site as adsorbed on
				self.fcc_ads_sites[ids] = True

				# Get ontop site array indices to block
				block_ids = ((self.ontop_neighbors + ids) % self.size).T
				
				# Block neighboring ontop sites using masking
				self.energy_grid_OH[tuple(block_ids)] = np.ma.masked
				
				# Mark the neighboring on-top sites as blocked
				#self.ontop_blocked_sites[tuple(block_ids)] = True

			else:

				# Get list index of minimum
				ids_min = np.argmin(self.energy_grid_OH)

				# Get grid indices of minimum
				ids = np.unravel_index(ids_min, self.size)

				# Get minimum energy
				E_min = self.energy_grid_OH[ids]

				# Block this ontop site because of adsorption
				self.energy_grid_OH[ids] = np.ma.masked
		
				# Mark this ontop site as adsorbed on
				self.ontop_ads_sites[ids] = True

				# Get fcc hollow site array indices to block
				block_ids = ((self.fcc_neighbors + ids) % self.size).T
				
				# Block neighboring hollow sites using masking
				self.energy_grid_O[tuple(block_ids)] = np.ma.masked
				
				# Mark the neighboring fcc sites as blocked
				#self.fcc_blocked_sites[tuple(block_ids)] = True
		
		return self
			
	def get_OH_energies(self):
		'Return the OH adsorption energies for which an adsorption has occured'
		return self.energy_grid_OH_orig[self.ontop_ads_sites]
	
	def get_O_energies(self):
		'Return the O adsorption energies for which an adsorption has occured'
		return self.energy_grid_O_orig[self.fcc_ads_sites]
		
	def show(self, lat=3.8, h_OH=2.0, h_O=1.2, orthogonal=False, size_include=None, randomize_H=False, show=True):
		'Plot and show surface as an ase atoms object'
		
		from ase import Atoms
		from ase.visualize import view
		
		if orthogonal and size_include is not None:
			print('[WARNING] Setting ´size_include´ and ´orthogonal´=True currently does not function properly')
		
		# Show the whole slap as default
		if size_include is None:
			size_include = self.grid.shape
		
		# Include all layers if the depth is not specified
		if len(size_include) == 2:
			size_include = (*size_include, self.grid.shape[2])
		
		# Get atomic symbols
		symbols_grid = np.array([[[self.metals[idx] for idx in col] for col in row] for row in self.grid])
		symbols_grid = symbols_grid[:size_include[0], :size_include[1], :size_include[2]]
		
		# Make symbols grid into a list
		symbols = np.transpose(symbols_grid, [2,0,1]).ravel()
		
		# Get atomic positions of metals
		positions = get_fcc111_positions(size=self.grid.shape, lat=lat, orthogonal=orthogonal)
		positions = positions[:size_include[0], :size_include[1], :size_include[2], :]
		
		# Define the slab of metals as an atoms object
		# while making sure to transpose and reshape the positions array
		# to follow the order of the symbols correctly
		atoms = Atoms(symbols, np.transpose(positions, [2,1,0,3]).reshape(-1, 3))

		# Get grid indices where OH has adsorbed at.
		# The grid is transposed to reflect how row and column numbers
		# transfer into y and x coordinates respectively
		include_ontop = self.ontop_ads_sites[:size_include[0], :size_include[1]]
		row_ids, col_ids = np.nonzero(include_ontop.T)
		
		# Get the number of OH adsorbates
		n_OH = len(row_ids)
		
		# Get coordinates of O and H in the OH adsorbates as the 
		# top layer coordinates with the height of the OH adsorbate added
		O_positions = positions[row_ids, col_ids, 0, :] + [0., 0., h_OH]
		
		if randomize_H:
			n_H = len(positions[row_ids, col_ids, 0, :])
			xs, ys = (np.random.rand(n_H, 2) - 0.5).T
			H_positions = positions[row_ids, col_ids, 0, :] + np.asarray([xs, ys, [h_OH + 0.7]*n_H]).T
		else:
			H_positions = positions[row_ids, col_ids, 0, :] + [0.5, 0.5, h_OH + 0.5]
		
		# Add OH adsorbates to atoms object
		Os = Atoms(['O']*n_OH, O_positions)
		Hs = Atoms(['H']*n_OH, H_positions)
		atoms += Os + Hs
		
		# Get grid indices where OH has adsorbed at
		# The grid is transposed to reflect how row and column numbers
		# transfer into y and x coordinates respectively
		include_fcc = self.fcc_ads_sites[:size_include[0], :size_include[1]]
		row_ids, col_ids = np.nonzero(include_fcc.T)
		
		# Get the number of O adsorbates
		n_O = len(row_ids)
		
		# Get coordinates of O adsorbates as the 
		# top layer coordinates with the height of the O adsorbate added
		O_positions = positions[row_ids, col_ids, 0, :] + [lat / sqrt(2), -lat / sqrt(6), h_O]
		
		# Add OH adsorbates to atoms object
		Os = Atoms(['O']*n_O, O_positions)
		atoms += Os
		
		if show:
			# Show structure
			view(atoms)
		
		return atoms

def get_transformation_matrix(lat):
	'Return matrix for converting from lattice coordinates to cartesian coordinates'
	
	# Define shorthand distances
	d = lat / sqrt(2)
	h = lat*sqrt(3) / (2*sqrt(2))
	z = lat / sqrt(3)
	b = lat / sqrt(24)
	
	# Define lattice vectors
	x1 = np.array([d, 0, 0])
	x2 = np.array([d/2, -h, 0])
	x3 = np.array([d/2, -b, -z])
	
	# Return coordinate transformation matrix
	return np.array([x1, x2, x3]).T

def get_fcc111_positions(size, lat, orthogonal=False):
	'Return x,y,z coordinates of atom positions of an fcc111 facet'
	
	# Get the lattice transformation matrix
	T = get_transformation_matrix(lat)
	
	# Initiate positions as a 4D array
	positions = np.zeros((*size, 3))
	
	# Get cartesian coordinates of all lattice points
	for vector in it.product(*[range(s) for s in size[::-1]]):
		
		# Reverse the order of ´vector´ in order to iterate over
		# coordinates in the order x,y,z instead of z,y,x)
		vector = vector[::-1]
		
		if orthogonal:
			n_move = vector[1] // 2
			
			if vector[0] >= size[0] - n_move:

				# Redefine ´vector´ in order to move this atom
				# according to the periodicity
				vector = tuple([vector[0] - size[0], *vector[1:]])

		# Get cartesian coordinates
		positions[vector] = np.dot(T, np.array(vector).T)

	# Return coordinates of lattice points
	return positions
