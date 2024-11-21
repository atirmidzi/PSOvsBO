from ase import Atoms, Atom
from ase.visualize import view
import itertools as it
import numpy as np
from math import factorial as fact
from random import seed, shuffle
from math import factorial

class Slab(object):
	'''Slab class for handling slabs and the environment around an adsorbate'''
	def __init__(self, atoms):
		
		# ase Atoms object
		self.atoms = atoms
		
		# number of atoms in slab
		self.n_atoms = len(atoms)
		
		# atoms repeated 3x3x1 times					
		self.slab_3x3 = atoms.repeat((3, 3, 1))
		
		# number of metals in slab
		self.n_atoms_metals = sum(1 for atom in atoms if atom.tag != 0)
		
		# number of adsorbate atoms in slab
		self.n_atoms_ads = self.n_atoms - self.n_atoms_metals
		
		# number of layers in slab
		tags = self.atoms.get_tags()
		tags_set = set(tags)
		self.n_layers = sum(1 for tag in tags_set if tag != 0)
		
		# number of atoms in each layer
		self.n_atoms_layer = self.n_atoms_metals // self.n_layers
		
		# adsorbing atom symbol (interpreted as the first adsorbate atom in the list of atoms)
		ads_symbol = [atom.symbol for atom in atoms if atom.tag == 0][0]

		# atom ids of all of the adsorbing atoms on the 3x3 slab
		ads_ids = np.where(np.asarray(self.slab_3x3.get_chemical_symbols()) == ads_symbol)[0]

		# atom id for the most centrally located adsorbing atom on the 3x3 slab
		self.ads_id = ads_ids[len(ads_ids) // 2]
		
		# xy coordinate of adorption site center
		self.center = None
	
	
	def view(self):
		'''view atoms object'''	
		view(self.atoms)
		
	def closest(self, start, stop, layer=None):
		''''return atom ids of the metals closest to the adsorbing atom for the 3x3 slab
		
		start	int		1: closest metal, 2: next closest, etc.
		
		stop	int		1: closest metal, 2: next closest, etc.
		
		layer	int		restrict the search to a given layer of the slab
		
		e.g. slab.closest(1, 3) returns the atom ids of the 1st, 2nd, and 3rd closest metals
			 slab.closest(3, 9) returns the atom ids of the 3rd, 4th, ..., and 9th
			 closest metals'''
		
		# atom ids of metals to measure distances to
		if layer is not None:
			ids = [atom.index for atom in self.slab_3x3 if atom.tag == layer]
		else:
			ids = [atom.index for atom in self.slab_3x3 if atom.tag != 0]
		
		# distances from adsorbing atom
		dists = self.slab_3x3.get_distances(self.ads_id, ids)
		
		# sort ids by distance and pick the indices given by start and stop
		return [idx for _, idx in sorted(zip(dists, ids))][start - 1: stop]
	
	
	def closest_to_site_center(self, start, stop, layer=None, ens_size=None):
		''''Return atom ids of the metals closest to the adsorption site center for the 3x3 slab
		
		start	int		1: closest metal, 2: next closest, etc.
		
		stop	int		1: closest metal, 2: next closest, etc.
		
		layer	int		Restrict the search to a given layer of the slab.
						1 is the surface, 2 is the subsurface, etc.
		
		ens_size int	Number of atoms in the adsorption site.
						This is necessary to define the center of the adsorption site.
						1 for on-top, 2 for bridge, and 3 for hollow sites.
		
		e.g. slab.closest(1, 3) returns the atom ids of the 1st, 2nd, and 3rd closest metals
			 slab.closest(3, 9) returns the atom ids of the 3rd, 4th, ..., and 9th
			 closest metals'''
		
		
		# xy coordinate of adsorption site center
		if self.center is None:
			if ens_size is None:
				raise ValueError('Adsorption center is not defined. Specify the size of the adsorption site, *ens_size*')
			self.center = self.site_center(start=1, stop=ens_size, layer=1)
		
		if layer is not None:
			
			# ids of atoms to measure distances to
			ids = [atom.index for atom in self.slab_3x3 if atom.tag == layer]
			
			# xy positions of atoms
			pos = np.asarray([atom.position[:2] for atom in self.slab_3x3 if atom.tag == layer])
		
		else:
			
			# ids of atoms to measure distances to
			ids = [atom.index for atom in self.slab_3x3 if atom.tag != 0]
			
			# xy positions of atoms
			pos = np.asarray([atom.position[:2] for atom in self.slab_3x3 if atom.tag != 0])
		
		# xy distances to adsorption site center
		dists = np.sqrt(np.sum(np.square(pos - self.center), axis=1))
		
		# sort ids by distance and pick the indices given by start and stop
		return [idx for _, idx in sorted(zip(dists, ids))][start - 1: stop]
		
		
	def distorted(self, expansion=1.10, slab_id=None):
		'''return True if a surface atom has distorted more than *expansion* in the z direction
		
		expansion	float	max allowed increase in slab height, 
							e.g. 1.10 allows for a 10% increase in the slab height
							
		slab_id		int		id to print in case the slab is distorted
		'''
		
		# number of layers
		max_tag = max([atom.tag for atom in self.atoms])
		
		# distance between two bottom layers
		z_bottom = min([atom.position[2] for atom in self.atoms if atom.tag == max_tag])
		z2 = max([atom.position[2] for atom in self.atoms if atom.tag == max_tag-1])
		layer_dist = z2 - z_bottom
		
		# initial slab height
		height_init = (max_tag-1)*layer_dist
		
		# height of relaxed slab
		z_surface = max([atom.position[2] for atom in self.atoms if atom.tag == 1])
		height = z_surface - z_bottom
		
		# expansion of height
		height_expand = height / height_init
		
		# if the slab height expansion is greater than the threshold
		if height_expand > expansion:
			if slab_id is not None:
				print('row id %d has expanded %.2f times (threshold: %.2f)'
					  %(slab_id, height_expand, expansion))
			return True
			
		# if the height is smaller than the threshold
		else:
			return False
			
	
	def site_center(self, start, stop, layer=None):
		'''return xy coordinate of the center of an ensemble of atoms
		
		start	int		1: closest metal, 2: next closest, etc.
		
		stop	int		1: closest metal, 2: next closest, etc.
		
		layer	int		index of layer (1 = surface, 2 = subsurface, ...).
						if not specified, then a search independent of layers is performed
		'''
		
		# get closest metal ids given *start*, *stop*, and *layer*
		metal_ids = self.closest(start, stop, layer)
		
		# position of atoms in zone
		pos = np.asarray([atom.position for atom in self.slab_3x3 if atom.index in metal_ids])
		
		# number of atoms in zone
		zone_size = stop - start + 1
		
		# xy-position of zone
		return np.sum(pos.T[:2], axis=1) / zone_size
	
	
	def dist_to_center(self, start, stop, layer=None):
		'''return xy-distance from the adsorbing atom to the center of an ensemble of atoms
		
		start	int		1: closest metal, 2: next closest, etc.
		
		stop	int		1: closest metal, 2: next closest, etc.
		
		layer	int		index of layer (1 = surface, 2 = subsurface, ...).
						if not specified, then a search independent of layers is performed
		'''
		
		# xy-position of adsorbate atom of 3x3 slab
		pos_ads = self.slab_3x3.get_positions()[self.ads_id][:2]

		# xy-coordinate of center of zone
		center = self.site_center(start, stop, layer)
		
		# xy-distance from adsorbate atom to center of zone
		return np.sqrt(sum(d**2 for d in (center - pos_ads)))
		
	
	def out_of_site(self, zone_sizes, distance=0.6, slab_id=None, row_id=None, facet=None):
		'''return True if the adsorbate is closer than *distance* (in xy-plane) from the center
		of the first zones in the two first layers.
		
		zone_sizes	int		number of atoms in each zone grouped into layers,
							e.g. [[1, 6], [3]]
		
		distance	float	limiting distance (A) to be classified as the site assumed by *ens_size*
		
		slab_id		int		id of slab to print if the adsorbate is out of site
		
		row_id		int		row id to print if the adsorbate is out of site
		
		facet		str		surface facet, e.g. '111', '100'
		'''

		# loop through layers
		for layer, zone_sizes_layer in enumerate(zone_sizes):
			
			# restrict to first two layers only
			if layer == 2: break
				
			# loop through zones in layer
			for zone, zone_size in enumerate(zone_sizes_layer):
				
				# restrict to only first zone
				if zone == 1: break
				
				# xy-distance from adsorbing atom to the center of the zone
				xy_dist = self.dist_to_center(start = 1 + sum(zone_sizes_layer[:zone]),
								   			  stop = sum(zone_sizes_layer[:zone + 1]),
								   			  layer = layer + 1)
				
				# if more than *distance* from the center of any zone
				if xy_dist > distance:
					
					# then the adsorbing atom is out of its intended site
					if slab_id is not None and facet is not None:
						print('row id {:d}, slab id {:d}, distance: {:.2f} A (threshold {:.2f} A). Site: {} (intended: {})'.format(row_id, slab_id, xy_dist, distance, self.get_site(facet), self.get_site(facet, zone_sizes)))
					
					elif slab_id is not None:
						print('Slab id: %d distance: %.2f A (threshold %.2f A)'
							  %(slab_id, xy_dist, distance))
					
					return True
		
		# if within *distance* for all zones, then the adsorbing is not out of its intended site
		return False
	
	def get_111_site(self, zone_sizes=None, force_hollow=False):
		'''return site of 111 surface as string: 'fcc', 'hcp', 'ontop', 'bridge' 
		
		zone_sizes	list of lists of ints	number of atoms in each zone grouped by layer.
											If specified, then a search is done based on this
											and not the atoms object, i.e. the intended site is
											found.
											
		force_hollow	bool 	if True, all sites will be classified as either 'fcc' or 'hcp'
		'''
		
		# if *zone_sizes* is given then return its corresponding site
		if zone_sizes is not None:
		
			# loop through layers
			for layer, layer_zone_sizes in enumerate(zone_sizes):
			
				# only the two top layers are necessary for 111 surfaces
				if layer == 2: break
				
				# loop through zones
				for zone, zone_size in enumerate(layer_zone_sizes):
					
					# only first zone in each layer are necessary for 111 surfaces
					if zone == 1: break
					
					if layer == 0 and zone_size == 1:
						return 'ontop'
					
					if layer == 0 and zone_size == 2:
						return 'bridge'
					
					if layer == 1 and zone_size == 1:
						return 'hcp'
						
					if layer == 1 and zone_size == 3:
						return 'fcc'
		
		
		# if classifying all sites as hollow sites
		if force_hollow:
			return self.get_hollow_site()
					
		## if *zone_sizes* is not specified:
		# adsorption ensemble sizes (ontop, bridge, hollow)
		ens_sizes = (1, 2, 3)
		
		# xy-distances to surface ensemble centers, 
		# i.e. distance to ontop, bridge, and hollow center
		xy_dists = [self.dist_to_center(1, ens_size, layer=1) for ens_size in ens_sizes]
		
		# the center closest to the adsorbate is defined as the adsorption site
		# index of minimum distance
		min_dist, idx = min((dist, idx) for (idx, dist) in enumerate(xy_dists))
		
		if idx == 0:
			return 'ontop'
		if idx == 1:
			return 'bridge'
		if idx == 2:
			return self.get_hollow_site()

	def get_hollow_site(self):
		'return whether the site is an fcc or hcp site'
		
		# check if ensemble in subsurface has size 1 (hcp) or 3 (fcc)
		ens_sizes = (1, 3)
		
		# xy-distances to ensemble centers
		xy_dists = [self.dist_to_center(1, ens_size, layer=2) for ens_size in ens_sizes]
		
		# index of minimum distance
		min_dist, idx = min((dist, idx) for (idx, dist) in enumerate(xy_dists))
		
		# if closest to a single atom in the subsurface
		if idx == 0:
			return 'hcp'
		
		# if closest to an ensemble of three atoms in the subsurface
		if idx == 1:
			return 'fcc'
				
	def get_site(self, facet='111', *args, **kwargs):
		'''return site of adsorbate as string, detected by closest symmetry point
		
		facet	str		surface facet, e.g. '111', '100'
		'''
	
		if facet == '111':
			return self.get_111_site(*args, **kwargs)

	
	def one_hot_fingerprint(self, metals, relative=True):
		'''return one-hot fingerprint of slab
		
		metals	list of Strings		metals in alloy
		
		relative bool				whether to consider the atom positions in the slab relative to the adsorbate
		'''
		
		metals = np.array(metals)
		
		# number of different metals
		n_metals = len(metals)
		
		# symbols of metal atoms
		symbols = np.array([atom.symbol for atom in self.atoms if atom.tag != 0])
		n_symbols = len(symbols)
		
		# initiate fingerprint
		fp = np.zeros(n_symbols*n_metals, dtype=int)
		
		if relative:
			
			# index of adsorbing atom
			ids_ads = [atom.index for atom in self.atoms if atom.tag == 0]
			idx_ads = min(ids_ads)
			
			# indices of metal atoms
			ids_metals = [atom.index for atom in self.atoms if atom.tag != 0]
			
			# one-hot encode metals according to layers
			# and according to their proximity to the adsorbing atom
			# this is to make the encoding usable also for e.g. hollow sites
			for layer_idx in range(self.n_layers):
				
				layer_metal_ids = ids_metals[layer_idx*self.n_atoms_layer : (layer_idx + 1)*self.n_atoms_layer]
				layer_metals = symbols[layer_metal_ids]
				
				dists = self.atoms.get_distances(idx_ads, layer_metal_ids, mic=True)
				
				# ids in ascending order of distance
				layer_ids = np.argsort(dists)
				
				# one-hot encode the layer
				for i, s in enumerate(layer_metals[layer_ids]):
					idx = np.where(metals == s)[0]
					fp[layer_idx*self.n_atoms_layer*n_metals + n_metals*i + idx] = 1
				
		else:		
		
			# one-hot encode the fingerprint
			for i, s in enumerate(symbols):
				idx = np.where(metals == s)[0]
				fp[i*n_metals+idx] = 1
			
		return fp
		
	def fingerprint(self, metals, zone_sizes, include, count_metals):
		'''return fingerprint of slab
		
		metals		list of Strings			metals in alloy
		
		zone_sizes	list of lists of ints	number of atoms in each zone orderer in  layers,
											e.g. [[1, 6], [3]]
											
		include		list of lists of ints	number of atoms to include from the zones,
											if atoms are missing, a random pick will be 
											made in the zone,
											e.g. [[1, 6], [1]]
											
		count_metals	list of lists of bools	assign for each zone whether counting of metals 											or combination parameters are wanted,
											e.g. [[False, True], [True]]
		'''
		
		# number of metals
		n_metals = len(metals)
		
		# number of layers
		n_layers = len(zone_sizes)
		
		# initiate fingerprint
		fp = []
		
		# loop through layers
		first_iter = True
		for layer in range(n_layers):
			
			zone_sizes_layer = zone_sizes[layer]
			include_layer = include[layer]
			count_metals_layer = count_metals[layer]
			
			# loop through zones in layer
			for zone, (zone_size, zone_include, zone_count_metals) \
			in enumerate(zip(zone_sizes_layer, include_layer, count_metals_layer)):
				
				# if the number of atoms to use in the zone is zero
				# then the fingerprint is dropped for this zone
				if zone_include == 0:
					continue
				
				if first_iter:
					# find xy coordinate of the center of the adsorption site
					self.center = self.site_center(start = 1 + sum(zone_sizes_layer[:zone]),
												   stop = sum(zone_sizes_layer[:zone+1]),
												   layer = layer+1)
					first_iter = False
				
				# atom ids on 3x3 slab
				ids_3x3 = self.closest_to_site_center(start = 1 + sum(zone_sizes_layer[:zone]),
													  stop = sum(zone_sizes_layer[:zone+1]),
													  layer = layer+1)
				
				# if not all atoms in the zone are to be included
				if zone_include != zone_size:
				
					# shuffle the ids
					shuffle(ids)
					
					# pick the *zone_include* first elements to use
					ids_3x3 = ids_3x3[:zone_include]
					
				# atomic symbols of zone
				symbols = self.slab_3x3[ids_3x3].get_chemical_symbols()
				
				# if fingerprint is constructed by counting metals
				if zone_count_metals:
					
					# count elements and append to total fingerprint
					fp.append(count_symbols(symbols, metals))
					
				# if fingerprint is constructed by combination parameters
				else:
					
					# sort symbols alphabetically
					symbols.sort()
					
					# listed possible zone ensembles
					ensembles = list(it.combinations_with_replacement(metals, zone_size))
					
					# index of the current zone ensemble
					ens_id = ensembles.index(tuple(symbols))
					
					# number of zone ensembles
					n_ens = len(ensembles)
					
					# initiate zone fingerprint
					zone_fp = [0]*n_ens
					
					# add a 1 at the current ensembles index
					zone_fp[ens_id] = 1
					
					# append to total fingerprint
					fp.append(zone_fp)
		
		# return the flattened fingerprint
		return list(it.chain.from_iterable(fp))
		
	
def count_symbols(symbols, reference):
	'''return list of counts of each  element in *symbols* ordered according to *reference*
	
	symbols		list of strs	elements symbols to count, e.g. (Ir, Pd)
	
	reference	list of strs	elements to count according to, e.g. (Ir, Pd, Pt, Rh, Ru)
	'''
	
	# initiate counter list
	counts = [0]*len(reference)
	
	# loop through elements
	for symbol in symbols:
		for i, ref in enumerate(reference):
			if symbol == ref:
				counts[i] += 1

	return counts
