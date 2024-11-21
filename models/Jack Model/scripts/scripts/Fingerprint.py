import numpy as np
import itertools as it

class Fingerprint():
	
	def __init__(self, metals, surface_grid, n_atoms_site):
		self.metals = metals
		self.n_metals = len(metals)
		self.surface_grid = np.asarray(surface_grid)
		self.surface_size = surface_grid.shape
		self.n_atoms_site = n_atoms_site
		
		# Get the unique on-top sites
		self.unique_sites = list(it.combinations_with_replacement(range(self.n_metals), self.n_atoms_site))

		# Get the number of unique sites 
		self.n_sites = len(self.unique_sites)
	
class SiteFingerprint(Fingerprint):
	
	def __init__(self, metals, surface_grid, n_atoms_site):
		super().__init__(metals, surface_grid, n_atoms_site)
		
	def get_fingerprint(self, coords, ads_pos, *zone_pos, return_site_idx=False):
		
		# Get metal ids of site
		ads_ids = self.surface_grid[tuple(((ads_pos + coords) % self.surface_size).T)]
		
		# Get the corresponding sorted atomic symbols of the site
		symbols = [self.metals[idx] for idx in ads_ids]
		
		# Sort symbols alphabetically
		site = ''.join(sorted(symbols))
		
		if return_site_idx:
			
			# Get site index of adsorption site
			site_idx = self.unique_sites.index(tuple(sorted(ads_ids)))
			
			# Return the adsorption site and its index
			return site, site_idx
		
		else:
		
			# Return the adsorption site
			return site
		
class ZoneCountingFingerprint(Fingerprint):
	
	def __init__(self, metals, surface_grid, n_atoms_site):
		super().__init__(metals, surface_grid, n_atoms_site)
	
	def get_fingerprint(self, coords, ads_pos, *zone_pos, return_site_idx=False):
	
		# Get element ids of neighbor atoms
		ads_ids = self.surface_grid[tuple(((ads_pos + coords) % self.surface_size).T)]
		zone_ids = [self.surface_grid[tuple(((pos + coords) % self.surface_size).T)] for pos in zone_pos]
		
		# Get site index of adsorption site
		site_idx = self.unique_sites.index(tuple(sorted(ads_ids)))
		
		# Get fingerprint
		fp_site = [0]*self.n_sites
		fp_site[site_idx] = 1
		fp_zones = [sum(zone == elem) for zone in zone_ids for elem in range(self.n_metals)]
		fp = fp_site + fp_zones

		if return_site_idx:
			return fp, site_idx
		else:
			return fp	
		
class FccZoneCountingFingerprint(ZoneCountingFingerprint):
	
	def __init__(self, metals, surface_grid):
		super().__init__(metals, surface_grid, n_atoms_site=3)
		
class OntopZoneCountingFingerprint(ZoneCountingFingerprint):
	
	def __init__(self, metals, surface_grid):
		super().__init__(metals, surface_grid, n_atoms_site=1)
