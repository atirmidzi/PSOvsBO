import sys
sys.path.append('../../..')
from scripts import Slab, metals, ontop_sites, hollow_sites
import numpy as np
from ase.db import connect
from collections import Counter
import itertools as it

# Get Pt(111) reference energies
refs = {}
with connect('../../../Pt_reference/Pt_out.db') as db:
	refs['OH'] = db.get(type='OH').energy - db.get(type='slab').energy
	refs['O'] = db.get(type='O').energy - db.get(type='slab').energy

# Maximum force allowed
fmax = 0.1

# Connect to slabs database
with connect('slabs.db') as db_slab:

	# Iterate through adsorbates
	for ads in ['OH', 'O']:
		
		# Define adsorbate-specific parameters
		if ads == 'OH':
			desired_site = 'ontop'
			n_atoms_site = 1
			sites = ontop_sites
			db_kwargs = dict(O=1, H=1)
			zone_kwargs = [dict(start=1, stop=1, layer=1),
						   dict(start=2, stop=7, layer=1),
						   dict(start=1, stop=3, layer=2),
						   dict(start=4, stop=6, layer=3)]
	
		elif ads == 'O':
			desired_site = 'fcc'
			n_atoms_site = 3
			sites = hollow_sites
			db_kwargs = dict(O=1, H=0)
			zone_kwargs = [dict(start=1, stop=3,  layer=1),
					 	   dict(start=4, stop=6,  layer=1),
					 	   dict(start=7, stop=12, layer=1),
					 	   dict(start=1, stop=3,  layer=2),
						   dict(start=4, stop=6,  layer=2)]
		
		# Connect to adsorbate database
		with connect(f'{ads}.db') as db_ads:

			# Get the number of unique sites
			n_sites = len(sites)

			# Define filename of output file
			filename = '{:s}.csv'.format(ads)
		
			# Open file in write mode. This will overwrite any previous file
			with open(filename, 'w') as file_:
		
				# Write header to file
				file_.write(f'# features, adsorption energy relative to Pt(111) (eV), {ads}.db row id, slab.db row id\n')
		
				# Iterate through the desired database entries
				for row in db_ads.select('energy', **db_kwargs):

					# Check that the maximum force is not too large
					if row.fmax > fmax:
						continue
			
					# Get atoms object
					atoms = db_ads.get_atoms(row.id)

					# Load slab object
					slab = Slab(atoms)
	
					# Get surface site
					site = slab.get_site()
	
					# Skip if the adsorbate is not adsorbed at the desired site
					if site != desired_site:
						continue
			
					# Get symbols of metals
					symbols = atoms.get_chemical_symbols()[:-len(ads)]
			
					# Count the number of each symbol
					symbols_count = Counter(symbols)
			
					# Get the corresponding slab without the adsorbate
					for row_slab in db_slab.select('energy', O=0, H=0, **symbols_count):
				
						# Check that the maximum force is not too large
						if row_slab.fmax > fmax:
							continue
				
						slab_atoms = db_slab.get_atoms(row_slab.id)
						slab_symbols = slab_atoms.get_chemical_symbols()
				
						if np.all(symbols == slab_symbols):
							slab_energy = row_slab.energy
							break
					else:
						print(f'[WARNING] No match found in slab.db for {ads}.db row {row.id}. Skipping..')
						continue

					# Get indices of site atoms in the 3x3 repeated surface
					zone_atom_ids = []
					for kwargs in zone_kwargs:
						atom_ids = slab.closest_to_site_center(**kwargs, ens_size=n_atoms_site)
						zone_atom_ids.append(atom_ids)
	
					# Get index of the adsorption site
					site_symbols = ''.join(sorted(slab.slab_3x3[zone_atom_ids[0]].get_chemical_symbols()))
					site_idx = sites.index(site_symbols)
	
					# Get the count of symbols for the remaining zones
					zone_counts = []
					for atom_ids in zone_atom_ids[1:]:
						count = Counter(slab.slab_3x3[atom_ids].get_chemical_symbols())
						zone_counts.append(count)

					# Make fingerprint
					fp_site = [0]*n_sites
					fp_site[site_idx] = 1
					fp_zones = [count[m] for count in zone_counts for m in metals]
					fp = fp_site + fp_zones
	
					# Get adsorption energy relative to the reference
					energy = row.energy - slab_energy - refs[ads]
		
					# Write line with fingerprint, energy, adsorbate row id, and slab row id to file.
					# The availability of the row ids makes it easier to locate the structures corresponding
					# to the rows afterwards
					file_.write(','.join(map('{:d}'.format, fp)) + f',{energy:.6f},{row.id},{row_slab.id}\n')
