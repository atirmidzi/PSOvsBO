import os
import sys
import itertools as it

def find_file(name, path):
	for root, dirs, files in os.walk(path):
		if name in files:
			return os.path.join(root, name)

# Locate the file defining the alloy system by
# looking for the file ´system.py´ in the parent
# directory of the current working directory
system_file = find_file('system.py', os.path.dirname(os.getcwd()))

if system_file is None:
	# Look in the grandparent directory
	system_file = find_file('system.py', os.path.dirname(os.path.dirname(os.getcwd())))

try:
	path = system_file.rsplit('/', 1)[0]
	sys.path.append(path)
except AttributeError:
	pass

try:
	from system import metals
except ImportError:
	pass

if 'metals' not in locals():
	default_alloy = 'AgIrPdPtRu'
	alloy = default_alloy
	import re
	metals = re.findall('[A-Z][^A-Z]*', alloy)	

# Get alloy string
alloy = ''.join(metals)

# Get number of metals
n_metals = len(metals)

# Get unique adsorption sites
ontop_sites = list(''.join(sym) for sym in it.combinations_with_replacement(metals, 1))
hollow_sites = list(''.join(sym) for sym in it.combinations_with_replacement(metals, 3))
sites = ontop_sites + hollow_sites

try:
	from system import gpr
except ImportError:
	gpr = None

try:
	from system import metal_colors
except ImportError:

	# Define colors of metals
	metal_colors = {'Ag': 'silver',
					'Ir': 'green',
					'Pd': 'dodgerblue',
					'Pt': 'orange',
					'Rh': 'lightskyblue',
					'Ru': 'orchid',
					'O' : 'red',
					'H' : 'white'}
