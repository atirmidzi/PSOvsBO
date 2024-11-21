'''
Find activity of molar fractions by iterating through a grid of evenly spaced molar fractions
'''

import multiprocessing as mp
import time
import sys
sys.path.append(r'C:\Users\ahmad.tirmidzi\OneDrive - Universitaet Bern\Documents\Others model\Jack Model\scripts\scripts')

from Surface_jack import BruteForceSurface
from system_importer import metals
from __init__jack import get_time_stamp,get_molar_fractions,get_activity
from regressor_jack import OntopRegressor,FccRegressor

import numpy as np
import itertools as it


metals = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru']
composition = [0.0, 0.0, 0.65, 0.05, 0.30]

def calculate_activity(metals, composition):
    # Number of processes
    n_processes = 4

    # Set random seed (XXX this is not enough to ensure exact reproducibility when parallelizing over more CPUs)
    np.random.seed(945)

    # Define slab size (no. of atom x no. of atoms)
    size = (100, 100)
    n_surface_atoms = np.prod(size)


    # 'step_size' must be a multiple of the number of metals
    step_size = 0.05
    fs = get_molar_fractions(step_size)
    n_fs = len(fs)

    # Load linear regression parameters for OH and O
    path = r'C:\Users\ahmad.tirmidzi\OneDrive - Universitaet Bern\Documents\Others model\Jack Model\1_AgIrPdPtRu\1_AgIrPdPtRu\2_machine_learning_model'
    params_OH = [np.loadtxt(f'{path}/OH_{metal}.csv', delimiter=',', usecols=1) for metal in metals]
    params_O = np.loadtxt(f'{path}/O.csv', delimiter=',', usecols=1)

    # Load adsorbate-specific regressors
    reg_OH = OntopRegressor(params_OH)
    reg_O = FccRegressor(params_O)

    # Define optimal OH adsorption energy (relative to Pt(111))
    E_opt = 0.1 # eV    

    # Define potential at which to evaluate the activity
    eU = 0.820
    
	# Convert into a molar fractions dictionary
    f = {m: f0 for m, f0 in zip(metals, composition)}

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
    
    # Cancel the random seed effect
    cancel = 1000 * time.time() # current time in milliseconds
    np.random.seed(int(cancel) % 2**32)
    
    return activity


#calculate_activity(metals, f = composition)



