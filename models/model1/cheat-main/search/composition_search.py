import sys
sys.path.append(r'C:\Users\ahmad.tirmidzi\OneDrive - Universitaet Bern\Documents\Others model\Christian Model\cheat-main\utils')
from surface import BruteForceSurface
from regression import load_GCN
from bayesian import expected_improvement, append_to_file, random_comp
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import time
import pickle
import torch
import numpy as np

# Objective function
def comp2act(comp):#, filename=None):
    # Construct surface
    surface = BruteForceSurface(comp, adsorbates, ads_atoms, sites, coordinates, height,
                                    regressor, 'graphs', 2, 'fcc111', surf_size, displace_e, scale_e)

    # Determine gross energies of surface sites
    surface.get_gross_energies()

    # Get net adsorption energy distributions upon filling the surface
    surface.get_net_energies()

    # Get activity of the net distribution of *OH adsorption energies
    activity = surface.get_activity(G_opt=E_opt, eU=eU, T=298.15, j_d=1)
    
    # Print sampled composition
    #f_str = ' '.join(f"{k}({v + 1e-5:.2f})" for k, v in comp.items())
    #print(f'{f_str}     A = {activity / pt_act * 100:.0f} %')
    
    return activity


# set adsorbate information
ads_atoms = ['O','H']  # adsorbate elements included
adsorbates = ['OH','O']  # adsorbates included
sites = ['ontop','fcc']  # sites of adsorption
coordinates = [([0,0,0],[0.65,0.65,0.40]), None]  # coordinates of multi-atom adsorbates
height = np.array([2,1.3])  # length of bond to surface

# displacement and scaling of adsorption energies
displace_e = [0.0, 0.0]
scale_e = [1, 0.5]


# load trained state
with open(r'C:\Users\ahmad.tirmidzi\OneDrive - Universitaet Bern\Documents\Others model\Christian Model\cheat-main\regression\model_states\GC3H0reludim18BS64lr0.001.state', 'rb') as input:
    trained_state = pickle.load(input)

# set model parameters and load trained model
kwargs = {'n_conv_layers': 3,  # number of gated graph convolution layers
          'n_hidden_layers': 0,  # number of hidden layers
          'conv_dim': 18,  # number of fully connected layers
          'act': 'relu' # activation function in hidden layers.
         }
regressor = load_GCN(kwargs,trained_state=trained_state)

# Define kernel to use for Gaussian process regressors

# Define exploration-exploitation trade-off
# The higher this number, the more exploration
xi = 0.01


# Define slab size (no. of atom x no. of atoms)
surf_size = (96, 96)

# Define optimal OH adsorption energy (relative to Pt(111))
E_opt = 0.100  # eV

# Define potential at which to evaluate the activity
eU = 0.820  # eV


# Reference activity of 2/3 *OH coverage on pure Pt(111)
j_ki = np.exp(-(np.abs(-E_opt) - 0.86 + eU) / (8.617e-5 * 298.15))
pt_act = 2 / 3 * np.sum(1 / (1 + 1 / j_ki))

def calculate_activity(elements, composition):
    # set random seed
    np.random.seed(14)
    
    # Calculate activity
    activity = comp2act(dict(zip(elements, composition)))
    
    
    # Cancel the random seed effect
    cancel = 1000 * time.time() # current time in milliseconds
    np.random.seed(int(cancel) % 2**32)
    return activity

# Define elements to use
elements = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru']
composition = [0.2, 0.2, 0.2, 0.2, 0.2]


#calculate_activity(elements, composition)

