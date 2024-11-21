import sys
sys.path.append('../../..')
from scripts import ontop_sites, hollow_sites, parity_plot, OntopRegressor, FccRegressor
import numpy as np
import itertools as it

# Set random seed
np.random.seed(81235)

# Specify adsorbates
adsorbates  = ['OH', 'O']

# Specify proportion of data to use for training
train_ratio = 0.8

# Initiate lists
calc_train = []
calc_test = []
pred_train = []
pred_test = []

# Iterate trough adsorbates
for ads in adsorbates:
	
	# Get unique adsorption sites
	if ads == 'OH':
		sites = ontop_sites
		reg = OntopRegressor()
		
	elif ads == 'O':
		sites = hollow_sites
		reg = FccRegressor()
	
	# Get number of unique sites
	n_sites = len(sites)
	
	# Load data from file
	data = np.loadtxt(f'../1_DFT_data/{ads}.csv', delimiter=',', skiprows=1)
	
	# Get number of samples
	n_samples = data.shape[0]
	
	# Get random indices to use for training and testing
	ids_train = np.random.choice(range(n_samples), size=int(train_ratio*n_samples), replace=False)
	ids_test = np.array([idx for idx in range(n_samples) if idx not in ids_train])
	
	# Get fingerprints and adsorption energies of the structures for the training set
	fps_train = data[ids_train, :-3]
	energies_train = data[ids_train, -3]
	calc_train.append(energies_train)
	
	# .. and for the test set
	fps_test = data[ids_test, :-3]
	energies_test = data[ids_test, -3]
	calc_test.append(energies_test)
	
	# Train the regressor on the training data
	reg.fit(fps_train, energies_train)
	
	# Get the predicted adsorption energies
	pred_train.append(reg.predict(fps_train))
	pred_test.append(reg.predict(fps_test))

# Make and save parity plot to the current directory
parity_plot(calc_train, pred_train, adsorbates, calc_test, pred_test, filename='parity_plot.png')
