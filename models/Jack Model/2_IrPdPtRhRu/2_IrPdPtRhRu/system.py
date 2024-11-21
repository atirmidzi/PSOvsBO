from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

# Define metals to be used in alloy
metals = ['Ir', 'Pd', 'Pt', 'Rh', 'Ru']

# Define colors of metals
metal_colors = {'Ag': 'silver',
				'Ir': 'green',
				'Pd': 'dodgerblue',
				'Pt': 'orange',
				'Rh': 'lightskyblue',
				'Ru': 'orchid',
				'O' : 'red',
				'H' : 'white'}

# Define kernel to use for Gaussian process regressors
kernel = C(constant_value=0.05, constant_value_bounds=(1e-5, 1e1))\
        *RBF(length_scale=0.5, length_scale_bounds=(1e-5, 1e1))\

# Define Gaussian process regressor
# alpha = 1e-5 is the approximate variance in the predicted current densities
# (the measured value for 100 samples of Ag20Pd80)
gpr = GPR(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)
