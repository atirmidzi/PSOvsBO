import numpy as np
from sklearn.linear_model import LinearRegression
import itertools as it
import sys
sys.path.append('../../..')
from scripts import metals, n_metals, ontop_sites, hollow_sites

def zero_mean_zones(params, n_atoms_zones, return_intercept=False, skip=0):
	'Return parameters with zero mean for each zone except the adsorption zone.'
	
	new_params = np.zeros(params.shape)
	intercept = 0
	for i, n_atoms in enumerate(n_atoms_zones):
		zone_params = params[skip + i*n_metals : skip + (i+1)*n_metals]
		mean_zone_params = np.mean(zone_params)
		new_params[skip + i*n_metals : skip + (i+1)*n_metals] = zone_params - mean_zone_params
		intercept += mean_zone_params*n_atoms

	if return_intercept:
		return new_params, intercept
	else:
		new_params[:skip] = params[:skip] + intercept
		return new_params
	
	
def pure_to_zero(params, n_atoms_zones, metal_idx):
	'Return parameters with the pure metal parameters set to zero.'
	
	new_params = np.zeros(params.shape)
	intercept = 0
	for i, n_atoms in enumerate(n_atoms_zones):
		zone_params = params[i*n_metals : (i+1)*n_metals]
		pure_zone_param = zone_params[metal_idx]
		new_params[i*n_metals : (i+1)*n_metals] = zone_params - pure_zone_param
		intercept += pure_zone_param*n_atoms
		
	return new_params, intercept

for ads in ['OH', 'O']:
	
	if ads == 'OH':
		
		# Define OH system parameters
		model_for_each_site = True
		
		# Define site labels
		sites = ontop_sites
		
		# Define zone names excluding the adsorption site zone
		zones = ['1b', '2a', '3a']
		
	elif ads == 'O':
		
		# Define O system parameters
		model_for_each_site = False
		
		# Define site labels
		sites = hollow_sites
		
		# Define zone names excluding the adsorption site zone
		zones = ['1b', '1c', '2a', '2b']
		
	# Load DFT data from csv file
	filename = f'../1_DFT_data/{ads}.csv'
	data = np.loadtxt(filename, delimiter=',')
	fps = data[:, :-3]
	energies = data[:, -3]
	
	# Get the number of unique adsorption sites
	n_sites = len(sites)

	if model_for_each_site:
	
		# Get the number of zones apart from the adsorption site zone
		n_zones = int((fps.shape[1] - n_sites) / n_metals)
		
		# Make linear models
		for site_idx in range(n_sites):
			
			all_params = np.zeros(n_zones*n_metals + 1)
			
			# Get fps and energies of the current site
			mask = fps[:, site_idx] == 1
			fps_site = fps[mask, n_sites:]
			energies_site = energies[mask]
			
			# Get linear regression parameters
			reg = LinearRegression(fit_intercept=False).fit(fps_site, energies_site)
			params = reg.coef_
			params, intercept = pure_to_zero(params, n_atoms_zones=[6,3,3], metal_idx=site_idx)
			
			# Define filename for output
			filename = '{:s}_{:s}.csv'.format(ads, metals[site_idx])
			
			with open(filename, 'w') as file_:
				file_.write(metals[site_idx] + ',{:.6f}'.format(intercept))
				
				for i, param in enumerate(params):
					file_.write('\n')
					zone = zones[int(i/n_metals)]
					metal = metals[i % n_metals]
					file_.write(zone + '_' + metal + ',' + '{:.6f}'.format(param))
	
	else:
		reg = LinearRegression(fit_intercept=False).fit(fps, energies)
		params = reg.coef_
		params = zero_mean_zones(params, [3,6,3,3], skip=n_sites)
		
		# Define filename for output
		filename = '{:s}.csv'.format(ads)
		
		with open(filename, 'w') as file_:
			file_.write('\n'.join(label + ',{:.6f}'.format(param) for label, param in zip(sites, params[:n_sites])))
			file_.write('\n')
			
			for i, param in enumerate(params[n_sites:]):
				zone = zones[int(i/n_metals)]
				metal = metals[i % n_metals]
				file_.write(zone + '_' + metal + ',' + '{:.6f}\n'.format(param))
