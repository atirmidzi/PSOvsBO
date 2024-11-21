import numpy as np
from sklearn.linear_model import LinearRegression
from system_importer import metals, n_metals, ontop_sites, hollow_sites

class Regressor():
	
	def fit(self, fps, energies, site, n_atoms_zones, filename=None):
		'''
		Train the multilinear regressor (a set of linear parameters for each adsorption site),
		and store the parameters as an instance variable 'self.params'
		'''
		if site == 'ontop':
			
			# Define OH system parameters
			model_for_each_site = True
			
			# Define site labels
			sites = ontop_sites
			
			# Define zone names excluding the adsorption site zone
			zones = ['1b', '2a', '3a']
			
		elif site == 'fcc':
			
			# Define O system parameters
			model_for_each_site = False
			
			# Define site labels
			sites = hollow_sites
			
			# Define zone names excluding the adsorption site zone
			zones = ['1b', '1c', '2a', '2b']
		
		else:
			raise ValueError("'site' must be either 'ontop' or 'fcc'")
		
		# Get the number of unique adsorption sites
		n_sites = len(sites)
		
		if model_for_each_site:
		
			# Get the number of zones apart from the adsorption site zone
			n_zones = int((fps.shape[1] - n_sites) / n_metals)
			self.params = np.zeros((n_sites, n_zones*n_metals+1))
			
			# Make linear models
			for site_idx in range(n_sites):
				
				# Get fps and energies of the current site
				mask = fps[:, site_idx] == 1
				fps_site = fps[mask, n_sites:]
				energies_site = energies[mask]
				
				# Get linear regression parameters
				reg = LinearRegression(fit_intercept=False).fit(fps_site, energies_site)
				params = reg.coef_
				params, intercept = pure_to_zero(params, n_atoms_zones=n_atoms_zones, metal_idx=site_idx)
				
				# Store parameters in the instance variable
				self.params[site_idx] = [intercept, *params]

				# Save the parameters to a csv file if the flag is given
				if filename is not None:
				
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
			params = zero_mean_zones(params, n_atoms_zones=n_atoms_zones, skip=n_sites)
			
			# Store parameters as instance an variable
			self.params = params
			
			# Save the parameters to a csv file if the flag is given
			if filename is not None:
				
				with open(filename, 'w') as file_:
					file_.write('\n'.join(label + ',{:.6f}'.format(param) for label, param in zip(sites, params[:n_sites])))
					file_.write('\n')
					
					for i, param in enumerate(params[n_sites:]):
						zone = zones[int(i/n_metals)]
						metal = metals[i % n_metals]
						file_.write(zone + '_' + metal + ',' + '{:.6f}\n'.format(param))
		
		return self

class OntopRegressor(Regressor):

	def __init__(self, params=None):
		'params: list of lists of parameters. One for each metal.'
		
		self.n_sites = len(ontop_sites)
		
		if params is not None:
			self.params = np.asarray(params)
			self.n_sites = self.params.shape[0]
	
	def fit(self, fps, energies, filename=None, n_atoms_zones=[6,3,3]):
		return super().fit(fps, energies, site='ontop', filename=filename,
						   n_atoms_zones=n_atoms_zones)
	
	def predict(self, fp):
		
		# Make into numpy
		fp = np.asarray(fp)

		# Expand axes if one-dimensional
		if fp.ndim == 1:
			fp = fp[None, :]
			
		# Get site ids
		site_ids = np.nonzero(fp[:, :self.n_sites])[1]

		# Predict using the site specific parameters for each on-top site
		n_samples = fp.shape[0]
		preds = np.zeros(n_samples)
		for i, site_idx in enumerate(site_ids):
		
			# Add the intercept parameter of the site itself
			preds[i] = self.params[site_idx, 0]
			
			# Add the contributions from the remaining zones
			preds[i] += np.dot(fp[i, self.n_sites:], self.params[site_idx, 1:])
		
		return preds
		
class FccRegressor(Regressor):

	def __init__(self, params=None):
		
		if params is not None:
			self.params = params
	
	def fit(self, fps, energies, filename=None, n_atoms_zones=[3,6,3,3]):
		return super().fit(fps, energies, site='fcc',
						   filename=filename, n_atoms_zones=n_atoms_zones)
	
	def predict(self, fp):
		fp = np.asarray(fp)
		return np.dot(fp, self.params)

class NormalDistributionSampler():

	def __init__(self, means, stds):
		
		# Means and standard deviations as dictionaries, one for each site
		self.means = means
		self.stds = stds
	
	def predict(self, site, n_samples=1):
		return np.random.normal(loc=self.means[site], scale=self.stds[site], size=n_samples)

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
