import numpy as np
from scipy.stats import norm

def expected_improvement(X, X_known, gpr, xi=0.01):
    '''
    Adaptation of http://krasserm.github.io/2018/03/21/bayesian-optimization/

    Args:
        X: Points at which EI shall be computed (m x d).
        X_known: Sample locations (n x d).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, std = gpr.predict(X, return_std=True)
    mu_known = gpr.predict(X_known)
	
    std = np.ravel(std)

    mu_known_opt = np.max(mu_known)

    temp = mu - mu_known_opt - xi
    Z = temp / std
    EI = temp * norm.cdf(Z) + std * norm.pdf(Z)
    
    EI[std == 0.0] = 0.0

    return EI
