import numpy as np
from matplotlib import pyplot as plt, rcParams
import os
import argparse

# Set plotting
params = {
        'axes.labelsize': 18,
        'font.size': 18,
        'legend.fontsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'text.usetex': False,
        'figure.figsize': [13,8],
        'axes.labelpad' : 10,
        'lines.linewidth' : 10,
        'legend.loc': 'lower left'
        }
rcParams['agg.path.chunksize'] = 10000
rcParams.update(params)
plt.style.use('bmh')

# Set-up GP
rbf_fn = lambda X1, X2: np.exp((np.dot(X1,(2*X2.T))-np.sum(X1*X1,1)[:,None]) - np.sum(X2*X2,1)[None,:])
gauss_kernel_fn = lambda X1, X2, ell, sigma_f: sigma_f**2 * rbf_fn(X1/(np.sqrt(2)*ell), X2/(np.sqrt(2)*ell))

# Pick some particular parameters for this demo:
k_fn = lambda X1, X2: gauss_kernel_fn(X1, X2, 30.0, 50.0)

# Pick the input locations that we want to see the function at.
X_grid = np.arange(1, 151, 1)[:,None]
# Define train data with respecitve indexes in X_grid (-1 here as only integers are used)
X_locs1 = np.array([1,50,75,100,150])[:,None]
idx = X_locs1 - 1
# Define Y_train data
f_locs1 = np.array([139.33,147.92,126.75,137.98,151.08])
# Add noise variance if required
noise_var = 0.0

# Plot training data
plt.plot(X_locs1, f_locs1, 'x', markersize=10, markeredgewidth=2, color="r")

# Compute covariance of function values for those points.
N_grid = X_grid.shape[0]
K_grid = k_fn(X_grid, X_grid) + 1e-9*np.eye(N_grid)

N_locs1 = idx.size
K_locs1 = k_fn(X_locs1, X_locs1)
L_locs1 = np.linalg.cholesky(K_locs1)

X_rest = np.delete(X_grid, idx, 0)
K_rest = k_fn(X_rest, X_rest)
K_rest_locs1 = k_fn(X_rest, X_locs1)
M = K_locs1 + noise_var*np.eye(N_locs1)

rest_cond_mu = np.dot(K_rest_locs1, np.linalg.solve(M, f_locs1))
rest_cond_cov = K_rest - np.dot(K_rest_locs1, np.linalg.solve(M, K_rest_locs1.T))

N_rest = X_rest.shape[0]
L_rest = np.linalg.cholesky(rest_cond_cov + 1e-9*np.eye(N_rest))

# thick black lines show +/- 2 standard deviations -- at any particular
# location, we have ~95% belief the function will lie in this range.
plt.plot(X_rest, rest_cond_mu, '-k', linewidth=2, label='mean completion')
rest_cond_std = np.sqrt(np.diag(rest_cond_cov))
plt.plot(X_rest, rest_cond_mu + 2*rest_cond_std, '--k', linewidth=2, label='credible band')
plt.plot(X_rest, rest_cond_mu - 2*rest_cond_std, '--k', linewidth=2, label='_nolegend_')
plt.fill_between(X_rest.flatten(), rest_cond_mu + 2*rest_cond_std, rest_cond_mu - 2*rest_cond_std, alpha=0.4, label='Observed values')

plt.legend(['mean completion', 'credible band', 'observed values'])
plt.xlabel('Beta')
plt.ylabel('FID')
plt.ylim((50, 200))

#plt.savefig("BOOO.pdf", bbox_inches='tight')

plt.show()