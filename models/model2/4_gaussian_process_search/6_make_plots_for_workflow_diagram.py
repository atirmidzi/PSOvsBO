import sys
sys.path.append('../../..')
from scripts import metals, n_metals, get_molar_fractions, get_random_molar_fractions,\
					get_molar_fractions_around, get_simplex_vertices,\
					make_ternary_contour_plot, molar_fractions_to_cartesians,\
					get_composition, gpr
import numpy as np
from iteround import saferound
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.stats import norm

# Get file name
folder = '2_0.010/2/'
filename_samples = 'samples.csv'
filename_surrogate = 'surrogate_max.csv'

# Load molar fractions and activities from file
data = np.loadtxt(folder + filename_samples, delimiter=',', skiprows=1)
fs_orig = data[:, :n_metals]
activities = data[:, -1]
n_samples = len(activities)

# Specify two metals to plot individually
metals_show = ('Ag', 'Pd')

# Set the number of ticks to make
n_ticks = 5
tick_labels = True

# Specify vertices as molar fractions
fs_vertices = [[1., 0., 0.],
			   [0., 1., 0.],
			   [0., 0., 1.]]

# Get height of triangle
h = 3**0.5/2

# Get cartesian coordinates of vertices
xs_vertices, ys_vertices = molar_fractions_to_cartesians(fs_vertices)

# Define padding to put the vertex text neatly
pad = [[-0.06, -0.06],
	   [ 0.06, -0.06],
	   [ 0.00,  0.08]]
has = ['right', 'left', 'center']
vas = ['top', 'top', 'bottom']

# Make ticks and tick labels on the triangle axes
left, right, top = np.concatenate((xs_vertices.reshape(-1,1), ys_vertices.reshape(-1,1)), axis=1)

#tick_size = 0.025
tick_size = 0.035
bottom_ticks = 0.8264*tick_size * (right - top)
right_ticks = 0.8264*tick_size * (top - left)
left_ticks = 0.8264*tick_size * (left - right)

def make_triangle_ticks(ax, start, stop, tick, n, offset=(0., 0.),
						fontsize=12, ha='center', tick_labels=True):
	r = np.linspace(0, 1, n+1)
	x = start[0] * (1 - r) + stop[0] * r
	x = np.vstack((x, x + tick[0]))
	y = start[1] * (1 - r) + stop[1] * r
	y = np.vstack((y, y + tick[1]))
	ax.plot(x, y, 'black', lw=1., zorder=0)
	
	if tick_labels:
	
		# Add tick labels
		for xx, yy, rr in zip(x[0], y[0], r):
			ax.text(xx+offset[0], yy+offset[1], f'{rr*100.:.0f}',
					fontsize=fontsize, ha=ha)

def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap

def prepare_triangle_plot(ax, elems):
	# Set axis limits
	ax.set_xlim(-0.05, 1.05)
	ax.set_ylim(-0.05, h+0.05)

	# Plot triangle edges
	ax.plot([0., 0.5], [0., h], '-', color='black', zorder=0)
	ax.plot([0.5, 1.], [h, 0.], '-', color='black', zorder=0)
	ax.plot([0., 1.], [0., 0.], '-', color='black', zorder=0)
	
	# Remove spines
	for direction in ['right', 'left', 'top', 'bottom']:
		ax.spines[direction].set_visible(False)
	
	# Remove tick and tick labels
	ax.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
	ax.set_aspect('equal')

	return ax

def expected_improvement(mu, std, y_known_max, xi=0.):
	diff = mu - y_known_max - xi
	temp = diff / std
	return diff*norm.cdf(temp) + std*norm.pdf(temp)
	
# Define color map of plot
cmap = truncate_colormap(plt.get_cmap('viridis'), minval=0.2, maxval=1.0, n=100)

# Define marker size
ms = 50

# Get molar fractions spanning the whole composition space uniformly
# to use for plotting
fs_span_orig = get_molar_fractions(0.05, n_elems=n_metals)

# Specify matplotlib color normalization
color_norm_act = mpl.colors.Normalize(vmin=0., vmax=0.17)
color_norm_unc = None
color_norm_acq = mpl.colors.LogNorm(vmin=0., vmax=1e-5)

# Sample molar fractions to show
show_n_samples = np.array([54])

# Iterate through selected samples
for f_idx in show_n_samples-1:
	
	print(f'({f_idx+1})')
	
	# Get maximal activity recorded for the samples so far
	act_sample_max = np.max(activities[:f_idx+1])
		
	# Update the regressor
	gpr.fit(fs_orig[:f_idx+1], activities[:f_idx+1])
	
	# Get activities of surrogate Gaussian process
	acts_orig, stds_orig = gpr.predict(fs_span_orig, return_std=True)

	# Get indices of the two specified metals
	m1_idx, m2_idx = [metals.index(m) for m in metals_show]
	remaining_ids = [idx for idx in range(n_metals) if idx not in [m1_idx, m2_idx]]
	
	# Make molar fractions into pseudo-ternary molar fractions
	# grouping all but the two specified elements
	remain = np.sum(fs_orig[:, remaining_ids], axis=1)
	fs = np.concatenate((fs_orig[:, m1_idx].reshape(-1, 1), fs_orig[:, m2_idx].reshape(-1, 1), remain.reshape(-1, 1)), axis=1)
	elems = [*metals_show, ''.join([metals[i] for i in remaining_ids])]
	
	remain = np.sum(fs_span_orig[:, remaining_ids], axis=1)
	fs_span = np.concatenate((fs_span_orig[:, m1_idx].reshape(-1, 1), fs_span_orig[:, m2_idx].reshape(-1, 1), remain.reshape(-1, 1)), axis=1)
	
	# Get cartesian coordinates corresponding to the molar fractions
	xs, ys = molar_fractions_to_cartesians(fs_span)

	# Make figure for plotting Gaussian process mean, uncertainty, and acquisition function
	fig_act, ax_act = plt.subplots(figsize=(4,4))
	fig_unc, ax_unc = plt.subplots(figsize=(4,4))
	fig_acq, ax_acq = plt.subplots(figsize=(4,4))
	
	# Apply shared settings to pseudo-ternary plot
	for ax in [ax_act, ax_unc, ax_acq]:
		prepare_triangle_plot(ax, elems)
		
	# Initialize z-values
	zs_act = np.zeros(len(fs_span))
	zs_unc = np.zeros(len(fs_span))
	zs_acq = np.zeros(len(fs_span))
		
	for f_ter_idx, f in enumerate(fs_span):
	
		# Get samples where the two explicit metals are at their given molar fractions
		mask = np.isclose(fs_span_orig[:, m1_idx], f[0])\
			   * np.isclose(fs_span_orig[:, m2_idx], f[1])
		# Get the maximum activity
		act = np.max(acts_orig[mask])
		
		# Get the uncertainty corresponding to the activity of the surrogate function
		std = np.max(stds_orig[mask])
		
		# Get the acquisition function value corresponding to the same molar fraction of maximal activity
		eis = expected_improvement(acts_orig[mask], stds_orig[mask], act_sample_max, xi=0.01)
		acq = np.max(eis)
		
		# Update lists
		zs_act[f_ter_idx] = act
		zs_unc[f_ter_idx] = std
		zs_acq[f_ter_idx] = acq

	# Plot surrogate/uncertainty/acquisition function as a contour plot
	plot_kwargs = dict(cmap=cmap, zorder=0)
	ax_act.tricontourf(xs, ys, zs_act, norm=color_norm_act, levels=10, **plot_kwargs)
	ax_unc.tricontourf(xs, ys, zs_unc, norm=color_norm_unc, levels=10, **plot_kwargs)
	
	# Acquisition function is hard to plot, because its scales are very diverse	
	plt.close(fig_act)
	plt.close(fig_unc)
	plt.close(fig_acq)
	
	min_val = 1e-15
	zs_acq[zs_acq<min_val] = min_val
	logs = np.log10(zs_acq)
	ax_acq.tricontourf(xs, ys, logs, levels=10, **plot_kwargs)
		
	# Get sampled compositions
	xs, ys = molar_fractions_to_cartesians(fs[:f_idx+1])
	
	# Plot sampled points
	mss = [50]*(f_idx+1)
	markers = ['o']*(f_idx+1)
	idx_max = np.argmax(activities[:f_idx+1])
	mss[idx_max] = 150
	markers[idx_max] = '*'
	
	plt_kwargs = dict(m=markers, zorder=1, s=mss, edgecolors='black', lw=0.8)
	mscatter(xs, ys, ax=ax_act, c=activities[:f_idx+1], cmap=cmap,
			 norm=color_norm_act, **plt_kwargs)
	mscatter(xs, ys, ax=ax_unc, c='black', **plt_kwargs)
	mscatter(xs, ys, ax=ax_acq, c='black', **plt_kwargs)

	# Save figures
	metals_str = ''.join(metals_show)
	fig_act.savefig(f'{folder}{metals_str}{f_idx+1}_act_workflow.png', bbox_inches='tight', dpi=300)
	fig_unc.savefig(f'{folder}{metals_str}{f_idx+1}_unc_workflow.png', bbox_inches='tight', dpi=300)
	fig_acq.savefig(f'{folder}{metals_str}{f_idx+1}_acq_workflow.png', bbox_inches='tight', dpi=300)
	
	# Close figures
	plt.close(fig_act)
	plt.close(fig_unc)
	plt.close(fig_acq)
