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
metals_shows = [('Ag', 'Pd'),
				('Pd', 'Ru')]

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
		
	make_triangle_ticks(ax, right, left, bottom_ticks, n_ticks, offset=(0.03, -0.08), ha='center', tick_labels=tick_labels)
	make_triangle_ticks(ax, left, top, left_ticks, n_ticks, offset=(-0.03, -0.015), ha='right', tick_labels=tick_labels)
	make_triangle_ticks(ax, top, right, right_ticks, n_ticks, offset=(0.015, 0.02), ha='left', tick_labels=tick_labels)

	# Show axis labels (i.e. atomic percentages)
	ax.text(0.5, -0.14, f'{elems[0]} content (at.%)', rotation=0., fontsize=12, ha='center', va='center')
	ax.text(0.9, 0.5, f'{elems[1]} content (at.%)', rotation=-60., fontsize=12, ha='center', va='center')
	ax.text(0.1, 0.5, f'{elems[2]} content (at.%)', rotation=60., fontsize=12, ha='center', va='center')
	
	# Show the chemical symbol as text at each vertex
	for idx, (x, y, (dx, dy)) in enumerate(zip(xs_vertices, ys_vertices, pad)):
		ax.text(x+dx, y+dy, s=elems[idx], fontsize=14, ha=has[idx], va=vas[idx])
	
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

# Sample molar fractions to show
show_n_samples = np.array([15, 28, 54, 150])

# Iterate through selected samples
for f_idx in show_n_samples-1:
	
	print(f'({f_idx+1})')
	
	# Get maximal activity recorded for the samples so far
	act_sample_max = np.max(activities[:f_idx+1])
		
	# Update the regressor
	gpr.fit(fs_orig[:f_idx+1], activities[:f_idx+1])
	
	# Get activities of surrogate Gaussian process
	acts_orig, stds_orig = gpr.predict(fs_span_orig, return_std=True)

	# Iterate through pairs of metals in pseudo-ternary plot
	for metals_show in metals_shows:
		
		print(metals_show)
		
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
		
		# Apply shared settings to pseudo-ternary plot
		prepare_triangle_plot(ax_act, elems)
			
		# Initialize z-values
		zs_act = np.zeros(len(fs_span))
			
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
		
		# Plot surrogate/uncertainty/acquisition function as a contour plot
		plot_kwargs = dict(cmap=cmap, zorder=0)
		ax_act.tricontourf(xs, ys, zs_act, norm=color_norm_act, levels=10, **plot_kwargs)
		
		# Acquisition function is hard to plot, because its scales are very diverse	
		plt.close(fig_act)
			
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
		
		# Show sample number above ternary plot
		ending = '' if f_idx==0 else 's'
		plot_kwargs = dict(s=f'{f_idx+1} sample{ending}', fontsize=14, ha='center', va='bottom',
						   bbox=dict(facecolor='white', alpha=1., edgecolor='black'))
		ax_act.text(x=0.5, y=1.2, transform=ax_act.transAxes, **plot_kwargs)
		
		# Save figures
		metals_str = ''.join(metals_show)
		fig_act.savefig(f'{folder}{metals_str}{f_idx+1}_act.png', bbox_inches='tight', dpi=300)
		
		# Close figures
		plt.close(fig_act)

# Load molar fractions and activities of surrogate function from file
data_sur = np.loadtxt(folder + filename_surrogate, delimiter=',', skiprows=1)
ns_samples = data_sur[:, 0]
fs_sur = data_sur[:, 1:n_metals+1]
activities_sur = data_sur[:, -3]
c_sur = data_sur[:, -2]
l_sur = data_sur[:, -1]

# Get unique sample numbers
unique_sample_ns = np.array(list(set(ns_samples)))
c_sur = [c_sur[ns_samples == n][0] for n in unique_sample_ns]
l_sur = [l_sur[ns_samples == n][0] for n in unique_sample_ns]

for write_labels in [False, True]:

	# Make convergence figure
	fig_con, (ax_con, ax_c, ax_l) = plt.subplots(figsize=(12,4), nrows=3, sharex=True,
		gridspec_kw=dict(height_ratios=(6,1,1), hspace=0.))

	# Make convergence plot
	n_samples = np.arange(1, 151, step=1)
	ax_con.plot(n_samples, -activities, lw=1., color='black')

	# Get locally optimal molar fractions groups
	fs_obj = np.array([[0.185, 0.000, 0.815, 0.000, 0.000], # Ag18Pd82
					   [0.000, 0.093, 0.641, 0.000, 0.265], # Ir9Pd64Ru27
					   [0.000, 0.485, 0.000, 0.515, 0.000]  # Ir49Pt51
					  ])

	# Get indices for each locally optimal molar fraction corersponding to each of the groups
	indices = [[] for _ in range(len(fs_obj))]
	for f_idx, f in enumerate(fs_sur):
	
		for f_obj_idx, f_obj in enumerate(fs_obj):
	
			# Get difference to objective molar fractions
			diff = np.sum(np.abs(f - f_obj))
		
			# If the difference is smaller than threshold, 
			# then append index
			if diff < 0.20:
				indices[f_obj_idx].append(f_idx)

	# Get x-axis limits
	xlim = ax_con.get_xlim()

	# Iterate through objective molar fractions
	for ids, f_obj in zip(indices, fs_obj):
	
		# Skip if this objective was not found
		if len(ids) == 0:
			continue
	
		# Extend dashed line until the right end of the plot
		xs = np.concatenate((ns_samples[ids], [xlim[-1]]))
		ys = -np.concatenate((activities_sur[ids], [activities_sur[ids[-1]]]))
		ax_con.plot(xs, ys, color='royalblue', ls='dashed', lw=1.0)

	# Reset x-axis limits
	ax_con.set_xlim(xlim)

	# Plot convergence of hyperparameters
	ax_c.plot(unique_sample_ns, c_sur, lw=1., color='green')
	ax_l.plot(unique_sample_ns, l_sur, lw=1., color='darkorange')

	# Set x-axis labels
	ax_l.set_xlabel('Number of samples', fontsize=14)

	# Set y-axis labels
	x = -0.1
	kwargs = dict(va='center', ha='center', fontsize=14, rotation=90.)
	fig_con.text(x=x, y=0.5, s='Current density\n(arb. units)',
				 transform=ax_con.transAxes, **kwargs)
	fig_con.text(x=x, y=0.5, s='$C^{2}$', transform=ax_c.transAxes, **kwargs)
	fig_con.text(x=x, y=0.5, s='$\ell$', transform=ax_l.transAxes, **kwargs)

	# Set axis limits
	ax_con.set_xlim((0, n_samples[-1]+1))
	ax_con.set_ylim((-0.27, 0.02))

	# Set tick parameters
	kwargs = dict(which='both', axis='both', left=True, right=True, bottom=True, top=True, direction='in', labelsize=12)
	ax_con.tick_params(**kwargs)
	ax_c.tick_params(**kwargs)
	ax_l.tick_params(**kwargs)

	# Set tick locations
	ax_con.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
	ax_con.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
	ax_con.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
	ax_con.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))

	ax_c.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.002))
	ax_c.set_ylim((0., 0.012))
	ax_c.yaxis.set_ticks([0.002, 0.008])
	ax_c.yaxis.set_ticklabels(['0.002', '0.008'])

	ax_l.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
	ax_l.set_ylim((0.2, 0.8))
	ax_l.yaxis.set_ticks([0.3, 0.6])
	ax_l.yaxis.set_ticklabels([0.3, 0.6])
	
	if write_labels:
		
		print('\nRunning maxima:')
		
		# Get the running maxima of the convergence plot
		running_maxs = {}
		for running_idx in range(n_samples[-1]):
	
			# Get the running maximum activity
			max_ = -max(activities[:running_idx+1])
	
			# Skip if smaller than a threshold value
			if max_ > -0.12:
				continue
	
			# If this running max has not been found before
			if max_ not in running_maxs.values():
	
				# Append maximum
				running_maxs[running_idx+1] = max_
		
				# Get composition
				composition = get_composition(fs_orig[running_idx], metals)#, return_latex=True)
		
				# Plot the running maximum composition
				print(f'({running_idx+1})'+composition)
				ax_con.annotate(text=f'({running_idx+1})'+composition, xy=(running_idx+1, max_),
								xytext=(running_idx+1, max_-0.03), fontsize=10,
								ha='center', va='top',
								arrowprops=dict(arrowstyle='->', color='black', relpos=(0.5, 1.0)))

	# Save and close figure
	if write_labels:
		ending = '_with_labels'
	else:
		ending = ''
		
	fig_con.savefig(f'{folder}convergence{ending}.png', bbox_inches='tight', dpi=300)
	plt.close(fig_con)
