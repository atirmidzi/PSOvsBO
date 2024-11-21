import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
import matplotlib.ticker as ticker
import numpy as np

def parity_plot(calc_train, pred_train, adsorbates, calc_test=[], pred_test=[],
				fontsize=14, pm=0.1, lw=0.5, filename='parity_plot.png'):
	'''
	calc		list of lists	calculated/expected/DFT values; a list for each adsorbate
	pred		list of lists	predicted values; a list for each adsorbate
	adsorbates	list of strings	adsorbates corresponding to the order of the calculated and
								predicted values
	fontsize	int				size of text
	pm			float			plus/minus y-value for dashed lines
	lw			float			line width
	'''
	
	# Get the number of plots to make
	n_ads = len(adsorbates)
	
	# Make figure
	fig, axes = plt.subplots(figsize=(4*n_ads, 3), ncols=n_ads, gridspec_kw=dict(wspace=0.3))
	
	# If a single axis is made, then make it iterable
	# for the below loop to work
	try:
		ax = iter(axes)
	except TypeError:
		axes = [axes]
	
	# Iterate through adsorbates and axes
	for ads_idx, (ads, ax, calc_train_, pred_train_, calc_test_, pred_test_)\
		in enumerate(zip(adsorbates, axes, calc_train, pred_train, calc_test, pred_test)):
		
		# Set axis labels
		if ads.endswith('H'):
			xlabel = '$\\rm \Delta E_{{\\rm ^{{\\ast}}{0:s}}}^{{\\rm DFT}} -\
					   \\rm \Delta E_{{\\rm ^{{\\ast}}{0:s}}}^{{\\rm Pt(111)}}$ (eV)'.format(ads)
			
			ylabel = '$\\rm \Delta E_{{\\rm ^{{\\ast}}{0:s}}}^{{\\rm pred}} -\
					   \\rm \Delta E_{{\\rm ^{{\\ast}}{0:s}}}^{{\\rm Pt(111)}}$ (eV)'.format(ads)
			
		else:
			xlabel = '$\\rm \Delta E_{{\\rm {0:s}^{{\\ast}}}}^{{\\rm DFT}} -\
					   \\rm \Delta E_{{\\rm {0:s}^{{\\ast}}}}^{{\\rm Pt(111)}}$ (eV)'.format(ads)
					   
			ylabel = '$\\rm \Delta E_{{\\rm {0:s}^{{\\ast}}}}^{{\\rm pred}} -\
					   \\rm \Delta E_{{\\rm {0:s}^{{\\ast}}}}^{{\\rm Pt(111)}}$ (eV)'.format(ads)
		
		ax.set_xlabel(xlabel, fontsize=fontsize-2)
		ax.set_ylabel(ylabel, fontsize=fontsize-2)
		
		# Make inset axis showing the prediction error as a histogram
		ax_inset = inset_axes(ax, width=0, height=0)
		margin = 0.015
		scale = 0.85
		width = 0.4*scale
		height = 0.3*scale
		pos = InsetPosition(ax,
			[margin, 1.0-height-margin, width, height])
		ax_inset.set_axes_locator(pos)
		
		# Make plus/minus 0.1 eV lines in inset axis
		ax_inset.axvline(pm, color='black', ls='--', dashes=(5, 5), lw=lw)
		ax_inset.axvline(-pm, color='black', ls='--', dashes=(5, 5), lw=lw)
		
		# Set x-tick label fontsize in inset axis
		ax_inset.tick_params(axis='x', which='major', labelsize=fontsize-6)
		
		# Remove y-ticks in inset axis
		ax_inset.tick_params(axis='y', which='major', left=False, labelleft=False)
	
		# Set x-tick locations in inset axis		
		ax_inset.xaxis.set_major_locator(ticker.MultipleLocator(0.50))
		ax_inset.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

		# Remove the all but the bottom spines of the inset axis
		for side in ['top', 'right', 'left']:
			ax_inset.spines[side].set_visible(False)
		
		# Make the background transparent in the inset axis
		ax_inset.patch.set_alpha(0.0)
		
		# Print 'pred-calc' below inset axis
		ax_inset.text(0.5, -0.5,
					  '$pred - DFT$ (eV)',
					  ha='center',
					  transform=ax_inset.transAxes,
					  fontsize=fontsize-7)
		
		# Iterate through training and test sets
		for calc, pred, color, y, label, marker, ms in zip([calc_train_, calc_test_], [pred_train_, pred_test_],\
														   ['deepskyblue', 'lightcoral'], [0.8, 0.6],
														   ['train', 'test'], ['o', 'X'], [10, 20]):
		
			# Get the number of data points
			n_samples = len(calc)
			
			# Make scatter parity plot
			ax.scatter(calc, pred,
					   marker=marker,
					   s=ms,
					   c=color,
					   edgecolors='black',
					   linewidths=0.1,
					   zorder=2,
					   label='{} ({:d} points)'.format(label, n_samples))
			
			# Get prediction errors
			errors = pred - calc
			
			# Make histogram of distribution of errors
			ax_inset.hist(errors,
					 	  bins=np.arange(-0.6, 0.6, 0.05),
					 	  color=color,
					 	  density=True,
					 	  alpha=0.7,
					 	  histtype='stepfilled',
					 	  ec='black',
					 	  lw=lw)
			
			# Print mean absolute error in plot
			mae = np.mean(np.absolute(errors))
			ax_inset.text(0.75, y,
						  'MAE({})={:.3f} eV'.format(label, mae),
						  ha='left',
						  color=color,
						  fontweight='bold',
						  transform=ax_inset.transAxes,
						  fontsize=fontsize-7)
					  
		# Set tick locations
		major_tick = 1.0
		ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25 * major_tick))
		ax.yaxis.set_major_locator(ticker.MultipleLocator(major_tick))
		ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25 * major_tick))
		
		# Format tick labels
		ax.xaxis.set_major_formatter('{x:.1f}')
		ax.yaxis.set_major_formatter('{x:.1f}')
		
		# Set tick label fontsize
		ax.tick_params(axis='both', which='major', labelsize=fontsize-5)
		ax.tick_params(axis='both', which='both', right=True, top=True, direction='in')
		
		# Get x and y limits		
		start = min([min(calc_train_), min(pred_train_), min(calc_test_), min(pred_test_)])
		stop = max([max(calc_train_), max(pred_train_), max(calc_test_), max(pred_test_)])
		diff = stop - start
		scale = 0.05
		stop += scale * diff
		start -= scale * diff
		lims = (start, stop)
		ax.set_xlim(lims)
		ax.set_ylim(lims)
		
		# Make central and plus/minus 0.1 eV lines in scatter plot
		ax.plot(lims, lims,
				lw=lw, color='black', zorder=1,
				label=r'$\rm \Delta E_{pred} = \Delta E_{DFT}$')
		
		# Make plus/minus 0.1 eV lines around y = x
		ax.plot(lims, [start+pm, stop+pm],
				lw=lw, ls='--', dashes=(5, 5), color='black', zorder=1,
				label=r'$\rm \pm$ {:.1f} eV'.format(pm))
				
		ax.plot([start, stop], [start-pm, stop-pm],
				lw=lw, ls='--', dashes=(5, 5), color='black', zorder=1)
		
		# Make legend
		ax.legend(frameon=False,
				  bbox_to_anchor=[0.45, 0.0],
				  loc='lower left',
				  handletextpad=0.2,
				  handlelength=1.0,
				  labelspacing=0.2,
				  borderaxespad=0.1,
				  markerscale=1.5,
				  fontsize=fontsize-5)
	
	fig.savefig(filename, dpi=300, bbox_inches='tight')
