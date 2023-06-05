import sys
sys.path.insert(0, '..')

from paus_utils import lya_redshift

import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
import matplotlib.colors
# matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# matplotlib.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 25})

import numpy as np


def main():
    # Initialize figure and axes
    fig, ax = plt.subplots(figsize=(20, 7))

    ###########################
    ### Transmission curves ###
    ###########################

    # Load transmission curves and filter info
    path_to_paus_data = '/home/alberto/almacen/PAUS_data'
    with open(f'{path_to_paus_data}/paus_tcurves.pkl', 'rb') as f:
        tcurves = pickle.load(f)
    filter_properties = pd.read_csv(
        f'{path_to_paus_data}/Filter_properties.csv')

    # Plot each curve
    for i, _ in enumerate(tcurves['tag']):
        ax.plot(tcurves['w'][i], tcurves['t'][i],
                lw=2, c=filter_properties['color'][i])

    ##################################
    ### Filter names and redshifts ###
    ##################################

    # First the NBs
    bar_NB_h = 0.75  # Height of the first bar
    bar_NB_h_delta = 0.01  # Increase of height

    height_list = [bar_NB_h,
                   bar_NB_h + bar_NB_h_delta * -1,
                   bar_NB_h + bar_NB_h_delta * -2,
                   bar_NB_h + bar_NB_h_delta * -1,
                   bar_NB_h,
                   bar_NB_h + bar_NB_h_delta * 1,
                   bar_NB_h + bar_NB_h_delta * 2,
                   bar_NB_h + bar_NB_h_delta * 1] * 5  # We have 40 NBs

    for i in range(40):
        w_eff = filter_properties['w_eff'][i]
        fwhm = filter_properties['fwhm'][i]
        color = matplotlib.colors.to_rgb(filter_properties['color'][i])
        btext_color = np.array(color) * 0.7
        this_patheffects = [patheffects.withStroke(linewidth=0.8,
                                                   foreground=btext_color)]

        ax.plot([w_eff - 0.5 * fwhm, w_eff + 0.5 * fwhm],
                [height_list[i]] * 2,
                lw=2.5, c=color)

        # Filter name
        nb_fontsize = 13.5

        filter_name_delta_h = 0.04
        ax.text(w_eff, bar_NB_h + filter_name_delta_h,
                filter_properties['name'][i],
                ha='center', va='bottom',
                # path_effects=this_patheffects, color=color,
                color='k',
                path_effects=[patheffects.withStroke(linewidth=0.3, foreground='k')],
                fontsize=nb_fontsize, rotation='vertical')

        # Lya redshift
        z_text = f'{lya_redshift(w_eff):0.2f}'
        z_text_delta_h = -0.035
        ax.text(w_eff, bar_NB_h + z_text_delta_h, z_text,
                ha='center', va='top',
                # path_effects=this_patheffects, color=color,
                color='k',
                path_effects=[patheffects.withStroke(linewidth=0.3, foreground='k')],
                fontsize=nb_fontsize, rotation='vertical')


    # Text indicating z_lya
    ax.text(4200, bar_NB_h + z_text_delta_h - 0.015,
            r'$z_{\mathrm{Ly}\alpha}$', fontsize=nb_fontsize+1,
            path_effects=[patheffects.withStroke(linewidth=0.3, foreground='k')],
            va='top')

    # Now, the BBs
    bar_BB_h = 0.925

    for i in range(40, 46):
        w_eff = filter_properties['w_eff'][i]
        fwhm = filter_properties['fwhm'][i]
        color = matplotlib.colors.to_rgb(filter_properties['color'][i])
        btext_color = np.array(color) * 0.6
        this_patheffects = [patheffects.withStroke(linewidth=0.3, foreground=color)]
        
        if i > 41:
            this_bar_BB_h = bar_BB_h + 0.015 * (i%2)
        else:
            this_bar_BB_h = bar_BB_h

        ax.errorbar(w_eff, this_bar_BB_h,
                    xerr=fwhm * 0.5,
                    capsize=3, capthick=3,
                    ecolor=color, elinewidth=3)

        ax.text(w_eff, bar_BB_h + 0.01 + 0.007,
                filter_properties['name'][i],
                ha='center', va='bottom',
                path_effects=this_patheffects, color=color,
                fontsize=nb_fontsize+2)


    #######################
    ### Axes properties ###
    #######################

    ax.set(xlim=(3000, 11000), ylim=(0, 1),
           xlabel='Wavelength [\AA]',
           ylabel='Response [A. U.]',
           facecolor='darkgrey')

    ax.set_xticks(np.arange(4000, 11000, 1000))

    plt.savefig('../figures/filter_transmission_curves.pdf',
                bbox_inches='tight', pad_inches=0.1, facecolor='w')


if __name__ == '__main__':
    main()
