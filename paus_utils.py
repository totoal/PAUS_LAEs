import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})

fil_properties_dir = '/home/alberto/almacen/PAUS_data/csv/Filter_properties.csv'
data_tab = pd.read_csv(fil_properties_dir)
w_central = data_tab['w_eff']

def plot_PAUS_source(flx, err, ax=None, set_ylim=True, e17scale=False, fs=15):
    '''
    Generates a plot with the JPAS data.
    '''

    if e17scale:
        flx = flx * 1e17
        err = err * 1e17

    cmap = data_tab['color']

    data_max = np.max(flx)
    data_min = np.min(flx)
    y_max = (data_max - data_min) * 2/3 + data_max
    y_min = data_min - (data_max - data_min) * 0.3

    # if not given any ax, create one
    if ax is None:
        ax = plt.gca()

    for i, w in enumerate(w_central[:-6]):
        ax.errorbar(w, flx[i], yerr=err[i],
                    marker='s', markeredgecolor='dimgray', markerfacecolor=cmap[i],
                    markersize=7, ecolor='dimgray', capsize=4, capthick=1, linestyle='',
                    zorder=-99)
    # If BBs are included, plot them
    if len(flx) > 40:
        for i, w in enumerate(w_central[-6:]):
            ax.errorbar(w_central[i - 6], flx[i - 6], yerr=err[i - 6],
                        markeredgecolor='dimgray',
                        fmt='^', markerfacecolor=cmap[i - 6], markersize=10,
                        ecolor='dimgray', capsize=4, capthick=1)

    try:
        if set_ylim:
            ax.set_ylim((y_min, y_max))
    except:
        pass

    ax.set_xlabel('$\lambda$ \AA', size=fs)
    if e17scale:
        ax.set_ylabel(
            r'$f_\lambda\cdot10^{17}$ [erg cm$^{-2}$ s$^{-1}$ \AA$^{-1}$]', size=fs)
    else:
        ax.set_ylabel(
            '$f_\lambda$ [erg cm$^{-2}$ s$^{-1}$ \AA$^{-1}$]', size=fs)

    return ax