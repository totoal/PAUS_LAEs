import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})

fil_properties_dir = '/home/alberto/almacen/PAUS_data/csv/Filter_properties.csv'
data_tab = pd.read_csv(fil_properties_dir)
w_central = np.array(data_tab['w_eff'])
fwhm_Arr = np.array(data_tab['fwhm'])

w_lya = 1215.67

def plot_PAUS_source(flx, err, ax=None, set_ylim=True, e17scale=True,
                     fs=15, sdss_range_mode=False, plot_BBs=True):
    '''
    Generates a plot with the JPAS data.
    '''

    if e17scale:
        flx = flx * 1e17
        err = err * 1e17

    cmap = np.array(data_tab['color'])

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
    if (len(flx) > 40) and plot_BBs:
        for i, w in enumerate(w_central[-6:]):
            if sdss_range_mode is True:
                if i == 0 or i == 5:
                    continue
            ax.errorbar(w_central[i - 6], flx[i - 6], yerr=err[i - 6],
                        markeredgecolor='dimgray',
                        fmt='^', markerfacecolor=cmap[i - 6], markersize=13,
                        ecolor='dimgray', capsize=4, capthick=1, alpha=0.8)

    try:
        if set_ylim:
            ax.set_ylim((y_min, y_max))
    except:
        pass

    ax.set_xlabel('$\lambda$ (\AA)', size=fs)
    if e17scale:
        ax.set_ylabel(
            r'$f_\lambda\cdot10^{17}$ [erg cm$^{-2}$ s$^{-1}$ \AA$^{-1}$]', size=fs)
    else:
        ax.set_ylabel(
            '$f_\lambda$ [erg cm$^{-2}$ s$^{-1}$ \AA$^{-1}$]', size=fs)

    return ax
    

def z_NB(cont_line_pos):
    '''
    Computes the Lyman-alpha redshift (z) for a given continuum narrowband (NB) index.

    Parameters
    ----------
    cont_line_pos : int or array-like of ints
        Index or indices of the continuum narrowband(s) to compute the redshift for.

    Returns
    -------
    z : float or array-like of floats
        The corresponding redshift(s) of the Lyman-alpha emission line.

    Notes
    -----
    This function assumes that the input continuum narrowband indices correspond to adjacent
    narrowbands centered at wavelengths increasing from the blue to the red end of the spectrum.
    '''
    cont_line_pos = np.atleast_1d(cont_line_pos)

    w1 = w_central[cont_line_pos.astype(int)]
    w2 = w_central[cont_line_pos.astype(int) + 1]

    w = (w2 - w1) * cont_line_pos % 1 + w1

    if len(w) > 1:
        return w / w_lya - 1
    else:
        return (w / w_lya - 1)[0]

def lya_redshift(w_obs):
    '''
    Returns the Ly alpha redshift of an observed wavelength
    '''
    return w_obs / w_lya - 1