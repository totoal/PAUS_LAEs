import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

from scipy.integrate import simpson

from jpasLAEs.utils import flux_to_mag


fil_properties_dir = '/home/alberto/almacen/PAUS_data/Filter_properties.csv'
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

    data_max = np.max(flx[err > 0])
    data_min = np.min(flx[err > 0])
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
            bb_flx = flx[i - 6]
            bb_err = err[i - 6]

            # Check if bb is well measured
            if bb_err < 0:
                continue

            ax.errorbar(w_central[i - 6], bb_flx, yerr=bb_err,
                        markeredgecolor='dimgray',
                        fmt='^', markerfacecolor=cmap[i - 6], markersize=13,
                        ecolor='dimgray', capsize=4, capthick=1, alpha=0.8)

    try:
        if set_ylim:
            ax.set_ylim((y_min, y_max))
    except:
        pass

    ax.set_xlabel('$\lambda$ [\AA]', size=fs)
    if e17scale:
        ax.set_ylabel(
            r'$f_\lambda\cdot10^{17}$ [erg cm$^{-2}$ s$^{-1}$ \AA$^{-1}$]', size=fs)
    else:
        ax.set_ylabel(
            '$f_\lambda$ [erg cm$^{-2}$ s$^{-1}$ \AA$^{-1}$]', size=fs)

    return ax
    

def z_NB(cont_line_pos, w0_line='Lya'):
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
    if w0_line == 'Lya':
        w0_line = w_lya
    else:
        if not isinstance(w0_line, (int, float)):
            raise TypeError('w0_line must be int or float')

    # Store a mask with the -1 values for the NB
    mask_nondetection = (cont_line_pos == -1)

    cont_line_pos = np.atleast_1d(cont_line_pos)

    w1 = w_central[cont_line_pos.astype(int)]
    w2 = w_central[cont_line_pos.astype(int) + 1]

    w = (w2 - w1) * cont_line_pos % 1 + w1

    Line_z_Arr = w / w0_line - 1
    Line_z_Arr[mask_nondetection] = -1

    if len(w) > 1:
        return Line_z_Arr
    else:
        return Line_z_Arr[0]

def NB_z(z, w0_line='Lya'):
    '''
    Takes a redshift as an argument and returns the corresponding NB to that redshift.
    Returns -1 if the Lya redshift is out of boundaries.
    '''
    z = np.atleast_1d(z)
    w_central_NB = w_central[:40]
    
    if w0_line == 'Lya':
        w_obs = (z + 1) * w_lya
    else:
        if not isinstance(w0_line, (int, float)):
            raise TypeError('w0_line must be int or float')
        w_obs = (z + 1) * w0_line

    n_NB = np.zeros(len(z)).astype(int)
    for i, w in enumerate(w_obs):
        w_diff = np.abs(w_central_NB - w)
        w_diff_argmin = np.argmin(w_diff)

        if w_diff[w_diff_argmin] > fwhm_Arr[w_diff_argmin] * 0.5:
            n_NB[i] = -1
        else:
            n_NB[i] = int(w_diff_argmin)

    # 39 It's too much, so let's assign also -1
    n_NB[(n_NB < 0) | (n_NB > 39)] = -1

    # If only one value passed, return as a number instead of numpy array
    if len(n_NB) == 1:
        n_NB = n_NB[0]

    return n_NB

def lya_redshift(w_obs):
    '''
    Returns the Ly alpha redshift of an observed wavelength
    '''
    return w_obs / w_lya - 1


def z_volume(z_min, z_max, area):
    '''
    Returns the comoving volume for an observation area between a range of redshifts
    '''
    z_x = np.linspace(z_min, z_max, 1000)
    dV = cosmo.differential_comoving_volume(z_x).to(u.Mpc**3 / u.sr).value
    area_rad = area * (2 * np.pi / 360) ** 2
    theta = np.arccos(1 - area_rad / (2 * np.pi))
    Omega = 2 * np.pi * (1 - np.cos(theta))
    vol = simpson(dV, z_x) * Omega

    return vol


def Lya_effective_volume(nb_min, nb_max, region_name=1):
    '''
    Due to NB overlap, specially when considering single filters, the volume probed by one
    NB has to be corrected because some sources could be detected in that NB or in either
    of the adjacent ones.
    '''
    area_dict = {
        'SFG': 400,
        'QSO_cont': 200,
        'QSO_LAEs_loL': 400,
        'QSO_LAEs_hiL': 4000,
        'GAL': 59.97,
        'W3': 17.02043,
        'W2': 10.36,
        'W1': 8.6303547,
    }
        
    try:
        area = area_dict[region_name]
    except:
        # If the survey name is not known, try to use the given value as area
        try:
            area = float(region_name)
        except:
            raise ValueError('Survey name not known or invalid area value')

    z_min_overlap = (w_central[nb_min] - fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max_overlap = (w_central[nb_max] + fwhm_Arr[nb_max] * 0.5) / w_lya - 1

    if nb_min == 0:
        z_min_abs = z_min_overlap
    else:
        z_min_abs = (w_central[nb_min - 1] +
                    fwhm_Arr[nb_min - 1] * 0.5) / w_lya - 1
    z_max_abs = (w_central[nb_max + 1] -
                 fwhm_Arr[nb_min + 1] * 0.5) / w_lya - 1

    volume_abs = z_volume(z_min_abs, z_max_abs, area)
    volume_overlap = (
        z_volume(z_min_overlap, z_min_abs, area)
        + z_volume(z_max_abs, z_max_overlap, area)
    )

    return volume_abs + volume_overlap * 0.5


def PAUS_monochromatic_Mag(cat, wavelength=1450):
    '''
    Calculate the absolute magnitude (M) and its error for sources in a catalog
    at a specified monochromatic wavelength in the rest-frame.

    Parameters:
    cat (dict): A dictionary containing information about sources, including:
        - 'z_NB' (numpy.ndarray): Redshift values for sources.
        - 'flx' (numpy.ndarray): Flux values for sources.
        - 'err' (numpy.ndarray): Flux error values for sources.
        - 'nice_lya' (numpy.ndarray): Boolean mask for selecting sources.

    wavelength (float, optional): The desired wavelength in Angstroms in the rest-frame.
        Default is 1450 Angstroms.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - M_Arr (numpy.ndarray): Absolute magnitude for sources. Sources without valid
          data are assigned a value of 99.
        - magAB_err_Arr (numpy.ndarray): Error in the absolute magnitude.
    '''
    # Find the NB of the specified wavelength in rest-frame
    nb_w_rest = NB_z(cat['z_NB'], wavelength)
    dist_lum_Arr = cosmo.luminosity_distance(cat['z_NB']).to(u.pc).value

    N_sources = len(cat['z_NB'])

    flambda_Arr = np.ones(N_sources).astype(float) * 99.
    flambda_err_Arr = np.copy(flambda_Arr)

    src_list = np.where(cat['nice_lya'])[0]
    for src in src_list:
        nb_min = np.max([nb_w_rest[src] - 1, 0])
        nb_max = nb_w_rest[src] + 1

        w = cat['err'][nb_min : nb_max, src] ** -2
        flambda_Arr[src] = np.average(cat['flx'][nb_min : nb_max, src],
                                      weights=w)
        flambda_err_Arr[src] = np.sum(w, axis=0) ** -0.5

    magAB_Arr = flux_to_mag(flambda_Arr, wavelength)
    magAB_err_Arr = magAB_Arr - flux_to_mag(flambda_Arr + flambda_err_Arr,
                                             wavelength)
    M_Arr = np.ones(N_sources) * 99

    mask = cat['nice_lya']
    M_Arr[mask] = magAB_Arr[mask] - 5 * (np.log10(dist_lum_Arr[mask]) - 1)

    return M_Arr, magAB_err_Arr