import os

import numpy as np
import pandas as pd

import pickle


def effective_wavelength(transmission, wavelength):
    '''
    Compute the effective wavelength of a filter given its transmission curve.

    Parameters:
    -----------
    transmission : array_like
        Array of transmission values for a filter as a function of wavelength.
    wavelength : array_like
        Array of wavelengths in units of angstroms corresponding to the transmission values.

    Returns:
    --------
    float
        The effective wavelength of the filter in units of angstroms.
    '''
    intergral_top = np.trapz(transmission * wavelength, wavelength)
    intergral_bot = np.trapz(transmission * 1. / wavelength, wavelength)

    lambda_pivot = np.sqrt(intergral_top * 1. / intergral_bot)

    return lambda_pivot


def wavelength_fwhm(transmission, wavelength):
    '''
    Compute the full-width at half-maximum (FWHM) of a filter given its transmission curve.

    Parameters:
    -----------
    transmission : array_like
        Array of transmission values for a filter as a function of wavelength.
    wavelength : array_like
        Array of wavelengths in units of angstroms corresponding to the transmission values.

    Returns:
    --------
    float
        The FWHM of the filter in units of angstroms.
    '''
    mask = transmission > np.amax(transmission) * 0.5

    fwhm = wavelength[mask][-1] - wavelength[mask][0]

    return fwhm


paus_tcurves_dir = '/home/alberto/almacen/PAUS_data/OUT_FILTERS'
tcurves_file_list = os.listdir(paus_tcurves_dir)
tcurves_file_list.sort()

paus_fil_names = []

for name in tcurves_file_list:
    if name[4] == 'D':
        this_name = f'NB{name[6:9]}'
    else:
        this_name = name[-5]

    paus_fil_names.append(this_name)

# Sort it so that the first 40 are NB and the last 6 are BBs
paus_fil_names = paus_fil_names[6:] + paus_fil_names[:6]
paus_fil_names[-6:] = (paus_fil_names[-2]
                       + paus_fil_names[-5]
                       + paus_fil_names[-3]
                       + paus_fil_names[-4]
                       + paus_fil_names[-1]
                       + paus_fil_names[-6])

# Now compute the central wavelength of each filter

w_central = np.zeros_like(paus_fil_names)
w_max_trans = np.zeros_like(paus_fil_names)
fwhm_Arr = np.zeros_like(paus_fil_names)

# Let's also generate the paus_tcurves dictionary
tcurves = {
    'tag': [],
    'w': [],
    't': []
}

for i, name in enumerate(paus_fil_names):
    if name[:2] == 'NB':
        dat = np.genfromtxt(f'{paus_tcurves_dir}/AOD_D_{name[2:]}.dat')
    else:
        dat = np.genfromtxt(f'{paus_tcurves_dir}/AOD_BBFL_{name}.txt')

    w_Arr = dat[:, 0]
    t_Arr = dat[:, 1]

    tcurves['w'].append(w_Arr * 10)
    tcurves['t'].append(t_Arr)
    tcurves['tag'].append(name)

    w_central[i] = effective_wavelength(t_Arr, w_Arr) * 10
    fwhm_Arr[i] = wavelength_fwhm(t_Arr, w_Arr) * 10
    w_max_trans[i] = w_Arr[np.argmax(t_Arr)] * 10

colors = [
    '#00B8FF',
    '#00FFFF',
    '#00FFB8',
    '#00FF69',
    '#00FF10',
    '#2BFF00',
    '#4CFF00',
    '#67FF00',
    '#85FF00',
    '#9CFF00',
    '#B3FF00',
    '#CEFF00',
    '#E5FF00',
    '#FFFF00',
    '#FFE300',
    '#FFC400',
    '#FFA600',
    '#FF8A00',
    '#FF6900',
    '#FF4600',
    '#FF1D00',
    '#FC0000',
    '#F70000',
    '#F10000',
    '#EC0000',
    '#E70000',
    '#E30000',
    '#DD0000',
    '#D80000',
    '#D30000',
    '#CD0000',
    '#C90000',
    '#C30000',
    '#BE0000',
    '#B80000',
    '#B30000',
    '#AE0000',
    '#A70053',
    '#A2004B',
    '#9C0043',
    '#B400FF',
    '#006600',
    '#FF0000',
    '#990033',
    '#610000',
    '#610000'
]


# Save all this info to a csv
data = {
    'name': paus_fil_names,
    'w_eff': w_central,
    'w_max_trans': w_max_trans,
    'fwhm': fwhm_Arr,
    'color': colors
}

path_to_paus_data = '/home/alberto/almacen/PAUS_data'
pd.DataFrame(data).to_csv(f'{path_to_paus_data}/Filter_properties.csv')
with open(f'{path_to_paus_data}/paus_tcurves.pkl', 'wb') as f:
    pickle.dump(tcurves, f)
