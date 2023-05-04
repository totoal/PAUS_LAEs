import os

import numpy as np
import pandas as pd


def effective_wavelength(transmission, wavelength):
    """
    Calculates the effective wavelength of an optical filter given its transmission curve.

    Parameters:
    transmission (array-like): The transmission values of the filter.
    wavelength (array-like): The wavelengths corresponding to the transmission values.

    Returns:
    float: The effective wavelength of the filter.
    """

    # Normalize the transmission curve.
    norm_transmission = transmission / np.trapz(transmission, wavelength)

    # Calculate the weighted average of the wavelength.
    weighted_wavelength = np.trapz(wavelength * norm_transmission, wavelength)

    # Calculate the area under the transmission curve.
    area = np.trapz(norm_transmission, wavelength)

    # Calculate the effective wavelength.
    effective_wavelength = weighted_wavelength / area

    return effective_wavelength


def fwhm(transmission, wavelength):
    """
    Calculates the Full Width at Half Maximum (FWHM) of an optical filter given its transmission curve.

    Parameters:
    transmission (array-like): The transmission values of the filter.
    wavelength (array-like): The wavelengths corresponding to the transmission values.

    Returns:
    float: The FWHM of the filter in units of wavelength.
    """

    # Find the maximum transmission value and its corresponding wavelength.
    max_trans = np.max(transmission)

    # Find the two wavelengths on either side of
    # the maximum where the transmission is closest to the half maximum value.
    left_idx = np.argsort(np.abs(max_trans / 2 - transmission))[:10].min()
    right_idx = np.argsort(np.abs(max_trans / 2 - transmission))[:10].max()

    # Get the wavelengths at the edges of the FWHM.
    left_wavelength = wavelength[left_idx]
    right_wavelength = wavelength[right_idx]

    # Compute the FWHM.
    fwhm = right_wavelength - left_wavelength

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

for i, name in enumerate(paus_fil_names):
    if name[:2] == 'NB':
        dat = np.genfromtxt(f'{paus_tcurves_dir}/AOD_D_{name[2:]}.dat')
    else:
        dat = np.genfromtxt(f'{paus_tcurves_dir}/AOD_BBFL_{name}.txt')

    w_Arr = dat[:, 0]
    t_Arr = dat[:, 1]

    w_central[i] = effective_wavelength(t_Arr, w_Arr) * 10
    fwhm_Arr[i] = fwhm(t_Arr, w_Arr) * 10
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

pd.DataFrame(data).to_csv('/home/alberto/almacen/PAUS_data/Filter_properties.csv')