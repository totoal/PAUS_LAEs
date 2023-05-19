from jpasLAEs import *

from load_paus_mocks import load_mocks_dict, add_errors
from paus_utils import *
from LAE_selection_method import *

import numpy as np
import pandas as pd

def compute_L_Lbin_err(cat, L_binning):
    '''
    Computes the errors due to dispersion of L_retrieved with some L_retrieved binning
    '''
    L_lya = cat['L_lya_spec']
    L_Arr = cat['L_lya']

    L_Lbin_err_plus = np.ones(len(L_binning) - 1) * np.inf
    L_Lbin_err_minus = np.ones(len(L_binning) - 1) * np.inf
    median = np.ones(len(L_binning) - 1) * np.inf
    last = [np.inf, np.inf]
    for i in range(len(L_binning) - 1):
        in_bin = (10 ** L_Arr >= L_binning[i]
                  ) & (10 ** L_Arr < L_binning[i + 1])
        if sum(in_bin) == 0:
            L_Lbin_err_plus[i] = last[0]
            L_Lbin_err_minus[i] = last[1]
            continue
        perc = np.nanpercentile((L_Arr - L_lya)[in_bin], [16, 50, 84])
        L_Lbin_err_plus[i] = perc[2] - perc[1]
        L_Lbin_err_minus[i] = perc[1] - perc[0]

        last = [L_Lbin_err_plus[i], L_Lbin_err_minus[i]]
        median[i] = perc[1]

    return L_Lbin_err_plus, L_Lbin_err_minus, median

def L_lya_bias(cat):
    # Compute and save L corrections and errors
    L_binning = np.logspace(40, 47, 25 + 1)
    L_bin_c = [L_binning[i: i + 2].sum() * 0.5 for i in range(len(L_binning) - 1)]

    Lmask = cat['nice_z'] & cat['nice_lya'] & (cat['L_lya_spec'] > 42.5)
    L_Lbin_err_plus, L_Lbin_err_minus, L_median =\
        compute_L_Lbin_err(cat, L_binning)

    # Correct L_Arr with the median
    mask_median_L = (L_median < 10)
    L_Arr_corr = (cat['L_lya']
                  - np.interp(cat['L_lya'],
                              np.log10(L_bin_c)[mask_median_L],
                              L_median[mask_median_L]))

    cat['L_lya_corr'] = L_Arr_corr
    cat['L_lya_corr_err'] = [L_Lbin_err_minus, L_Lbin_err_plus]

    return cat


def compute_LF_corrections(mocks_dict, field_name, nb_min, nb_max):
    # Modify the mocks adding errors according to the corresponding field
    for mock_name, mock in mocks_dict.items():
        print(mock_name)
        mock['flx'], mock['err'] = add_errors(mock['flx_0'], field_name)

        ## Now we have the mock with the errors, do everything else for
        ## each mock

        ## First select LAEs and estimate L_lya etc.
        mock = select_LAEs(mock, nb_min, nb_max,
                           ew0min_lya=30, ewmin_other=100,
                           check_nice_z=True)
        print(f'N nice_lya = {sum(mock["nice_lya"])}')

        # Now produce the correction matrices



def main(nb_min, nb_max):
    # State the mock area in deg²:
    gal_fraction = 0.01

    SFG_area = 400
    QSO_cont_area = 200
    QSO_LAEs_loL_area = 400
    QSO_LAEs_hiL_area = 4000
    GAL_area = 59.97 * gal_fraction

    # Load the mocks
    source_cats_dir = '/home/alberto/almacen/Source_cats'
    mock_SFG_path = f'{source_cats_dir}/LAE_12.5deg_z2.55-5_PAUS_0'
    mock_QSO_cont_path = f'{source_cats_dir}/QSO_PAUS_contaminants_2'
    mock_QSO_LAEs_loL_path = f'{source_cats_dir}/QSO_PAUS_LAES_2'
    mock_QSO_LAEs_hiL_path = f'{source_cats_dir}/QSO_PAUS_LAES_hiL_2'
    mock_GAL_path = '/home/alberto/almacen/PAUS_data/catalogs/LightCone_mock.fits'
    mocks_dict = load_mocks_dict(mock_SFG_path, mock_QSO_cont_path,
                                 mock_QSO_LAEs_loL_path, mock_QSO_LAEs_hiL_path,
                                 mock_GAL_path, gal_fraction=gal_fraction)

    # List of PAUS fields
    # field_list = ['foo', 'bar']
    # for field_name in field_list:
    #     compute_LF_corrections(mocks_dict, field_name,
    #                            nb_min, nb_max)

    return

if __name__ == '__main__':
    main(1, 5)