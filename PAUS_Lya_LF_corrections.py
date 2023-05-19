from jpasLAEs.utils import smooth_Image, bin_centers

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


def puricomp_corrections(mock_dict, area_dict, L_bins, r_bins,
                         nb_min, nb_max, ew0_min=30):
    # Perturb L
    N_iter = 500

    hist_dict = {}
    for key in mock_dict.keys():
        hist_dict[f'{key}_nice'] = np.empty((len(L_bins) - 1,
                                             len(r_bins) - 1, N_iter))
        hist_dict[f'{key}_sel'] = np.empty((len(L_bins) - 1,
                                            len(r_bins) - 1, N_iter))

    for mock_name, mock in mock_dict.items():
        N_sources = len(mock['zspec'])
        for k in range(N_iter):
            # Generate random numbers
            randN = np.random.randn(N_sources)
            L_perturbed = np.empty(N_sources)
            L_perturbed[randN <= 0] = (mock['L_lya_corr']
                                    + mock['L_lya_corr_err'][0] * randN)[randN <= 0]
            L_perturbed[randN > 0] = (mock['L_lya_corr']
                                    + mock['L_lya_corr_err'][1] * randN)[randN > 0]
            L_perturbed[np.isnan(L_perturbed)] = 0.

            nice_mask = (mock['nice_lya'] & mock['nice_z']
                         & (mock['lya_NB'] >= nb_min) & (mock['lya_NB'] <= nb_max)
                         & (mock['EW0_lya_spec'] >= ew0_min))
            hist_dict[f'{mock_name}_nice'][..., k] =\
                np.histogram2d(L_perturbed[nice_mask], mock['r_mag'][nice_mask],
                               bins=[L_bins, r_bins])[0]
            
            sel_mask = (mock['nice_lya']
                        & (mock['lya_NB'] >= nb_min) & (mock['lya_NB'] <= nb_max))
            hist_dict[f'{mock_name}_sel'][..., k] =\
                np.histogram2d(L_perturbed[sel_mask], mock['r_mag'][sel_mask],
                               bins=[L_bins, r_bins])[0]

        # Apply area factor
        hist_dict[f'{mock_name}_nice'] /= area_dict[mock_name]
        hist_dict[f'{mock_name}_sel'] /= area_dict[mock_name]
        
        # Take the median
        hist_dict[f'{mock_name}_nice'] = np.median(hist_dict[f'{mock_name}_nice'],
                                                   axis=2)
        hist_dict[f'{mock_name}_sel'] = np.median(hist_dict[f'{mock_name}_sel'],
                                                  axis=2)

        # Compute parent histograms
        parent_mask = ((NB_z(mock['zspec']) >= nb_min)
                       & (NB_z(mock['zspec']) <= nb_max)
                       & (mock['EW0_lya_spec'] >= ew0_min))
        hist_dict[f'{mock_name}_parent'] =\
            np.histogram2d(mock['L_lya_spec'][parent_mask],
                           mock['r_mag'][parent_mask],
                           bins=[L_bins, r_bins]) / area_dict[mock_name]

    h2d_nice = np.zeros((len(L_bins) - 1, len(r_bins) - 1))
    h2d_sel = np.zeros((len(L_bins) - 1, len(r_bins) - 1))
    h2d_parent = np.zeros((len(L_bins) - 1, len(r_bins) - 1))

    for key in mock_dict.keys():
        h2d_nice += hist_dict[f'{mock_name}_nice']
        h2d_sel += hist_dict[f'{mock_name}_sel']
        h2d_parent += hist_dict[f'{mock_name}_parent']

    # Make the mats smooooooth
    r_bins_c = bin_centers(r_bins)
    L_bins_c = bin_centers(L_bins)

    h2d_nice_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_nice, 0.15, 0.3)
    h2d_sel_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_sel, 0.15, 0.3)
    h2d_parent_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_parent, 0.15, 0.3)

    # np.save(f'{dirname}/h2d_nice_smooth_{survey_name}', h2d_nice_smooth)
    # np.save(f'{dirname}/h2d_sel_smooth_{survey_name}', h2d_sel_smooth)
    # np.save(f'{dirname}/h2d_parent_smooth_{survey_name}', h2d_parent_smooth)

    puri2d = (1 + h2d_sel_smooth / h2d_nice_smooth) ** -1
    comp2d = h2d_nice_smooth / h2d_parent_smooth

    return puri2d, comp2d


def compute_LF_corrections(mocks_dict, area_dict,
                           field_name, nb_min, nb_max, mag_min, mag_max):
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

        # L_lya bias ocrrection
        mock = L_lya_bias(mock)

        # Now produce the correction matrices
        r_bins = np.linspace(mag_min, mag_max, 200 + 1)
        L_bins = np.linspace(40, 47, 200 + 1)




def main(nb_min, nb_max):
    # State the mock area in deg²:
    gal_fraction = 0.01

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

    area_dict = {
        'SFG': 400,
        'QSO_cont': 200,
        'QSO_LAEs_loL': 400,
        'QSO_LAEs_hiL': 4000,
        'GAL': 59.97 * gal_fraction
    }

    # List of PAUS fields
    # field_list = ['foo', 'bar']
    # for field_name in field_list:
    #     compute_LF_corrections(mocks_dict, area_dict, field_name,
    #                            nb_min, nb_max)

    return

if __name__ == '__main__':
    main(1, 5)