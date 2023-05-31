#!/home/alberto/miniconda3/bin/python3

from jpasLAEs.utils import smooth_Image, bin_centers, mag_to_flux, hms_since_t0

from load_paus_mocks import load_mocks_dict, add_errors
from paus_utils import *
from LAE_selection_method import *

from scipy.stats import binned_statistic

import pickle
import time
import sys

import numpy as np

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

def L_lya_bias_estimation(cat):
    '''
    Compute and save L corrections and errors
    '''
    L_binning = np.logspace(40, 47, 25 + 1)

    L_Lbin_err_plus, L_Lbin_err_minus, L_median =\
        compute_L_Lbin_err(cat, L_binning)

    corr_dir = f'/home/alberto/almacen/PAUS_data/LF_corrections'
    np.save(f'{corr_dir}/L_nb_err_plus.npy', L_Lbin_err_plus)
    np.save(f'{corr_dir}/L_nb_err_minus.npy', L_Lbin_err_minus)
    np.save(f'{corr_dir}/L_bias.npy', L_median)
    np.save(f'/{corr_dir}/L_nb_err_binning.npy', L_binning)


def L_lya_bias_apply(cat):
    '''
    Applies the bias sustraction and estimates errors based in the
    computations made by L_lya_bias_estimation.
    '''
    corr_dir = f'/home/alberto/almacen/PAUS_data/LF_corrections'
    L_Lbin_err_plus = np.load(f'{corr_dir}/L_nb_err_plus.npy')
    L_Lbin_err_minus = np.load(f'{corr_dir}/L_nb_err_minus.npy')
    L_median = np.load(f'{corr_dir}/L_bias.npy')
    L_binning = np.load(f'/{corr_dir}/L_nb_err_binning.npy')
    L_bin_c = [L_binning[i: i + 2].sum() * 0.5 for i in range(len(L_binning) - 1)]

    # Correct L_Arr with the median
    mask_median_L = (L_median < 10)
    L_Arr_corr = (cat['L_lya']
                  - np.interp(cat['L_lya'],
                              np.log10(L_bin_c)[mask_median_L],
                              L_median[mask_median_L]))

    # Apply bin err
    L_binning_position = binned_statistic(10 ** cat['L_lya'], None,
                                          'count', bins=L_binning).binnumber
    L_binning_position[L_binning_position > len(L_binning) - 2] = len(L_binning) - 2
    L_e_Arr_pm = [L_Lbin_err_minus[L_binning_position],
                  L_Lbin_err_plus[L_binning_position]]

    cat['L_lya_corr'] = L_Arr_corr
    cat['L_lya_corr_err'] = L_e_Arr_pm

    return cat


def puricomp_corrections(mock_dict, L_bins, r_bins,
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
        # If mock is `QSO_LAEs_loL` add extra mask to cut out high L_lya sources
        if mock_name == 'QSO_LAEs_loL':
            mask_hiL = mock['L_lya_spec'] < 44
        else:
            mask_hiL = np.ones_like(mock['L_lya_spec']).astype(bool)

        area_obs = mock['area']

        N_sources = len(mock['zspec'])
        for k in range(N_iter):
            if (k + 1) % 50 == 0:
                print(f'{mock_name} correction matrices: {k + 1} / {N_iter}',
                      end=('\r' if k + 1 < N_iter else ''))
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
                         & (mock['EW0_lya_spec'] >= ew0_min) & mask_hiL)
            hist_dict[f'{mock_name}_nice'][..., k] =\
                np.histogram2d(L_perturbed[nice_mask], mock['r_mag'][nice_mask],
                               bins=[L_bins, r_bins])[0]
            
            sel_mask = (mock['nice_lya']
                        & (mock['lya_NB'] >= nb_min) & (mock['lya_NB'] <= nb_max)
                        & mask_hiL)
            hist_dict[f'{mock_name}_sel'][..., k] =\
                np.histogram2d(L_perturbed[sel_mask], mock['r_mag'][sel_mask],
                               bins=[L_bins, r_bins])[0]

        # Apply area factor
        hist_dict[f'{mock_name}_nice'] /= area_obs
        hist_dict[f'{mock_name}_sel'] /= area_obs
        
        # Take the median
        hist_dict[f'{mock_name}_nice'] = np.median(hist_dict[f'{mock_name}_nice'],
                                                   axis=2)
        hist_dict[f'{mock_name}_sel'] = np.median(hist_dict[f'{mock_name}_sel'],
                                                  axis=2)

        # Compute parent histograms
        parent_mask = ((NB_z(mock['zspec']) >= nb_min)
                       & (NB_z(mock['zspec']) <= nb_max)
                       & (mock['EW0_lya_spec'] >= ew0_min)
                       & mask_hiL)
        hist_dict[f'{mock_name}_parent'] =\
            np.histogram2d(mock['L_lya_spec'][parent_mask],
                           mock['r_mag'][parent_mask],
                           bins=[L_bins, r_bins])[0] / area_obs

    h2d_nice = np.zeros((len(L_bins) - 1, len(r_bins) - 1))
    h2d_sel = np.zeros((len(L_bins) - 1, len(r_bins) - 1))
    h2d_parent = np.zeros((len(L_bins) - 1, len(r_bins) - 1))

    for key in mock_dict.keys():
        h2d_nice = h2d_nice + hist_dict[f'{key}_nice']
        h2d_sel = h2d_sel + hist_dict[f'{key}_sel']
        h2d_parent = h2d_parent + hist_dict[f'{key}_parent']

    # Make the mats smooooooth
    r_bins_c = bin_centers(r_bins)
    L_bins_c = bin_centers(L_bins)

    h2d_nice_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_nice, 0.15, 0.3)
    h2d_sel_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_sel, 0.15, 0.3)
    h2d_parent_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_parent, 0.15, 0.3)

    # np.save(f'{dirname}/h2d_nice_smooth_{survey_name}', h2d_nice_smooth)
    # np.save(f'{dirname}/h2d_sel_smooth_{survey_name}', h2d_sel_smooth)
    # np.save(f'{dirname}/h2d_parent_smooth_{survey_name}', h2d_parent_smooth)

    puri2d = np.empty_like(h2d_nice_smooth)
    comp2d = np.empty_like(h2d_nice_smooth)

    mask_nonzero_nice = (h2d_nice_smooth > 0)
    mask_nonzero_parent = (h2d_parent_smooth > 0)
    puri2d[mask_nonzero_nice] = (1 + h2d_sel_smooth[mask_nonzero_nice]\
                                 / h2d_nice_smooth[mask_nonzero_nice]) ** -1
    comp2d[mask_nonzero_parent] =\
        h2d_nice_smooth[mask_nonzero_parent] / h2d_parent_smooth[mask_nonzero_parent]

    puri2d[~mask_nonzero_nice] = np.nan
    comp2d[~mask_nonzero_parent] = np.nan

    return puri2d, comp2d


def compute_LF_corrections(mocks_dict, field_name,
                           nb_min, nb_max, r_min, r_max):
    # Modify the mocks adding errors according to the corresponding field
    for mock_name, mock in mocks_dict.items():
        ## PROVISIONAL ERRORS FOR TESTING
        nominal_errs = mag_to_flux(23, w_central) / 3
        mock['err'] = np.ones_like(mock['flx_0']) * nominal_errs.reshape(-1, 1)
        mock['flx'] = mock['flx_0'] + mock['err'] * np.random.normal(size=mock['flx_0'].shape)
        # TODO: add_errors function
        # mock['flx'], mock['err'] = add_errors(mock['flx_0'], field_name)

        # Compute r_mag
        mock['r_mag'] = flux_to_mag(mock['flx'][-4], w_central[-4])

        ## Now we have the mock with the errors, do everything else for
        ## each mock

        ## First select LAEs and estimate L_lya etc.
        print(f'{mock_name}: Selecting candidates')
        mock = select_LAEs(mock, nb_min, nb_max, r_min, r_max,
                           ew0min_lya=30, ewmin_other=100,
                           check_nice_z=True)

    # L_lya bias correction with the QSO LAEs catalog as reference
    L_lya_bias_estimation(mocks_dict['QSO_LAEs_loL'])

    # Now apply the bias correction and compute L statistical errors
    for _, mock in mocks_dict.items():
        mock = L_lya_bias_apply(mock)

    # Now compute the correction matrices
    r_bins = np.linspace(r_min, r_max, 200 + 1)
    L_bins = np.linspace(40, 47, 200 + 1)
    puri2d, comp2d = puricomp_corrections(mocks_dict, L_bins, r_bins,
                                          nb_min, nb_max, ew0_min=30)
    savedir = '/home/alberto/almacen/PAUS_data/LF_corrections'
    np.save(f'{savedir}/puricomp2D_L_bins.npy', L_bins)
    np.save(f'{savedir}/puricomp2D_r_bins.npy', r_bins)
    np.save(f'{savedir}/puri2D_{field_name}_nb{nb_min}-{nb_max}.npy', puri2d)
    np.save(f'{savedir}/comp2D_{field_name}_nb{nb_min}-{nb_max}.npy', comp2d)

    # Finally, let's save the full mocks in the final state to later compute
    # 1D purity, completeness and other stuff.
    with open(f'{savedir}/mock_dict_{field_name}_nb{nb_min}-{nb_max}.pkl', 'wb') as f:
        pickle.dump(mocks_dict, f)




def main(nb_min, nb_max, r_min, r_max):
    # Load only a fraction of the GAL mock because it's too heavy
    gal_fraction = 1.
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

    # State the mock area in deg²:
    area_dict = {
        'SFG': 400,
        'QSO_cont': 200,
        'QSO_LAEs_loL': 400,
        'QSO_LAEs_hiL': 4000,
        'GAL': 59.97 * gal_fraction
    }

    for mock_name, area in area_dict.items():
        mocks_dict[mock_name]['area'] = area

    # List of PAUS fields
    field_list = ['foo']
    for field_name in field_list:
        print(f'Field: {field_name}')
        print('----------------------')
        compute_LF_corrections(mocks_dict, field_name,
                               nb_min, nb_max, r_min, r_max)

    return

if __name__ == '__main__':
    print('Computing Lya LF corrections')

    t00 = time.time()

    r_min, r_max = 17, 24

    [nb_min, nb_max] = [int(nb) for nb in sys.argv[1].split()]

    args = (nb_min, nb_max, r_min, r_max)

    if args[0] == args[1]:
        print(f'NB: {args[0]}\n')
    else:
        print(f'NB: {args[0]}-{args[1]}\n')

    main(*args)

    print('Done in {0}h {1}m {2}s'.format(*hms_since_t0(t00)))