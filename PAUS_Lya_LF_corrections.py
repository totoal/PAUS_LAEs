#!/home/alberto/miniconda3/bin/python3

from jpasLAEs.utils import smooth_Image, bin_centers, hms_since_t0
from K_corr import UV_magnitude_K

from load_paus_mocks import load_mock_dict, add_errors
from paus_utils import *
from LAE_selection_method import *

from scipy.stats import binned_statistic

import pickle
import time
import sys
import os

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

def L_lya_bias_estimation(cat, field_name, nb_min, nb_max):
    '''
    Compute and save L corrections and errors
    '''
    L_binning = np.logspace(40, 47, 25 + 1)

    L_Lbin_err_plus, L_Lbin_err_minus, L_median =\
        compute_L_Lbin_err(cat, L_binning)

    corr_dir = f'/home/alberto/almacen/PAUS_data/LF_corrections'
    os.makedirs(corr_dir, exist_ok=True)
    surname = f'{field_name}_nb{nb_min}-{nb_max}'
    np.save(f'{corr_dir}/L_nb_err_plus_{surname}.npy', L_Lbin_err_plus)
    np.save(f'{corr_dir}/L_nb_err_minus_{surname}.npy', L_Lbin_err_minus)
    np.save(f'{corr_dir}/L_bias_{surname}.npy', L_median)
    np.save(f'/{corr_dir}/L_nb_err_binning.npy', L_binning)


def L_lya_bias_apply(cat, field_name, nb_min, nb_max):
    '''
    Applies the bias sustraction and estimates errors based in the
    computations made by L_lya_bias_estimation.
    '''
    corr_dir = f'/home/alberto/almacen/PAUS_data/LF_corrections'

    if nb_min != nb_max:
        nb_min_corr = nb_min
        nb_max_corr = nb_max
    elif nb_min % 2:
        nb_min_corr = nb_min - 1
        nb_max_corr = nb_min + 1
    elif nb_min < 18:
        nb_min_corr = nb_min
        nb_max_corr = nb_min + 2
    elif nb_min == 18:
        nb_min_corr = nb_min - 2
        nb_max_corr = nb_min
    else:
        raise Exception('No puricomp2d corrections available in this NB range.')

    surname = f'{field_name}_nb{nb_min_corr}-{nb_max_corr}'
    L_Lbin_err_plus = np.load(f'{corr_dir}/L_nb_err_plus_{surname}.npy')
    L_Lbin_err_minus = np.load(f'{corr_dir}/L_nb_err_minus_{surname}.npy')
    L_median = np.load(f'{corr_dir}/L_bias_{surname}.npy')
    L_binning = np.load(f'/{corr_dir}/L_nb_err_binning.npy')
    L_bin_c = bin_centers(L_binning)

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
    cat['L_lya_corr_err'] = np.array(L_e_Arr_pm)

    return cat


def puricomp_corrections(mock_dict, L_bins, r_bins,
                         nb_min, nb_max, ew0_min=0,
                         LF_kind='Lya'):
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
        elif mock_name == 'QSO_LAEs_hiL':
            mask_hiL = mock['L_lya_spec'] >= 44
        else:
            mask_hiL = np.ones_like(mock['L_lya_spec']).astype(bool)

        area_obs = mock['area']

        N_sources = len(mock['zspec'])
        for k in range(N_iter):
            if (k + 1) % 50 == 0:
                print(f'{mock_name} correction matrices: {k + 1} / {N_iter}',
                      end=('\r' if k + 1 < N_iter else '\n'))
            # Generate random numbers
            randN = np.random.randn(N_sources)

            if LF_kind == 'Lya':
                L_perturbed = np.empty(N_sources)
                L_perturbed[randN <= 0] = (mock['L_lya_corr']
                                        + mock['L_lya_corr_err'][0] * randN)[randN <= 0]
                L_perturbed[randN > 0] = (mock['L_lya_corr']
                                        + mock['L_lya_corr_err'][1] * randN)[randN > 0]
                L_perturbed[np.isnan(L_perturbed)] = 0.

                L_spec = mock['L_lya_spec']
            elif LF_kind == 'UV':
                L_perturbed = mock['M_UV'] - randN * mock['M_UV_err']

                L_spec = mock['M_UV_spec']
            else:
                raise Exception('what?')

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
        hist_dict[f'{mock_name}_nice'] = np.average(hist_dict[f'{mock_name}_nice'],
                                                    axis=2)
        hist_dict[f'{mock_name}_sel'] = np.average(hist_dict[f'{mock_name}_sel'],
                                                   axis=2)

        # Compute parent histograms
        parent_mask = ((NB_z(mock['zspec']) >= nb_min)
                       & (NB_z(mock['zspec']) <= nb_max)
                       & (mock['EW0_lya_spec'] >= ew0_min)
                       & mask_hiL)
        hist_dict[f'{mock_name}_parent'] =\
            np.histogram2d(L_spec[parent_mask],
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

    if LF_kind == 'Lya':
        DL = 0.2
    elif LF_kind == 'UV':
        DL = 0.5
    Dr = 0.5

    h2d_nice_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_nice, DL, Dr)
    h2d_sel_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_sel, DL, Dr)
    h2d_parent_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_parent, DL, Dr)

    puri2d = np.empty_like(h2d_nice_smooth).astype(float)
    comp2d = np.empty_like(h2d_nice_smooth).astype(float)

    mask_nonzero_nice = (h2d_nice_smooth > 0)
    mask_nonzero_parent = (h2d_parent_smooth > 0)
    puri2d[mask_nonzero_nice] =\
        h2d_nice_smooth[mask_nonzero_nice] / h2d_sel_smooth[mask_nonzero_nice]
    comp2d[mask_nonzero_parent] =\
        h2d_nice_smooth[mask_nonzero_parent] / h2d_parent_smooth[mask_nonzero_parent]

    puri2d[~mask_nonzero_nice] = np.nan
    comp2d[~mask_nonzero_parent] = np.nan

    return puri2d, comp2d


def compute_LF_corrections(mock_dict, field_name,
                           nb_min, nb_max, r_min, r_max):
    # Modify the mocks adding errors according to the corresponding field
    for mock_name, mock in mock_dict.items():
        # Add errors
        mock['flx'], mock['err'] = add_errors(mock['flx_0'], field_name,
                                              add_errors=True)

        # # Compute r_mag
        # mock['r_mag'] = flux_to_mag(mock['flx'][-4], w_central[-4])
        #### Testing with a synthetic BB ####
        stack_nb_ids = np.arange(12, 26 + 1)
        synth_BB_flx = np.average(mock['flx'][stack_nb_ids],
                                weights=mock['err'][stack_nb_ids] ** -2,
                                axis=0)
        mock['r_mag'] = flux_to_mag(synth_BB_flx, w_central[-4])
        ################################################

        ## Now we have the mock with the errors, do everything else for
        ## each mock

        ## First select LAEs and estimate L_lya etc.
        print(f'{mock_name}: Selecting candidates')
        mock = select_LAEs(mock, nb_min, nb_max, r_min, r_max,
                           check_nice_z=True)

    # L_lya bias correction with the QSO LAEs catalog as reference
    L_lya_bias_estimation(mock_dict['QSO_LAEs_loL'], field_name,
                          nb_min, nb_max)

    # Now apply the bias correction and compute L statistical errors
    for mock_name, mock in mock_dict.items():
        mock = L_lya_bias_apply(mock, field_name, nb_min, nb_max)

        M_UV_spec_Arr, _ = PAUS_monochromatic_Mag(mock, wavelength=1450,
                                                  flx_cat_key='flx_0',
                                                  redshift_cat_key='zspec',
                                                  only_nice=False)
        mock['M_UV_spec'] = M_UV_spec_Arr

    # Save M_UV and M_UV_spec
    where_save = '/home/alberto/almacen/PAUS_data/K_corrections'
    os.makedirs(where_save, exist_ok=True)
    np.save(f'{where_save}/r_synth_nb{nb_min}-{nb_max}.npy', mock_dict['QSO_LAEs_loL']['r_mag'])
    np.save(f'{where_save}/M_UV_spec_nb{nb_min}-{nb_max}.npy', mock_dict['QSO_LAEs_loL']['M_UV_spec'])
    np.save(f'{where_save}/zspec_nb{nb_min}-{nb_max}.npy', mock_dict['QSO_LAEs_loL']['zspec'])

    for mock_name, mock in mock_dict.items():
        mock = L_lya_bias_apply(mock, field_name, nb_min, nb_max)
        ## Compute UV magnitudes
        M_UV_Arr, M_UV_err_Arr = PAUS_monochromatic_Mag(mock, wavelength=1450)
        # M_UV_Arr, M_UV_err_Arr = UV_magnitude_K(mock['r_mag'], mock['z_NB'], nb_min, nb_max,
        #                                         nice_mask=mock['nice_lya'])
        mock['M_UV'] = M_UV_Arr
        mock['M_UV_err'] = M_UV_err_Arr

    # Now compute the correction matrices
    r_bins = np.linspace(r_min, r_max, 200 + 1)
    L_bins = np.linspace(40, 47, 200 + 1)
    M_UV_bins = np.linspace(-29, -16, 300 + 1)


    # For the Lya LF
    puri2d, comp2d = puricomp_corrections(mock_dict, L_bins, r_bins,
                                          nb_min, nb_max, LF_kind='Lya')
    savedir = '/home/alberto/almacen/PAUS_data/LF_corrections'
    os.makedirs(savedir, exist_ok=True)
    np.save(f'{savedir}/puricomp2D_L_bins.npy', L_bins)
    np.save(f'{savedir}/puricomp2D_r_bins.npy', r_bins)
    np.save(f'{savedir}/puri2D_{field_name}_nb{nb_min}-{nb_max}.npy', puri2d)
    np.save(f'{savedir}/comp2D_{field_name}_nb{nb_min}-{nb_max}.npy', comp2d)

    # For the UV LF
    puri2d_uv, comp2d_uv = puricomp_corrections(mock_dict, M_UV_bins, r_bins,
                                                nb_min, nb_max, LF_kind='UV')
    np.save(f'{savedir}/puricomp2D_M_UV_bins.npy', M_UV_bins)
    np.save(f'{savedir}/puri2D_{field_name}_nb{nb_min}-{nb_max}_UV.npy', puri2d_uv)
    np.save(f'{savedir}/comp2D_{field_name}_nb{nb_min}-{nb_max}_UV.npy', comp2d_uv)

    # Finally, let's save the full mocks in the final state to later compute
    # 1D purity, completeness and other stuff.
    # BUT, let's save a reduced version of the mock, without the fluxes
    reduced_mock_dict = {}
    keys_to_save = ['nice_lya', 'nice_lya_0', 'zspec', 'r_mag', 'lya_NB',
                    'EW0_lya_spec', 'L_lya_spec', 'EW0_lya',
                    'L_lya', 'L_lya_corr', 'area', 'flx', 'err', 'M_UV', 'M_UV_spec']
    for mock_name in mock_dict.keys():
        reduced_mock_dict[mock_name] = {}
        for key in keys_to_save:
            reduced_mock_dict[mock_name][key] = mock_dict[mock_name][key]

    with open(f'{savedir}/mock_dict_{field_name}_nb{nb_min}-{nb_max}.pkl', 'wb') as f:
        pickle.dump(reduced_mock_dict, f)


def main(nb_min, nb_max, r_min, r_max):
    # Load only a fraction of the GAL mock because it's too heavy
    gal_fraction = 1.
    # Load the mocks
    source_cats_dir = '/home/alberto/almacen/Source_cats'
    mock_SFG_path = f'{source_cats_dir}/LAE_12.5deg_z2.55-5_PAUS_0'
    mock_QSO_cont_path = f'{source_cats_dir}/QSO_PAUS_contaminants_2'
    mock_QSO_LAEs_loL_path = f'{source_cats_dir}/QSO_PAUS_LAES_2'
    mock_QSO_LAEs_hiL_path = f'{source_cats_dir}/QSO_PAUS_LAES_hiL_2'
    mock_GAL_dir = '/home/alberto/almacen/PAUS_data/Dani_Lightcone/columns'  # path where the data is store
    mock_GAL_suff = '_magCut[PAUS_BBF_i_25]_LC_chunks[0-511].npy'
    mock_dict = load_mock_dict(mock_SFG_path, mock_QSO_cont_path,
                                 mock_QSO_LAEs_loL_path, mock_QSO_LAEs_hiL_path,
                                 mock_GAL_dir, mock_GAL_suff, gal_fraction=gal_fraction,
                                 load_artifact_mock=False)

    # State the mock area in deg²:
    area_dict = {
        # 'SFG': 400,
        'SFG': 1e99,
        'QSO_cont': 1000,
        'QSO_LAEs_loL': 1000,
        'QSO_LAEs_hiL': 5000,
        'GAL': 59.97 * gal_fraction,
    }

    for mock_name, area in area_dict.items():
        mock_dict[mock_name]['area'] = area

    # List of PAUS fields
    field_list = ['W1', 'W2', 'W3']
    # field_list = ['W3']
    for field_name in field_list:
        print(f'Field: {field_name}')
        print('----------------------')
        compute_LF_corrections(mock_dict, field_name,
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