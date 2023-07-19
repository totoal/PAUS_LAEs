#!/home/alberto/miniconda3/bin/python3

from load_paus_mocks import add_errors, load_mock_dict
from load_paus_cat import load_paus_cat

import numpy as np
 
import time
import os
import sys
import pickle

from scipy.stats import binned_statistic_2d

from jpasLAEs.utils import hms_since_t0, flux_to_mag

from paus_utils import *
from LAE_selection_method import select_LAEs
from PAUS_Lya_LF_corrections import L_lya_bias_apply


def puricomp2d_weights(r_Arr, L_lya_Arr, puri_mat, comp_mat,
                       puricomp2d_L_bins, puricomp2d_r_bins):
    '''
    Returns the array of weights, purity and completeness as estimated
    by the puricomp2d matrices.
    '''
    # If L_Arr is empty, return empty weights lists
    if len(L_lya_Arr) == 0:
        return np.array([]), np.array([]), np.array([])

    bs = binned_statistic_2d(L_lya_Arr, r_Arr, None, 'count',
                             bins=[puricomp2d_L_bins, puricomp2d_r_bins],
                             expand_binnumbers=True)

    puri_mat[np.isnan(puri_mat) | np.isinf(puri_mat)] = 0.
    comp_mat[np.isnan(comp_mat) | np.isinf(comp_mat)] = 0.

    puri_mat = np.vstack([puri_mat, np.zeros(puri_mat.shape[1])])
    puri_mat = np.hstack([puri_mat, np.zeros(puri_mat.shape[0]).reshape(-1, 1)])
    comp_mat = np.vstack([comp_mat, np.zeros(comp_mat.shape[1])])
    comp_mat = np.hstack([comp_mat, np.zeros(comp_mat.shape[0]).reshape(-1, 1)])

    xx, yy = bs.binnumber

    return puri_mat[xx - 1, yy - 1], comp_mat[xx - 1, yy - 1]


def Lya_LF_weights(r_Arr, L_lya_Arr, puri2d, comp2d,
                   puricomp2d_L_bins, puricomp2d_r_bins):
    '''
    Computes the weights for every source to be included in the
    Lya LF computation.
    '''
    # TODO: Add the rest of weights. By now, only accounting for 
    # puricomp2D weights.
    p1, c1 = puricomp2d_weights(r_Arr, L_lya_Arr, puri2d, comp2d,
                                puricomp2d_L_bins, puricomp2d_r_bins)

    # Combine all contributions of purity and completeness
    puri = p1
    comp = c1

    return puri, comp



def Lya_LF_matrix(cat, L_bins, nb_min, nb_max, LF_savedir,
                  N_iter=100, N_boots=5):
    '''
    Makes a matrix of Lya LFs. Each row is a LF made perturbing the L_lya estimate
    with its bin error.
    '''
    # Load the field correction matrices
    corr_dir = '/home/alberto/almacen/PAUS_data/LF_corrections'
    
    # For nb_min = nb_max, we load the corrections for three consecutive filters
    # for better precision
    if nb_min != nb_max:
        nb_min_corr = nb_min
        nb_max_corr = nb_max
    elif nb_min % 2:
        nb_min_corr = nb_min - 1
        nb_max_corr = nb_min + 1
    elif nb_min < 16:
        nb_min_corr = nb_min
        nb_max_corr = nb_min + 2
    elif nb_min == 16:
        nb_min_corr = nb_min - 2
        nb_max_corr = nb_min
    else:
        raise Exception('No puricomp2d corrections available in this NB range.')

    puri2d = np.load(f'{corr_dir}/puri2D_{field_name}_nb{nb_min_corr}-{nb_max_corr}.npy')
    comp2d = np.load(f'{corr_dir}/comp2D_{field_name}_nb{nb_min_corr}-{nb_max_corr}.npy')

    puricomp2d_L_bins = np.load(f'{corr_dir}/puricomp2D_L_bins.npy')
    puricomp2d_r_bins = np.load(f'{corr_dir}/puricomp2D_r_bins.npy')

    N_bins = len(L_bins) - 1

    L_Arr = cat['L_lya_corr']
    L_e_Arr = cat['L_lya_corr_err']
    
    # This is the global nice_lya array. We the selection in N_boots subsamples.
    total_nice_lya = cat['nice_lya']

    hist_i_mat = np.zeros((N_iter, N_bins))

    unique_pointing_ids = np.unique(cat['pointing_id'])
    N_boots = 10
    pointings_per_boot, remainder = divmod(len(unique_pointing_ids), N_boots)
    N_pointings_Arr = np.ones(N_boots).astype(int) * pointings_per_boot
    N_pointings_Arr[-1] += remainder

    for boot_i in range(N_boots + 1):
        print(f'Subregion: {boot_i}')
        if boot_i == 0:
            # First compute the LF with all the sources
            boot_nice_lya = total_nice_lya
        else:
            point_ids_boot = np.random.choice(unique_pointing_ids,
                                              size=N_pointings_Arr[boot_i - 1],
                                              replace=True)
            
            boot_nice_lya = np.concatenate([
                np.where(cat['pointing_id'] == int(pid))[0] for pid in point_ids_boot
            ]).astype(int)

        # Preliminar completeness
        _, comp_preliminar =\
            Lya_LF_weights(cat['r_mag'][boot_nice_lya], L_Arr[boot_nice_lya],
                           puri2d, comp2d,
                           puricomp2d_L_bins, puricomp2d_r_bins)
        pre_comp_mask = (comp_preliminar > 0.05)

        for k in range(N_iter):
            if (k + 1) % 50 == 0:
                print(f'Progress: {k + 1} / {N_iter}',
                        end=('\r' if k + 1 < N_iter else '\n'))

            randN = np.random.randn(len(L_Arr))
            L_perturbed = np.empty_like(L_Arr)
            L_perturbed[randN <= 0] = (L_Arr + L_e_Arr[0] * randN)[randN <= 0]
            L_perturbed[randN > 0] = (L_Arr + L_e_Arr[1] * randN)[randN > 0]
            L_perturbed[np.isnan(L_perturbed)] = 0.

            puri_k, comp_k =\
                Lya_LF_weights(cat['r_mag'][boot_nice_lya], L_perturbed[boot_nice_lya],
                               puri2d, comp2d,
                               puricomp2d_L_bins, puricomp2d_r_bins)

            # The array of weights w
            w = np.random.rand(len(puri_k))
            # Mask very low completeness and randomly according to purity
            include_mask = (w <= puri_k) & (comp_k > 0.05) & pre_comp_mask
            w[~include_mask] = 0.
            w[include_mask] = 1. / comp_k[include_mask]
            w[np.isnan(w) | np.isinf(w)] = 0. # Just in case

            # Store the realization of the LF in the hist matrix
            hist_i_mat[k], _ = np.histogram(L_perturbed[boot_nice_lya],
                                            bins=L_bins, weights=w)
            
        # Save hist_i_mat
        np.save(f'{LF_savedir}/hist_i_mat_{boot_i}.npy', hist_i_mat)
        


def main(nb_min, nb_max, r_min, r_max, field_name):
    # First load the PAUS catalog of the desired field

    print(f'\nField: {field_name}')
    print('----------------------')
    mock_list = ['SFG', 'QSO_cont', 'QSO_LAEs_loL', 'QSO_LAEs_hiL',
                   'GAL']
    PAUS_field_names = ['W1', 'W3']

    if field_name in mock_list:
        print('Loading catalog (mock)')
        source_cats_dir = '/home/alberto/almacen/Source_cats'
        mock_SFG_path = f'{source_cats_dir}/LAE_12.5deg_z2.55-5_PAUS_0'
        mock_QSO_cont_path = f'{source_cats_dir}/QSO_PAUS_contaminants_2'
        mock_QSO_LAEs_loL_path = f'{source_cats_dir}/QSO_PAUS_LAES_2'
        mock_QSO_LAEs_hiL_path = f'{source_cats_dir}/QSO_PAUS_LAES_hiL_2'
        mock_GAL_path = '/home/alberto/almacen/PAUS_data/catalogs/LightCone_mock.fits'
        mocks_dict = load_mock_dict(mock_SFG_path, mock_QSO_cont_path,
                                    mock_QSO_LAEs_loL_path, mock_QSO_LAEs_hiL_path,
                                    mock_GAL_path, gal_fraction=1.)
        cat = mocks_dict[field_name]
    elif field_name in PAUS_field_names:
        cats_dir = '/home/alberto/almacen/PAUS_data/catalogs'
        path_to_cat_list = [f'{cats_dir}/PAUS_3arcsec_{field_name}.csv']
        cat = load_paus_cat(path_to_cat_list)
    else:
        raise ValueError(f'Field name `{field_name}` not valid')

    # cat['r_mag'] = flux_to_mag(cat['flx'][-4], w_central[-4])
    ################################################
    #### Testing with a synthetic BB ####
    stack_nb_ids = np.arange(12, 16 + 1)
    synth_BB_flx = np.average(cat['flx'][stack_nb_ids],
                              weights=cat['err'][stack_nb_ids] ** -2,
                              axis=0)
    cat['r_mag'] = flux_to_mag(synth_BB_flx, w_central[-4])
    ################################################

    # TEMPORARILY limit the catalog to objs with all the NBs
    mask_NB_number = (cat['NB_number'] > 39)
    cat['flx'] = cat['flx'][:, mask_NB_number]
    cat['err'] = cat['err'][:, mask_NB_number]
    cat['NB_mask'] = cat['NB_mask'][:, mask_NB_number]
    for key in cat.keys():
        if key in ['flx', 'err', 'NB_mask', 'area']:
            continue
        cat[key] = cat[key][mask_NB_number]

    # Load the pointing_ids
    load_path = f'/home/alberto/almacen/PAUS_data/catalogs/pointing_ids_{field_name}.npy'
    cat['pointing_id'] = np.load(load_path)[mask_NB_number].astype(int)

    # Select LAEs in the observational catalogs
    print('Selecting LAEs')
    cat = select_LAEs(cat, nb_min, nb_max, r_min, r_max)
    print(f'N nice_lya = {sum(cat["nice_lya"])}')

    # Apply the bias correction and compute L statistical errors
    cat = L_lya_bias_apply(cat, field_name, nb_min, nb_max)


    #######################
    ### Make the Lya LF ###   
    #######################

    # Define the LF L binning
    L_min, L_max = 40, 47
    N_bins = 50
    L_bins = np.linspace(L_min, L_max, N_bins + 1)

    # Dir to save the LFs and subproducts
    LF_name = f'Lya_LF_nb{nb_min}-{nb_max}_{field_name}'
    LF_savedir = f'/home/alberto/almacen/PAUS_data/Lya_LFs/{LF_name}'
    os.makedirs(LF_savedir, exist_ok=True)

    # Save the bins
    np.save(f'{LF_savedir}/LF_L_bins.npy', L_bins)

    print('Making the LF')
    Lya_LF_matrix(cat, L_bins, nb_min, nb_max, LF_savedir)

    # Save a dictionary with useful data about the selection
    reduced_cat = {}
    keys_to_save = ['ref_id', 'r_mag', 'lya_NB', 'EW0_lya', 'EW0_lya_err',
                    'L_lya', 'L_lya_corr', 'L_lya_corr_err']
    for key in keys_to_save:
        reduced_cat[key] = cat[key][..., cat['nice_lya']]
    with open(f'{LF_savedir}/selection.pkl', 'wb') as f:
        pickle.dump(reduced_cat, f)


if __name__ == '__main__':
    print('\n##########################')
    print('Computing the Lya LF')

    field_list = ['W3']

    t00 = time.time()

    for field_name in field_list:
        t0 = time.time()

        r_min, r_max = 17, 24

        [nb_min, nb_max] = [int(nb) for nb in sys.argv[1].split()]

        args = (nb_min, nb_max, r_min, r_max, field_name)

        if args[0] == args[1]:
            print(f'NB: {args[0]}')
        else:
            print(f'NB: {args[0]}-{args[1]}')

        main(*args)

        print('Done in {0}h {1}m {2}s'.format(*hms_since_t0(t0)))

    
    print('\nEverything done in {0}h {1}m {2}s'.format(*hms_since_t0(t00)))