from load_paus_mocks import load_qso_mock ## Provisional

import numpy as np
 
import time
import os

from scipy.stats import binned_statistic_2d

from jpasLAEs.utils import mag_to_flux, hms_since_t0, flux_to_mag

from paus_utils import *
from LAE_selection_method import select_LAEs
from PAUS_Lya_LF_corrections import L_lya_bias_apply


def puricomp2d_weights(r_Arr, L_lya_Arr, puri2d, comp2d,
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

    puri_mat = puri2d
    comp_mat = comp2d
    
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



def Lya_LF_matrix(cat, L_bins, field_name, LF_savedir,
                  N_iter=500):
    '''
    Makes a matrix of Lya LFs. Each row is a LF made perturbing the L_lya estimate
    with its bin error.
    '''
    # Load the field correction matrices
    corr_dir = '/home/alberto/almacen/PAUS_data/LF_corrections'
    puri2d = np.load(f'{corr_dir}/puri2D_foo.npy')
    comp2d = np.load(f'{corr_dir}/comp2D_foo.npy')
    puricomp2d_L_bins = np.load(f'{corr_dir}/puricomp2D_L_bins.npy')
    puricomp2d_r_bins = np.load(f'{corr_dir}/puricomp2D_r_bins.npy')

    N_bins = len(L_bins) - 1

    L_Arr = cat['L_lya_corr']
    L_e_Arr = cat['L_lya_corr_err']
    nice_lya = cat['nice_lya']

    hist_i_mat = np.zeros((N_iter, N_bins))

    # Initialize array to save the purity of each LF realization
    puri_list = []

    for k in range(N_iter):
        randN = np.random.randn(len(L_Arr))
        L_perturbed = np.empty_like(L_Arr)
        L_perturbed[randN <= 0] = (L_Arr + L_e_Arr[0] * randN)[randN <= 0]
        L_perturbed[randN > 0] = (L_Arr + L_e_Arr[1] * randN)[randN > 0]
        L_perturbed[np.isnan(L_perturbed)] = 0.

        puri_k, comp_k = Lya_LF_weights(cat['r_mag'], L_perturbed,
                                        puri2d, comp2d,
                                        puricomp2d_L_bins, puricomp2d_r_bins)

        # The array of weights w
        w = np.random.rand(len(puri_k))
        # Mask very low completeness and randomly according to purity
        include_mask = (w < puri_k) & (comp_k > 0.01)
        w[:] = 1.
        w[~include_mask] = 0.
        w[include_mask] = 1. / comp_k[include_mask]
        w[np.isnan(w) | np.isinf(w)] = 0.

        # Store the realization of the LF in the hist matrix
        hist_i_mat[k], _ = np.histogram(L_perturbed[nice_lya],
                                        bins=L_bins, weights=w[nice_lya])
        
        # Store purity of this realization
        puri_list.append(puri_k)

    # Save hist_i_mat
    np.save(f'{LF_savedir}/hist_i_mat_{field_name}.npy', hist_i_mat)
    # Save the purity
    np.save(f'{LF_savedir}/estimated_purity_field_{field_name}',
            np.array(puri_list))
        


def main(nb_min, nb_max, r_min, r_max, field_name):
    # First load the PAUS catalog of the desired field

    # TODO: Load the actual catalogs. For now, we test with the QSO mock
    print(f'\nField: {field_name}')
    print('----------------------')
    if field_name == 'foo':
        source_cats_dir = '/home/alberto/almacen/Source_cats'
        mock_path = f'{source_cats_dir}/QSO_PAUS_LAES_2'
        cat = load_qso_mock(mock_path)
        ## PROVISIONAL ERRORS FOR TESTING
        nominal_errs = mag_to_flux(23, w_central) / 3
        cat['err'] = np.ones_like(cat['flx_0']) * nominal_errs.reshape(-1, 1)
        cat['flx'] = cat['flx_0'] + cat['err'] * np.random.normal(size=cat['flx_0'].shape)
        cat['r_mag'] = flux_to_mag(cat['flx'][-4], w_central[-4])
    else:
        raise ValueError(f'Field name `{field_name}` not valid')

    # Select LAEs in the observational catalogs
    print('Selecting LAEs')
    cat = select_LAEs(cat, nb_min, nb_max, r_min, r_max,
                      ew0min_lya=30, ewmin_other=100)
    print(f'N nice_lya = {sum(cat["nice_lya"])}')

    # Apply the bias correction and compute L statistical errors
    cat = L_lya_bias_apply(cat)


    #######################
    ### Make the Lya LF ###   
    #######################

    # Define the LF L binning
    L_min, L_max = 40, 47
    N_bins = 30
    L_bins = np.linspace(L_min, L_max, N_bins + 1)

    # Dir to save the LFs and subproducts
    LF_name = f'Lya_LF_nb{nb_min}-{nb_max}_{field_name}'
    LF_savedir = f'/home/alberto/almacen/PAUS_data/Lya_LFs/{LF_name}'
    os.makedirs(LF_savedir, exist_ok=True)

    # Save the bins
    np.save(f'{LF_savedir}/LF_L_bins.npy', L_bins)


    print('Making the LF')
    Lya_LF_matrix(cat, L_bins, field_name, LF_savedir)



if __name__ == '__main__':
    print('Computing the Lya LF')

    field_list = ['foo']

    t00 = time.time()

    for field_name in field_list:
        t0 = time.time()

        main(1, 10, 17, 24, field_name)

        print('Done in {0}h {1}m {2}s'.format(*hms_since_t0(t0)))

    
    print('\nEverything done in {0}h {1}m {2}s'.format(*hms_since_t0(t00)))