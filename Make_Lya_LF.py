from load_paus_mocks import load_qso_mock ## Provisional

import numpy as np

from scipy.stats import binned_statistic_2d

from jpasLAEs.utils import mag_to_flux

from paus_utils import *
from LAE_selection_method import select_LAEs


def puricomp2d_weights(r_Arr, L_lya_Arr, puri2d, comp2d,
                       puricomp2d_L_bins, puricomp2d_r_bins):
    '''
    Returns the array of weights, purity and completeness as estimated
    by the puricomp2d matrices.
    '''
    w_mat = np.zeros_like(puri2d)
    mask_nonzero_comp2d = comp2d > 0
    w_mat[mask_nonzero_comp2d] =\
        puri2d[mask_nonzero_comp2d] / comp2d[mask_nonzero_comp2d]

    # Add a zeros row & column to w_mat for perturbed luminosities exceeding the binning
    w_mat = np.vstack([w_mat, np.zeros(w_mat.shape[1])])
    w_mat = np.hstack([w_mat, np.zeros(w_mat.shape[0]).reshape(-1, 1)])

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

    return w_mat[xx - 1, yy - 1], puri_mat[xx - 1, yy - 1], comp_mat[xx - 1, yy - 1]


def Lya_LF_weights(r_Arr, L_lya_Arr, puri2d, comp2d,
                   puricomp2d_L_bins, puricomp2d_r_bins):
    '''
    Computes the weights for every source to be included in the
    Lya LF computation.
    '''
    # TODO: Add the rest of weights. By now, only accounting for 
    # puricomp2D weights.
    w1, p1, c1 = puricomp2d_weights(r_Arr, L_lya_Arr, puri2d, comp2d,
                                    puricomp2d_L_bins, puricomp2d_r_bins)

    return w1



def Lya_LF_matrix(cat, L_bins, N_iter=500):
    '''
    Makes a matrix of Lya LFs. Each row is a LF made perturbing the L_lya estimate
    with its bin error.
    '''
    N_bins = len(L_bins) - 1

    L_Arr = cat['L_lya_corr']
    L_e_Arr = cat['L_lya_corr_err']

    hist_i_mat = np.zeros((N_iter, N_bins))

    for k in range(N_iter):
        randN = np.random.randn(len(L_Arr))
        L_perturbed = np.empty_like(L_Arr)
        L_perturbed[randN <= 0] = (L_Arr + L_e_Arr[0] * randN)[randN <= 0]
        L_perturbed[randN > 0] = (L_Arr + L_e_Arr[1] * randN)[randN > 0]
        L_perturbed[np.isnan(L_perturbed)] = 0.


def main(nb_min, nb_max):
    # First load the PAUS catalog of the desired field

    # TODO: Load the actual catalogs. For now, we test with the QSO mock
    source_cats_dir = '/home/alberto/almacen/Source_cats'
    mock_path = f'{source_cats_dir}/QSO_PAUS_LAES_2'
    cat = load_qso_mock(mock_path)
    ## PROVISIONAL ERRORS FOR TESTING
    nominal_errs = mag_to_flux(23, w_central) / 3
    cat['err'] = np.ones_like(cat['flx_0']) * nominal_errs.reshape(-1, 1)
    cat['flx'] = cat['flx_0'] + cat['err'] * np.random.normal(size=cat['flx_0'].shape)

    # Load the field correction matrices
    corr_dir = '/home/alberto/almacen/PAUS_data/LF_corrections'
    puri2d = np.load(f'{corr_dir}/puri2D_foo.npy')
    comp2d = np.load(f'{corr_dir}/comp2D_foo.npy')
    puricomp2d_L_bins = np.load(f'{corr_dir}/puricomp2D_L_bins.npy')
    puricomp2d_r_bins = np.load(f'{corr_dir}/puricomp2D_r_bins.npy')

    # Select LAEs in the observational catalogs
    cat = select_LAEs(cat, nb_min, nb_max,
                      ew0min_lya=30, ewmin_other=100)
    print(f'N nice_lya = {sum(cat["nice_lya"])}')



if __name__ == '__main__':
    main(1, 10)