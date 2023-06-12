#!/home/alberto/miniconda3/bin/python3

import numpy as np

from paus_utils import Lya_effective_volume

import os
import sys

# List of subregions
region_list_0 = np.array(['W3'])

def bootstrapped_LFs(nb_list, region_list_indices):
    '''
    Returns a matrix of N iterations of the LF using the fields given by
    region_list_indices.
    '''
    region_list = region_list_0[region_list_indices]
    volume_list = []

    this_hist = None
    for jj, region in enumerate(region_list):
        for [nb1, nb2] in nb_list:
            volume_list.append(Lya_effective_volume(nb1, nb2, region))

            LF_name = f'Lya_LF_nb{nb1}-{nb2}_{region}'
            pathname = f'/home/alberto/almacen/PAUS_data/Lya_LFs/{LF_name}'
            filename_hist = f'{pathname}/hist_i_mat.npy'
            hist_i_mat = np.load(filename_hist)

            L_bins = np.load(f'{pathname}/LF_L_bins.npy')
            bin_width = [L_bins[i + 1] - L_bins[i] for i in range(len(L_bins) - 1)]

            this_field_LF = hist_i_mat / volume_list[jj] / bin_width

            if this_hist is None:
                this_hist = hist_i_mat
                field_LF_mat = this_field_LF
            else:
                this_hist += hist_i_mat
                field_LF_mat = np.vstack([field_LF_mat, this_field_LF])

    total_volume = np.sum(volume_list)

    return this_hist / bin_width / total_volume


if __name__ == '__main__':
    if sys.argv[1] == 'combi':
        print('\n##########################')
        print('\nBootstrapping combined sky regions')

        nb_list = [[0, 2], [2, 4], [4, 6], [6, 8],
                [8, 10], [10, 12], [12, 14], [14, 16]]

        hist_mat = None
        
        N_realizations = 100
        for iter_i in range(N_realizations):
            print(f'{iter_i + 1} / {N_realizations}', end='\r')

            # boots = np.random.choice(np.arange(5), 5, replace=True)
            # TODO: by now we only have the 5 mocks
            boots = np.array([0, 0, 0, 0, 0])

            this_hist_mat = bootstrapped_LFs(nb_list, boots)

            if hist_mat is None:
                hist_mat = this_hist_mat
            else:
                hist_mat = np.vstack([hist_mat, this_hist_mat])
        print('\n')

        L_LF_err_percentiles = np.percentile(hist_mat, [16, 50, 84], axis=0)
        LF_err_plus = L_LF_err_percentiles[2] - L_LF_err_percentiles[1]
        LF_err_minus = L_LF_err_percentiles[1] - L_LF_err_percentiles[0]

        name = f'bootstrap_errors'
        pathname = f'/home/alberto/almacen/PAUS_data/Lya_LFs/{name}'
        os.makedirs(pathname, exist_ok=True)

        np.save(f'{pathname}/LF_err_plus_combi', LF_err_plus)
        np.save(f'{pathname}/LF_err_minus_combi', LF_err_minus)
        np.save(f'{pathname}/median_LF_combi', L_LF_err_percentiles[1])

        sys.exit(0)

    print('\n##########################')
    print('\nBootstrapping sky regions')

    [nb1, nb2] = [int(nb) for nb in sys.argv[1].split()]

    if nb1 == nb2:
        print(f'NB: {nb1}')
    else:
        print(f'NB: {nb1}-{nb2}')

    hist_mat = None
    
    N_realizations = 1000
    for iter_i in range(N_realizations):
        print(f'{iter_i + 1} / {N_realizations}', end='\r')

        # boots = np.random.choice(np.arange(5), 5, replace=True)
        # TODO: by now we only have the 5 mocks
        boots = np.array([0, 0, 0, 0, 0])

        this_hist_mat = bootstrapped_LFs(nb1, nb2, boots)

        if hist_mat is None:
            hist_mat = this_hist_mat
        else:
            hist_mat = np.vstack([hist_mat, this_hist_mat])
    print('\n')

    L_LF_err_percentiles = np.percentile(hist_mat, [16, 50, 84], axis=0)
    LF_err_plus = L_LF_err_percentiles[2] - L_LF_err_percentiles[1]
    LF_err_minus = L_LF_err_percentiles[1] - L_LF_err_percentiles[0]

    name = f'bootstrap_errors'
    pathname = f'/home/alberto/almacen/PAUS_data/Lya_LFs/{name}'
    os.makedirs(pathname, exist_ok=True)

    np.save(f'{pathname}/LF_err_plus_nb{nb1}-{nb2}', LF_err_plus)
    np.save(f'{pathname}/LF_err_minus_nb{nb1}-{nb2}', LF_err_minus)
    np.save(f'{pathname}/median_LF_nb{nb1}-{nb2}', L_LF_err_percentiles[1])