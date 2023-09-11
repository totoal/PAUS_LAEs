#!/home/alberto/miniconda3/bin/python3

import numpy as np

from paus_utils import Lya_effective_volume

from jpasLAEs.utils import bin_centers

import os
import sys

def bootstrapped_LFs(nb_list, region_list, boot_i,
                     combined_LF=False):
    '''
    Returns a matrix of N iterations of the LF using the fields given by
    region_list_indices.
    '''
    this_hist = None
    masked_volume = None
    for region_name in region_list:
        for [nb1, nb2] in nb_list:
            LF_name = f'Lya_LF_nb{nb1}-{nb2}_{region_name}'
            pathname = f'/home/alberto/almacen/PAUS_data/Lya_LFs/{LF_name}'
            filename_hist = f'{pathname}/hist_i_mat_{boot_i}.npy'

            L_bins = np.load(f'{pathname}/LF_L_bins.npy')
            L_bins_c = bin_centers(L_bins)

            hist_i_mat = np.load(filename_hist)

            if combined_LF:
                this_volume = Lya_effective_volume(nb1, nb2, region_name)
                vol_Arr = this_volume * np.ones_like(L_bins_c)
                N_median_hist = np.nanmedian(hist_i_mat, axis=0)
                vol_Arr[N_median_hist <= -1] = 0
                hist_i_mat[:, N_median_hist <= -1] = 0

                if masked_volume is None:
                    masked_volume = vol_Arr
                else:
                    masked_volume += vol_Arr

            if this_hist is None:
                this_hist = hist_i_mat
            else:
                this_hist += hist_i_mat


    bin_width = np.array([L_bins[i + 1] - L_bins[i] for i in range(len(L_bins) - 1)])

    if combined_LF:
        eff_vol = masked_volume
    else:
        eff_vol = 0.
        for region_name in region_list:
            for [nb1, nb2] in nb_list:
                eff_vol += Lya_effective_volume(nb1, nb2, region_name) * np.ones_like(L_bins_c)


    lum_func = np.zeros_like(this_hist).astype(float)
    lum_func[:, eff_vol > 0] = this_hist[:, eff_vol > 0]\
        / bin_width[eff_vol > 0] / eff_vol[eff_vol > 0]

    return lum_func


def count_N_boots(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename[-5] == '0':
            continue
        if filename.startswith('hist_i_mat') and os.path.isfile(os.path.join(directory, filename)):
            count += 1
    return count



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

            LF_name = f'Lya_LF_nb0-2_W3'
            pathname = f'/home/alberto/almacen/PAUS_data/Lya_LFs/{LF_name}'
            N_boots = count_N_boots(pathname)
            boots_ids = np.random.choice(np.arange(N_boots), N_boots,
                                         replace=True) + 1
            region_list = ['W3', 'W1', 'W2']

            this_hist_mat = 0.
            for boot_i in boots_ids:
                this_hist_mat += bootstrapped_LFs(nb_list, region_list,
                                                  boot_i, combined_LF=True)

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
        np.save(f'{pathname}/hist_mat_boots_combi', hist_mat)

        sys.exit(0)

    print('\n##########################')
    print('\nBootstrapping sky regions')

    [nb1, nb2] = [int(nb) for nb in sys.argv[1].split()]

    if nb1 == nb2:
        print(f'NB: {nb1}')
    else:
        print(f'NB: {nb1}-{nb2}')

    hist_mat = None
    
    N_realizations = 100
    for iter_i in range(N_realizations):
        print(f'{iter_i + 1} / {N_realizations}', end='\r')

        LF_name = f'Lya_LF_nb{nb1}-{nb2}_W3'
        pathname = f'/home/alberto/almacen/PAUS_data/Lya_LFs/{LF_name}'
        N_boots = count_N_boots(pathname)
        boots_ids = np.random.choice(np.arange(N_boots), N_boots,
                                     replace=True) + 1
        region_list = ['W3', 'W1', 'W2']

        this_hist_mat = 0.
        for boot_i in boots_ids:
            this_hist_mat += bootstrapped_LFs([[nb1, nb2]], region_list,
                                              boot_i, combined_LF=False)

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
    np.save(f'{pathname}/hist_mat_boots_nb{nb1}-{nb2}', hist_mat)