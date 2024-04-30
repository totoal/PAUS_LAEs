import numpy as np
from jpasLAEs.utils import bin_centers
from paus_utils import Lya_effective_volume

def load_combined_LF(region_list, NB_list, combined_LF=False,
                     LF_kind='Lya'):
    this_hist = None
    masked_volume = None
    for region_name in region_list:
        for [nb1, nb2] in NB_list:
            LF_name = f'Lya_LF_nb{nb1}-{nb2}_{region_name}'
            pathname = f'/home/alberto/almacen/PAUS_data/Lya_LFs/{LF_name}'

            if LF_kind == 'Lya':
                kind_surname = ''
                L_bins = np.load(f'{pathname}/LF_L_bins.npy')
            elif LF_kind == 'UV':
                kind_surname = '_M'
                L_bins = np.load(f'{pathname}/M_UV_bins.npy')
            else:
                raise ValueError('Unknown LF_kind.')

            L_bins_c = bin_centers(L_bins)

            filename_hist = f'{pathname}/hist_i_mat_0{kind_surname}.npy'

            hist_i_mat = np.load(filename_hist)

            if combined_LF:
                this_volume = Lya_effective_volume(nb1, nb2, region_name)
                vol_Arr = this_volume * np.ones_like(L_bins_c)

                if masked_volume is None:
                    masked_volume = vol_Arr
                else:
                    masked_volume += vol_Arr

            if this_hist is None:
                this_hist = hist_i_mat
            else:
                this_hist += hist_i_mat


    eff_vol = 0
    if combined_LF:
        eff_vol = masked_volume
    else:
        for region_name in region_list:
            for [nb1, nb2] in NB_list:
                eff_vol += Lya_effective_volume(nb1, nb2, region_name) * np.ones_like(L_bins_c)

    bin_width = np.array([L_bins[i + 1] - L_bins[i] for i in range(len(L_bins) - 1)])

    hist_median = np.percentile(this_hist, 50, axis=0)

    boots_path = f'/home/alberto/almacen/PAUS_data/Lya_LFs/bootstrap_errors'
    if combined_LF:
        err_surname = f'combi{kind_surname}'
    elif len(region_list) == 1:
        err_surname = f'nb{nb1}-{nb2}_{region_list[0]}{kind_surname}'
    else:
        err_surname = f'nb{nb1}-{nb2}{kind_surname}'

    yerr_minus = np.load(f'{boots_path}/LF_err_minus_{err_surname}.npy')
    yerr_plus = np.load(f'{boots_path}/LF_err_plus_{err_surname}.npy')
    LF_boots = np.load(f'{boots_path}/median_LF_{err_surname}.npy')

    this_LF = np.zeros_like(hist_median).astype(float)
    poisson_err = np.zeros_like(hist_median).astype(float)
    this_LF[eff_vol > 0] = hist_median[eff_vol > 0] / bin_width[eff_vol > 0] / eff_vol[eff_vol > 0]
    poisson_err[eff_vol > 0] = hist_median[eff_vol > 0] ** 0.5\
        / bin_width[eff_vol > 0] / eff_vol[eff_vol > 0]
    # Fix yerr_minus when LF_boots == 0
    yerr_minus[LF_boots == 0] = this_LF[LF_boots == 0]

    this_LF_dict = {
        'LF_bins': L_bins_c,
        'LF_total': this_LF,
        'LF_boots': LF_boots,
        'LF_total_err': [yerr_plus, yerr_minus],
        'poisson_err': poisson_err
    }

    return this_LF_dict


if __name__ == '__main__':
    lf = load_combined_LF(['W1'], [[2, 4]])