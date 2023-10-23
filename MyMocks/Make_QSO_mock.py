import os

import numpy as np
import pandas as pd
import pickle

from scipy.integrate import dblquad
from scipy.interpolate import RectBivariateSpline

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

from jpasLAEs.utils import mag_to_flux


fil_properties_dir = '/home/alberto/almacen/PAUS_data/Filter_properties.csv'
data_tab = pd.read_csv(fil_properties_dir)
w_central = np.array(data_tab['w_eff'])
path_to_paus_data = '/home/alberto/almacen/PAUS_data'
with open(f'{path_to_paus_data}/paus_tcurves.pkl', 'rb') as f:
    tcurves = pickle.load(f)


def main(z_min, z_max, r_min, r_max, L_min, L_max, area_obs,
         surname='', contaminants=False, train_or_test='none'):
    # Load the SDSS catalog
    filename_pm_DR16 = '/home/alberto/almacen/PAUS_data/PAUS-PHOTOSPECTRA_QSO_Superset_DR16_v2.csv'

    pm_SEDs_DR16 = pd.read_csv(
        filename_pm_DR16, usecols=np.arange(1, 50)).to_numpy()[:, 0:46]

    def format_string4(x): return '{:04d}'.format(int(x))
    def format_string5(x): return '{:05d}'.format(int(x))
    convert_dict = {
        122: format_string4,
        123: format_string5,
        124: format_string4
    }
    plate_mjd_fiber = pd.read_csv(
        filename_pm_DR16, sep=',', usecols=[47, 48, 49],
        converters=convert_dict
    ).to_numpy().T

    plate_mjd_fiber = plate_mjd_fiber[np.array([1, 0, 2])]
    plate = plate_mjd_fiber[0].astype(int)
    mjd = plate_mjd_fiber[1].astype(int)
    fiber = plate_mjd_fiber[2].astype(int)

    # z_Arr of SDSS sources

    Lya_fts = pd.read_csv('/home/alberto/cosmos/LAEs/csv/Lya_fts_DR16_v2.csv')
    z_Arr = Lya_fts['Lya_z'].to_numpy().flatten()
    # z_Arr[z_Arr == 0] = -1

    F_line = np.array(Lya_fts['LyaF']) * 1e-17
    F_line_err = np.array(Lya_fts['LyaF_err']) * 1e-17
    EW0 = np.array(Lya_fts['LyaEW']) / (1 + z_Arr)
    dL = cosmo.luminosity_distance(z_Arr).to(u.cm).value
    L = np.log10(F_line * 4*np.pi * dL ** 2)

    F_line_NV = np.array(Lya_fts['NVF']) * 1e-17
    F_line_NV_err = np.array(Lya_fts['NVF_err']) * 1e-17
    EW0_NV = np.array(Lya_fts['NVEW']) / (1 + z_Arr)

    L_NV = np.ones_like(F_line_NV) * -99.
    mask_positive_NV = (F_line_NV > 0)
    L_NV[mask_positive_NV] = np.log10(F_line_NV[mask_positive_NV]
                    * 4*np.pi * dL[mask_positive_NV] ** 2)

    model = pd.read_csv('/home/alberto/cosmos/LAEs/MyMocks/csv/PD2016-QSO_LF.csv')
    counts_model_2D = model.to_numpy()[:-1, 1:-1].astype(float) * 1e-4 * area_obs
    r_yy = np.arange(15.75, 24.25, 0.5)
    z_xx = np.arange(0.5, 6, 1)
    # Interpolate 2D the model
    f_counts = RectBivariateSpline(z_xx, r_yy, counts_model_2D.T,
                                   kx=1, ky=1)

    N_src = int(dblquad(f_counts, r_min, r_max, z_min, z_max)[0] * 2)

    # Re-bin distribution
    z_xx_new = np.linspace(z_min, z_max, 10000)
    r_yy_new = np.linspace(r_min, r_max, 10001)
    model_2D_interpolated = f_counts(z_xx_new, r_yy_new).T.flatten()
    model_2D_interpolated /= np.sum(model_2D_interpolated) # Normalize
    idx_sample = np.random.choice(np.arange(len(model_2D_interpolated)), N_src,
                                    p=model_2D_interpolated)
    idx_sample = np.unravel_index(idx_sample, (len(r_yy_new), len(z_xx_new)))
    out_z_Arr = z_xx_new[idx_sample[1]]
    out_r_Arr = r_yy_new[idx_sample[0]]

    r_flx_Arr = pm_SEDs_DR16[:, -4]
    out_r_flx_Arr = mag_to_flux(out_r_Arr, w_central[-4])

    # Look for the closest source of SDSS in redshift
    out_sdss_idx_list = np.zeros(out_z_Arr.shape).astype(int)
    print('Looking for the sources in SDSS catalog')
    if contaminants:
        general_mask = np.ones_like(z_Arr).astype(bool)
    else:
        general_mask = (L > 40) & (EW0 > 0)

    # Add extra constraint to general mask if the mock is going to be used for a ML
    # train/test set
    # Let's use 80% train, 20% test
    if len(train_or_test) > 0:
        N_sources_SDSS = len(general_mask)
        np.random.seed(685757192)
        rand_ids = np.random.choice(np.arange(N_sources_SDSS),
                                    size=int(N_sources_SDSS * 0.8),
                                    replace=False)

        tt_mask = np.ones_like(general_mask).astype(bool)
        tt_mask[rand_ids] = False

        if train_or_test == 'train':
            pass
        elif train_or_test == 'test':
            tt_mask = ~tt_mask
        else:
            raise Exception('What?')

        general_mask = general_mask & tt_mask

    for src in range(N_src):
        if src % 500 == 0:
            print(f'{src} / {N_src}')
        # Select sources with a redshift closer than 0.06
        closest_z_Arr = np.where((np.abs(z_Arr - out_z_Arr[src]) < 0.1)
                                 & general_mask)[0]
        # If less than 10 objects found with that z_diff, then select the 10 closer
        if len(closest_z_Arr) < 10:
            closest_z_Arr = np.abs(z_Arr - out_z_Arr[src]).argsort()[:10]

        # Select one random source from those
        out_sdss_idx_list[src] = np.random.choice(closest_z_Arr, 1)
        # Modify z_out_Arr to match the new values
        out_z_Arr[src] = z_Arr[out_sdss_idx_list[src]]

    # Correction factor to match rSDSS
    r_corr_factor = out_r_flx_Arr / r_flx_Arr[out_sdss_idx_list]

    # Output PM array
    pm_flx_0 = pm_SEDs_DR16[out_sdss_idx_list] * r_corr_factor.reshape(-1, 1)
    out_EW = EW0[out_sdss_idx_list] * (1 + z_Arr[out_sdss_idx_list]) / (1 + out_z_Arr)
    out_L = L[out_sdss_idx_list] + np.log10(r_corr_factor)
    out_Flambda = F_line[out_sdss_idx_list] * r_corr_factor
    out_Flambda_err = F_line_err[out_sdss_idx_list] * r_corr_factor

    out_EW_NV = EW0_NV[out_sdss_idx_list] * (1 + z_Arr[out_sdss_idx_list]) / (1 + out_z_Arr)
    out_L_NV = L_NV[out_sdss_idx_list] + np.log10(r_corr_factor)
    out_Flambda_NV = F_line_NV[out_sdss_idx_list] * r_corr_factor
    out_Flambda_NV_err = F_line_NV_err[out_sdss_idx_list] * r_corr_factor

    out_mjd = mjd[out_sdss_idx_list]
    out_fiber = fiber[out_sdss_idx_list]
    out_plate = plate[out_sdss_idx_list]

    # Make the pandas df
    print('Saving files')
    cat_name = f'QSO_{surname}'
    dirname = f'/home/alberto/almacen/Source_cats/{cat_name}'
    os.makedirs(dirname, exist_ok=True)

    # Withour errors
    subpart = str(int(z_min * 100)) + str(int(z_max * 100))
    print(f'Subpart = {subpart}')
    filename = f'{dirname}/data{subpart}.csv'
    hdr = (
        tcurves['tag']
        + [s + '_e' for s in tcurves['tag']]
        + ['zspec', 'EW0', 'L_lya', 'F_line', 'F_line_err']
        + ['EW0_NV', 'L_NV', 'F_line_NV', 'F_line_NV_err']
        + ['mjd', 'fiber', 'plate']
    )

    # Mask according to L lims
    if L_min > 0 and L_max > 0:
        L_mask = (out_L >= L_min) & (out_L < L_max)
    else:
        L_mask = np.ones_like(out_L).astype(bool)
    print(f'Final N_sources = {sum(L_mask)}')

    df = pd.DataFrame(
        data=np.hstack(
            (
                pm_flx_0[L_mask], pm_flx_0[L_mask] * 0,
                out_z_Arr[L_mask].reshape(-1, 1),
                out_EW[L_mask].reshape(-1, 1),
                out_L[L_mask].reshape(-1, 1),
                out_Flambda[L_mask].reshape(-1, 1),
                out_Flambda_err[L_mask].reshape(-1, 1),
                out_EW_NV[L_mask].reshape(-1, 1),
                out_L_NV[L_mask].reshape(-1, 1),
                out_Flambda_NV[L_mask].reshape(-1, 1),
                out_Flambda_NV_err[L_mask].reshape(-1, 1),
                out_mjd[L_mask].reshape(-1, 1),
                out_fiber[L_mask].reshape(-1, 1),
                out_plate[L_mask].reshape(-1, 1),
            )
        )
    )
    df.to_csv(filename, header=hdr)

if __name__ == '__main__':
    zs_list = [[0., 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.], [1., 1.25],
               [1.25, 1.5], [1.5, 1.75], [1.75, 2.], [2., 2.25], [2.25, 2.50],
               [2.50, 2.75]]
    # for z_min, z_max in zs_list:
    #     r_min = 16
    #     r_max = 24
    #     L_min = 0
    #     L_max = 0
    #     area_obs = 200
    #     surname = 'PAUS_contaminants_2'
    #     main(z_min, z_max, r_min, r_max, L_min, L_max, area_obs,
    #          surname=surname, contaminants=True)

    # zs_list = [[2.25, 2.5], [2.5, 2.75], [2.75, 3],
    #            [3, 3.25], [3.25, 3.5], [3.5, 3.75], [3.75, 4], [4, 4.5]]
    # for z_min, z_max in zs_list:
    #     r_min = 16
    #     r_max = 24
    #     L_min = 40
    #     L_max = 47
    #     area_obs = 400
    #     surname = 'PAUS_LAES_2'
    #     main(z_min, z_max, r_min, r_max, L_min, L_max, area_obs,
    #          surname=surname, contaminants=False)

    #     r_min = 16
    #     r_max = 24
    #     L_min = 44
    #     L_max = 47
    #     area_obs = 4000
    #     surname = 'PAUS_LAES_hiL_2'
    #     main(z_min, z_max, r_min, r_max, L_min, L_max, area_obs,
    #          surname=surname, contaminants=False)


    for t_or_t in ['train', 'test']:
        for z_min, z_max in zs_list:
            r_min = 16
            r_max = 24
            L_min = 0
            L_max = 0
            area_obs = 200
            surname = f'PAUS_contaminants_2_{t_or_t}'
            main(z_min, z_max, r_min, r_max, L_min, L_max, area_obs,
                surname=surname, contaminants=True, train_or_test=t_or_t)

        zs_list = [[2.25, 2.5], [2.5, 2.75], [2.75, 3],
                [3, 3.25], [3.25, 3.5], [3.5, 3.75], [3.75, 4], [4, 4.5]]
        for z_min, z_max in zs_list:
            r_min = 16
            r_max = 24
            L_min = 40
            L_max = 47
            area_obs = 400
            surname = f'PAUS_LAES_2_{t_or_t}'
            main(z_min, z_max, r_min, r_max, L_min, L_max, area_obs,
                surname=surname, contaminants=False, train_or_test=t_or_t)

            r_min = 16
            r_max = 24
            L_min = 44
            L_max = 47
            area_obs = 4000
            surname = f'PAUS_LAES_hiL_2_{t_or_t}'
            main(z_min, z_max, r_min, r_max, L_min, L_max, area_obs,
                surname=surname, contaminants=False, train_or_test=t_or_t)