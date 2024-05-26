import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo


nb_list = [[0, 3], [2, 5], [4, 7], [6, 9], [8, 11], [10, 13], [12, 15], [14, 18]]
where_save = '/home/alberto/almacen/PAUS_data/K_corrections'


def UV_magnitude_K(r_Arr_in, z_NB_Arr, nb_min, nb_max, nice_mask=None):
    r_mag = np.load(f'{where_save}/r_synth_nb{nb_min}-{nb_max}.npy')
    M_UV = np.load(f'{where_save}/M_UV_spec_nb{nb_min}-{nb_max}.npy')
    zspec = np.load(f'{where_save}/zspec_nb{nb_min}-{nb_max}.npy')

    mask = (M_UV < 0) & np.isfinite(r_mag) & (r_mag > 16) & (r_mag < 25)
    this_K = np.nanmean((r_mag - cosmo.distmod(zspec).value - M_UV)[mask])
    this_K_std = np.nanstd((r_mag - cosmo.distmod(zspec).value - M_UV)[mask])
    
    MUV_out = r_Arr_in - cosmo.distmod(z_NB_Arr).value - this_K
    MUV_err_out = np.ones_like(MUV_out).astype(float) * this_K_std

    if nice_mask is not None:
        MUV_out[~nice_mask] = 99.
        MUV_err_out[~nice_mask] = 99.

    return MUV_out, MUV_err_out

if __name__ == '__main__':
    #### For debugging
    nb_min, nb_max = 4, 7
    r_mag = np.load(f'{where_save}/r_synth_nb{nb_min}-{nb_max}.npy')
    M_UV = np.load(f'{where_save}/M_UV_spec_nb{nb_min}-{nb_max}.npy')
    zspec = np.load(f'{where_save}/zspec_nb{nb_min}-{nb_max}.npy')

    mask = (M_UV < 0) & np.isfinite(r_mag) & (r_mag > 16) & (r_mag < 25)
    this_K = np.nanmedian((r_mag - cosmo.distmod(zspec).value - M_UV)[mask])
    this_K_std = np.nanstd((r_mag - cosmo.distmod(zspec).value - M_UV)[mask])
    
    print(this_K)

    plt.hist((r_mag - cosmo.distmod(zspec).value - M_UV)[mask])
    plt.show()