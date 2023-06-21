import pandas as pd
import numpy as np

from paus_utils import w_central, data_tab

from jpasLAEs.utils import mag_to_flux


c = 29979245800
flx_u_constant = 1.445439770746259e-22 * c

def paus_flux_units(paus_flx, w):
    '''
    Units such: m_AB = 26 - 2.5 * log10(flux)
    '''
    return flx_u_constant * paus_flx * w ** -2


def load_paus_cat(cat_paths_list):
    tab = pd.concat([pd.read_csv(path) for path in cat_paths_list])

    # Stack the NBs and BBs
    flx_mat = np.array([]).reshape(0, len(tab))
    flx_err_mat = np.array([]).reshape(0, len(tab))
    for jj, filter_name in enumerate(data_tab['name']):
        if filter_name[:2] == 'NB':
            flx_mat = np.vstack([flx_mat,
                                 np.array(tab[filter_name])])
            flx_err_mat = np.vstack([flx_err_mat,
                                     np.array(tab[f'{filter_name}_error'])])
        elif len(filter_name) == 1:
            mag_cat_name = f'mag_{filter_name.lower()}'
            if mag_cat_name in tab.keys():
                this_bb_mag = np.array(tab[mag_cat_name])
                this_bb_mag_err = np.array(tab[f'magerr_{filter_name.lower()}'])
                this_bb_flx = mag_to_flux(this_bb_mag, data_tab['w_eff'][jj])
                this_bb_flx_err = this_bb_flx * this_bb_mag_err # magerr is approx. flx_relerr
            else:
                this_bb_flx = np.ones(len(tab)).astype(float) * 1e-99
                this_bb_flx_err = np.ones(len(tab)).astype(float) * 1e-99

            flx_mat = np.vstack([flx_mat, this_bb_flx])
            flx_err_mat = np.vstack([flx_err_mat, this_bb_flx_err])


    # Convert fluxes from PAUS units to erg/s/cm/A (only NBs)
    flx_mat[:40] = paus_flux_units(flx_mat[:40],
                                   w_central[:40].reshape(-1, 1))
    flx_err_mat[:40] = paus_flux_units(flx_err_mat[:40],
                                       w_central[:40].reshape(-1, 1))


    # Define the catalog dictionary
    cat = {} # Initialize catlalog dict

    # Mask where the flux is measured
    measured_mask = np.isfinite(flx_mat)
    # Number of available NBs
    measured_NBs = np.sum(measured_mask[:40], axis=0)

    cat['flx'] = flx_mat
    cat['err'] = flx_err_mat
    cat['ref_id'] = np.array(tab['ref_id'])
    cat['r_mag'] = np.array(tab['mag_r'])
    cat['NB_mask'] = measured_mask
    cat['NB_number'] = measured_NBs
    cat['RA'] = np.array(tab['alpha_j2000'])
    cat['DEC'] = np.array(tab['delta_j2000'])
    cat['bulge_fraction'] = np.array(tab['bulge-fraction'])
    cat['flattening'] = 1 - np.array(tab['b_world']) / np.array(tab['a_world'])
    # TODO: areas!
    # cat['area'] = 16.1 # This is provisionally the area of W3

    return cat


if __name__ == '__main__':
    path = '/home/alberto/almacen/PAUS_data/catalogs/PAUS_W1.csv'
    load_paus_cat(path)