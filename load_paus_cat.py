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


def load_paus_cat(path_to_cat):
    tab = pd.read_csv(path_to_cat)

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



    # Convert fluxes from PAUS units to erg/s/cm/A
    flx_mat = paus_flux_units(flx_mat, w_central.reshape(-1, 1))
    flx_err_mat = paus_flux_units(flx_err_mat, w_central.reshape(-1, 1))


    # Define the catalog dictionary
    cat = {} # Initialize catlalog dict

    cat['flx'] = flx_mat
    cat['err'] = flx_err_mat
    cat['ref_id'] = np.array(tab['ref_id'])
    # TODO: add morphology index

    return cat


if __name__ == '__main__':
    path = '/home/alberto/almacen/PAUS_data/catalogs/PAUS_W3.csv'
    load_paus_cat(path)