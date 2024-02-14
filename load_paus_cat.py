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
    extension = cat_paths_list[0].split('.')[-1]
    if extension == 'pq':
        pd_read_func = pd.read_parquet
    elif extension == 'csv':
        pd_read_func = pd.read_csv
    else:
        raise Exception('Catalog format not known.')

    tab = pd.concat([pd_read_func(path) for path in cat_paths_list])

    # Read class star from a different file
    class_star = pd.DataFrame()
    for path in cat_paths_list:
        class_star_filename = f'{path.split('/')[-1][:15]}_class_star.csv'
        class_star_path = '/'.join(path.split('/')[:-1]) + f'/{class_star_filename}'
        class_star = pd.concat([class_star,
                                pd.read_csv(class_star_path, index_col=0)])
    tab = tab.merge(class_star, on='ref_id', how='left')

    # Check if there's a "mask" column
    if 'mask' in tab.keys():
        tab = tab[(tab['mask'].values & 16412) == 0]
    print(f'Parent catalog length: {len(tab)}')

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
            magname = filter_name.lower()
            if not f'mag_{magname}' in tab.keys():
                magname = f'gaap_{filter_name.lower()}'
                gaap = '_gaap'
            else:
                gaap = ''
            mag_cat_name = f'mag_{magname}'
            if mag_cat_name in tab.keys():
                this_bb_mag = np.array(tab[mag_cat_name])
                this_bb_mag_err = np.array(tab[f'magerr_{magname}'])
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

    # If available, use ref_id column.
    # If not, make a fake one
    if 'ref_id' in tab.keys():
        ref_id_Arr = np.array(tab['ref_id'])
    else:
        try:
            ref_id_Arr = np.array(tab.index)
        except:
            print('Warning: \'ref_id\' not found in catalog. Making up fake IDs.')
            ref_id_Arr = (np.array(tab['alpha_j2000'] * 10000).astype(int)
                        + np.array(tab['delta_j2000'] * 10000).astype(int) * 10000000)

    cat['flx'] = flx_mat
    cat['err'] = flx_err_mat
    cat['ref_id'] = ref_id_Arr
    cat['r_mag'] = np.array(tab[f'mag{gaap}_r'])
    cat['NB_mask'] = measured_mask
    cat['NB_number'] = measured_NBs
    cat['RA'] = np.array(tab['alpha_j2000'])
    cat['DEC'] = np.array(tab['delta_j2000'])
    cat['class_star'] = np.array(tab['class_star'])
    
    if 'sg_flag' in tab.keys():
        star_flag_name = 'sg_flag'
        cat['sg_flag'] = np.abs(np.array(tab[star_flag_name]) - 1)
    elif 'star_flag' in tab.keys():
        star_flag_name = 'star_flag'
        cat['sg_flag'] = np.array(tab[star_flag_name])
    else:
        raise ValueError('No star flag.')

    return cat


if __name__ == '__main__':
    path = ['/home/alberto/almacen/PAUS_data/catalogs/PAUS_3arcsec_W3_extinction_corrected.pq']
    cat = load_paus_cat(path)
    print(cat['ref_id'])