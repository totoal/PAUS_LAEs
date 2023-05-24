import pandas as pd
import glob
from paus_utils import w_central
from jpasLAEs.utils import flux_to_mag, mag_to_flux
import numpy as np
from astropy.table import Table

'''
Functions to load the mocks.
The mocks must be dictionaries with this structure:
    'flx_0': (46, N_sources) matrix with the fluxes
    'zspec': (N_sources,) Array with the true redshifts of the sources
    'L_lya': (N_sources,) Array with the intrinsic Lya Luminosities
    'EW_lya': (N_sources,) Array with the intrinsic Lya rest-frame EWs
'''

def load_qso_mock(path_to_mock):
    files = glob.glob(f'{path_to_mock}/data*')
    files.sort()
    fi = []

    for name in files:
        fi.append(pd.read_csv(name))

    qso_data = pd.concat(fi, axis=0, ignore_index=True)

    cat = {}

    cat['flx_0'] = qso_data.to_numpy()[:, 1 : 1 + 46].T
    # inf values set to 99.
    cat['flx_0'][~np.isfinite(cat['flx_0'])] = 99.

    cat['zspec'] = qso_data['zspec']
    cat['L_lya_spec'] = qso_data['L_lya']
    cat['EW0_lya_spec'] = qso_data['EW0']

    return cat

def load_sfg_mock(path_to_mock):
    files = glob.glob(f'{path_to_mock}/data*')
    files.sort()
    fi = []

    for name in files:
        fi.append(pd.read_csv(name))

    sfg_data = pd.concat(fi, axis=0, ignore_index=True)

    cat = {}


    cat['flx_0'] = sfg_data.to_numpy()[:, 1 : 1 + 46].T

    cat['zspec'] = sfg_data['zspec']
    cat['L_lya_spec'] = sfg_data['L_lya']
    cat['EW0_lya_spec'] = sfg_data['EW0']

    return cat

def load_gal_mock(path_to_mock, cat_fraction):
    tab = Table.read(path_to_mock).to_pandas().to_numpy()
    mock_size = len(tab)

    np.random.seed(1312)
    sel = np.random.randint(0, mock_size, size=int(mock_size * cat_fraction))
    
    cat = {}

    cat['flx_0'] = mag_to_flux(tab[sel, 11 : 11 + 40],
                                    w_central[:-6]).T

    # Add BBs
    cat['flx_0'] = np.vstack([cat['flx_0'],
                             mag_to_flux(tab[sel, -5:].T,
                                         w_central[-6:-1].reshape(-1,1)),
                             np.zeros(len(sel))])

    # Precompute r and mask by r. Otherwise the mock is too large
    r_mag = flux_to_mag(cat['flx_0'][-4], w_central[-4])
    i_mag = flux_to_mag(cat['flx_0'][-3], w_central[-3])
    mag_mask = np.array(r_mag < 24.3) & np.array(i_mag < 23.3)
    cat['flx_0'] = cat['flx_0'][:, mag_mask]

    cat['zspec'] = np.array(tab[:, 4])[sel][mag_mask]
    
    # L_lya and EW0_lya are zero for all these objects.
    cat['L_lya_spec'] = np.zeros_like(cat['zspec'])
    cat['EW0_lya_spec'] = np.zeros_like(cat['zspec'])

    return cat


def load_mocks_dict(mock_SFG_path, mock_QSO_cont_path, mock_QSO_LAEs_loL_path,
                    mock_QSO_LAEs_hiL_path, mock_GAL_path, gal_fraction=0.1):
    '''
    Loads all the mocks needed to compute the Lya LF corrections, and returns
    a dictionary of mocks.
    '''
    mock_path_list = [mock_SFG_path,
                      mock_QSO_cont_path,
                      mock_QSO_LAEs_loL_path,
                      mock_QSO_LAEs_hiL_path,
                      mock_GAL_path]
    mock_name_list = ['SFG', 'QSO_cont', 'QSO_LAEs_loL', 'QSO_LAEs_hiL',
                   'GAL']

    mocks_dict = {}
    for i, (mock_path, mock_name) in enumerate(zip(mock_path_list, mock_name_list)):
        if i == 0:
            mocks_dict[mock_name] = load_sfg_mock(mock_path)
        elif i > 0 and i < 4:
            mocks_dict[mock_name] = load_qso_mock(mock_path)
        elif i == 4:
            mocks_dict[mock_name] = load_gal_mock(mock_path, gal_fraction)

    print(f'Mock len: {len(mocks_dict["SFG"]["zspec"])}')
    
    return mocks_dict


################################################

def add_errors(flx_0, field_name):
    '''
    Returns the matrix of perturbed fluxes and errors: flx, err
    '''
    return
