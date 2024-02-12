import pandas as pd
import numpy as np

import glob

from paus_utils import fwhm_Arr, z_NB
from lightcone_mock.read_lightcone import load_lightcone_mock

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

'''
Functions to load the mocks.
The mocks must be dictionaries with this structure:
    'flx_0': (46, N_sources) matrix with the fluxes
    'zspec': (N_sources,) Array with the true redshifts of the sources
    'L_lya': (N_sources,) Array with the intrinsic Lya Luminosities
    'EW_lya': (N_sources,) Array with the intrinsic Lya rest-frame EWs
'''

def load_qso_mock(*path_list):
    '''
    Load QSO mock data from multiple files and return a dictionary containing relevant information.

    Args:
        *path_list: Variable-length argument list of paths to the mock data files.

    Returns:
        dict: A dictionary containing the following keys and corresponding values:
            - 'flx_0': A 2D NumPy array representing the flux data from the mock files. 
                       The array shape is (46, N), where N is the total number of objects.
                       Infinite values are replaced with 99.
            - 'zspec': A Pandas Series representing the spectroscopic redshifts of the objects.
            - 'L_lya_spec': A Pandas Series representing the Lyman-alpha luminosities of the objects.
            - 'EW0_lya_spec': A Pandas Series representing the Lyman-alpha equivalent widths of the objects.
    '''
    fi = []

    for path_to_mock in path_list:
        files = glob.glob(f'{path_to_mock}/data*')
        files.sort()

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
    '''
    Load star-forming galaxy (SFG) mock data from files and return a dictionary containing relevant information.

    Args:
        path_to_mock (str): The path to the directory containing the mock data files.

    Returns:
        dict: A dictionary containing the following keys and corresponding values:
            - 'flx_0': A 2D NumPy array representing the flux data from the mock files.
                       The array shape is (46, N), where N is the total number of objects.
            - 'zspec': A Pandas Series representing the spectroscopic redshifts of the objects.
            - 'L_lya_spec': A Pandas Series representing the Lyman-alpha luminosities of the objects.
            - 'EW0_lya_spec': A Pandas Series representing the Lyman-alpha equivalent widths of the objects.
    '''
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


def load_gal_mock(directory, suffix):
    '''
    Previous version substituted with this one.
    '''
    return load_lightcone_mock(directory, suffix)

def add_artifacts(mock, min_fake_L_lya=43.5, max_fake_L_lya=46,
                  nb_min=0, nb_max=18):
    '''
    Adds random fluxes to random NBs in a mock catalog to simulate
    cosmic rays and artifacts.
    '''
    N_sources = mock['flx_0'].shape[1]

    random_fake_L_Lya = np.random.uniform(min_fake_L_lya, max_fake_L_lya,
                                          size=N_sources)
    random_NB_idx = np.random.randint(nb_min, nb_max + 1, size=N_sources)

    fake_NB_redshift = z_NB(random_NB_idx)
    fake_lumdist = cosmo.luminosity_distance(fake_NB_redshift).to(u.cm).value

    artifact_Flux_Arr = 10**random_fake_L_Lya / (4 * np.pi * fake_lumdist**2)

    for src in range(N_sources):
        this_NB = random_NB_idx[src]
        this_F = artifact_Flux_Arr[src]

        mock['flx_0'][this_NB, src] += this_F / fwhm_Arr[this_NB]

    # Add two columns with introduced artifact info.
    mock['fake_L_lya'] = random_fake_L_Lya
    mock['artifact_NB'] = random_NB_idx

    return mock



def load_mock_dict(mock_SFG_path, mock_QSO_cont_path, mock_QSO_LAEs_loL_path,
                    mock_QSO_LAEs_hiL_path, mock_GAL_dir, mock_GAL_suff,
                    gal_fraction=1.):
    '''
    Loads all the mocks needed to compute the Lya LF corrections, and returns
    a dictionary of mocks.
    '''
    mock_path_list = [mock_SFG_path,
                      mock_QSO_cont_path,
                      mock_QSO_LAEs_loL_path,
                      mock_QSO_LAEs_hiL_path,
                      (mock_GAL_dir, mock_GAL_suff)]
    mock_name_list = ['SFG', 'QSO_cont', 'QSO_LAEs_loL', 'QSO_LAEs_hiL',
                   'GAL']

    mock_dict = {}
    for i, (mock_path, mock_name) in enumerate(zip(mock_path_list, mock_name_list)):
        if i == 0:
            mock_dict[mock_name] = load_sfg_mock(mock_path)
        elif i > 0 and i < 4:
            mock_dict[mock_name] = load_qso_mock(mock_path)
        elif i == 4:
            mock_dict[mock_name] = load_gal_mock(*mock_path)

    return mock_dict


################################################
def expfit(x, a, b, c):
    return a * np.exp(b * x + c)

def add_errors(flx_0, field_name, add_errors=False):
    '''
    Returns the matrix of perturbed fluxes and errors: flx, err
    '''
    # Load the fit parameters
    path_to_fit = '/home/alberto/almacen/PAUS_data/catalogs/error_distribution'
    fit_params = np.load(f'{path_to_fit}/fit_params_{field_name}.npy')

    mask_bad_flx = (flx_0 > 0)

    log_flx_0 = np.ones_like(flx_0) * -99.
    flx_err_mat = np.empty_like(flx_0)

    log_flx_0[mask_bad_flx] = np.log10(flx_0[mask_bad_flx])

    for filter_i in range(46):
        params_i = fit_params[filter_i]

        # First compute the 5 sigma level (relerr=0.2)
        log_flx_x = np.linspace(-21, -16, 10000)
        relerr_y = expfit(log_flx_x, *params_i)
        sigma5_flux = 10 ** log_flx_x[np.argmin(np.abs(relerr_y - 0.2))]
        
        mask_5sigma = (flx_0[filter_i] > sigma5_flux)

        this_flx_err = np.empty(flx_0.shape[1])
        this_flx_err[mask_5sigma] =\
            expfit(log_flx_0[filter_i, mask_5sigma], *params_i)\
                * flx_0[filter_i, mask_5sigma]
        this_flx_err[~mask_5sigma] = sigma5_flux * 0.2

        flx_err_mat[filter_i] = this_flx_err

        if add_errors:
            flx_with_err = flx_0 + np.random.normal(size=flx_0.shape) * flx_err_mat
        else:
            flx_with_err = flx_0

    return flx_with_err, flx_err_mat


if __name__ == '__main__':
    ## Debugging
    mock_GAL_path = '/home/alberto/almacen/PAUS_data/catalogs/LightCone_mock.fits'
    load_gal_mock(mock_GAL_path)