from astropy.table import Table
from paus_utils import w_central
import numpy as np


c = 29979245800
flx_u_constant = 1.445439770746259e-22 * c

def paus_flux_units(paus_flx, w):
    '''
    Units such: m_AB = 26 - 2.5 * log10(flux)
    '''
    return flx_u_constant * paus_flx * w ** -2


def load_paus_cat(path_to_cat):
    tab = Table.read(path_to_cat).to_pandas()

    cat = {} # Initialize catlalog dict

    cat['ref_id'] = np.array(tab['ref_id'])
    cat['photoz'] = np.array(tab['zb'])
    cat['photoz_odds'] = np.array(tab['odds'])
    cat['ra'] = np.array(tab['ra'])
    cat['dec'] = np.array(tab['dec'])
    cat['type'] = np.array(tab['type'])
    cat['zspec'] = np.array(tab['zspec'])
    
    # Flux units have to be converted to erg s^-1 A^-1
    cat['flx'] = paus_flux_units(tab.to_numpy()[:, 7 : 7 + 46],
                                 w_central)
    cat['err'] = paus_flux_units(tab.to_numpy()[:, 7 + 46 : 7 + 46 + 46],
                                 w_central)

    return cat