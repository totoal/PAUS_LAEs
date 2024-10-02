import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})

import sys
sys.path.insert(0, '..')

from jpasLAEs.utils import bin_centers
from paus_utils import NB_z, z_NB, PAUS_monochromatic_Mag

area_dict = {
    'SFG': 400,
    'QSO_cont': 1000,
    'QSO_LAEs_loL': 1000,
    'QSO_LAEs_hiL': 5000,
    'GAL': 59.97
}

def plot_puricomp1d(field_name, nb_min, nb_max, r_min, r_max,
                    L_bins=None, LF_kind='Lya'):
    corr_dir = '/home/alberto/almacen/PAUS_data/LF_corrections'
    with open(f'{corr_dir}/mock_dict_{field_name}_nb{nb_min}-{nb_max}.pkl', 'rb') as f:
        mocks_dict = pickle.load(f)


    if L_bins is None:
        L_bins = np.linspace(42, 46, 20)
    L_bins_c = bin_centers(L_bins)

    h_sel = np.zeros_like(L_bins_c)
    h_nice = np.zeros_like(L_bins_c)
    h_nice_spec = np.zeros_like(L_bins_c)
    h_parent = np.zeros_like(L_bins_c)

    for mock_name, mock in mocks_dict.items():
        # Define nice_z
        nice_z = np.abs(mock['zspec'] - z_NB(mock['lya_NB'])) < 0.12

        mask_r = (mock['r_mag'] >= r_min) & (mock['r_mag'] <= r_max)
        mask_sel = (mock['nice_lya']
                    & (mock['lya_NB'] >= nb_min)
                    & (mock['lya_NB'] <= nb_max)
                    & mask_r)
        mask_parent = ((mock['EW0_lya_spec'] > 20)
                       & (NB_z(mock['zspec']) >= nb_min)
                       & (NB_z(mock['zspec']) <= nb_max)
                       & mask_r)
        mask_nice = (mask_sel & nice_z)

        area_obs = area_dict[mock_name]

        if LF_kind == 'Lya':
            L_key = 'L_lya_corr'
            L_spec_key = 'L_lya_spec'
        elif LF_kind == 'UV':
            L_key = 'M_UV'
            L_spec_key = 'M_UV_spec'
        elif LF_kind == 'r':
            L_key = 'r_mag'
            L_spec_key = 'r_mag'
        else:
            raise ValueError('What?')

        h_sel += np.histogram(mock[L_key][mask_sel],
                              L_bins)[0] / area_obs
        h_nice += np.histogram(mock[L_key][mask_nice],
                                    L_bins)[0] / area_obs
        h_nice_spec += np.histogram(mock[L_spec_key][mask_nice & mask_parent],
                                    L_bins)[0] / area_obs
        h_parent += np.histogram(mock[L_spec_key][mask_parent],
                                  L_bins)[0] / area_obs

    puri1d = np.zeros_like(h_sel)
    comp1d = np.zeros_like(h_sel)

    mask_nonzero_sel = (h_sel > 0)
    mask_nonzero_parent = (h_parent > 0)
    puri1d[mask_nonzero_sel] =\
        h_nice[mask_nonzero_sel] / h_sel[mask_nonzero_sel]
    comp1d[mask_nonzero_parent] =\
        h_nice_spec[mask_nonzero_parent] / h_parent[mask_nonzero_parent]

    return puri1d, comp1d


def plot_puricomp2d(field_name, nb_min, nb_max, r_min, r_max,
                    L_bins=None, L_bins2=None, LF_kind='Lya'):
    corr_dir = '/home/alberto/almacen/PAUS_data/LF_corrections'
    with open(f'{corr_dir}/mock_dict_{field_name}_nb{nb_min}-{nb_max}.pkl', 'rb') as f:
        mocks_dict = pickle.load(f)

    if L_bins is None:
        L_bins = np.linspace(42, 46, 20)
    if L_bins2 is None:
        L_bins2 = np.linspace(0, 1, 20)  # Change this based on your second variable's range

    L_bins_c = bin_centers(L_bins)
    L_bins2_c = bin_centers(L_bins2)

    h_sel = np.zeros((len(L_bins_c), len(L_bins2_c)))
    h_nice = np.zeros((len(L_bins_c), len(L_bins2_c)))
    h_nice_spec = np.zeros((len(L_bins_c), len(L_bins2_c)))
    h_parent = np.zeros((len(L_bins_c), len(L_bins2_c)))

    for mock_name, mock in mocks_dict.items():
        # Define nice_z
        nice_z = np.abs(mock['zspec'] - z_NB(mock['lya_NB'])) < 0.12

        mask_r = (mock['r_mag'] >= r_min) & (mock['r_mag'] <= r_max)
        mask_sel = (mock['nice_lya']
                    & (mock['lya_NB'] >= nb_min)
                    & (mock['lya_NB'] <= nb_max)
                    & mask_r)
        mask_parent = ((mock['EW0_lya_spec'] > 20)
                       & (NB_z(mock['zspec']) >= nb_min)
                       & (NB_z(mock['zspec']) <= nb_max)
                       & mask_r)
        mask_nice = (mask_sel & nice_z)

        area_obs = area_dict[mock_name]

        if LF_kind == 'Lya':
            L_key = 'L_lya_corr'
            L_spec_key = 'L_lya_spec'
        elif LF_kind == 'UV':
            L_key = 'M_UV'
            L_spec_key = 'M_UV_spec'
        elif LF_kind == 'r':
            L_key = 'r_mag'
            L_spec_key = 'r_mag'
        else:
            raise ValueError('What?')

        # Assuming L_key2 and L_spec_key2 correspond to the second variable
        L_key2 = 'r_mag'
        L_spec_key2 = 'r_mag'

        h_sel += np.histogram2d(mock[L_key][mask_sel],
                                mock[L_key2][mask_sel],
                                bins=[L_bins, L_bins2])[0] / area_obs
        h_nice += np.histogram2d(mock[L_key][mask_nice],
                                 mock[L_key2][mask_nice],
                                 bins=[L_bins, L_bins2])[0] / area_obs
        h_nice_spec += np.histogram2d(mock[L_spec_key][mask_nice & mask_parent],
                                      mock[L_spec_key2][mask_nice & mask_parent],
                                      bins=[L_bins, L_bins2])[0] / area_obs
        h_parent += np.histogram2d(mock[L_spec_key][mask_parent],
                                   mock[L_spec_key2][mask_parent],
                                   bins=[L_bins, L_bins2])[0] / area_obs

    puri2d = np.zeros_like(h_sel)
    comp2d = np.zeros_like(h_sel)

    mask_nonzero_sel = (h_sel > 0)
    mask_nonzero_parent = (h_parent > 0)
    puri2d[mask_nonzero_sel] =\
        h_nice[mask_nonzero_sel] / h_sel[mask_nonzero_sel]
    comp2d[mask_nonzero_parent] =\
        h_nice_spec[mask_nonzero_parent] / h_parent[mask_nonzero_parent]

    return puri2d, comp2d, L_bins_c, L_bins2_c
