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
from paus_utils import NB_z, z_NB

def plot_puricomp1d(field_name, nb_min, nb_max, r_min, r_max, ax=None):
    corr_dir = '/home/alberto/almacen/PAUS_data/LF_corrections'
    with open(f'{corr_dir}/mock_dict_{field_name}_nb{nb_min}-{nb_max}.pkl', 'rb') as f:
        mocks_dict = pickle.load(f)

    L_bins = np.linspace(42, 46, 20)
    L_bins_c = bin_centers(L_bins)

    h_sel = np.zeros_like(L_bins_c)
    h_nice = np.zeros_like(L_bins_c)
    h_parent = np.zeros_like(L_bins_c)

    for _, mock in mocks_dict.items():
        # Define nice_z
        nice_z = np.abs(mock['zspec'] - z_NB(mock['lya_NB'])) < 0.6

        mask_r = (mock['r_mag'] >= r_min) & (mock['r_mag'] <= r_max)
        mask_sel = (mock['nice_lya']
                    & (mock['lya_NB'] >= nb_min)
                    & (mock['lya_NB'] <= nb_max)
                    & mask_r)
        mask_parent = ((mock['EW0_lya_spec'] > 30)
                       & (NB_z(mock['zspec']) >= nb_min)
                       & (NB_z(mock['zspec']) <= nb_max)
                       & mask_r)
        mask_nice = (mask_parent
                     & mock['nice_lya'] & nice_z)

        area_obs = mock['area']

        h_sel += np.histogram(mock['L_lya_spec'][mask_sel],
                              L_bins)[0] / area_obs
        h_nice += np.histogram(mock['L_lya_spec'][mask_nice],
                               L_bins)[0] / area_obs
        h_parent += np.histogram(mock['L_lya_spec'][mask_parent],
                                  L_bins)[0] / area_obs

    puri1d = np.zeros_like(h_sel)
    comp1d = np.zeros_like(h_sel)

    mask_nonzero_sel = (h_sel > 0)
    mask_nonzero_parent = (h_parent > 0)
    puri1d[mask_nonzero_sel] =\
        h_nice[mask_nonzero_sel] / h_sel[mask_nonzero_sel]
    comp1d[mask_nonzero_parent] =\
        h_nice[mask_nonzero_parent] / h_parent[mask_nonzero_parent]

    # Plot the puri/comp 1D
    if ax is None:
        ax = plt.gca()

    ax.plot(L_bins_c, puri1d, ls='-', label='Purity')
    ax.plot(L_bins_c, comp1d, ls='-', label='Completeness')

    ax.set(xlabel=r'$L_{\mathrm{Ly}\alpha}$',
           ylim=(0, 1))

    return ax, puri1d, comp1d, L_bins_c


main = plot_puricomp1d

if __name__ == '__main__':
    main('foo', 0, 10, 17, 24)