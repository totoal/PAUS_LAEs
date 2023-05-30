import sys
sys.path.insert(0, '..')

from paus_utils import w_central

from jpasLAEs.utils import bin_centers

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 16})

import pickle

import numpy as np

color_code = {
    'SFG': 'C1',
    'QSO': 'C0',
    'GAL': 'C2'
}

line_dict = {
    r'Ly$\alpha$': 1215.67,
    'CIV': 1549.48,
    'CIII': 1908.73,
    'CII': 2326.00,
    'MgII': 2799.12,
    'OII': (3727.09 + 3729.88) * 0.5,
    r'H$\beta$': 4862.68,
    'OIII': (4960.30 + 5008.24) * 0.5,
}

gal_line_w = [2799, 4861, 3727, 5008]
gal_line_name = ['MgII', r'H$\beta$', 'OII', 'OIII']
qso_line_w = [1549.48, 1908.73, 2326.00, 1215.67]
qso_line_name = ['CIV', 'CIII', 'CII',
                    r'Ly$\alpha$']
line_w = gal_line_w + qso_line_w
line_name = gal_line_name + qso_line_name


def main(field_name, nb1, nb2):
    # First load the mock dict
    path_to_mock_dict = '/home/alberto/almacen/PAUS_data/LF_corrections/'
    with open(f'{path_to_mock_dict}/mock_dict_{field_name}_nb{nb1}-{nb2}.pkl', 'rb') as f:
        mock_dict = pickle.load(f)

    w_obs_bins = np.linspace(1000, 7000, 100)
    w_obs_bins_c = bin_centers(w_obs_bins)

    h_dict = {
        'QSO': np.zeros(len(w_obs_bins) - 1)
    } # initialize histograms

    for mock_name, mock in mock_dict.items():
        lya_lines = mock['lya_NB']

        to_hist = (w_central[lya_lines] / (1 + mock['zspec']))[lya_lines >= 0]
        h = np.histogram(to_hist, w_obs_bins)[0]

        if mock_name[:3] == 'QSO':
            print('qso')
            h_dict['QSO'] += h / mock['area']
        else:
            h_dict[mock_name] = h / mock['area']


    fig, ax = plt.subplots(figsize=(6, 3))

    for name, hist in h_dict.items():
        ax.plot(w_obs_bins_c, hist, drawstyle='steps-mid',
                c=color_code[name], lw=2, label=name)

    h_max = np.max(h_dict['QSO'])
    for line_name, line_w0 in line_dict.items():
        if line_name == r'H$\beta$':
            x_text = line_w0 - 200
        elif line_name == 'OIII':
            x_text = line_w0 - 50
        elif line_name == r'Ly$\alpha$':
            x_text = line_w0 - 150
        else:
            x_text = line_w0 - 120
        y_text = h_max * 1.1

        ax.axvline(line_w0, ls=':', c='dimgray')
        ax.text(x_text, y_text, line_name, va='bottom',
                fontsize=10)

    ax.legend(fontsize=12)

    ax.set(xlim=(1000, 7000),
           ylim=(0, h_max * 1.1),
           xlabel=r'$\lambda_\mathrm{obs}$ (\AA)',
           ylabel=r'deg$^{-2}$')

    fig.savefig('../figures/contaminants_hist.pdf', bbox_inches='tight',
                pad_inches=0.1, facecolor='w')




if __name__ == '__main__':
    field_name = 'foo' # Test data
    nb1, nb2 = 0, 2
    main(field_name, nb1, nb2)