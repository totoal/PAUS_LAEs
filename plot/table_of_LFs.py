import os
import sys
sys.path.insert(0, '..')

import numpy as np

from load_lya_LF import load_combined_LF
from paus_utils import z_NB


survey_list = ['W3']

LF_table = None
LF_err_up_table = None
LF_err_down_table = None

for iii in range(17):
    LyaLF = load_combined_LF(survey_list, [[iii, iii]])

    LF_bins = LyaLF['LF_bins']
    LF_total = LyaLF['LF_boots'] # NOTE: Boots?

    if LF_table is None:
        LF_table = LF_total
        LF_err_up_table = LyaLF['LF_total_err'][0]
        LF_err_down_table = LyaLF['LF_total_err'][1]
    else:
        LF_table = np.vstack([LF_table, LF_total])
        LF_err_up_table = np.vstack([LF_err_up_table, LyaLF['LF_total_err'][0]])
        LF_err_down_table = np.vstack([LF_err_down_table, LyaLF['LF_total_err'][1]])

# Mask empty LFs
mask_empty = np.any(LF_table, axis=0)
LF_table = LF_table[:, mask_empty]
LF_bins = LF_bins[mask_empty]
LF_err_up_table = LF_err_up_table[:, mask_empty]
LF_err_down_table = LF_err_down_table[:, mask_empty]

filename = '../tables/single_NB_LFs.tab'
with open(filename, 'w') as file:
    row = LF_bins
    row_str = ' & '.join(f'{val:0.2f}' for val in row)
    file.write(r'\toprule' + '\n')
    file.write(r'$\log_{10}(L_{\mathrm{Ly}\alpha}/\mathrm{erg\,s}^{-1})$ & ' + f'{row_str}\\\\\n\n')
    file.write(r'\midrule' + '\n')

    for j, (row1, row2, row3) in enumerate(zip(LF_table, LF_err_up_table, LF_err_down_table)):
        this_z = z_NB(j)
        file.write(f'z = {this_z:0.2f}\\\\\n')
        for ri, row in enumerate([row1, row2, row3]):
            if ri == 0:
                leftcol_text = r'$\Phi$ ($10^6\,$Mpc$^{-3}$\,$\Delta\log_{10}L^{-1}$)'
            elif ri == 1:
                leftcol_text = 'err up'
            elif ri == 2:
                leftcol_text = 'err down'

            row_str = ' & '.join(f'{val * 1e6:0.3f}' if val > 0 else ' ' for val in row)
            file.write(f'{leftcol_text} & ' + row_str + '\\\\\n')
        file.write('\midrule\\\\\n')
