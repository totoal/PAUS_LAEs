from jpasLAEs import *

from load_paus_mocks import load_mocks_dict, add_errors
from paus_utils import *
from LAE_selection_method import *

import numpy as np
import pandas as pd


def compute_LF_corrections(mocks_dict, field_name, nb_min, nb_max):
    # Modify the mocks adding errors according to the corresponding field
    for mock_name, mock in mocks_dict.items():
        print(mock_name)
        mock['flx'], mock['err'] = add_errors(mock['flx_0'], field_name)

        ## Now we have the mock with the errors, do everything else for
        ## each mock

        ## First select LAEs
        mock = select_LAEs(mock, nb_min, nb_max,
                           ew0min_lya=30, ewmin_other=100)
        print(f'N nice_lya = {sum(mock["nice_lya"])}')
        


def main(nb_min, nb_max):
    # Load the mocks
    mock_SFG_path = ''
    mock_QSO_cont_path = ''
    mock_QSO_LAEs_loL_path = ''
    mock_QSO_LAEs_hiL_path = ''
    mock_GAL_path = ''
    mocks_dict = load_mocks_dict(mock_SFG_path, mock_QSO_cont_path,
                                 mock_QSO_LAEs_loL_path, mock_QSO_LAEs_hiL_path,
                                 mock_GAL_path)

    # List of PAUS fields
    field_list = ['foo', 'bar']
    for field_name in field_list:
        compute_LF_corrections(mocks_dict, field_name,
                               nb_min, nb_max)

    return

if __name__ == '__main__':
    main(1, 5)