from jpasLAEs import *

from load_paus_mocks import load_mocks_dict, add_errors
from paus_utils import *

import numpy as np
import pandas as pd


def compute_LF_corrections(mocks_dict, field_name):
    # Modify the mocks adding errors according to the corresponding field
    for mock in mocks_dict:
        mock['flx'], mock['err'] = add_errors(mock['flx_0'], field_name)


    ## Now we have the mock with the errors, do everything else

    # First select LAEs (code the functions to do so)
        
    return


def main():
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
        compute_LF_corrections(mocks_dict, field_name)

    return

if __name__ == '__main__':
    main()