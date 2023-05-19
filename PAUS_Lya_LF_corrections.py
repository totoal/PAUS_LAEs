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
    # State the mock area in deg²:
    gal_fraction = 0.01

    SFG_area = 400
    QSO_cont_area = 200
    QSO_LAEs_loL_area = 400
    QSO_LAEs_hiL_area = 4000
    GAL_area = 59.97 * gal_fraction

    # Load the mocks
    source_cats_dir = '/home/alberto/almacen/Source_cats'
    mock_SFG_path = f'{source_cats_dir}/LAE_12.5deg_z2.55-5_PAUS_0'
    mock_QSO_cont_path = f'{source_cats_dir}/QSO_PAUS_contaminants_2'
    mock_QSO_LAEs_loL_path = f'{source_cats_dir}/QSO_PAUS_LAES_2'
    mock_QSO_LAEs_hiL_path = f'{source_cats_dir}/QSO_PAUS_LAES_hiL_2'
    mock_GAL_path = '/home/alberto/almacen/PAUS_data/catalogs/LightCone_mock.fits'
    mocks_dict = load_mocks_dict(mock_SFG_path, mock_QSO_cont_path,
                                 mock_QSO_LAEs_loL_path, mock_QSO_LAEs_hiL_path,
                                 mock_GAL_path, gal_fraction=gal_fraction)

    # List of PAUS fields
    # field_list = ['foo', 'bar']
    # for field_name in field_list:
    #     compute_LF_corrections(mocks_dict, field_name,
    #                            nb_min, nb_max)

    return

if __name__ == '__main__':
    main(1, 5)