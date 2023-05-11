'''
Functions to load the mocks.
The mocks must be dictionaries with this structure:
    'flx_0': (46, N_sources) matrix with the fluxes
    'mock_z': (N_sources,) Array with the true redshifts of the sources
    'L_lya': (N_sources,) Array with the intrinsic Lya Luminosities
    'EW_lya': (N_sources,) Array with the intrinsic Lya rest-frame EWs
'''

def load_qso_mock(path_to_mock):
    return

def load_sfg_mock(path_to_mock):
    return

def load_gal_mock(path_to_mock):
    return


def load_mocks_dict(mock_SFG_path, mock_QSO_cont_path, mock_QSO_LAEs_loL_path,
                    mock_QSO_LAEs_hiL_path, mock_GAL_path):
    mock_path_list = [mock_SFG_path,
                      mock_QSO_cont_path,
                      mock_QSO_LAEs_loL_path,
                      mock_QSO_LAEs_hiL_path,
                      mock_GAL_path]
    mock_name_list = ['SFG', 'QSO_cont', 'QSO_LAEs_LoL', 'QSO_LAEs_hiL',
                   'GAL']

    mocks_dict = {}
    for i, (mock_path, mock_name) in enumerate(zip(mock_path_list, mock_name_list)):
        if i == 0:
            mocks_dict[mock_name] = load_sfg_mock(mock_path)
        elif i > 0 and i < 4:
            mocks_dict[mock_name] = load_qso_mock(mock_path)
        elif i == 4:
            mocks_dict[mock_name] = load_gal_mock(mock_path)
    
    return mocks_dict


################################################

def add_errors(flx_0, field_name):
    '''
    Returns the matrix of perturbed fluxes and errors: flx, err
    '''
    return