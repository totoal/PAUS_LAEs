import sys
sys.path.insert(0, '..')

from paus_utils import w_central

import numpy as np
from jpasLAEs.utils import mag_to_flux



#------------------------------------------------------------------------#
#-------------------------   TOOLS / ROUTINES   -------------------------#
#------------------------------------------------------------------------#
def convert_absolute_to_observed_magnitudes(absmags=None, dist=None):
    """
    This routine transforms absolute magnitude(s) into observed magnitude(s) provided the distance(s) of the source(s)

    Parameters
    -------------
    absmags  : float, int; list or array. Magnitude(s) of the source(s)
    dist     : float, int; list or array. Y coordinate(s) of the point

    Outputs
    obsmags  : float. Observed magnitude(s)
    """
    if (absmags is None) or (dist is None):
        print("\nError in 'convert_absolute_to_observed_magnitudes' - Missing inputs!! Either absmags or dist were not given!!!\n\n")
        exit()

    absmags = np.atleast_1d(absmags)
    dist    = np.atleast_1d(dist)
    obsmags = 5 * (np.log10(dist) - 1.) + absmags

    # setting all problematic values to 99
    mask = (absmags > 98.)
    obsmags[mask] = 99.
    mask = (absmags < -98.)
    obsmags[mask] = 99.
    mask = np.isinf(absmags)
    obsmags[mask] = 99.
    mask = np.isnan(absmags)
    obsmags[mask] = 99.

    return obsmags



def get_ra_dec_from_cartesian(x=None, y=None, z=None, d=None, units='deg', verbose=False):
    """
    This routine computes RA and DEC from a set of X, Y and Z coordinates

    Parameters
    -------------
    x  : float, int; list or array. X coordinate(s) of the point
    y  : float, int; list or array. Y coordinate(s) of the point
    z  : float, int; list or array. Z coordinate(s) of the point
    d  : float, int; list or array. 3D distance of the point from the origin

    Outputs
    ra  : float; list or array. RA coordinate(s) of the point
    dec : float; list or array. dec coordinate(s) of the point
    """

    if ((x is None) or (y is None) or (y is None)):
        print("\nError in 'get_ra_dec_from_cartesian': either X, Y or Z coordinate(s) were not given!!!\n\n")
        exit()
    if (units != 'deg') & (units != 'rad'): 
        print("\nError in 'get_ra_dec_from_cartesian' - Unrecognized 'units': %s. They can only be 'deg' or 'rad'!!!\n"%units)
        exit()

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    if (len(x) != len(y)) or (len(z) != len(y)) or (len(x) != len(z)):
        print("\nError in 'get_ra_dec_from_cartesian' : X, Y and Z coordinate(s) MUST all have the same dimensions!!!\n\n")
        exit()

    if (d == None): d = np.sqrt( x**2. + y**2. + z**2.)

    dec = np.arcsin(z/d)
    if verbose:  print("DEC (rad ; deg): ",dec,dec*180/np.pi)
    cosdec = np.cos(dec)

    ra_arccos = np.arccos( x/(d*cosdec) )
    if verbose:  print("RA from cos(DEC) (rad ; deg):  ",ra_arccos, ra_arccos*180/np.pi)
    ra_arcsin = np.arcsin( y/(d*cosdec) )
    if verbose:  print("RA from sin(DEC) (rad ; deg):  ",ra_arcsin, ra_arcsin*180/np.pi)

    ra = 1.*ra_arccos
    # if (ra_arcsin < 0.): ra = 2*np.pi - ra_arccos
    mask = (ra_arcsin < 0.)
    ra[mask] = 2*np.pi - ra_arccos[mask]
    if verbose:  print("FINAL RA (rad ; deg):  ",ra, ra*180/np.pi)

    if units == 'deg':
        ra = ra * 180/np.pi
        dec = dec * 180/np.pi

    if (len(ra) == 1):
        ra = ra[0]
        dec = dec[0]
    
    return ra, dec


### Function to read them by ATT
def load_lightcone_mock(directory, suffix):
    column_names = ['Pos', 'ObsMagDust', 'Redshift']

    mock = dict()
    for nm in column_names:
        fname = f'{directory}/{nm}{suffix}'
        mock[nm] = np.load(fname)

    mock['RA'], mock['DEC'] =\
        get_ra_dec_from_cartesian(x=mock['Pos'][:,0], y=mock['Pos'][:,1],
                                  z=mock['Pos'][:,2], d=None,
                                  units='deg', verbose=False)

    distance = np.sqrt(mock['Pos'][:,0]**2
                       + mock['Pos'][:,1]**2
                       + mock['Pos'][:,2]**2) * 1.e6 / 0.673
    distance_matrix = np.array([distance,] * mock['ObsMagDust'].shape[1]).transpose()
    mock['ObsAppMagsDust'] =\
        convert_absolute_to_observed_magnitudes(absmags=mock['ObsMagDust'],
                                                dist=distance_matrix)

    # AT: the filter order I prefer
    filter_order = np.concatenate([np.arange(1, 3),
                                   np.arange(4, 21),
                                   np.arange(22, 35),
                                   np.arange(36, 44),
                                   [0, 3, 21, 35, 44, 45]])

    # Apply magnitude cuts
    r_mag = mock['ObsAppMagsDust'].T[filter_order][-4]
    i_mag = mock['ObsAppMagsDust'].T[filter_order][-3]
    mag_mask = np.array(r_mag < 24.3) & np.array(i_mag < 23.3)

    # Prepare output catalog
    OUT = dict()
    OUT['flx_0'] = mag_to_flux(mock['ObsAppMagsDust'].T[filter_order],
                               w_central.reshape(-1, 1))[:, mag_mask]
    OUT['zspec'] = mock['Redshift'][mag_mask]

    # L_lya and EW0_lya are zero for all these objects.
    OUT['L_lya_spec'] = np.zeros_like(OUT['zspec'])
    OUT['EW0_lya_spec'] = np.zeros_like(OUT['zspec'])
    
    return OUT


#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
#------------------------------------------------------------------------#

if __name__ == '__main__':
    directory = "/home/alberto/almacen/PAUS_data/20240207_mock_data_with_FluxLines_columns_MR_150vols_3x3deg_z0-5"  # path where the data is store
    suff_name = "_magCut[PAUS_BBF_i_25]_LC_chunks[0-150].npy"

    load_lightcone_mock(directory, suff_name)