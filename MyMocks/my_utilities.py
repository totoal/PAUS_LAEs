import numpy as np

import pandas as pd

from scipy.stats import norm
from scipy.integrate import simpson
from scipy.optimize import fsolve

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.table import Table

#import JBOSS as jp

import time
#==============================================================#
#==============================================================#
#==============================================================#


def Load_Filter(filter_name):
    '''
        Returns the response of a filter for several wavelengths.

        Input : string : A filter name, e.g. J0378

        Output : 2 arrays:
            first : wavelength (amstrongs)
            second : system respose
    '''

    Trans_dir = './'

    Trans_name = Trans_dir + 'Transmission_Curves_20170316/' + \
        'JPAS_' + filter_name + '.tab'

    lambda_Amstrongs, Response = np.loadtxt(Trans_name, unpack=True)

    return lambda_Amstrongs, Response
#==============================================================#
#==============================================================#
#==============================================================#


def FWHM_lambda_pivot_filter(filter_name):
    '''
        Return the lambda pivot and the FWHM of a filter

        Input : string : A filter name, e.g. J0378

        Output : 2 floats:
            first : lambda pivot (amstrongs)
            second : FWHM ( amstrongs )
    '''

    lambda_Arr, Transmission = Load_Filter(filter_name)

    # lambda_pivot

    intergral_up = np.trapz(Transmission * lambda_Arr, lambda_Arr)
    intergral_bot = np.trapz(Transmission * 1. / lambda_Arr, lambda_Arr)

    lambda_pivot = np.sqrt(intergral_up * 1. / intergral_bot)

    # FWHM

    mask = Transmission > np.amax(Transmission) * 0.5

    FWHM = lambda_Arr[mask][-1] - lambda_Arr[mask][0]

    FWHM = np.absolute(FWHM)

    return lambda_pivot, FWHM
#==============================================================#
#==============================================================#
#==============================================================#


def Synthetic_Photometry_measure_flux_simple(lambda_Arr, f_lambda_Arr, filter_name):
    '''
        Synthetic fotometry for a spectrum. ( Advanced function ( lv2 ) ).
        Use this when running low amount of synthetic photometries.

        Input :
            lambda_Arr   : An array with the wavelenths of the spectrum
            f_lambda_Arr : An array with the fluxes  (per unit of amstrong) of the spectrum

            filter_name  : An string with the filter name. E.g. J0378

        Output :
            f_lambda_mean : A float with the synthetic photometry at the NB
    '''

    lambda_Arr_f, Transmission_Arr_f = Load_Filter(filter_name)

    lambda_pivot, FWHM = FWHM_lambda_pivot_filter(filter_name)

    f_lambda_mean = Synthetic_Photometry_measure_flux(
        lambda_Arr, f_lambda_Arr, lambda_Arr_f, Transmission_Arr_f, lambda_pivot, FWHM)

    return f_lambda_mean
#==============================================================#
#==============================================================#
#==============================================================#


def Synthetic_Photometry_measure_flux(lambda_Arr, f_lambda_Arr, lambda_Arr_f, Transmission_Arr_f, lambda_pivot, FWHM):
    '''
        Synthetic fotometry for a spectrum. ( Basic function ( lv1 ) ).
        Use this when running several synthetic photometries to avoid
        loading each time the filters.

        Input :
            lambda_Arr   : An array with the wavelenths of the spectrum
            f_lambda_Arr : An array with the fluxes  (per unit of amstrong) of the spectrum

            lambda_Arr_f       : An array with the wavelenths of the filter
            Transmission_Arr_f : An array with the response   of the filter

            lambda_pivot : A float with the lambda pivot of the filter
            FWHM         : A float with the    FHWM      of the filter

        Output :
            f_lambda_mean : A float with the synthetic photometry at the NB
    '''
    bin_lambda_filter = lambda_Arr_f[1] - lambda_Arr_f[0]

    bin_lambda_spectrum = lambda_Arr[1] - lambda_Arr[0]

    if bin_lambda_filter > bin_lambda_spectrum:

        mask_integration_spect = (
            lambda_Arr > lambda_pivot - FWHM) * (lambda_Arr < lambda_pivot + FWHM)

        LAMBDA_to_use = lambda_Arr[mask_integration_spect]

        SPECTRUM_to_use = f_lambda_Arr[mask_integration_spect]

        TRANMISSION_to_use = np.interp(
            LAMBDA_to_use, lambda_Arr_f, Transmission_Arr_f, left=0, right=0)

    else:

        mask_integration_filter = (
            lambda_Arr_f > lambda_pivot - FWHM) * (lambda_Arr_f < lambda_pivot + FWHM)

        LAMBDA_to_use = lambda_Arr_f[mask_integration_filter]

        TRANMISSION_to_use = Transmission_Arr_f[mask_integration_filter]

        SPECTRUM_to_use = np.interp(
            LAMBDA_to_use, lambda_Arr, f_lambda_Arr, left=0, right=0)

    numerador = np.trapz(LAMBDA_to_use * SPECTRUM_to_use *
                         TRANMISSION_to_use, LAMBDA_to_use)

    denominador = np.trapz(LAMBDA_to_use * TRANMISSION_to_use, LAMBDA_to_use)

    f_lambda_mean = numerador * 1. / denominador

    return f_lambda_mean
#==============================================================#
#==============================================================#
#==============================================================#


def IGM_TRANSMISSION(redshift_Arr, A, B):

    Transmission_Arr = np.exp(A * (1 + redshift_Arr)**B)

    return Transmission_Arr
##==============================================================#
##==============================================================#
##==============================================================#


def Load_BC03_grid_data():

    path = 'TAU_PROJECT/BC03_Interpolation/'

    name = 'data_from_BC03.npy'

    file_name = path + '/' + name

    loaded_model = np.load(file_name, allow_pickle=True,
                           encoding='latin1').item()

    return loaded_model
#==============================================================#
#==============================================================#
#==============================================================#


def Interpolate_Lines_Arrays_3D_grid_MCMC(Met_value, Age_value, Ext_value, Grid_Dictionary):

    Grid_Line = Grid_Dictionary['grid']

    met_Arr_Grid = Grid_Dictionary['met_Arr']
    age_Arr_Grid = Grid_Dictionary['age_Arr']
    ext_Arr_Grid = Grid_Dictionary['ext_Arr']

    w_Arr = Grid_Dictionary['w_Arr']

    aux_line = Linear_3D_interpolator(
        Met_value, Age_value, Ext_value, met_Arr_Grid, age_Arr_Grid, ext_Arr_Grid, Grid_Line)

    return w_Arr, aux_line
#==============================================================#
#==============================================================#
#==============================================================#


def Linear_3D_interpolator(X_prob, Y_prob, Z_prob, X_grid, Y_grid, Z_grid, Field_in_grid):

    INDEX_X = np.where((X_grid < X_prob))[0][-1]
    INDEX_Y = np.where((Y_grid < Y_prob))[0][-1]
    INDEX_Z = np.where((Z_grid < Z_prob))[0][-1]

    dX_grid = X_grid[INDEX_X + 1] - X_grid[INDEX_X]
    dY_grid = Y_grid[INDEX_Y + 1] - Y_grid[INDEX_Y]
    dZ_grid = Z_grid[INDEX_Z + 1] - Z_grid[INDEX_Z]

    X_min_grid = X_grid[INDEX_X]
    Y_min_grid = Y_grid[INDEX_Y]
    Z_min_grid = Z_grid[INDEX_Z]

    Xprob_X0 = (X_prob - X_min_grid) * 1. / dX_grid
    Yprob_Y0 = (Y_prob - Y_min_grid) * 1. / dY_grid
    Zprob_Z0 = (Z_prob - Z_min_grid) * 1. / dZ_grid

    Vol1 = (1. - Xprob_X0) * (1. - Yprob_Y0) * (1. - Zprob_Z0)
    Vol2 = (1. - Xprob_X0) * (Yprob_Y0) * (1. - Zprob_Z0)
    Vol3 = (1. - Xprob_X0) * (Yprob_Y0) * (Zprob_Z0)
    Vol4 = (1. - Xprob_X0) * (1. - Yprob_Y0) * (Zprob_Z0)

    Vol5 = (Xprob_X0) * (1. - Yprob_Y0) * (1. - Zprob_Z0)
    Vol6 = (Xprob_X0) * (Yprob_Y0) * (1. - Zprob_Z0)
    Vol7 = (Xprob_X0) * (Yprob_Y0) * (Zprob_Z0)
    Vol8 = (Xprob_X0) * (1. - Yprob_Y0) * (Zprob_Z0)

    Field1 = Field_in_grid[INDEX_X, INDEX_Y, INDEX_Z]
    Field2 = Field_in_grid[INDEX_X, INDEX_Y + 1, INDEX_Z]
    Field3 = Field_in_grid[INDEX_X, INDEX_Y + 1, INDEX_Z + 1]
    Field4 = Field_in_grid[INDEX_X, INDEX_Y, INDEX_Z + 1]
    Field5 = Field_in_grid[INDEX_X + 1, INDEX_Y, INDEX_Z]
    Field6 = Field_in_grid[INDEX_X + 1, INDEX_Y + 1, INDEX_Z]
    Field7 = Field_in_grid[INDEX_X + 1, INDEX_Y + 1, INDEX_Z + 1]
    Field8 = Field_in_grid[INDEX_X + 1, INDEX_Y, INDEX_Z + 1]

    Field_at_the_prob_point = Vol1 * Field1 + Vol2 * Field2 + Vol3 * Field3 + \
        Vol4 * Field4 + Vol5 * Field5 + Vol6 * Field6 + Vol7 * Field7 + Vol8 * Field8

    return Field_at_the_prob_point
#======================================================#
#======================================================#
#======================================================#


def gaussian_f(x_Arr, mu, sigma, Amp):

    y_Arr = norm.pdf(x_Arr, mu, sigma) * Amp

    return y_Arr
#======================================================#
#======================================================#
#======================================================#


def plot_a_rebinned_line(new_wave_Arr, binned_line, Bin):

    DD = Bin * 1e-10

    XX_Arr = np.zeros(len(new_wave_Arr) * 2)
    YY_Arr = np.zeros(len(new_wave_Arr) * 2)

    for i in range(0, len(new_wave_Arr)):

        i_0 = 2 * i
        i_1 = 2 * i + 1

        XX_Arr[i_0] = new_wave_Arr[i] - 0.5 * Bin + DD
        XX_Arr[i_1] = new_wave_Arr[i] + 0.5 * Bin

        YY_Arr[i_0] = binned_line[i]
        YY_Arr[i_1] = binned_line[i]

    return XX_Arr, YY_Arr
#======================================================#
#======================================================#
#======================================================#


def compute_cumulative(x_Arr, y_Arr):

    cum_Arr = np.zeros(len(x_Arr))

    for i in range(1, len(x_Arr)):

        cum_Arr[i] = cum_Arr[i-1] + y_Arr[i]

    cum_Arr = cum_Arr * 1. / np.amax(cum_Arr)

    return cum_Arr
#======================================================#
#======================================================#
#======================================================#


def generate_random_number_from_distribution(x_Arr, Dist_Arr, N_random):

    cum_Arr = compute_cumulative(x_Arr, Dist_Arr)

    random_Arr = np.random.rand(N_random)

    my_random_varible_Arr = np.interp(random_Arr, cum_Arr, x_Arr)

    return my_random_varible_Arr
#======================================================#
#======================================================#
#======================================================#


def generate_spectrum(LINE, my_z, my_ew, my_flux_g, my_widths, my_noises,
                      my_MET, my_AGE, my_EXT, w_Arr, Grid_Dictionary, Noise_w_Arr, Noise_Arr,
                      T_A, T_B, gSDSS_data):

    if LINE == 'Lya':
        w_line = 1215.68

    if LINE == 'OII':
        w_line = 0.5 * (3727.092 + 3729.875)

    cat_w_Arr, cat_rest_spectrum = Interpolate_Lines_Arrays_3D_grid_MCMC(
        my_MET, my_AGE, my_EXT, Grid_Dictionary
    )
    obs_frame_spectrum = np.interp(
        w_Arr, cat_w_Arr * (1 + my_z), cat_rest_spectrum)
    IGM_obs_continum = np.copy(obs_frame_spectrum)

    if LINE == 'Lya':
        redshift_w_Arr = w_Arr * 1. / w_line - 1.
        IGM_T_w_Arr = IGM_TRANSMISSION(redshift_w_Arr, T_A, T_B)
        mask_IGM = w_Arr < w_line * (1 + my_z)
        IGM_obs_continum[mask_IGM] = IGM_obs_continum[mask_IGM] * \
            IGM_T_w_Arr[mask_IGM]

    noisy_spectrum = np.random.normal(0.0, my_noises, len(w_Arr))

    NOISE_w = True
    if NOISE_w:
        Noise_in_my_w_Arr = np.interp(w_Arr, Noise_w_Arr, Noise_Arr)

        Delta_w_noise = 50.0  # A

        w_Lya_observed = (my_z + 1.) * w_line

        mask_noise_norms = (w_Arr > w_Lya_observed - 0.5*Delta_w_noise)\
            * (w_Arr < w_Lya_observed + 0.5*Delta_w_noise)

        I_noise_Arr = np.trapz(Noise_in_my_w_Arr[mask_noise_norms],
                               w_Arr[mask_noise_norms]) * 1. / Delta_w_noise

        Noise_in_my_w_Arr = my_noises * Noise_in_my_w_Arr * 1. / I_noise_Arr

        noisy_spectrum = np.random.normal(0.0, Noise_in_my_w_Arr, len(w_Arr))

    g_w_Arr = gSDSS_data['lambda_Arr_f']
    g_T_Arr = gSDSS_data['Transmission_Arr_f']
    g_w = gSDSS_data['lambda_pivot']
    g_FWHM = gSDSS_data['FWHM']

    # Noises_flux_g = Synthetic_Photometry_measure_flux(
    #     w_Arr, noisy_spectrum, g_w_Arr, g_T_Arr, g_w, g_FWHM
    #     )
    # source_flux_g = Synthetic_Photometry_measure_flux(
    #     w_Arr, IGM_obs_continum, g_w_Arr, g_T_Arr, g_w, g_FWHM
    #     )

    ## Synthetic NB arround emission line##
    snb_w_Arr = g_w_Arr
    snb_T_Arr = np.zeros(snb_w_Arr.shape)
    snb_w = w_line * (1 + my_z)
    snb_T_Arr[np.where(np.abs(snb_w_Arr - snb_w) < 74.)] = 1.
    snb_FWHM = 148.

    Noises_flux_snb = Synthetic_Photometry_measure_flux(
        w_Arr, noisy_spectrum, snb_w_Arr, snb_T_Arr, snb_w, snb_FWHM
    )
    source_flux_snb = Synthetic_Photometry_measure_flux(
        w_Arr, IGM_obs_continum, snb_w_Arr, snb_T_Arr, snb_w, snb_FWHM
    )

    Continum_normalization = (
        my_flux_g - Noises_flux_snb) * 1. / (source_flux_snb)

    cont_around_line = (
        Continum_normalization
        * obs_frame_spectrum[np.where(np.abs(w_Arr-w_line*(1+my_z)) <= 6)]
    )

    obs_lya_line_Arr = norm.pdf(w_Arr, w_line * (1 + my_z), my_widths)
    my_flux_f = np.mean(cont_around_line) * my_ew * (1 + my_z)
    obs_lya_line_Arr = np.absolute(obs_lya_line_Arr * my_flux_f)

    catalog_obs_spectrum_No_IGM = noisy_spectrum + \
        obs_lya_line_Arr + Continum_normalization * obs_frame_spectrum
    catalog_obs_spectrum = noisy_spectrum + obs_lya_line_Arr + \
        Continum_normalization * IGM_obs_continum
    catalog_obs_spectrum_No_Line = noisy_spectrum + \
        Continum_normalization * IGM_obs_continum

    return catalog_obs_spectrum, catalog_obs_spectrum_No_IGM, catalog_obs_spectrum_No_Line
#======================================================#
#======================================================#
#======================================================#

# Function to compute a volume from z interval


def z_volume(z_min, z_max, area):
    z_x = np.linspace(z_min, z_max, 1000)
    dV = cosmo.differential_comoving_volume(z_x).to(u.Mpc**3 / u.sr).value
    area *= (2 * np.pi / 360) ** 2
    theta = np.arccos(1 - area / (2 * np.pi))
    Omega = 2 * np.pi * (1 - np.cos(theta))
    vol = simpson(dV, z_x) * Omega
    # print('Volume = {0:3e} Mpc3'.format(vol))
    return vol

# Function to calculate EW from line flux


def L_flux_to_g(L_Arr, rand_z_Arr, rand_EW_Arr):
    dL_Arr = cosmo.luminosity_distance(rand_z_Arr).to(u.cm).value
    return 10**L_Arr / ((1 + rand_z_Arr) * rand_EW_Arr * 4*np.pi * dL_Arr**2)

# Computes z from L, EW, g


def at_which_redshift(L_Arr, EW0_Arr, f_Arr):
    p0 = np.ones(L_Arr.shape) * 2.5

    z_x = np.linspace(0.1, 10, 100000)
    d_x = cosmo.luminosity_distance(z_x)

    def f(z): return np.interp(((x / (1 + z)) ** 0.5), d_x, z_x) - z

    x = 10 ** L_Arr / (EW0_Arr * 4*np.pi * f_Arr) * u.cm ** 2

    return fsolve(f, p0)


def JPAS_synth_phot(SEDs, w_Arr, tcurves, which_filters=[]):
    phot_len = len(tcurves['tag'])
    pm = np.zeros(phot_len)

    if len(which_filters) == 0:
        which_filters = np.arange(phot_len)

    for fil in which_filters:
        w = np.array(tcurves['w'][fil])
        t = np.array(tcurves['t'][fil])

        # Cut w and t where the transmission is grater than some value for
        # performance and bugs
        cut_t_curve = (t > 0.05)
        w = w[cut_t_curve]
        t = t[cut_t_curve]

        sed_interp = np.interp(w, w_Arr, SEDs, left=np.inf)

        sed_int = np.trapz(w * t * sed_interp, w)
        t_int = np.trapz(w * t, w)

        pm[fil] = sed_int / t_int
    return pm[which_filters]


def mag_to_flux(m, w):
    c = 29979245800
    return 10**((m + 48.60) / (-2.5)) * c/w**2 * 1e8


def flux_to_mag(f, w):
    c = 29979245800
    return -2.5 * np.log10(f * w**2/c * 1e-8) - 48.60


def schechter(L, phistar, Lstar, alpha):
    '''
    Just the regular Schechter function
    '''
    return (phistar / Lstar) * (L / Lstar)**alpha * np.exp(-L / Lstar)


def central_wavelength():
    data_tab = Table.read('../fits/FILTERs_table.fits', format='fits')
    w_central = data_tab['wavelength']

    return np.array(w_central)


def Zero_point_error(tile_id_Arr, catname):
    w_central = central_wavelength()

    # Load Zero Point magnitudes
    zpt_cat = pd.read_csv(
        f'../csv/{catname}.CalibTileImage.csv', sep=',', header=1)

    zpt_err = zpt_cat['ERRZPT'].to_numpy()

    ones = np.ones((len(w_central), len(zpt_err)))

    zpt_err = ones * zpt_err

    # Duplicate rows to match the tile_ID of each source
    idx = np.empty(tile_id_Arr.shape).astype(int)

    zpt_id = zpt_cat['TILE_ID'].to_numpy()
    for src in range(len(tile_id_Arr)):
        idx[src] = np.where(
            (zpt_id == tile_id_Arr[src]) & (
                zpt_cat['IS_REFERENCE_METHOD'] == 1)
        )[0][0]

    zpt_err = zpt_err[:, idx]

    return zpt_err

def nb_fwhm(nb_ind, give_fwhm=True):
    '''
    Returns the FWHM of a filter in tcurves if give_fwhm is True. If it is False, the
    function returns a tuple with (w_central - fwhm/2, w_central + fwhm/2)
    '''
    data_tab = Table.read('../fits/FILTERs_table.fits', format='fits')
    w_central = data_tab['wavelength'][nb_ind]
    fwhm = data_tab['width'][nb_ind]

    if give_fwhm == False:
        return w_central + fwhm / 2, w_central - fwhm / 2
    if give_fwhm == True:
        return fwhm