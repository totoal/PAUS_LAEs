import os
import sys
sys.path.insert(0, '..')

from astropy.io import fits

from paus_utils import Lya_effective_volume
from jpasLAEs.utils import bin_centers

import csv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})

import corner

from scipy import linalg, stats

from multiprocessing import Pool
from autoemcee import ReactiveAffineInvariantSampler

from load_lya_LF import load_combined_LF


def covmat_simple(xiallreal):
    '''
    Computes the covariance matrix of the matrix of histograms
    '''
    nbins = len(xiallreal)

    ximeans = np.nanmean(xiallreal, axis=1)
    xistds = np.nanstd(xiallreal, axis=1, ddof=1)
    covmat = np.empty((nbins, nbins), float)
    corrmat = np.empty((nbins, nbins), float)

    for i in range(nbins):
        for j in range(nbins):
            sigma = ((xiallreal[i] - ximeans[i])
                        * (xiallreal[j] - ximeans[j]))
            nreals = sum(np.isfinite(sigma))
            covmat[i, j] = np.nansum(sigma) / (nreals - 1)
            corrmat[i, j] = covmat[i, j]/(xistds[i]*xistds[j])

    return covmat, corrmat


def compute_invcovmat(hist_mat, where_fit, poisson_err):
    '''
    Computes the inverse covariance matrix of hist_mat in a range defined by the mask where_fit
    '''
    this_hist_mat = hist_mat.T[where_fit]
    this_hist_mat[~np.isfinite(this_hist_mat)] = np.nan
    covmat = covmat_simple(this_hist_mat)[0]

    # Add poisson errors
    covmat = (covmat**2 + np.eye(sum(where_fit)) * poisson_err[where_fit]**2) ** 0.5

    invcovmat = linalg.inv(covmat)
    return invcovmat, covmat

def load_and_compute_invcovmat(nb_list, where_fit, region_list):
    if len(nb_list) == 1:
        [nb1, nb2] = nb_list[0]
        combined_LF = False
    else:
        combined_LF = True

    # Load the matrix of LF realizations
    name = f'bootstrap_errors'
    pathname = f'/home/alberto/almacen/PAUS_data/Lya_LFs/{name}'
    if combined_LF:
        filename_hist = f'{pathname}/hist_mat_boots_combi_M.npy'
    else:
        filename_hist = f'{pathname}/hist_mat_boots_nb{nb1}-{nb2}_M.npy'

    hist_i_mat = np.load(filename_hist)

    # Load Poisson errors
    LF_dict = load_combined_LF(region_list, nb_list, combined_LF=combined_LF,
                                   LF_kind='UV')
    poisson_err = np.log10(1 + LF_dict['poisson_err'] / LF_dict['LF_boots'])


    return compute_invcovmat(np.log10(hist_i_mat), where_fit, poisson_err)



##########################

def double_power_law(M, Phistar, Mbreak, beta, gamma):
    exp1 = 0.4 * (Mbreak - M) * (beta - 1)
    exp2 = 0.4 * (Mbreak - M) * (gamma - 1)

    return 10. ** Phistar / (10. ** exp1 + 10. ** exp2)

# The fitting curve
def dpl_fit(*args):
    return np.log10(double_power_law(*args))

################################################
### A set of functions to compute MCMC stuff ###
################################################

def chi2_fullmatrix(data_vals, inv_covmat, model_predictions):
    """
    Given a set of data points, its inverse covariance matrix and the
    corresponding set of model predictions, computes the standard chi^2
    statistic (using the full covariances)
    """

    y_diff = data_vals - model_predictions
    return np.dot(y_diff, np.dot(inv_covmat, y_diff))

################################################
################################################


def transform(theta):
    '''
    Transform features to match the priors
    '''
    theta_trans = np.empty_like(theta)
    
    # Flat Priors
    Phistar_range = [-10, -1]
    Mbreak_range = [-27, -15]
    beta_range = [-5, 0]
    gamma_range = [0, 5]

    theta_trans[0] = Phistar_range[0] + (Phistar_range[1] - Phistar_range[0]) * theta[0]
    theta_trans[1] = Mbreak_range[0] + (Mbreak_range[1] - Mbreak_range[0]) * theta[1]
    theta_trans[2] = beta_range[0] + (beta_range[1] - beta_range[0]) * theta[2]
    theta_trans[3] = gamma_range[0] + (gamma_range[1] - gamma_range[0]) * theta[3]

    return theta_trans

# Main function
def run_mcmc_fit(nb_list, region_list):
    # Load the Lya LF
    if len(nb_list) == 1:
        combined_LF = False
    else:
        combined_LF = True
    if len(nb_list) == 3:
        last_bin = True
    else:
        last_bin = False

    if last_bin:
        ## For the last bin

        N_bins_UV = 23 + 1
        M_UV_bins = np.linspace(-30, -19, N_bins_UV)
        LF_bins = bin_centers(M_UV_bins)

        vi_cat_hiz = fits.open('/home/alberto/almacen/PAUS_data/catalogs/LAE_selection_VI_hiZ_with_MUV.fits')[1].data
        hiz_mask = vi_cat_hiz['is_hiZ_LAE']
        nb_min = vi_cat_hiz['lya_NB_1'][hiz_mask].min()
        nb_max = vi_cat_hiz['lya_NB_VI'][hiz_mask].max()

        vol_hiz = 0.
        for field_name in ['W1', 'W2', 'W3']:
            vol_hiz += Lya_effective_volume(nb_min, nb_max, field_name)

        MUV_Arr_hiz = [-24.85033822, -73.68420369, -23.65994873, -23.69145089, -23.96237285,
                       -73.57526006, -73.46105176, -73.57526006, -23.61200385, -23.7234547,
                       -23.77824481, -23.81355431, -23.67256559, -26.60930838, -24.55089336,
                     -73.46105176, -23.8043054,  -23.97002486]
        MUV_e_Arr = [0.07865987, 0.75257499, 0.26237464, 0.20637891, 0.17488506, 0.75257499,
                    0.75257499, 0.75257499, 0.17609249, 0.19124034, 0.22245714, 0.22387602,
                    0.20254465, 0.01569782, 0.10766078, 0.75257499, 0.20308774, 0.19353003]
        LF_mat_hiz = []
        for jjj in range(100):
            MUV_Arr = np.random.choice(MUV_Arr_hiz, len(MUV_Arr_hiz), replace=True)

            randN = np.random.randn(len(MUV_Arr))
            MUV_perturbed = np.empty_like(MUV_Arr)
            MUV_perturbed = MUV_Arr + MUV_e_Arr * randN
            MUV_perturbed[np.isnan(MUV_perturbed)] = 0.

            LF_mat_hiz.append(np.histogram(MUV_perturbed, M_UV_bins)[0] / (M_UV_bins[1] - M_UV_bins[0]) / vol_hiz)

        LF_phi = np.mean(LF_mat_hiz, axis=0)
        yerr = np.std(LF_mat_hiz, axis=0)
        yerr = np.log10(LF_phi + yerr) - np.log10(LF_phi)

        # In which LF bins fit
        where_fit = np.isfinite(yerr) & (LF_bins > -28) & (LF_bins < -15) & (yerr > 0) & np.isfinite(LF_phi)

        covmat = np.eye(sum(where_fit)) * yerr[where_fit]**2
        invcovmat = linalg.inv(covmat)
    else:
        uv_LF = load_combined_LF(region_list, nb_list, combined_LF=combined_LF,
                                LF_kind='UV')
        [yerr_up, yerr_down] = uv_LF['LF_total_err']
        yerr_up = (yerr_up**2 + uv_LF['poisson_err']**2)**0.5
        yerr_down = (yerr_down**2 + uv_LF['poisson_err']**2)**0.5
        LF_phi = uv_LF['LF_total']
        LF_bins = uv_LF['LF_bins']
        

        # Error to use
        yerr = (yerr_up + yerr_down) * 0.5
        yerr[LF_phi == 0] = np.inf

        # In which LF bins fit
        where_fit = np.isfinite(yerr) & (LF_bins > -28) & (LF_bins < -15) & (uv_LF['poisson_err'] > 0) & np.isfinite(LF_phi)

        invcovmat, _ = load_and_compute_invcovmat(nb_list, where_fit, region_list)

    # Define the name of the fit parameters
    paramnames = ['Phistar', 'Mbreak', 'beta', 'gamma']

    Lx = LF_bins[where_fit]
    Phi = np.log10(LF_phi[where_fit])

    def log_like(theta):
        Phistar0 = theta[0]
        Mbreak0 = theta[1]
        beta0 = theta[2]
        gamma0 = theta[3]

        model_Arr = dpl_fit(Lx, Phistar0, Mbreak0, beta0, gamma0)
        model_Arr[~np.isfinite(model_Arr)] = 1e-99

        chi2 = chi2_fullmatrix(Phi, invcovmat, model_Arr)

        return -0.5 * chi2


    # Define the sampler
    sampler = ReactiveAffineInvariantSampler(paramnames,
                                             log_like,
                                             transform=transform)
    # Run the sampler
    sampler.run(max_ncalls=1e7, progress=False)
    # Print the results
    sampler.print_results()

    # Plot results
    if last_bin:
        nb1 = nb_min
        nb2 = nb_max
    else:
        nb1 = np.array(nb_list).flatten()[0]
        nb2 = np.array(nb_list).flatten()[-1]
    fig = corner.corner(sampler.results['samples'], labels=paramnames,
                        show_titles=True, truths=sampler.results['posterior']['median'])
    fig.savefig(f'figures/corner_UV_nb{nb1}-{nb2}.pdf', pad_inches=0.1,
                bbox_inches='tight', facecolor='w')
    plt.close()

    
    # Save the chain
    flat_samples = sampler.results['samples']
    np.save(f'chains/mcmc_UV_dpl_fit_chain_nb{nb1}-{nb2}', flat_samples)

    # Obtain the fit parameters
    fit_params = sampler.results['posterior']['median']
    fit_params_perc84 = np.percentile(flat_samples, [84], axis=0)[0]
    fit_params_perc16 = np.percentile(flat_samples, [16], axis=0)[0]
    fit_params_err_up = fit_params_perc84 - fit_params
    fit_params_err_down = fit_params - fit_params_perc16

    return fit_params, fit_params_err_up, fit_params_err_down


def initialize_csv(filename, columns):
    if os.path.exists(filename):
        # If the file exists, overwrite it with an empty file
        open(filename, 'w').close()

    # Initialize the file with column names
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)



if __name__ == '__main__':
    region_list = ['W3', 'W1', 'W2']

    nb_list = [[0, 3], [2, 5], [4, 7], [6, 9], [8, 11], [10, 13], [12, 15], [14, 18]]

    # Add individual NB LFs
    nb_list = [[nbl] for nbl in nb_list] + [nb_list] + [[1, 1, 1]]

    # Initialize file to write the fit parameters
    param_filename = 'UV_dpl_fit_parameters.csv'
    columns = ['nb_min', 'nb_max', 'Phistar', 'Mbreak', 'beta',
               'gamma', 'Phistar_err_up', 'Mbreak_err_up', 'beta_err_up',
               'gamma_err_up', 'Phistar_err_down', 'Mbreak_err_down',
               'beta_err_down', 'gamma_err_down']
    initialize_csv(param_filename, columns)

    for nbl in nb_list:
        print(nbl)
        # Run the MCMC fit
        fit_params, fit_params_err_up, fit_params_err_down =\
            run_mcmc_fit(nbl, region_list)

        # Append the parameters to csv file
        with open(param_filename, 'a', newline='') as param_file:
            writer = csv.writer(param_file)
            if len(nbl) < 2:
                row_to_write = np.concatenate([
                    [nbl[0][0], nbl[-1][-1]],
                    fit_params,
                    fit_params_err_up,
                    fit_params_err_down
                ])
            else:
                row_to_write = np.concatenate([
                    [19, 50],
                    fit_params,
                    fit_params_err_up,
                    fit_params_err_down
                ])
            
            writer.writerow(row_to_write)
