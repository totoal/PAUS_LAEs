import os
import sys
sys.path.insert(0, '..')

from paus_utils import Lya_effective_volume
from jpasLAEs.utils import bin_centers

from astropy.io import fits

import csv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})

import corner

from scipy import linalg

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
    covmat = (covmat**2 + np.eye(sum(where_fit)) * poisson_err[where_fit]**4) ** 0.5

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
        filename_hist = f'{pathname}/hist_mat_boots_combi.npy'
    else:
        filename_hist = f'{pathname}/hist_mat_boots_nb{nb1}-{nb2}.npy'

    hist_i_mat = np.load(filename_hist)

    # Load Poisson errors
    LF_dict = load_combined_LF(region_list, nb_list, combined_LF=combined_LF,
                                   LF_kind='Lya')
    poisson_err = LF_dict['poisson_err']


    return compute_invcovmat(hist_i_mat, where_fit, poisson_err)



##########################

def schechter(L, phistar, Lstar, alpha):
    '''
    Just the regular Schechter function
    '''
    return (phistar / Lstar) * (L / Lstar)**alpha * np.exp(-L / Lstar)


# The fitting curve
def sch_fit(Lx, Phistar, Lstar, alpha):
    Phi = schechter(Lx, 10 ** Phistar, 10 ** Lstar, alpha) * Lx * np.log(10)
    # Phi = schechter(Lx, 10 ** -6.721022310205154, 10 ** Lstar, alpha) * Lx * np.log(10)
    # Phi = schechter(Lx, 10 ** Phistar, 10 ** 45., alpha) * Lx * np.log(10)
    # Phi = schechter(Lx, 10 ** Phistar, 10 ** Lstar, -1.69) * Lx * np.log(10)
    return Phi


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
    Phistar_range = [-9, -3]
    Lstar_range = [43, 45.5]
    alpha_range = [-3, -1]
    theta_trans[0] = Phistar_range[0] + (Phistar_range[1] - Phistar_range[0]) * theta[0]
    theta_trans[1] = Lstar_range[0] + (Lstar_range[1] - Lstar_range[0]) * theta[1]
    theta_trans[2] = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * theta[2]

    return theta_trans

# Main function
def run_mcmc_fit(nb_list, region_list, suffix=''):
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
        vi_cat_hiz = fits.open('/home/alberto/almacen/PAUS_data/catalogs/LAE_selection_VI_hiZ.fits')[1].data
        hiz_mask = vi_cat_hiz['is_hiZ_LAE']

        L_min, L_max = 40, 47
        N_bins = 50
        L_bins = np.linspace(L_min, L_max, N_bins + 1)
        LF_bins = bin_centers(L_bins)

        nb_min = vi_cat_hiz['lya_NB'][hiz_mask].min()
        nb_max = vi_cat_hiz['lya_NB_VI'][hiz_mask].max()

        vol_hiz = 0.
        for field_name in ['W1', 'W2', 'W3']:
            vol_hiz += Lya_effective_volume(nb_min, nb_max, field_name)

        L_Arr_hiz = vi_cat_hiz['L_lya_corr'][hiz_mask]
        L_e_Arr = [vi_cat_hiz['L_lya_corr_err_down'][hiz_mask], vi_cat_hiz['L_lya_corr_err_up'][hiz_mask]]
        LF_mat_hiz = []
        for _ in range(100):
            L_Arr = np.random.choice(L_Arr_hiz, len(L_Arr_hiz), replace=True)

            randN = np.random.randn(len(L_Arr))
            L_perturbed = np.empty_like(L_Arr)
            L_perturbed[randN <= 0] = (L_Arr + L_e_Arr[0] * randN)[randN <= 0]
            L_perturbed[randN > 0] = (L_Arr + L_e_Arr[1] * randN)[randN > 0]
            L_perturbed[np.isnan(L_perturbed)] = 0.

            LF_mat_hiz.append(np.histogram(L_perturbed, L_bins)[0] / (L_bins[1] - L_bins[0]) / vol_hiz)

        LF_phi = np.mean(LF_mat_hiz, axis=0)
        yerr = np.std(LF_mat_hiz, axis=0)

        # In which LF bins fit
        where_fit = np.isfinite(yerr) & (LF_bins > 43) & (LF_bins < 45.5) & (yerr > 0)

        covmat = np.eye(sum(where_fit)) * yerr[where_fit]**2
        invcovmat = linalg.inv(covmat)
    else:
        LyaLF = load_combined_LF(region_list, nb_list, combined_LF=combined_LF)
        [yerr_up, yerr_down] = LyaLF['LF_total_err']
        yerr_up = (yerr_up**2 + LyaLF['poisson_err']**2)**0.5
        yerr_down = (yerr_down**2 + LyaLF['poisson_err']**2)**0.5
        LF_phi = LyaLF['LF_total']
        LF_bins = LyaLF['LF_bins']

        # Error to use
        yerr = (yerr_up + yerr_down) * 0.5
        yerr[LF_phi == 0] = np.inf

        # In which LF bins fit
        where_fit = np.isfinite(yerr) & (LF_bins > 44) & (LF_bins < 45.5)

        invcovmat, _ = load_and_compute_invcovmat(nb_list, where_fit, region_list)


    # Define the name of the fit parameters
    paramnames = ['Phistar', 'Lstar', 'alpha']

    Lx = 10**LF_bins[where_fit]
    Phi = LF_phi[where_fit]

    def log_like(theta):
        Phistar0 = theta[0]
        Lstar0 = theta[1]
        alpha0 = theta[2]

        model_Arr = sch_fit(Lx, Phistar0, Lstar0, alpha0)
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
    fig.savefig(f'figures/corner_nb{nb1}-{nb2}{suffix}.pdf', pad_inches=0.1,
                bbox_inches='tight', facecolor='w')
    plt.close()

    
    # Save the chain
    flat_samples = sampler.results['samples']
    np.save(f'chains/mcmc_schechter_fit_chain_nb{nb1}-{nb2}{suffix}', flat_samples)

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
    region_list = ['W3', 'W2', 'W1']

    nb_list = [[0, 3], [2, 5], [4, 7], [6, 9], [8, 11], [10, 13], [12, 15], [14, 18]]
    # nb_list = [[1, 1, 1]]

    # Add individual NB LFs
    # nb_list = [[nbl] for nbl in nb_list] + [[[n, n]] for n in range(18 + 1)] + [nb_list]
    nb_list = [[nbl] for nbl in nb_list] + [nb_list] + [[1, 1, 1]]

    # suffix = '_fixed_Lstar'
    # suffix = '_fixed_alpha'
    suffix = ''

    # Initialize file to write the fit parameters
    param_filename = f'schechter_fit_parameters{suffix}.csv'
    columns = ['nb_min', 'nb_max', 'Phistar', 'Lstar', 'alpha',
               'Phistar_err_up', 'Lstar_err_up', 'alpha_err_up',
               'Phistar_err_down', 'Lstar_err_down', 'alpha_err_down']
    initialize_csv(param_filename, columns)

    for nbl in nb_list:
        print(nbl)
        # Run the MCMC fit
        fit_params, fit_params_err_up, fit_params_err_down =\
            run_mcmc_fit(nbl, region_list, suffix)

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
