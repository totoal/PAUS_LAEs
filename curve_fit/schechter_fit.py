import sys
sys.path.insert(0, '..')

import pickle

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})

import emcee
import corner

import pandas as pd

from scipy import linalg
from scipy import stats

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


def compute_invcovmat(hist_mat, where_fit):
    '''
    Computes the inverse covariance matrix of hist_mat in a range defined by the mask where_fit
    '''
    this_hist_mat = hist_mat.T[where_fit]
    this_hist_mat[~np.isfinite(this_hist_mat)] = np.nan
    covmat = covmat_simple(this_hist_mat)[0]
    invcovmat = linalg.inv(covmat)
    return invcovmat, covmat

def load_and_compute_invcovmat(nb1, nb2, region_list, where_fit):
    # Load the matrix of LF realizations
    hist_i_mat = None
    for region_name in region_list:
        LF_name = f'Lya_LF_nb{nb1}-{nb2}_W3'
        pathname = f'/home/alberto/almacen/PAUS_data/Lya_LFs/{LF_name}'
        filename_hist = f'{pathname}/hist_i_mat_0.npy'

        if hist_i_mat is None:
            hist_i_mat = np.load(filename_hist)
        else:
            hist_i_mat += np.load(filename_hist)

    return compute_invcovmat(np.log10(hist_i_mat), where_fit)



##########################

def schechter(L, phistar, Lstar, alpha):
    '''
    Just the regular Schechter function
    '''
    return (phistar / Lstar) * (L / Lstar)**alpha * np.exp(-L / Lstar)


# The fitting curve
def sch_fit(Lx, Phistar, Lstar, alpha):
    Phi = schechter(Lx, 10 ** Phistar, 10 ** Lstar, alpha) * Lx * np.log(10)
    return np.log10(Phi)


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
    Lstar_range = [40, 47]
    alpha_range = [-4, 2]
    theta_trans[0] = Phistar_range[0] + (Phistar_range[1] - Phistar_range[0]) * theta[0]
    theta_trans[1] = Lstar_range[0] + (Lstar_range[1] - Lstar_range[0]) * theta[1]
    theta_trans[2] = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * theta[2]

    return theta_trans

# Main function
def run_mcmc_fit(nb1, nb2, region_list):
    # Load the Lya LF
    LyaLF = load_combined_LF(region_list, [[nb1, nb2]])
    LF_yerr_minus = LyaLF['LF_total_err'][0]
    LF_yerr_plus = LyaLF['LF_total_err'][1]
    LF_phi = LyaLF['LF_total']
    LF_bins = LyaLF['LF_bins']
    

    # Error to use
    yerr = (LF_yerr_plus + LF_yerr_minus) * 0.5
    yerr[LF_phi == 0] = np.inf

    # In which LF bins fit
    where_fit = np.isfinite(yerr) & (LF_bins > 43) & (LF_bins < 46)# & (yerr > 0)

    invcovmat, _ = load_and_compute_invcovmat(nb1, nb2, region_list, where_fit)

    # Define the name of the fit parameters
    paramnames = ['Phistar', 'Lstar', 'alpha']

    Lx = 10**LF_bins[where_fit]
    Phi = np.log10(LF_phi[where_fit])

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
    sampler.run(max_ncalls=1e7)
    # Print the results
    sampler.print_results()

    # Plot results
    fig = corner.corner(sampler.results['samples'], labels=paramnames,
                        show_titles=True, truths=sampler.results['posterior']['median'])
    fig.savefig(f'figures/corner_nb{nb1}-{nb2}.pdf', pad_inches=0.1,
                bbox_inches='tight', facecolor='w')
    plt.close()

    
    # Save the chain
    flat_samples = sampler.results['samples']
    np.save(f'chains/mcmc_schechter_fit_chain_nb{nb1}-{nb2}', flat_samples)



if __name__ == '__main__':
    region_list = ['W3']
    run_mcmc_fit(0, 2, region_list)