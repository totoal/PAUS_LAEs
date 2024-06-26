import pickle

import numpy as np

from paus_utils import *
from jpasLAEs.utils import flux_to_mag

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


# Line rest-frame wavelengths (Angstroms)
w_lya = 1215.67
w_lyb = 1025.7220
w_SiIV = 1397.61
w_CIV = 1549.48
w_CIII = 1908.73
w_MgII = 2799.12


def IGM_TRANSMISSION(w_Arr, A=-0.001845, B=3.924):
    '''
    Returns the IGM transmission associated to the Lya Break.
    '''
    return np.exp(A * (w_Arr / w_lya) ** B)


def estimate_continuum(NB_flx, NB_err, N_nb=7, IGM_T_correct=True,
                       only_right=False, N_nb_min=0, N_nb_max=39):
    '''
    Returns a matrix with the continuum estimate at any NB in all sources.
    '''
    NB_flx = NB_flx[:40]
    NB_err = NB_err[:40]

    cont_est = np.ones(NB_flx.shape) * 99.
    cont_err = np.ones(NB_flx.shape) * 99.

    for nb_idx in range(N_nb_min, N_nb_max + 1):
        if (nb_idx < N_nb) or only_right:
            if IGM_T_correct and nb_idx > 0:
                IGM_T = IGM_TRANSMISSION(
                    np.array(w_central[: nb_idx - 1])
                ).reshape(-1, 1)
            else:
                IGM_T = 1.
            
            # Stack filters at both sides or only at the right of the central one
            if only_right or (nb_idx == 0):
                NBs_to_avg = NB_flx[nb_idx + 2: nb_idx + N_nb + 1]
                NBs_errs = NB_err[nb_idx + 2: nb_idx + N_nb + 1]
            else:
                NBs_to_avg = np.vstack((
                    NB_flx[: nb_idx - 1] / IGM_T,
                    NB_flx[nb_idx + 2: nb_idx + N_nb + 1]
                ))
                NBs_errs = np.vstack((
                    NB_err[: nb_idx - 1] / IGM_T,
                    NB_err[nb_idx + 2: nb_idx + N_nb + 1]
                ))

        elif (N_nb <= nb_idx) & (nb_idx < (NB_flx.shape[0] - N_nb)) and not only_right:
            if IGM_T_correct:
                IGM_T = IGM_TRANSMISSION(
                    np.array(w_central[nb_idx - N_nb: nb_idx - 1])
                ).reshape(-1, 1)
            else:
                IGM_T = 1.
            NBs_to_avg = np.vstack((
                NB_flx[nb_idx - N_nb: nb_idx - 1] / IGM_T,
                NB_flx[nb_idx + 2: nb_idx + N_nb + 1]
            ))
            NBs_errs = np.vstack((
                NB_err[nb_idx - N_nb: nb_idx - 1] / IGM_T,
                NB_err[nb_idx + 2: nb_idx + N_nb + 1]
            ))

        elif nb_idx >= (NB_flx.shape[0] - N_nb):
            if IGM_T_correct:
                IGM_T = IGM_TRANSMISSION(
                    np.array(w_central[nb_idx - N_nb: nb_idx - 1])
                ).reshape(-1, 1)
            else:
                IGM_T = 1.
            NBs_to_avg = np.vstack((
                NB_flx[nb_idx - N_nb: nb_idx - 1] / IGM_T,
                NB_flx[nb_idx + 2:]
            ))
            NBs_errs = np.vstack((
                NB_err[nb_idx - N_nb: nb_idx - 1] / IGM_T,
                NB_err[nb_idx + 2:]
            ))

        # Weights for the average
        w = NBs_errs ** -2

        cont_est[nb_idx] = np.average(NBs_to_avg, weights=w, axis=0)
        cont_err[nb_idx] = np.sum(w, axis=0) ** -0.5

    return cont_est, cont_err


##### LOOK FOR LINES #####

def is_there_line(pm_flx, pm_err, cont_est, cont_err, ew0min,
                  mask=None, obs=False, sigma=3):
    '''
    Returns a matrix of the same shape of pm_flx, with bool values whether
    there is a significant flux excess or not.
    '''
    if not obs:
        z_nb_Arr = (w_central[:40] / w_lya - 1)
        ew_Arr = ew0min * (1 + z_nb_Arr)
    if obs:
        ew_Arr = ew0min

    if mask is None:
        mask = np.ones(pm_flx.shape[1]).astype(bool)

    line = (
        # 3-sigma flux excess
        (
            pm_flx[:40] - cont_est > sigma * \
            (pm_err[:40]**2 + cont_err**2) ** 0.5
        )
        # EW0 min threshold
        & (
            pm_flx[:40] / cont_est > 1 + (ew_Arr / fwhm_Arr[:40]).reshape(-1, 1)
        )
        & (
            pm_flx[:40] > cont_est
        )
        # Masks
        & (
            mask
        )
        # Check that cont_est is ok
        & (
            np.isfinite(cont_est)
        )
    )
    return line

def identify_lines(line_mat, flux_mat, continuum_flux,
                   mult_lines=False):
    '''
    Returns a list of positions with the maximum flux for each source.

    Input:
    line_mat: Bool matrix of shape (N_filters, N_sources) indicating detections
    flux_mat: Matrix of shape (N_filters, N_sources) containing flux values
    mult_lines: Boolean flag indicating whether to return positions of all detections or just the maximum flux position

    Output:
    A list of length N_sources. If mult_lines is True, each element is a list containing the positions of all detections for the corresponding source. Otherwise, each element is the position of the maximum flux for the corresponding source.
    '''
    
    if (line_mat.shape != continuum_flux.shape):
        raise ValueError('All input matrices must have the same dimensions.')

    if (line_mat.shape != flux_mat.shape):
        raise ValueError('All input matrices must have the same dimensions.')

    N_filters, N_src = line_mat.shape
    line_positions = []

    flux_diff_mat = flux_mat - continuum_flux

    for src in range(N_src):
        this_src_detections = line_mat[:, src]

        if mult_lines:
            line_pos_list = []
            is_contiguous = False

            for pos in range(N_filters):
                if this_src_detections[pos]:
                    if not is_contiguous:
                        is_contiguous = True
                        start_pos = pos
                else:
                    if is_contiguous:
                        is_contiguous = False
                        end_pos = pos

                        pos_Arr = np.arange(start_pos, end_pos)
                        this_flux_diff = [flux_diff_mat[:, src][i] for i in pos_Arr]
                        line_pos = pos_Arr[np.argmax(this_flux_diff)]
                        
                        line_pos_list.append(line_pos)

            line_positions.append(line_pos_list)
        else:
            if not np.any(this_src_detections):
                line_positions.append(-1)
                continue

            where_detection = np.where(this_src_detections)[0]
            line_pos = where_detection[np.argmax(flux_mat[:, src][where_detection])]
            # line_pos = where_detection[0]
            line_positions.append(line_pos)

    if not mult_lines:
        line_positions = np.array(line_positions)

    return line_positions


def nice_lya_select(lya_lines, other_lines, pm_flx, pm_err,
                    z_Arr, mask=None):
    '''
    Apply selection criteria to a list of Lyman-alpha (Lya) line candidates.

    Parameters:
    -----------
    lya_lines : array_like
        List of Lya line candidates.
    other_lines : list of list
        List of lists with the indices of the other lines
        detected for each Lya line candidate.
    pm_flx : array_like
        Matrix of the photometric flux values.
    z_Arr : array_like
        Array of redshift values.
    mask : array_like, optional
        Boolean array to be used as a mask.

    Returns:
    --------
    nice_lya : array_like
        A boolean array of the same length as `lya_lines`, where True means a good candidate.
    color_mask : array_like
        A boolean array indicating if the color criteria is met.
    mlines_mask : array_like
        A boolean array indicating if the mask for the other lines is met.
    '''
    # TODO: Check colors of PAUS. Extend to all BBs
    # i = flux_to_mag(pm_flx[-3], w_central[-3])
    r = flux_to_mag(pm_flx[-4], w_central[-4])
    g = flux_to_mag(pm_flx[-5], w_central[-5])
    u = flux_to_mag(pm_flx[-6], w_central[-6])
    gr = g - r
    ug = u - g

    color_aux = np.ones_like(gr).astype(bool)
    # NBs: 0-2
    mask_llines = (lya_lines >= 0) & (lya_lines < 2)
    # color_aux[mask_llines] = ((gr > -0.23) & (ug > 0.024))[mask_llines]
    # NBs: 2-4
    mask_llines = (lya_lines >= 2) & (lya_lines < 4)
    # color_aux[mask_llines] = ((gr > -0.19) & (ug > 0.18))[mask_llines]
    # NBs: 4-6
    mask_llines = (lya_lines >= 4) & (lya_lines < 6)
    # color_aux[mask_llines] = ((gr > -0.16) & (ug > 0.35))[mask_llines]
    # NBs: 6-8
    mask_llines = (lya_lines >= 6) & (lya_lines < 8)
    # color_aux[mask_llines] = (ug < 0.5)[mask_llines]
    # NBs: 8-10
    mask_llines = (lya_lines >= 8) & (lya_lines < 10)
    # color_aux[mask_llines] = (ug < 0.42)[mask_llines]
    # NBs: 10-12
    mask_llines = (lya_lines >= 10) & (lya_lines < 12)
    # color_aux[mask_llines] = (ug < 0.37)[mask_llines]
    # NBs: 12-14
    mask_llines = (lya_lines >= 12) & (lya_lines < 14)
    # color_aux[mask_llines] = (ug < 0.34)[mask_llines]
    # NBs: 14-16
    mask_llines = (lya_lines >= 14) & (lya_lines <= 16)
    # color_aux[mask_llines] = (ug < 0.45)[mask_llines]
    # NBs: 16-18
    mask_llines = (lya_lines >= 16) & (lya_lines <= 18)
    # color_aux[mask_llines] = (ug < 0.47)[mask_llines]
    # NBs: >18
    mask_llines = (lya_lines > 18)
    color_aux[mask_llines] = ((pm_flx[-6] / pm_err[-6] < 3))[mask_llines]


    N_sources = pm_flx.shape[1]

    color_mask = np.ones(N_sources).astype(bool)
    mlines_mask = np.ones(N_sources).astype(bool)

    for src in np.where(np.array(lya_lines) != -1)[0]:
        z_src = z_Arr[src]

        w_obs_lya = (1 + z_src) * w_lya
        w_obs_lyb = (1 + z_src) * w_lyb
        w_obs_SiIV = (1 + z_src) * w_SiIV
        w_obs_CIV = (1 + z_src) * w_CIV
        w_obs_CIII = (1 + z_src) * w_CIII
        # w_obs_MgII = (1 + z_src) * w_MgII

        for l in other_lines[src]:
            # Ignore very red and very blue NBs
            if (l > 40) | (l < 0):
                continue

            w_obs_l = w_central[l]
            fwhm = fwhm_Arr[l]

            good_l = (
                (np.abs(w_obs_l - w_obs_lyb) < fwhm * 1.)
                | (np.abs(w_obs_l - w_obs_lya) < fwhm * 1.)
                # | (((w_obs_CIV - w_obs_l) < 326.) & ((w_obs_CIV - w_obs_l) > 0))
                | (((w_obs_CIV - w_obs_l) < 260.) & ((w_obs_CIV - w_obs_l) > 0))
                | (np.abs(w_obs_l - w_obs_SiIV) < fwhm * 2.)
                | (((w_obs_l - w_obs_CIV) < 71.) & ((w_obs_l - w_obs_CIV) > 0))
                | (((w_obs_CIII - w_obs_l) < 399.) & ((w_obs_CIII - w_obs_l) > 0))
                | (((w_obs_l - w_obs_CIII) < 125.) & ((w_obs_l - w_obs_CIII) > 0))
            )
            if ~good_l:
                mlines_mask[src] = False
                break

        if len(other_lines[src]) > 1 and lya_lines[src] < 18:
            pass
        else:
            good_colors = color_aux[src]

            if ~good_colors:
                color_mask[src] = False

    # Define nice_lya
    nice_lya = (lya_lines >= 0)
    if mask is not None:
        nice_lya = nice_lya & mask

    return nice_lya, color_mask, mlines_mask


def ML_LAE_class(cat, p_min=0.0, nice_col_name='nice_lya'):
    '''
    This function applies a pre-trained machine learning classifier to predict the class of a given input catalog.

    Parameters:
    cat (numpy.ndarray): The input catalog containing relevant data for classification.

    Returns:
    numpy.ndarray: An array representing the predicted classes for the input catalog.
    '''
    # import the classifier
    save_dir = '/home/alberto/almacen/PAUS_data/ML_classifier'
    with open(f'{save_dir}/source_classifier.sav', 'rb') as file:
        classifier = pickle.load(file)
    with open(f'{save_dir}/source_scaler.sav', 'rb') as file:
        scaler = pickle.load(file)

    # Pre-processing
    selection = cat[nice_col_name]
    dataset = np.hstack([
        cat['flx'][:40, selection].T * 1e17,
        cat['lya_NB'][selection].reshape(-1, 1),
        cat['r_mag'][selection].reshape(-1, 1),
        cat['flx'][40:45, selection].T * 1e17, # BBs
    ])

    # Apply scaler and PCA
    dataset[:, :40] /= np.sum(dataset[:, :40], axis=1).reshape(-1, 1)
    dataset[:, 42:47] /= np.sum(dataset[:, 42:47], axis=1).reshape(-1, 1)
    dataset[:, :47] = scaler.transform(dataset[:, :47])

    prediction = classifier.predict(dataset)
    log_p = classifier.predict_log_proba(dataset)

    for src in range(len(prediction)):
        if prediction[src] == 4:
            pred_i = 2
        elif prediction[src] == 5:
            pred_i = 3
        else:
            pred_i = prediction[src] - 1

        class_log_p = log_p[src, pred_i]
        if p_min > 0:
            if class_log_p < np.log(p_min):
                prediction[src] = 6

    return prediction, log_p


def ML_z_Lya_regression(cat, nice_col_name='nice_lya'):
    '''
    This function applies a pre-trained machine learning classifier to predict the class of a given input catalog.

    Parameters:
    cat (numpy.ndarray): The input catalog containing relevant data for classification.

    Returns:
    numpy.ndarray: An array representing the predicted classes for the input catalog.
    '''
    # import the classifier
    save_dir = '/home/alberto/almacen/PAUS_data/ML_z_reg'
    with open(f'{save_dir}/z_fit_NN_reg.sav', 'rb') as file:
        regressor = pickle.load(file)

    # Pre-processing
    selection = cat[nice_col_name]
    dataset = np.hstack([
        cat['flx'][:40, selection].T * 1e17,
        cat['r_mag'][selection].reshape(-1, 1),
        cat['flx'][40:45, selection].T * 1e17, # BBs
        cat['lya_NB'][selection].reshape(-1, 1),
    ])

    # Apply scaler and PCA
    dataset[:, :40] /= np.sum(dataset[:, :40], axis=1).reshape(-1, 1)
    dataset[:, 41:46] /= np.sum(dataset[:, 41:46], axis=1).reshape(-1, 1)
    dataset[:, 40] /= 30
    dataset[:, 46] /= 30

    predicted_z = regressor.predict(dataset)


    return predicted_z


def select_LAEs(cat, nb_min, nb_max, r_min, r_max, ew0min_lya=20,
                ewmin_other=100, check_nice_z=False):
    N_sources = cat['flx'].shape[1]
    # Estimate continuum
    cont_est, cont_err = estimate_continuum(cat['flx'], cat['err'],
                                            IGM_T_correct=True, N_nb=6)
    cont_est_other, cont_err_other = estimate_continuum(cat['flx'], cat['err'],
                                                        IGM_T_correct=False,
                                                        N_nb=6)

    # Identify NB excess
    is_line_lya = is_there_line(cat['flx'], cat['err'], cont_est, cont_err,
                                ew0min=ew0min_lya)
    is_line_other = is_there_line(cat['flx'], cat['err'], cont_est_other,
                                  cont_err_other, ew0min=ewmin_other, obs=True)
    lya_lines = identify_lines(is_line_lya, cat['flx'][:40],
                               cont_est, mult_lines=False)
    other_lines = identify_lines(is_line_other, cat['flx'][:40],
                                 cont_est_other, mult_lines=True)

    # Estimate redshift (z_Arr)
    z_Arr = z_NB(lya_lines)

    snr = np.empty(N_sources)
    for src in range(N_sources):
        l = lya_lines[src]
        snr[src] = cat['flx'][l, src] / cat['err'][l, src]

    nb_mask = (lya_lines >= nb_min) & (lya_lines <= nb_max)
    snr_mask = (snr >= 6)

    nice_lya_mask = snr_mask & nb_mask\
        & (cat['r_mag'] >= r_min) & (cat['r_mag'] <= r_max)

    nice_lya, color_mask, ml_mask =\
        nice_lya_select(lya_lines, other_lines, cat['flx'], cat['err'],
                        z_Arr, mask=nice_lya_mask)

    # Add columns to cat
    cat['nice_lya'] = nice_lya
    cat['nice_color'] = color_mask
    cat['nice_ml'] = ml_mask
    cat['z_NB'] = z_Arr
    cat['lya_NB'] = lya_lines
    cat['other_lines_NBs'] = other_lines
    cat['lya_snr'] = snr

    z_ML = ML_z_Lya_regression(cat)
    cat['z_ML'] = np.ones_like(cat['z_NB']) * -1
    cat['z_ML'][nice_lya] = z_ML
    mask_nice_z_ML = np.abs(z_Arr - cat['z_ML']) < 0.24
    cat['z_NB'][mask_nice_z_ML] = cat['z_ML'][mask_nice_z_ML]

    # Machine learning classification
    prediction, _ = ML_LAE_class(cat)
    # If classified as Galaxy, nice_lya = False
    # 1 for QSO_cont
    # 2 for QSO_LAEs
    # 4 for GAL
    class_mask = np.zeros_like(nice_lya).astype(bool)
    class_pred = np.ones_like(nice_lya).astype(int) * -1
    class_mask[nice_lya] = prediction == 2 # 2 for LAEs
    class_pred[nice_lya] = prediction

    # Mask based on class_star if not mock
    if check_nice_z:
        class_star_mask = np.ones_like(nice_lya).astype(bool)
    else:
        class_star_mask = cat['class_star'] > 0.1
    
    # Update cat
    cat['nice_lya_0'] = np.copy(cat['nice_lya'])
    cat['nice_lya'] = nice_lya & color_mask & ml_mask & class_mask & class_star_mask
    cat['class_pred'] = class_pred

    if check_nice_z:
        nice_z = np.abs(z_Arr - cat['zspec']) < 0.115
        cat['nice_z'] = nice_z
    
    # Estimate L_lya, F_lya and EW0_lya
    cat = Lya_L_estimation(cat, cont_est, cont_err)

    return cat


def LumDist(z):
    '''
    Computes the luminosity distance for a given redshift.

    Parameters:
        z (float or array-like): Redshift value(s) for which to compute the luminosity distance.

    Returns:
        float or array-like: Luminosity distance(s) in centimeters.
    '''
    return cosmo.luminosity_distance(z).to(u.cm).value


def Lya_L_estimation(cat, cont_est, cont_est_err):
    '''
    Returns a catalog including L_lya and EW0_lya estimations
    '''
    mask_selected_NB = (cat['lya_NB'], np.arange(len(cat['lya_NB'])))
    nice_lya = cat['nice_lya_0']

    Flambda_lya = (
        cat['flx'][mask_selected_NB] - cont_est[mask_selected_NB]
    ) * fwhm_Arr[cat['lya_NB']]
    Flambda_lya_err = (
        cat['err'][mask_selected_NB] ** 2 + cont_est_err[mask_selected_NB] ** 2
    ) ** 0.5 * fwhm_Arr[cat['lya_NB']]

    # Flambda to zero if no selection
    Flambda_lya[~nice_lya] = 0.
    Flambda_lya_err[~nice_lya] = 0.

    dL = LumDist(cat['z_NB'])
    dL_err = (
        LumDist(lya_redshift(w_central[cat['lya_NB']]
                         + 0.5 * fwhm_Arr[cat['lya_NB']]))
        - LumDist(lya_redshift(w_central[cat['lya_NB']]))
    )

    z_NB_err = (lya_redshift(w_central[cat['lya_NB']]) + 0.5 * fwhm_Arr[cat['lya_NB']]
                - lya_redshift(w_central[cat['lya_NB']]) - 0.5 * fwhm_Arr[cat['lya_NB']]) * 0.5

    # L_lya to 99. if no selection
    L_lya = np.ones_like(Flambda_lya) * 99
    L_lya_err = np.zeros_like(Flambda_lya)

    L_lya[nice_lya] = np.log10(Flambda_lya[nice_lya]
                               * 4 * np.pi * dL[nice_lya]**2)
    L_lya_err[nice_lya] = ((dL[nice_lya] * Flambda_lya_err[nice_lya]) ** 2
                 + (2 * Flambda_lya[nice_lya] * dL_err[nice_lya]) ** 2
                ) ** 0.5 * 4 * np.pi * dL[nice_lya]


    EW0_lya = np.zeros_like(L_lya)
    EW0_lya_err = np.zeros_like(L_lya)
    EW0_lya[nice_lya] = Flambda_lya[nice_lya] / cont_est[mask_selected_NB][nice_lya]\
         / (1 + cat['z_NB'][nice_lya])
    EW0_lya_err[nice_lya] = (
        (1. / Flambda_lya[nice_lya] * Flambda_lya_err[nice_lya]) ** 2
        + (1. / cont_est[mask_selected_NB][nice_lya]
           * cont_est_err[mask_selected_NB][nice_lya]) ** 2
        + (1. / (1 + cat['z_NB'][nice_lya]) * z_NB_err[nice_lya]) ** 2
    ) ** 0.5 * EW0_lya[nice_lya]

    cat['L_lya'] = L_lya
    cat['L_lya_err'] = L_lya_err
    cat['EW0_lya'] = EW0_lya
    cat['EW0_lya_err'] = EW0_lya_err
    cat['F_lya'] = Flambda_lya
    cat['F_lya_err'] = Flambda_lya_err

    return cat