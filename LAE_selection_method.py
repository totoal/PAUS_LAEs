import numpy as np

from paus_utils import *
from jpasLAEs.utils import flux_to_mag

# Line rest-frame wavelengths (Angstroms)
w_lya = 1215.67
w_lyb = 1025.7220
w_lya = 1215.67
w_SiIV = 1397.61
w_CIV = 1549.48
w_CIII = 1908.73
w_MgII = 2799.12


def IGM_TRANSMISSION(w_Arr, A=-0.001845, B=3.924):
    '''
    Returns the IGM transmission associated to the Lya Break.
    '''
    return np.exp(A * (w_Arr / w_lya)**B)

def estimate_continuum(NB_flx, NB_err, N_nb=7, IGM_T_correct=True,
                       only_right=False, N_nb_min=0, N_nb_max=30):
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

        if (N_nb <= nb_idx) & (nb_idx < (NB_flx.shape[0] - N_nb)) and not only_right:
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

        if nb_idx >= (NB_flx.shape[0] - N_nb):
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
        cont_err[nb_idx] = np.sum(NBs_errs ** -2, axis=0) ** -0.5

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


def identify_lines(line_Arr, qso_flx, cont_flx, nb_min=0, first=False):
    '''
    Returns a list of N lists with the index positions of the lines.

    Input: 
    line_Arr: Bool array of 3sigma detections in sources. Dim N_filters x N_sources
    qso_flx:  Flambda data
    nb_min
    '''
    N_fil, N_src = line_Arr.shape
    line_list = []
    line_len_list = []
    line_cont_list = []

    for src in range(N_src):
        fil = 0
        this_src_lines = []  # The list of lines
        this_cont_lines = []  # The list of continuum indices of lines
        this_src_line_flx = []  # The list of lengths of this src lines

        while fil < N_fil:
            this_line = []  # The list of contiguous indices of this line
            while ~line_Arr[fil, src]:
                fil += 1
                if fil == N_fil - 1:
                    break
            if fil == N_fil - 1:
                break
            while line_Arr[fil, src]:
                this_line.append(fil)
                fil += 1
                if fil == N_fil - 1:
                    break
            if fil == N_fil - 1:
                break

            aux = -len(this_line) + nb_min + fil

            if first:  # If first=True, append continuum index to list
                this_cont_lines.append(
                    np.average(
                        np.array(this_line),
                        weights=qso_flx[np.array(this_line), src] ** 2
                    )
                )
            # Append index of the max flux of this line to the list
            this_src_lines.append(
                np.argmax(qso_flx[np.array(this_line) + nb_min, src]) + aux
            )
            this_src_line_flx.append(
                qso_flx[np.array(this_line) + nb_min, src].sum())

        if first:
            try:
                idx = np.argmax(
                    np.array(this_src_line_flx)
                    - cont_flx[np.array(this_src_lines), src]
                )

                line_list.append(this_src_lines[idx])
                line_len_list.append(this_src_lines)
                line_cont_list.append(this_cont_lines[idx])
            except:
                line_list.append(-1)
                line_len_list.append([-1])
                line_cont_list.append(-1)

        if not first:
            line_list.append(this_src_lines)

    if first:
        return np.array(line_list)
    else:
        return line_list


def nice_lya_select(lya_lines, other_lines, pm_flx, z_Arr, mask=None):
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
    i = flux_to_mag(pm_flx[-3], w_central[-3])
    r = flux_to_mag(pm_flx[-4], w_central[-4])
    g = flux_to_mag(pm_flx[-5], w_central[-5])
    gr = g - r
    ri = r - i
    # For z > 3
    color_aux1 = (ri < 0.6) & (gr < 1.5)
    # For z < 3
    color_aux2 = (ri < 0.6) & (gr < 0.6)

    color_mask = np.ones_like(color_aux2).astype(bool)
    mlines_mask = np.ones_like(color_aux2).astype(bool)

    for src in np.where(np.array(lya_lines) != -1)[0]:
        z_src = z_Arr[src]

        w_obs_lya = (1 + z_src) * w_lya
        w_obs_lyb = (1 + z_src) * w_lyb
        w_obs_SiIV = (1 + z_src) * w_SiIV
        w_obs_CIV = (1 + z_src) * w_CIV
        w_obs_CIII = (1 + z_src) * w_CIII
        w_obs_MgII = (1 + z_src) * w_MgII

        for l in other_lines[src]:
            # Ignore very red and very blue NBs
            if (l > 50) | (l < 1):
                continue

            w_obs_l = w_central[l]
            fwhm = fwhm_Arr[l]

            good_l = (
                (np.abs(w_obs_l - w_obs_lya) < fwhm * 1.)
                | (np.abs(w_obs_l - w_obs_lyb) < fwhm * 1.)
                | (np.abs(w_obs_l - w_obs_SiIV) < fwhm * 1.)
                | (np.abs(w_obs_l - w_obs_CIV) < fwhm * 1.)
                | (np.abs(w_obs_l - w_obs_CIII) < fwhm * 1.)
                | (np.abs(w_obs_l - w_obs_MgII) < fwhm * 1.)
                | (w_obs_l > w_obs_MgII + fwhm)
            )

            if ~good_l:
                mlines_mask[src] = False
                break

        if len(other_lines[src]) > 1:
            pass
        else:
            if z_src > 3.:
                good_colors = color_aux2[src]
            else:
                good_colors = color_aux1[src]

            if ~good_colors:
                color_mask[src] = False

    # Define nice_lya
    nice_lya = (lya_lines >= 0) & color_mask & mlines_mask
    if mask is not None:
        nice_lya = nice_lya & mask

    return nice_lya, color_mask, mlines_mask