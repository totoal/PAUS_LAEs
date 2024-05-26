import os
import sys
sys.path.insert(0, '..')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 13})

import numpy as np

import pandas as pd

from load_paus_cat import load_paus_cat
from paus_utils import w_central, plot_PAUS_source
from jpasLAEs.utils import flux_to_mag, mag_to_flux, rebin_1d_arr
from LAE_selection_method import estimate_continuum

from astropy.table import Table

w_lya = 1215.67
w_CIV = 1549.48
w_CIII = 1908.734
w_lyb = 1025.72
w_SiIV = 1399.8

cutouts_dir = '/home/alberto/almacen/PAUS_data/cutouts'
NB_wav_Arr = np.arange(455, 855, 10).astype(int)

def nanomaggie_to_flux(nmagg, wavelength):
    mAB = -2.5 * np.log10(nmagg * 1e-9)
    flx = mag_to_flux(mAB, wavelength)
    return flx


if __name__ == '__main__':
    # Load the PAUS cat

    for field_name in  ['W3', 'W2', 'W1']:
        path_to_cat = [f'/home/alberto/almacen/PAUS_data/catalogs/PAUS_3arcsec_{field_name}_extinction_corrected.pq']
        cat = load_paus_cat(path_to_cat)

        mask_NB_number = (cat['NB_number'] > 39)
        cat['flx'] = cat['flx'][:, mask_NB_number]
        cat['err'] = cat['err'][:, mask_NB_number]
        cat['NB_mask'] = cat['NB_mask'][:, mask_NB_number]
        for key in cat.keys():
            if key in ['flx', 'err', 'NB_mask', 'area']:
                continue
            cat[key] = cat[key][mask_NB_number]

        stack_nb_ids = np.arange(12, 26 + 1)
        synth_BB_flx = np.average(cat['flx'][stack_nb_ids],
                                weights=cat['err'][stack_nb_ids] ** -2,
                                axis=0)

        # Load selection
        save_to_path = '/home/alberto/almacen/PAUS_data/catalogs/'
        selection = pd.read_csv(f'{save_to_path}/LAE_selection_vi.csv')
        selection = selection[selection['nice_lya']].reset_index(drop=True)
        print(f'{len(selection)=}')

        cont_est, cont_err = estimate_continuum(cat['flx'], cat['err'],
                                                IGM_T_correct=True, N_nb=6)

        # Directory of the spectra .fits files
        fits_dir = '/home/alberto/almacen/SDSS_spectra_fits/DR16/QSO'
        
        # Dir to save the figures
        fig_save_dir = '/home/alberto/almacen/PAUS_data/candidates'
        # Make dirs if they don't exist
        os.makedirs(f'{fig_save_dir}/paper_examples', exist_ok=True)

        sel_to_plot = selection['ref_id']

        for sel_src, refid in enumerate(sel_to_plot):
            try:
                cat_src = np.where((refid == cat['ref_id'])
                                   & (selection['field'][sel_src] == field_name))[0][0]
            except:
                print(f'{refid=} not found in {field_name}.')
                continue
            flx = cat['flx'][:, cat_src]
            err = cat['err'][:, cat_src]
            r_synth_mag = synth_BB_flx[cat_src]
            cat['r_mag'] = flux_to_mag(synth_BB_flx, w_central[-4])

            fig, ax = plt.subplots(figsize=(6, 2))

            #### Plot the P-spectra ####
            plot_PAUS_source(flx, err, ax=ax, markersize=5,
                             set_ylim=False, fs=12, plot_BBs=False)
            
            # Plot continuum only in the Lya position
            lya_NB = selection['lya_NB'][sel_src]
            ax.errorbar(w_central[lya_NB], cont_est[lya_NB, cat_src] * 1e17,
                        yerr=cont_err[lya_NB, cat_src] * 1e17,
                        fmt='s', ms=7, c='k')

            data_max = np.max(flx[err > 0]) * 1e17
            data_min = np.min(flx[err > 0]) * 1e17
            y_max = (data_max - data_min) * 0.3 + data_max
            y_min = data_min - (data_max - data_min) * 0.05


            #### Plot SDSS spectrum if available ####
            plate = selection['plate'][sel_src]
            mjd = selection['mjd'][sel_src]
            fiber = selection['fiber'][sel_src]
            spec_name = f'spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits'
            print(spec_name)
            spec_bool = True
            try:
                spec_sdss = Table.read(f'{fits_dir}/{spec_name}', hdu=1, format='fits')
                sdss_bbs = Table.read(f'{fits_dir}/{spec_name}', hdu=2, format='fits')['SPECTROFLUX']
                r_band_sdss = nanomaggie_to_flux(np.array(sdss_bbs)[0][2], 6250)
            except:
                print('Couldn\'t load the SDSS spectrum.')
                spec_bool = False
                # plt.close()
                # continue # NOTE: By now, only plotting sources with SDSS spectrum

            if spec_bool:
                # Normalizing factor:
                norm = r_synth_mag / r_band_sdss
                spec_flx_sdss = spec_sdss['FLUX'] * norm
                spec_w_sdss = 10 ** spec_sdss['LOGLAM']

                rebin_factor = 10
                spec_flx_sdss_rb, _ = rebin_1d_arr(spec_flx_sdss,
                                                spec_w_sdss,
                                                rebin_factor)

                spec_w_sdss_rb = np.empty_like(spec_flx_sdss_rb)
                for i in range(len(spec_flx_sdss_rb)):
                    spec_w_sdss_rb[i] = spec_w_sdss[i * rebin_factor]

                ax.plot(spec_w_sdss_rb, spec_flx_sdss_rb,
                        c='dimgray', zorder=-99, alpha=0.7)

            # Draw CIV and CIII positions
            line_dict = {r'Ly$\alpha$': 1215.67, 'LyC': 912., 'CIV': w_CIV, 'CIII': w_CIII,
                         r'Ly$\beta$': w_lyb, 'SiIV': w_SiIV}
            for linename, w in line_dict.items():
                this_w_obs = w * (1 + selection['z_NB'][sel_src])
                if this_w_obs > 4000 and this_w_obs < 9000:
                    ax.axvline(this_w_obs,
                               ls=':', color='dimgray')
                    ax.text(this_w_obs + 10, ax.get_ylim()[0] + 0.1,
                            linename, verticalalignment='bottom', 
                            fontsize=10)


            # Mark the zero flux level
            ax.axhline(0, color='k', zorder=-9999)


            ## Info text
            text = (f'field: {selection["field"][sel_src]}\n ID: {selection['ref_id'][sel_src]}\n'
                    r'$z_{\rm phot} = $ ' + f'{selection["z_NB"][sel_src]:0.2f}\n'
                    r'$\log L_{{\rm Ly}\alpha} =$ ' + f'{selection["L_lya_corr"][sel_src]:0.2f}\n'
                    r'$M_{\rm UV}=$ ' + f'{selection["M_UV"][sel_src]:0.2f}')

            ax.set_ylim(y_min, y_max)
            ax.set_xlim(3600, 10900)
                

            ypos = ax.get_ylim()[1] * 0.97
            xpos = 8500
            ax.text(xpos, ypos, text,
                    verticalalignment='top', horizontalalignment='left',
                    fontsize=12)


            ax.tick_params(direction='in', which='both')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)

                #########################################

            savename = f'{sel_src+1:04d}-{field_name}_{selection["ref_id"][sel_src]}.pdf'
            subfolder = 'paper_examples'
            fig.savefig(f'{fig_save_dir}/{subfolder}/{savename}',
                        pad_inches=0.1, bbox_inches='tight', facecolor='w')

            savename = f'{sel_src+1:04d}-{field_name}_{selection["ref_id"][sel_src]}.png'
            subfolder = 'paper_examples'
            fig.savefig(f'{fig_save_dir}/{subfolder}/{savename}',
                        pad_inches=0.1, bbox_inches='tight', facecolor='w')
            plt.close()

    print('\n\nDone.\n')
