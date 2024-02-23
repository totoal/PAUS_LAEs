import os
import sys
sys.path.insert(0, '..')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})

import numpy as np

import pandas as pd

from load_paus_cat import load_paus_cat
from paus_utils import w_central, plot_PAUS_source
from jpasLAEs.utils import flux_to_mag, mag_to_flux, rebin_1d_arr

from astropy.table import Table

w_lya = 1215.67
w_CIV = 1549.48
w_CIII = 1908.734

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

        # Directory of the spectra .fits files
        fits_dir = '/home/alberto/almacen/SDSS_spectra_fits/DR16/QSO'
        
        # Dir to save the figures
        fig_save_dir = '/home/alberto/almacen/PAUS_data/candidates'
        # Make dirs if they don't exist
        os.makedirs(f'{fig_save_dir}/with_spec', exist_ok=True)
        os.makedirs(f'{fig_save_dir}/no_spec', exist_ok=True)

        for sel_src, refid in enumerate(selection['ref_id']):
            try:
                cat_src = np.where(refid == cat['ref_id'])[0][0]
            except:
                print(f'{refid=} not found in {field_name}.')
                continue
            flx = cat['flx'][:, cat_src]
            err = cat['err'][:, cat_src]
            r_synth_mag = synth_BB_flx[cat_src]
            cat['r_mag'] = flux_to_mag(synth_BB_flx, w_central[-4])

            fig, axes = plt.subplots(1, 3, figsize=(16, 4),
                                   width_ratios=[0.7, 0.15, 0.15])

            #### Plot the P-spectra ####
            ax = axes[0]
            plot_PAUS_source(flx, err, set_ylim=True, ax=ax)

            #### Mark the selected NB ####
            ax.axvline(w_lya * (selection['z_NB'][sel_src] + 1),
                    c='r', ls='--', lw=2)

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

                rebin_factor = 2
                spec_flx_sdss_rb, _ = rebin_1d_arr(spec_flx_sdss,
                                                spec_w_sdss,
                                                rebin_factor)

                spec_w_sdss_rb = np.empty_like(spec_flx_sdss_rb)
                for i in range(len(spec_flx_sdss_rb)):
                    spec_w_sdss_rb[i] = spec_w_sdss[i * rebin_factor]

                ax.plot(spec_w_sdss_rb, spec_flx_sdss_rb,
                        c='dimgray', zorder=-99, alpha=0.7)

            #### Info text ####
            ypos = ax.get_ylim()[1] * 1.05

            # Define the ML predicted class
            cl_num = selection["class_pred"][sel_src]
            if cl_num == 1:
                ml_class = 'QSO Cont.'
            elif cl_num == 2:
                ml_class = 'LAE'
            elif cl_num == 4:
                ml_class = 'Low-z Gal.'
            elif cl_num == 5 or cl_num < 0:
                ml_class = '?'
            else:
                raise Exception(f'I don\'t know this class: {cl_num}')

            text1 = (f'REF_ID: {selection["ref_id"][sel_src]}\n'
                    f'RA: {selection["RA"][sel_src]:0.2f}\n'
                    f'DEC: {selection["DEC"][sel_src]:0.2f}\n'
                    f'r_synth = {cat["r_mag"][cat_src]:0.1f}\n'
                    f'star-galaxy = {cat["sg_flag"][cat_src]}')

            text2 = (f'L_lya = {selection["L_lya_corr"][sel_src]:0.2f}\n'
                    f'EW0_lya = {selection["EW0_lya"][sel_src]:0.2f}' + r'\AA'
                    f'\npred_class = {ml_class}\n'
                    f'z_NB = {selection["z_NB"][sel_src]:0.2f}')

            text3 = ('SDSS\n'
                    f'MJD: {selection["mjd"][sel_src]}\n'
                    f'fiber: {selection["fiber"][sel_src]}\n'
                    f'plate: {selection["plate"][sel_src]}\n'
                    f'L_lya = {selection["L_lya_SDSS"][sel_src]:0.2f}\n'
                    f'z_spec = {selection["z_best"][sel_src]:0.2f}')

            text_to_plot = [[3500, text1],
                            [5500, text2],
                            [7400, text3]]
            for [xpos, txt] in text_to_plot:
                ax.text(xpos, ypos, txt, fontsize=12)

            
            # Draw CIV and CIII positions
            for w in [w_CIV, w_CIII]:
                ax.axvline(w * (1 + selection['z_NB'][sel_src]),
                        ls=':', color='dimgray')

            #########################################

            savename = f'{field_name}_{sel_src}-{selection["ref_id"][sel_src]}.png'
            if spec_bool:
                subfolder = 'with_spec'
            else:
                subfolder = 'no_spec'
            fig.savefig(f'{fig_save_dir}/{subfolder}/{savename}',
                        pad_inches=0.1, bbox_inches='tight', facecolor='w')
            plt.close()

        break