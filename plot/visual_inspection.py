import os
import sys
sys.path.insert(0, '..')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.gridspec as gridspec

import numpy as np

import pandas as pd

from load_paus_cat import load_paus_cat
from paus_utils import w_central, plot_PAUS_source, z_NB
from jpasLAEs.utils import flux_to_mag, mag_to_flux, rebin_1d_arr
from LAE_selection_method import estimate_continuum

from astropy.table import Table
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u

w_lya = 1215.67
w_CIV = 1549.48
w_CIII = 1908.734

cutouts_dir = '/home/alberto/almacen/PAUS_data/cutouts/out_cutouts'
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
        os.makedirs(f'{fig_save_dir}/with_spec', exist_ok=True)
        os.makedirs(f'{fig_save_dir}/no_spec', exist_ok=True)

        for sel_src, refid in enumerate(selection['ref_id']):
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

            fig = plt.figure(figsize=(16, 7))
            gs = gridspec.GridSpec(2, 6, figure=fig)

            #### Plot the P-spectra ####
            ax = fig.add_subplot(gs[:, :4])
            plot_PAUS_source(flx, err, set_ylim=True, ax=ax)
            ax.errorbar(w_central[:35], cont_est[:35, cat_src] * 1e17,
                        yerr=cont_err[:35, cat_src] * 1e17)

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

                rebin_factor = 5
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
            lya_NB = int(selection['lya_NB'][sel_src])
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
                
            # Signal to noise BBs
            SN_u = flx[-6] / err[-6]
            SN_g = flx[-5] / err[-5]
            SN_r = flx[-4] / err[-4]
            SN_i = flx[-3] / err[-3]

            text1 = (f'REF_ID: {selection["ref_id"][sel_src]}\n'
                    f'RA: {selection["RA"][sel_src]:0.5f}\n'
                    f'DEC: {selection["DEC"][sel_src]:0.5f}\n'
                    f'r_synth = {cat["r_mag"][cat_src]:0.1f}\n'
                    f'class_star = {cat["class_star"][cat_src]:0.2f}\n'
                    f'S/N(u, g, r, i) =\n{SN_u:0.1f}, {SN_g:0.1f}, {SN_r:0.1f}, {SN_i:0.1f}')

            text2 = (f'L_lya = {selection["L_lya_corr"][sel_src]:0.2f}\n'
                    f'EW0_lya = {selection["EW0_lya"][sel_src]:0.2f}' + r'\AA'
                    f'\npred_class = {ml_class}\n'
                    f'lya_NB = {lya_NB}\n'
                    f'z_phot = {selection["z_NB"][sel_src]:0.2f}')

            text3 = ('SDSS\n'
                    f'MJD: {selection["mjd"][sel_src]}\n'
                    f'fiber: {selection["fiber"][sel_src]}\n'
                    f'plate: {selection["plate"][sel_src]}\n'
                    f'z_spec = {selection["z_spec"][sel_src]:0.2f}')

            text_to_plot = [[3500, text1],
                            [5700, text2],
                            [7600, text3]]
            for [xpos, txt] in text_to_plot:
                ax.text(xpos, ypos, txt, fontsize=12)

            
            # Draw CIV and CIII positions
            for linename, w in {'LyC': 912., 'CIV': w_CIV, 'CIII': w_CIII}.items():
                this_w_obs = w * (1 + selection['z_NB'][sel_src])
                if this_w_obs > 4000 and this_w_obs < 9000:
                    ax.axvline(this_w_obs,
                               ls=':', color='dimgray')
                    ax.text(this_w_obs + 10, ax.get_ylim()[0] * 1.1,
                            linename)
            # Lya NB
            ax.axvline(w_lya * (1 + z_NB(selection['lya_NB'][sel_src])),
                    ls=':', color='dimgray')

            #########################################
            ax_NB_img = fig.add_subplot(gs[0, 4])
            ax_cont_img = fig.add_subplot(gs[0, 5])
            ax_r_img = fig.add_subplot(gs[1, 4])
            ax_r_wht = fig.add_subplot(gs[1, 5])

            RA = selection['RA'][sel_src]
            DEC = selection['DEC'][sel_src]

            # Load NB cutout
            cutout_square_size = 10 * u.arcsec
            try:
                lya_NB = int(selection['lya_NB'][sel_src])
                ref_id = int(selection['ref_id'][sel_src])
                NB_int_wav = NB_wav_Arr[lya_NB]

                cutout_path = f'{cutouts_dir}/NB{NB_int_wav}/coadd_cutout_{ref_id}.fits'
                cutout = fits.open(cutout_path)
                img = cutout[0].data
                coords = SkyCoord(RA, DEC, unit='deg')
                wcs = WCS(cutout[0])
                cutout_img = Cutout2D(img, coords, size=cutout_square_size,
                                  wcs=wcs, mode='partial', fill_value=0.).data
                [vmin, vmax] = ZScaleInterval(contrast=0.1).get_limits(cutout_img.flatten())
                ax_NB_img.imshow(cutout_img, vmin=vmin, vmax=vmax,
                                rasterized=True, interpolation='nearest')

                r_synth_folder = 'NB575_585_595_605_615_625_635_645_655_665_675_685_695_705_715'
                cutout_path = f'{cutouts_dir}/{r_synth_folder}/coadd_cutout_{ref_id}.fits'
                cutout = fits.open(cutout_path)
                img = cutout[0].data
                coords = SkyCoord(RA, DEC, unit='deg')
                wcs = WCS(cutout[0])
                cutout_img = Cutout2D(img, coords, size=cutout_square_size,
                                  wcs=wcs, mode='partial', fill_value=0.).data
                [vmin, vmax] = ZScaleInterval(contrast=0.1).get_limits(cutout_img.flatten())
                ax_r_img.imshow(cutout_img, vmin=vmin, vmax=vmax,
                                rasterized=True, interpolation='nearest')

                
                # Where to look for the continuum cutout
                NB_Arr_cont = np.concatenate(
                    [np.arange(max(0, lya_NB - 6), max(0, lya_NB - 1)),
                    np.arange(min(lya_NB + 2, 39), min(lya_NB + 6 + 1, 39))]       
                )
                NB_wav_Arr_cont = NB_wav_Arr[NB_Arr_cont]
                save_coadds_to = f'NB' + '_'.join(NB_wav_Arr_cont.astype(str))

                cutout_path = f'{cutouts_dir}/{save_coadds_to}/coadd_cutout_{ref_id}.fits'
                cutout = fits.open(cutout_path)
                img = cutout[0].data
                coords = SkyCoord(RA, DEC, unit='deg')
                wcs = WCS(cutout[0])
                cutout_img = Cutout2D(img, coords, size=cutout_square_size,
                                  wcs=wcs, mode='partial', fill_value=0.).data
                [vmin, vmax] = ZScaleInterval(contrast=0.1).get_limits(cutout_img.flatten())
                ax_cont_img.imshow(cutout_img, vmin=vmin, vmax=vmax,
                                rasterized=True, interpolation='nearest')

                ax_NB_img.set_title('Lya NB')
                ax_cont_img.set_title('Continuum')
                ax_r_img.set_title('r_synth')


                for ax in [ax_NB_img, ax_cont_img, ax_r_img, ax_r_wht]:
                    ax.set_yticks([])
                    ax.set_xticks([])

                    # Add circumference showing aperture 3arcsec diameter
                    aper_r_px = 1.5 / 0.2645
                    circ1 = plt.Circle(np.array(cutout_img.shape).T / 2,
                                    radius=aper_r_px, ec='r', fc='none')
                    circ2 = plt.Circle(np.array(cutout_img.shape).T / 2,
                                    radius=aper_r_px, ec='r', fc='none')
                    ax.add_patch(circ1)
                    ax.add_patch(circ2)
            except FileNotFoundError as err:
                print(err)
            except:
                raise

            #########################################

            savename = f'{field_name}_{sel_src}-{selection["ref_id"][sel_src]}.png'
            if spec_bool:
                subfolder = 'with_spec'
            else:
                subfolder = 'no_spec'
            fig.savefig(f'{fig_save_dir}/{subfolder}/{savename}',
                        pad_inches=0.1, bbox_inches='tight', facecolor='w')
            plt.close()
