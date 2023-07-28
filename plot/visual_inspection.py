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

def nanomaggie_to_flux(nmagg, wavelength):
    mAB = -2.5 * np.log10(nmagg * 1e-9)
    flx = mag_to_flux(mAB, wavelength)
    return flx


if __name__ == '__main__':
    # Load the PAUS cat

    field_name = 'W3'
    path_to_cat = [f'/home/alberto/almacen/PAUS_data/catalogs/PAUS_3arcsec_{field_name}.csv']
    cat = load_paus_cat(path_to_cat)

    mask_NB_number = (cat['NB_number'] > 39)
    cat['flx'] = cat['flx'][:, mask_NB_number]
    cat['err'] = cat['err'][:, mask_NB_number]
    cat['NB_mask'] = cat['NB_mask'][:, mask_NB_number]
    for key in cat.keys():
        if key in ['flx', 'err', 'NB_mask', 'area']:
            continue
        cat[key] = cat[key][mask_NB_number]

    stack_nb_ids = np.arange(12, 16 + 1)
    synth_BB_flx = np.average(cat['flx'][stack_nb_ids],
                            weights=cat['err'][stack_nb_ids] ** -2,
                            axis=0)

    # Load selection
    save_to_path = '/home/alberto/almacen/PAUS_data/catalogs/'
    selection = pd.read_csv(f'{save_to_path}/LAE_selection.tsv', sep='\t')

    # Directory of the spectra .fits files
    fits_dir = '/home/alberto/almacen/SDSS_spectra_fits/DR16/QSO'

    for sel_src, refid in enumerate(selection['ref_id']):
        cat_src = np.where(refid == cat['ref_id'])[0][0]
        flx = cat['flx'][:, cat_src]
        err = cat['err'][:, cat_src]
        r_synth_mag = synth_BB_flx[cat_src]

        fig, ax = plt.subplots(figsize=(12, 7))

        #### Plot the P-spectra ####
        plot_PAUS_source(flx, err, ax=ax, set_ylim=False)

        #### Mark the selected NB ####
        ax.axvline(w_lya * (selection['z_NB'][sel_src] + 1),
                   c='r', ls='--', lw=2)

        #### Plot SDSS spectrum if available ####
        plate = selection['plate'][sel_src]
        mjd = selection['mjd'][sel_src]
        fiber = selection['fiberid'][sel_src]
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

        #########################################
        plt.show()

        break