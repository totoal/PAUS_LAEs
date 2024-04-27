from load_paus_cat import load_paus_cat
from paus_utils import PAUS_monochromatic_Mag, z_NB

from astropy.io import fits

import numpy as np


vi_cat = fits.open('/home/alberto/almacen/PAUS_data/catalogs/LAE_selection_VI_hiZ.fits')[1].data

MUV = []
MUV_err = []

for field_name in ['W1', 'W2', 'W3']:
    path_to_cat = [f'/home/alberto/almacen/PAUS_data/catalogs/PAUS_3arcsec_{field_name}_extinction_corrected.pq']
    cat = load_paus_cat(path_to_cat)
    print(f'{sum(cat["NB_number"] > 39)=}')

    mask = vi_cat['is_hiZ_LAE'] & (vi_cat['field'] == field_name)

    lya_NB = np.array(vi_cat['lya_NB'][mask])
    lya_NB[vi_cat['lya_NB_VI'][mask] > 0] = vi_cat['lya_NB_VI'][mask][vi_cat['lya_NB_VI'][mask] > 0]

    redshifts = z_NB(lya_NB)
    print(redshifts)

    LAE_vi_IDs = np.array(vi_cat['ref_id'][mask])

    cat['z_NB'] = np.ones(cat['flx'].shape[1]) * -1
    cat['nice_lya'] = np.zeros(cat['flx'].shape[1]).astype(bool)

    where_LAEs_in_cat = np.empty_like(LAE_vi_IDs).astype(int)
    for i, thisid in enumerate(LAE_vi_IDs):
        where_LAEs_in_cat[i] = np.where(thisid == cat['ref_id'])[0][0]
        cat['z_NB'][where_LAEs_in_cat[i]] = redshifts[i]
        cat['nice_lya'][where_LAEs_in_cat[i]] = True

    this_MUV, this_MUV_err = PAUS_monochromatic_Mag(cat)
    MUV.append(this_MUV)
    MUV_err.append(this_MUV_err)

print(np.concatenate(MUV)[np.concatenate(MUV) < 0])
print(np.concatenate(MUV_err)[np.concatenate(MUV) < 0])