import pandas as pd
import xarray as xr

from scipy.interpolate import RectBivariateSpline


# Load mask
field_name = 'W2'

mask_path_dict = {
    'W1': 'coadd_W1_1015_40NB_5_arcs.nc',
    'W2': 'coadd_W2_1057_40NB_external_16412.nc',
    'W3': 'coadd_W3_1012_40NB_5_arcs.nc'
}

mask_path = f'/home/alberto/almacen/PAUS_data/masks/{mask_path_dict[field_name]}'
field_mask = xr.open_dataarray(mask_path)


# Interpolate random RA,DEC points to the mask
def radec_mask_interpolator(ra_Arr, dec_Arr, mask):
    interpolator = RectBivariateSpline(mask.ra, mask.dec, mask.data.T)
    mask_values = interpolator(ra_Arr, dec_Arr, grid=False)

    masked_radec = (mask_values > 0.9)

    return masked_radec


# Load PAUS
path_to_paus_cat = f'/home/alberto/almacen/PAUS_data/catalogs/PAUS_3arcsec_{field_name}_unmask_extinction_corrected.pq'
cat = pd.read_parquet(path_to_paus_cat)

mask_cat = radec_mask_interpolator(cat['alpha_j2000'], cat['delta_j2000'], field_mask)

cat[mask_cat].to_parquet(f'/home/alberto/almacen/PAUS_data/catalogs/PAUS_3arcsec_{field_name}_extinction_corrected.pq')