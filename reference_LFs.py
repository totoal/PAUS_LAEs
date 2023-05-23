'''
This module loads some Lya LFs in the literature.
'''

import numpy as np

from astropy.cosmology import LambdaCDM
from astropy.cosmology import Planck18 as my_cosmo

import pandas as pd


# First a couple of functions to convert the cosmology of each LF to Planck18

def convert_cosmology_luminosity(log_L_Arr, redshift, this_H0, this_Om0, this_Ode0):
    this_cosmo = LambdaCDM(H0=this_H0, Om0=this_Om0, Ode0=this_Ode0)
    
    my_dL_Arr = my_cosmo.luminosity_distance(redshift).value
    this_dL_Arr = this_cosmo.luminosity_distance(redshift).value


    return log_L_Arr - np.log10((my_dL_Arr / this_dL_Arr) ** 2)

def convert_cosmology_Phi(Phi_Arr, redshift, new_H0, new_Om0, new_Ode0):
    this_cosmo = LambdaCDM(H0=new_H0, Om0=new_Om0, Ode0=new_Ode0)

    my_dV = my_cosmo.differential_comoving_volume(redshift).value
    this_dV = this_cosmo.differential_comoving_volume(redshift).value

    return Phi_Arr / my_dV * this_dV

# Load reference LFs
pathname = '/home/alberto/almacen/literature_LF_data'

# Blanc 2011 (z=1.9--3.8)
filename = f'{pathname}/blanc2011_allz.txt'
df = pd.read_table(filename, delimiter='\t')
b11 = {
    'logL': df['LogLya'],
    'Phi': df['phi'],
    'yerr_plus': df['phi_err_up'] - df['phi'],
    'yerr_minus': df['phi'] - df['phi_err_down'],
    'z': (3.8 + 1.9) * 0.5,
    'label': 'Blanc 2011 ($z=1.9-3.8$)',
    'fmt': 'h',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Gronwall 2007 (z=3.1)
filename = f'{pathname}/gronwall2007_z3.1.txt'
df = pd.read_table(filename, delimiter='\t')
g07 = {
    'logL': df['LogLya'],
    'Phi': df['phi'],
    'yerr_plus': df['phi_err_up'] - df['phi'],
    'yerr_minus': df['phi'] - df['phi_err_down'],
    'z': 3.1,
    'label': 'Gronwall 2007 ($z=3.1$)',
    'fmt': 'x',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Konno 2016 (z=2.2)
filename = f'{pathname}/konno2016_z2.2.txt'
df = pd.read_table(filename, delimiter=',')
k16 = {
    'logL': df['LogLya'],
    'Phi': df['phi'],
    'yerr_plus': df['phi_err_down'] - df['phi'],
    'yerr_minus': df['phi'] - df['phi_err_up'],
    'z': 2.2,
    'label': 'Konno 2016 ($z=2.2$)',
    'fmt': '^',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Matthee 2017 (z=2.2)
filename = f'{pathname}/matthee2017_z2.2.txt'
df = pd.read_table(filename, delimiter='\t')
m17a = {
    'logL': df['LogLya'],
    'Phi': df['phi'],
    'yerr_plus': df['phi_err_up'] - df['phi'],
    'yerr_minus': df['phi'] - df['phi_err_down'],
    'z': 2.2,
    'label': 'Matthee 2017 ($z=2.2$)',
    'fmt': '*',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Matthee 2017 (z=2.4)
filename = f'{pathname}/matthee2017_z2.4.txt'
df = pd.read_table(filename, delimiter='\t')
m17b = {
    'logL': df['LogLya'],
    'Phi': df['phi'],
    'yerr_plus': df['phi_err_up'] - df['phi'],
    'yerr_minus': df['phi'] - df['phi_err_down'],
    'z': 2.4,
    'label': 'Matthee 2017 ($z=2.4$)',
    'fmt': '*',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Ouchi 2008 (z=3.1)
filename = f'{pathname}/ouchi2008_z3.1.txt'
df = pd.read_table(filename, delimiter='\t')
u08 = {
    'logL': df['LogLya'],
    'Phi': df['phi'],
    'yerr_plus': df['phi_err_up'] - df['phi'],
    'yerr_minus': df['phi'] - df['phi_err_down'],
    'z': 3.1,
    'label': 'Ouchi 2008 ($z=3.1$)',
    'fmt': 'o',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Sobral 2016 (z=2.2)
filename = f'{pathname}/sobral2016_z2.2.txt'
df = pd.read_table(filename, delimiter='\t')
s16 = {
    'logL': df['LogLya'],
    'Phi': df['phi'],
    'yerr_plus': df['phi_err_up'] - df['phi'],
    'yerr_minus': df['phi'] - df['phi_err_down'],
    'z': 2.2,
    'label': 'Sobral 2016 ($z=2.2$)',
    'fmt': 'D',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Sobral 2017 (z=2.2)
filename = f'{pathname}/sobral2017_z2.2.txt'
df = pd.read_table(filename, delimiter='\t')
s17 = {
    'logL': df['LogLya'],
    'Phi': df['phi'],
    'yerr_plus': df['phi_err_up'] - df['phi'],
    'yerr_minus': df['phi'] - df['phi_err_down'],
    'z': 2.2,
    'label': 'Sobral 2017 ($z=2.2$)',
    'fmt': 'D',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Sobral 2018 (z=2.5)
filename = f'{pathname}/sobral2018_z2.5.txt'
df = pd.read_table(filename, delimiter='\t')
s18a = {
    'logL': df['LogLya'],
    'Phi': df['phi'],
    'yerr_plus': df['phi_err_up'] - df['phi'],
    'yerr_minus': df['phi'] - df['phi_err_down'],
    'z': 2.5,
    'label': 'Sobral 2018 ($z=2.5$)',
    'fmt': 'D',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Sobral 2018 (z=3.0)
filename = f'{pathname}/sobral2018_z3.0.txt'
df = pd.read_table(filename, delimiter='\t')
s18b = {
    'logL': df['LogLya'],
    'Phi': df['phi'],
    'yerr_plus': df['phi_err_up'] - df['phi'],
    'yerr_minus': df['phi'] - df['phi_err_down'],
    'z': 3.0,
    'label': 'Sobral 2018 ($z=3.0$)',
    'fmt': 'D',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Sobral 2018 (z=3.2)
filename = f'{pathname}/sobral2018_z3.2.txt'
df = pd.read_table(filename, delimiter='\t')
s18c = {
    'logL': df['LogLya'],
    'Phi': df['phi'],
    'yerr_plus': df['phi_err_up'] - df['phi'],
    'yerr_minus': df['phi'] - df['phi_err_down'],
    'z': 3.2,
    'label': 'Sobral 2018 ($z=3.2$)',
    'fmt': 'D',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Sobral 2018 (z=2.8)
filename = f'{pathname}/sobral2018_z2.8.txt'
df = pd.read_table(filename, delimiter=',')
s18d = {
    'logL': df['LogLya'],
    'Phi': 10 ** (df['phi']),
    'yerr_plus': 10 ** df['phi_err_up'] - 10 ** df['phi'],
    'yerr_minus': 10 ** df['phi'] - 10 ** df['phi_err_down'],
    'z': 2.8,
    'label': 'Sobral 2018 ($z=2.8$)',
    'fmt': 'D',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Sobral 2018 (z=3.3)
filename = f'{pathname}/sobral2018_z3.3.txt'
df = pd.read_table(filename, delimiter=',')
s18e = {
    'logL': df['LogLya'],
    'Phi': 10 ** (df['phi']),
    'yerr_plus': 10 ** df['phi_err_up'] - 10 ** df['phi'],
    'yerr_minus': 10 ** df['phi'] - 10 ** df['phi_err_down'],
    'z': 3.3,
    'label': 'Sobral 2018 ($z=3.3$)',
    'fmt': 'D',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Sobral 2018 (z=3.7)
filename = f'{pathname}/sobral2018_z3.7.txt'
df = pd.read_table(filename, delimiter=',')
s18f = {
    'logL': df['LogLya'],
    'Phi': 10 ** (df['phi']),
    'yerr_plus': 10 ** df['phi_err_up'] - 10 ** df['phi'],
    'yerr_minus': 10 ** df['phi'] - 10 ** df['phi_err_down'],
    'z': 3.7,
    'label': 'Sobral 2018 ($z=3.7$)',
    'fmt': 'D',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}


# Spinoso 2020
fnam = '/home/alberto/almacen/literature_LF_data/LF_data_SpinosoEtAl2020/20200518_J0395_SNR5_LFdata.txt'
loglya, ModLF, ModLF_pc16, ModLF_pc84 = np.genfromtxt(fnam, skip_header=3, usecols=(0,6,7,8), unpack=True)
dLogL = loglya[1] - loglya[0]
mask = ModLF_pc16 > 0
snr = np.zeros_like(ModLF)
snr[mask] = ModLF[mask] / ModLF_pc16[mask]
ds20_z225 = {
    'logL': loglya[snr > 1],
    'Phi': ModLF[snr > 1],
    'yerr_plus': ModLF_pc84[snr > 1],
    'yerr_minus': ModLF_pc16[snr > 1],
    'z': 2.25,
    'label': 'Spinoso 2020 ($z=2.25$)',
    'fmt': 'X',
    'H0': 67.3,
    'Om0': 0.315,
    'Ode0': 0.685
}

fnam = '/home/alberto/almacen/literature_LF_data/LF_data_SpinosoEtAl2020/20200518_J0410_SNR5_LFdata.txt'
loglya, ModLF, ModLF_pc16, ModLF_pc84 = np.genfromtxt(fnam, skip_header=3, usecols=(0,6,7,8), unpack=True)
dLogL = loglya[1] - loglya[0]
mask = ModLF_pc16 > 0
snr = np.zeros_like(ModLF)
snr[mask] = ModLF[mask] / ModLF_pc16[mask]
ds20_z237 = {
    'logL': loglya[snr > 1],
    'Phi': ModLF[snr > 1],
    'yerr_plus': ModLF_pc84[snr > 1],
    'yerr_minus': ModLF_pc16[snr > 1],
    'z': 2.37,
    'label': 'Spinoso 2020 ($z=2.37$)',
    'fmt': 'X',
    'H0': 67.3,
    'Om0': 0.315,
    'Ode0': 0.685
}

fnam = '/home/alberto/almacen/literature_LF_data/LF_data_SpinosoEtAl2020/20200518_J0430_SNR5_LFdata.txt'
loglya, ModLF, ModLF_pc16, ModLF_pc84 = np.genfromtxt(fnam, skip_header=3, usecols=(0,6,7,8), unpack=True)
dLogL = loglya[1] - loglya[0]
mask = ModLF_pc16 > 0
snr = np.zeros_like(ModLF)
snr[mask] = ModLF[mask] / ModLF_pc16[mask]
ds20_z254 = {
    'logL': loglya[snr > 1],
    'Phi': ModLF[snr > 1],
    'yerr_plus': ModLF_pc84[snr > 1],
    'yerr_minus': ModLF_pc16[snr > 1],
    'z': 2.54,
    'label': 'Spinoso 2020 ($z=2.54$)',
    'fmt': 'X',
    'H0': 67.3,
    'Om0': 0.315,
    'Ode0': 0.685
}

fnam = '/home/alberto/almacen/literature_LF_data/LF_data_SpinosoEtAl2020/20200518_J0515_SNR5_LFdata.txt'
loglya, ModLF, ModLF_pc16, ModLF_pc84 = np.genfromtxt(fnam, skip_header=3, usecols=(0,6,7,8), unpack=True)
dLogL = loglya[1] - loglya[0]
mask = ModLF_pc16 > 0
snr = np.zeros_like(ModLF)
snr[mask] = ModLF[mask] / ModLF_pc16[mask]
ds20_z324 = {
    'logL': loglya[snr > 1],
    'Phi': ModLF[snr > 1],
    'yerr_plus': ModLF_pc84[snr > 1],
    'yerr_minus': ModLF_pc16[snr > 1],
    'z': 3.24,
    'label': 'Spinoso 2020 ($z=3.24$)',
    'fmt': 'X',
    'H0': 67.3,
    'Om0': 0.315,
    'Ode0': 0.685
}

# Zhang 2021 (z=2.0-3.5)
df = pd.read_csv('/home/alberto/cosmos/LAEs/csv/Zhang2021_LF.csv')
z21 = {
    'logL': df['Llya'],
    'Phi': df['Phi'],
    'yerr_plus': df['yerr_plus'] - df['Phi'],
    'yerr_minus': df['Phi'] - df['yerr_minus'],
    'z': (2.0 + 3.5) * 0.5,
    'label': 'Zhang 2021 ($z=2.0-3.5$)',
    'fmt': 'd',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Liu 2022 (z=1.88-3.53)
df = pd.read_csv('/home/alberto/cosmos/LAEs/csv/Liu_LF.csv')
l22 = {
    'logL': df['logLya'],
    'Phi': df['Phi'],
    'yerr_plus': df['yerr'],
    'yerr_minus': df['yerr'],
    'z': (1.88 + 3.53) * 0.5,
    'label': 'Liu 2022 ($z=1.9-3.5$)',
    'fmt': 'v',
    'H0': 70,
    'Om0': 0.3,
    'Ode0': 0.7
}

# Assign colors
LF_ref_list = [b11, g07, k16, m17a, m17b, u08, s16, s17, s18a,
               s18b, s18c, s18d, s18e, s18f, ds20_z225, ds20_z237,
               ds20_z254, ds20_z324, z21, l22]
for i, lf_dict in enumerate(LF_ref_list):
    lf_dict['color'] = f'C{i}'

# Convert to my cosmology
for lf_dict in LF_ref_list:
    args = (lf_dict['logL'], lf_dict['z'], lf_dict['H0'],
            lf_dict['Om0'], lf_dict['Ode0'])
    lf_dict['logL'] = convert_cosmology_luminosity(*args)

    args = (lf_dict['Phi'], lf_dict['z'], lf_dict['H0'],
            lf_dict['Om0'], lf_dict['Ode0'])
    lf_dict['Phi'] = convert_cosmology_Phi(*args)