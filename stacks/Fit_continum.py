from astropy.table import Table 

import numpy as np

from pylab import *

import JPAS_utils as jp

from scipy.optimize import curve_fit

from QSO_mock_cat import load_QSO_mock
from QSO_mock_cat import add_errors
from QSO_mock_cat import w_central
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#############################################################
#############################################################
#############################################################
def Load_conv_specs_NB():

    cat = load_QSO_mock('./QSO_Jspectra_QSO_z2-5/')
    #dict_keys(['flx', 'Lya_EW', 'z', 'L_lya', 'L_NV', 'NV_EW'])

    f_names = [ 'J0348','J0378','J0390','J0400','J0410','J0420','J0430','J0440','J0450','J0460','J0470','J0480','J0490','J0500','J0510','J0520','J0530','J0540','J0550','J0560','J0570','J0580','J0590','J0600','J0610','J0620','J0630','J0640','J0650','J0660','J0670','J0680','J0690','J0700','J0710','J0720','J0730','J0740','J0750','J0760','J0770','J0780','J0790','J0800','J0810','J0820','J0830','J0840','J0850','J0860','J0870','J0880','J0890','J0900','J0910','J1007','uJPAS','gSDSS','rSDSS','iSDSS']
  
    flx, flx_err = add_errors(cat['flx'], 'minijpasAEGIS002')

    flx = flx.T
    flx_err = flx_err.T
 
    flx_AL_Mat = flx[:, 2:56]

    print( 'flx_AL_Mat.shape' , flx_AL_Mat.shape )
    print( 'flx' , flx.shape )
    
    # redshift
    z_best_Arr = cat['z']

    dic_filters = jp.get_dic_of_filters_ids( MODE=None )
 
    w_Arr = np.zeros( len(flx_AL_Mat[0]) )
    c_Arr = []
   
    filter_names = f_names[2:56]
 
    for i , key in enumerate( filter_names ):
    
        w_Arr[i] = dic_filters[ key + '_w' ]
        c_Arr += [ dic_filters[ key + '_color' ] ]

    return cat , flx_AL_Mat , filter_names , w_Arr , c_Arr
#############################################################
#############################################################
#############################################################

TABLE , flx_AL_Mat , filter_names , w_Arr , c_Arr = Load_conv_specs_NB()

print( 'TABLE[z]' , np.amin( TABLE['z'] ) , np.amax( TABLE['z'] ) )

#############################################################
#############################################################
#############################################################
w_rest_Arr = np.linspace( 910 , 3000 , 1000 )

w_NORMALIZATION = 1450 #2400.0 #A

F_rest_Mat = np.zeros( len(flx_AL_Mat) * len(w_rest_Arr) ).reshape( len(flx_AL_Mat) , len(w_rest_Arr) )

for i in range( 0 , len(flx_AL_Mat) ):

    w_SDSS_rest_Arr = w_Arr * 1. / ( 1 + TABLE['z'][i] )

    f_rest_Arr = np.interp( w_rest_Arr , w_SDSS_rest_Arr , flx_AL_Mat[i] , left=np.nan , right=np.nan )

    Amp_at_w_NORM = np.interp( w_NORMALIZATION , w_rest_Arr , f_rest_Arr )#, left=np.nan , right=np.nan )

    f_rest_Arr = f_rest_Arr * 1. / Amp_at_w_NORM

    F_rest_Mat[i] = f_rest_Arr
#############################################################
#############################################################
#############################################################

f_z_all_Q50_Arr = np.nanpercentile( F_rest_Mat , 50 , axis=0 )

print( f_z_all_Q50_Arr.shape )

#windows_list = [  [ 1432.4617298846242 , 1487.2228715783126 ] ,
#                  [ 1680.9441218610968 , 1832.5446798883947 ] ,
#                  [ 1975.7041324183697 , 2590.7407058291237 ] ]

lya_pos = 1215.67
civ_pos = 1546.39
cii_pos = 1906.50

windows_list = [  [ lya_pos + 90  , lya_pos + 130 ] ,
                  [ civ_pos - 110 , civ_pos - 70 ] ,
                  [ civ_pos + 155 , cii_pos - 70 ] ,
                  [ cii_pos + 70  , 2450         ] ]

mask_fitting = np.zeros( len(f_z_all_Q50_Arr) ).astype(bool)

for i in range( 0 , len( windows_list ) ):

    xmin = windows_list[i][0]
    xmax = windows_list[i][1]

    mini_mask = ( w_rest_Arr > xmin ) * ( w_rest_Arr < xmax )

    mask_fitting[mini_mask] = True
#############################################################
#############################################################
#############################################################
def QSO_cont(x, a, b ):
    return b * ( x**a )
#############################################################
#############################################################
#############################################################
popt, pcov = curve_fit( QSO_cont , w_rest_Arr[ mask_fitting ] , f_z_all_Q50_Arr[ mask_fitting ] , p0=[-2,1e6] )

print( 'popt = ' , popt )

f_fit_Arr = QSO_cont( w_rest_Arr , popt[0] , popt[1] )
#############################################################
#############################################################
#############################################################
#Measuring:

w_T_min = 1100.00
w_T_max = 1125.00

w_T_mean = 0.5 * ( w_T_min + w_T_max )
#############################################################
#############################################################
#############################################################
figure( figsize=(10,5) )

plot(    w_rest_Arr                 , f_z_all_Q50_Arr                 , '-' , lw=8 , color='seagreen' , label='Stack spectrum')
plot(    w_rest_Arr                 , f_fit_Arr                       , '--', lw=2 , color='k'        , label=r'fit: $a \times \lambda ^b $')

scatter( w_rest_Arr[ mask_fitting ] , f_z_all_Q50_Arr[ mask_fitting ] , s=15 , c='gold' , label='Stack spectrum to fit', zorder=100 )

for i in range( 0 , len( windows_list ) ):
    xmin = windows_list[i][0]
    xmax = windows_list[i][1]
    if i==0 : 
        label='Fit window'
    else:
        label=None
    fill_between( [xmin,xmax] , [0.0,0.0] , [3.5,3.5] , color='gold' , alpha=0.3 , label=label)

fill_between( [ w_T_min , w_T_max ] , [0.0,0.0] , [3.5,3.5] , color='k' , alpha=0.3 , label='Measuring window')

ylim( 0 , 3.25 )
xlim( 900 , 3000 )

legend(loc=0)

xlabel('restframe wavelegth[A]' , size=20 )
ylabel('$f_{\lambda}$[a.u.] ' , size=20 )
savefig( 'fig_MOCK_spectrum_rest_zall_fit.pdf' , bbox_inches='tight' )
clf()
#############################################################
#############################################################
#############################################################


Continum_IGM_free_at_W = QSO_cont( w_T_mean , popt[0] , popt[1] ) 
#############################################################
#############################################################
#############################################################

NN_bins = 10

z_edges = np.linspace( 2.25 , 4.5 , NN_bins+1 )

print( z_edges )

T_QSO_Arr = np.zeros( len(z_edges)-1 )
z_QSO_Arr = np.zeros( len(z_edges)-1 )

cmap = get_cmap( 'rainbow' )

c_Arr = [] 
for i in range( 0 , NN_bins ):

    zmin = z_edges[i  ]
    zmax = z_edges[i+1]

    z_bin_mean = 0.5 * ( zmin + zmax )

    w_window_obs = w_T_mean * ( 1+ z_bin_mean )

    z_QSO_Arr[i] = w_window_obs / 1215.67 - 1. 

    z_mask_1 = ( TABLE['z'] > zmin ) * ( TABLE['z'] < zmax )

    f_1_Arr = np.nanpercentile( F_rest_Mat[ z_mask_1 ] , 50 , axis=0 )

    mask_T = ( w_rest_Arr > w_T_min ) * ( w_rest_Arr < w_T_max )

    T_QSO_Arr[i] = np.nanmean( f_1_Arr[ mask_T ] ) * 1. / Continum_IGM_free_at_W

    color = cmap( i * 1. / ( NN_bins-1 ) )

    c_Arr += [ color ]

    label = 'z in [' + str( np.round(zmin,1) ) + ',' + str( np.round(zmax,1) ) + ']'

    plot( w_rest_Arr , f_1_Arr , color=color , label=label )

plot( w_rest_Arr , f_fit_Arr , '--', lw=2 , color='k' , label=r'fit: $a \times \lambda ^b $' )

for i in range( 0 , len( windows_list ) ):
    xmin = windows_list[i][0]
    xmax = windows_list[i][1]
    if i==0 :
        label='Fit window'
    else:
        label=None
    fill_between( [xmin,xmax] , [0.0,0.0] , [3.5,3.5] , color='gold' , alpha=0.3 , label=label)

fill_between( [ w_T_min , w_T_max ] , [0.0,0.0] , [3.5,3.5] , color='k' , alpha=0.3 , label='Measuring window')

ylim( 0 , 3.25 )
xlim( 900 , 3000 )
legend(loc=0, ncol=2)
xlabel('restframe wavelegth[A]' , size=20 )
ylabel('$f_{\lambda}$[a.u.] ' , size=20 )
savefig( 'fig_MOCK_spectrum_rest_zslides_fit.pdf' , bbox_inches='tight' )
#clf()

xlim( 900 , 1500 )
savefig( 'fig_MOCK_spectrum_rest_zslides_fit_ZOOM_IN.pdf' , bbox_inches='tight' )
clf()

#############################################################
#############################################################
#############################################################
def IGM_TRANSMISSION( redshift_Arr , A , B ):

    Transmission_Arr = np.exp( A * ( 1 + redshift_Arr )**B )

    return Transmission_Arr
#############################################################
#############################################################
#############################################################
def Mukae_T( redshift_Arr ):

    A = -0.001845
    B = 3.924

    Transmission = IGM_TRANSMISSION( redshift_Arr , A , B )

    return Transmission
#############################################################
#############################################################
#############################################################
figure()

redshift_th_Arr = np.linspace( 0.0 , 7.0 )

T_th_Arr = Mukae_T( redshift_th_Arr )

plot( redshift_th_Arr , T_th_Arr , label='Faucher-Giguere et al. (2008)')

scatter( z_QSO_Arr , T_QSO_Arr , c=c_Arr , s=100 , label='SDSS JPAS QSO')

xlim(0.0,6.5)
ylim(0.0,1.05)

xlabel('IGM redhift ' , size=20)
ylabel('IGM Transmission' , size=20)

legend(loc=0)

savefig( 'fig_MOCK_test_T.pdf' , bbox_inches='tight' )  
clf()






