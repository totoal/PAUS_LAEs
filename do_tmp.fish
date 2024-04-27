#!/bin/fish

# py qso_phot_dr16.ipynb
clear
cd MyMocks/ &&\
py Make_QSO_mock.py &&\
cd .. &&\
py PAUS_Lya_LF_corrections.py "0 18" &&\
cd ML_class/ &&\
py NN_classifier.ipynb &&\
py NN_z_fit.ipynb &&\
cd .. &&\
./do_everything.fish &&\
cd curve_fit/ &&\
py schechter_fit.py &&\
py dpl_fit_UV.py