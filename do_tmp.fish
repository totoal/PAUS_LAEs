#!/bin/fish

clear && cd MyMocks/ && py Make_QSO_mock.py && cd .. && py PAUS_Lya_LF_corrections.py "0 16" && cd ML_class/ && py NN_classifier.ipynb && cd .. && ./do_everything.fish && cd curve_fit/ && py schechter_fit.py
