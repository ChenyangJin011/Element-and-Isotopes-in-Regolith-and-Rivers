# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 13:15:01 2025
Compared with jia, this one upgrates the calculation function of sec. cal. prec. in groundwater
@author: Chenyang

1. Dissolution, in soil water, open system (for carbon)
2. Dissolution and precipitation, in groundwater, closed system
3. Degassing, in river
4. Degassing and precipitation, in river
"""

import pandas as pd
# import glob

import os
import numpy as np
# import pandas as pd
# import csv
from sympy import symbols, Eq, solve
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset

# import matplotlib.patches as patches
#%% File save poisition
df =  pd.read_csv(r"E:\Doctorat\Codes\Data_python.csv",sep=',',header='infer')
list_spring = df.Spring.unique()[0:28] #make 'Spring'column as an array
save_path = "E:\Doctorat\Codes\C isotopes\comb"   
file_name = 'concentration_gamma_initial.csv'
full_path = os.path.join(save_path, file_name)
#%%
df['Ca/Sr']=df['Ca*']/df['Sr_uM']
df['Sr/Ca']=df['Sr_uM']/df['Ca*']
df['Mg/Ca']=df['Mg*']/df['Ca*']
df['Ca/Mg']=df['Ca*']/df['Mg*']
df['d44/42Ca']=-df['d42/44']
df['Alk/Ca']=df['Alkalinity_uM']/2*df['Ca']
df0 = df[0:28] # database
df1 = df[0:10] # Mekong Tributary data
df2 = df[10:13] # Mekong Mainstem data
df3 = df[13:24] # Salween Tributary data
df4 = df[24:28] # Salween Mainstem data
#%%
def calculate_ED_ratios(df, element1, element2, leachates):
    ratios = {}
    for leachate in leachates:
        column1 = f"{element1}_{leachate}"
        column2 = f"{element2}_{leachate}"
        ratio_key = f"{element1}{element2}_{leachate.lower()}"
        ratios[ratio_key] = {
            'mean': np.mean(df[column1] / df[column2]),
            'max': (df[column1] / df[column2]).max(),
            'min': (df[column1] / df[column2]).min(),
        }
    return ratios
def calculate_EDs_open(ECa_carb, ECa_sil, XCa_carb, XCa_sil):
    return ECa_carb * XCa_carb + XCa_sil * ECa_sil
def calculate_EDs_closed(ECa_carb, ECa_sil, ECa_gyp, XCa_carb, XCa_sil, XCa_gyp):
    return ECa_carb * XCa_carb + XCa_sil * ECa_sil + ECa_gyp * XCa_gyp

def calculate_ED_ratio_inverses(weighted_means):
    return {key: 1 / value for key, value in weighted_means.items()}

def calculate_ED_isotopes(df, element, isotopes, inverse=False):

    isotope_ratios = {}
    for isotope in isotopes:
        isotope_key = f"{isotope}_{element.lower()}"
        column = f"{isotope}"
        isotope_ratios[isotope_key] = {
            'mean': np.mean(df[column]),
            'max': df[column].min(),
            'min': df[column].max(),
        }
        if inverse:
            isotope_ratios[isotope_key] = {
                'mean': -np.mean(df[column]),
                'max': -df[column].min(),
                'min': -df[column].max(),
             }
    return isotope_ratios

#%% Parameters
K1 = 3.69e-7  # First dissociation constant of carbonic acid (mol/L) at 10°C, hendy 1971
K2 = 3.451e-11  # Second dissociation constant of carbonic acid (mol/L) at 10°C, hendy 1971
Kw = 2.92e-15  # Water dissociation constant (mol^2/L^2) at at 10°C, hendy 1971
KCO2 = 5.236e-2  # Henry's Law constant for CO2 at 10°C (mol·L^-1·atm^-1), hendy 1971
Ksp = 5.25e-9  # Solubility product for calcite at 10°C (mol^2/L^2), hendy 1971

pH_increment = 0.001  # Increment for pH

A = 0.5115  # L^1/2/mol^1/2 at 10°C
B = 0.3291  # nm^-1 L^1/2/mol^1/2 at 10°C

# Initial activity coefficients (all set to 1)
gamma_H = 1.0
gamma_OH = 1.0
gamma_HCO3 = 1.0
gamma_CO3 = 1.0
gamma_CO2_aq = 1.0
gamma_Ca = 1.0

# Ionic radius, in nm. 1 nm = 10 angstrom. Kielland 1937
a_H = 0.9  # nm
a_OH = 0.35  # nm
a_HCO3 = 0.45  # nm
a_CO3 = 0.45  # nm
a_Ca = 0.6  # nm

p_CO2_air_ppm = 420 # ppm
# p_CO2_soil_ppm = 9000 # Soil CO2 partial pressure, ppm

# dC_CO2_soil = -26 # unit of ‰, Liu 2005
dC_CO2_soil = -25 # dC_CO2_g, unit of ‰, Fohlmeister 2011
dC_calcite = 1 # ‰， Fohlmeister 2011
pMC_CO2_soil = 106 # pMC, Fohlmeister 2011
pMC_calcite = 0 # pMC, Fohlmeister 2011
# C isotope fractionation factor in equilibrium system, at the temperature of 10 degree, Mook, 2000, p93, 13e_g_HCO3 = -9.6 at 10 ℃
aC_CO2g_HCO3 = -9.6
aC_CO2aq_CO2g = -1.13
aC_CO2aq_HCO3 = -10.72
aC_CO3_HCO3 = -0.54
aC_calcite_HCO3 = 0.15
aC_calcite_CO2g = 9.85

# 13C composition in open system 
dC_CO2_aq_open = dC_CO2_soil + aC_CO2aq_CO2g
dC_HCO3_open = dC_CO2_soil - aC_CO2g_HCO3 
dC_CO3_open = dC_HCO3_open - aC_CO3_HCO3

aC_CO2aq_CO3 = aC_CO2aq_HCO3 - aC_CO3_HCO3
aC_HCO3_CO3 = 0.54
aC_calcite_CO3 = aC_calcite_HCO3 - aC_CO3_HCO3

aC_prec_calcite_HCO3 = 1 # Romanek 1992, e_calcite_HCO3 =1‰
aC_prec_calcite_CO2g = 10.78  # Romanek 1992
aC_prec_CO2aq_HCO3 = aC_prec_calcite_HCO3 - aC_prec_calcite_CO2g
aC_prec_calcite_CO3 = aC_prec_calcite_HCO3 + aC_HCO3_CO3
# 14C isotope fractionation factor, 14e = 2.3 * 13e / 10, Fohlmeister 2011
apMC_CO2g_HCO3 = aC_CO2g_HCO3 * 2.3 /10
apMC_CO2aq_HCO3 = aC_CO2aq_HCO3 * 2.3 /10
apMC_CO3_HCO3 = aC_CO3_HCO3 * 2.3 /10
apMC_calcite_HCO3 = aC_calcite_HCO3 * 2.3 /10

apMC_CO2aq_CO3 = aC_CO2aq_CO3 * 2.3 /10 
apMC_HCO3_CO3 = aC_HCO3_CO3 * 2.3 /10
apMC_calcite_CO3 = aC_calcite_CO3 * 2.3 /10

apMC_CO2aq_CO2g = aC_CO2aq_CO2g * 2.3 /10
apMC_calcite_CO2g = aC_calcite_CO2g * 2.3 /10

apMC_prec_calcite_HCO3 = aC_prec_calcite_HCO3 * 2.3 /10
apMC_prec_CO2aq_HCO3 = aC_prec_calcite_CO3 * 2.3 /10
apMC_prec_calcite_CO3 = aC_prec_calcite_CO3 * 2.3 /10

# 14C composition in C open system 
pMC_CO2_aq_open = pMC_CO2_soil + apMC_CO2aq_CO2g
pMC_HCO3_open = pMC_CO2_soil - apMC_CO2g_HCO3
pMC_CO3_open = pMC_HCO3_open + apMC_CO3_HCO3

#%% Common eqs

def ionic_strength(concentrations, charges):
    return 0.5 * sum(c * z**2 for c, z in zip(concentrations, charges))

def calculate_activity_coefficient(A, B, z_i, I, a_i):
    log_gamma = - (A * z_i**2 * np.sqrt(I)) / (1 + B * a_i * np.sqrt(I))
    return 10**log_gamma
T = 283.15 # in Kelvin, 10 degree celsius
R = 0.0821

# def calculate_activity_coefficient_Güntelberg(A, B, z_i, I, a_i):
#     log_gamma = - A * z_i**2 * np.sqrt(I) / (1 + np.sqrt(I))
#     return 10**log_gamma

# def calculate_activity_coefficient_Davies(A, B, z_i, I, a_i):
#     log_gamma = - A * z_i**2 * ( np.sqrt(I) / (1 + np.sqrt(I)) - 0.2 * I)
#     return 10**log_gamma

def calculate_c_alk(c_HCO3, c_CO3, c_OH, c_H):
    return c_HCO3 + 2*c_CO3 + c_OH - c_H

def calculate_c_DIC(c_CO2_aq, c_HCO3, c_CO3):
    return c_CO2_aq + c_HCO3 + c_CO3

def calculate_CO2_aq_initial(c_H, Kw, K1, K2, gamma_H, gamma_OH, gamma_HCO3, gamma_CO3, gamma_CO2_aq):
    numerator = c_H * ((c_H)**2 - Kw / (gamma_H * gamma_OH))
    denominator = (K1 * gamma_CO2_aq / gamma_H) * (c_H / gamma_HCO3 + 2 * K2 / (gamma_H * gamma_CO3))
    return numerator / denominator

def calculate_CO2_aq_open(p_CO2_soil_atm, KCO2):
    return p_CO2_soil_atm * KCO2

def calculate_CO2_aq_closed(Kw, K1, c_H, gamma_H, gamma_OH, gamma_CO2_aq, gamma_HCO3, sigma_DIC_initial):
    numerator = Kw / (c_H * gamma_H * gamma_OH) - c_H + 2 * sigma_DIC_initial
    denominator = 2 + (K1 * gamma_CO2_aq) / (c_H * gamma_H * gamma_HCO3)
    return numerator / denominator


    
def calculate_p_CO2(concentration, unit, R, T):
    # PV = nRT, P: partial pressure; V: volume; n: mol; R: iead gas constant (0.0821 L*atm/(mol*K)); T: temperature, K
    # Ideal gas constant in L·atm/(mol·K)
    if unit == 'atm-ppm':
        return concentration * 1e6
    elif unit == 'atm-mol':
        return concentration / (R * T)
    elif unit == 'ppm-atm':
        return concentration / 1e6
    elif unit == 'ppm-mol':
        return concentration / (R * T * 1e6)
    elif unit == 'mol-ppm':
        return concentration * R * T * 1e6
    elif unit == 'mol-atm':
        return concentration * R * T  
    
def calculate_c_HCO3(c_H, c_CO2_aq, K1, gamma_H, gamma_HCO3, gamma_CO2_aq):
    return (K1 * c_CO2_aq * gamma_CO2_aq) / (c_H * gamma_H * gamma_HCO3)

def calculate_c_CO3(c_H, c_CO2_aq, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq):
    return (K1 * K2 * c_CO2_aq * gamma_CO2_aq) / (c_H**2 * gamma_H**2 * gamma_CO3)

def calculate_c_OH(c_H, Kw, gamma_H, gamma_OH):
    return Kw / (c_H * gamma_H * gamma_OH)

def calculate_c_H(c_HCO3, c_CO2_aq, K1, gamma_H, gamma_HCO3, gamma_CO2_aq):
    return (K1 * c_CO2_aq * gamma_CO2_aq) / (c_HCO3 * gamma_H * gamma_HCO3)

def residual_c_H_single(c_H, c_CO2_aq, c_alk, Kw, K1, K2, gamma_H, gamma_OH, gamma_HCO3, gamma_CO3, gamma_CO2_aq):
    c_OH = calculate_c_OH(c_H, Kw, gamma_H, gamma_OH)
    c_HCO3 = calculate_c_HCO3(c_H, c_CO2_aq, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
    c_CO3 = calculate_c_CO3(c_H, c_CO2_aq, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
    
    alkalinity_residual = c_HCO3 + 2 * c_CO3 + c_OH - c_H - c_alk
    return alkalinity_residual

def residual_c_H(vars, c_alk, c_DIC, Kw, K1, K2, gamma_H, gamma_OH, gamma_HCO3, gamma_CO3, gamma_CO2_aq):
    c_H = vars[0]
    c_CO2_aq = vars[1]
    
    c_OH = calculate_c_OH(c_H, Kw, gamma_H, gamma_OH)
    c_HCO3 = calculate_c_HCO3(c_H, c_CO2_aq, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
    c_CO3 = calculate_c_CO3(c_H, c_CO2_aq, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
    
    alkalinity_residual = c_HCO3 + 2 * c_CO3 + c_OH - c_H - c_alk
    c_DIC_residual = c_CO2_aq + c_HCO3 + c_CO3 - c_DIC
    return [alkalinity_residual, c_DIC_residual]

def calculate_sigma_DIC(c_CO2_aq, c_HCO3, c_CO3):
    return c_CO2_aq + c_HCO3 + c_CO3

def calculate_sigma_DIC_inter(c_CO2_aq, c_HCO3, c_CO3, c_cation):
    return c_CO2_aq + c_HCO3 + c_CO3 - c_cation

def calculate_dC_DIC_open(c_CO2_aq, c_HCO3, c_CO3, dC_CO2_aq, dC_HCO3, dC_CO3):
    return (c_CO2_aq * dC_CO2_aq + c_HCO3 * dC_HCO3 + c_CO3 * dC_CO3) / (c_CO2_aq + c_HCO3 + c_CO3) 
 
def calculate_dC_DIC_closed(c_CO2_aq, c_HCO3, c_CO3, dC_CO2_aq, dC_HCO3, dC_CO3):
    return (c_CO2_aq * dC_CO2_aq + c_HCO3 * dC_HCO3 + c_CO3 * dC_CO3) / (c_CO2_aq + c_HCO3 + c_CO3)

def calculate_dC_DIC_dis(dC_DIC, c_DIC, c_DIC_1, dC_calcite, c_Ca, c_Ca_1, XCa_carb_closed):
    return (dC_DIC * c_DIC + dC_calcite * (c_Ca_1 - c_Ca)*XCa_carb_closed) / c_DIC_1

def calculate_dC_CO2_aq(c_CO2_aq, c_HCO3, c_CO3, dC_DIC, aC_CO2aq_HCO3, aC_CO2aq_CO3):
    return ((c_CO2_aq + c_HCO3 + c_CO3) * dC_DIC + aC_CO2aq_HCO3 * c_HCO3 + aC_CO2aq_CO3 * c_CO3) / (c_CO2_aq + c_HCO3 + c_CO3)

def calculate_dC_HCO3(dC_CO2_aq, aC_CO2aq_HCO3):
    return dC_CO2_aq - aC_CO2aq_HCO3

def calculate_dC_CO3(dC_CO2_aq, aC_CO2aq_CO3):
    return dC_CO2_aq - aC_CO2aq_CO3

def calculate_c_cation(c_OH, c_HCO3, c_CO3, c_H):
    return (c_OH + c_HCO3 + 2*c_CO3 - c_H) # without rain input

def calculate_CO2_aq_closed_inter(Kw, K1, K2, c_H, gamma_H, gamma_OH, gamma_CO2_aq, gamma_HCO3, gamma_CO3, 
                                  c_DIC_open_last, c_cation_open_last, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed,
                                  MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp, XSO4_gyp_closed):
    
    phi = XCa_carb_closed * 2 * (1 + MgCa_carb + SrCa_carb) + XCa_sil_closed * (2 * (1 + MgCa_sil + SrCa_sil) + NaCa_sil + KCa_sil) + XCa_gyp_closed * 2 * (1 + MgCa_gyp + SrCa_gyp)
    psi =  2 * (XCa_gyp_closed / XSO4_gyp_closed)
    sigma = XCa_carb_closed * (1 + MgCa_carb + SrCa_carb)
    
    numerator = Kw / (c_H * gamma_H * gamma_OH) - c_H + ((phi - psi)/sigma) * c_DIC_open_last - c_cation_open_last
    denominator = ((phi - psi)/sigma) + ((phi - psi)/sigma - 1) * (K1 * gamma_CO2_aq) / (c_H * gamma_H * gamma_HCO3) + ((phi - psi)/sigma - 2) * (K1 * K2 * gamma_CO2_aq) / (c_H**2 * gamma_H**2 * gamma_CO3)
    return numerator / denominator

def calculate_c_Ca_closed(sum_cation_SO4, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed,  XSO4_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp):
    phi = XCa_carb_closed * 2 * (1 + MgCa_carb + SrCa_carb) + XCa_sil_closed * (2 * (1 + MgCa_sil + SrCa_sil) + NaCa_sil + KCa_sil) + XCa_gyp_closed * 2 * (1 + MgCa_gyp + SrCa_gyp)
    psi =  2 * (XCa_gyp_closed / XSO4_gyp_closed)
    return sum_cation_SO4 / (phi - psi)

def calculate_c_Ca_inter_closed(sum_cation_SO4, c_cation_open_last, c_Ca_open_last, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed,  XSO4_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp):
    phi = XCa_carb_closed * 2 * (1 + MgCa_carb + SrCa_carb) + XCa_sil_closed * (2 * (1 + MgCa_sil + SrCa_sil) + NaCa_sil + KCa_sil) + XCa_gyp_closed * 2 * (1 + MgCa_gyp + SrCa_gyp)
    psi =  2 * (XCa_gyp_closed / XSO4_gyp_closed)
    return (sum_cation_SO4 - c_cation_open_last) / (phi - psi) + c_Ca_open_last

def calculate_c_ion_inter_closed(c_Ca, c_Ca_open_last, ionCa_rock_closed, c_ion_open_last):
    return (c_Ca - c_Ca_open_last) * ionCa_rock_closed + c_ion_open_last

def calculate_c_cation_from_sum(sum_cation_SO4, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, XSO4_gyp_closed, c_cation_open_last,  MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil):
    # phi = XCa_carb_closed * 2 * (1 + MgCa_carb + SrCa_carb) + XCa_sil_closed * (2 * (1 + MgCa_sil + SrCa_sil) + NaCa_sil + KCa_sil)
    phi = XCa_carb_closed * 2 * (1 + MgCa_carb + SrCa_carb) + XCa_sil_closed * (2 * (1 + MgCa_sil + SrCa_sil) + NaCa_sil + KCa_sil) + XCa_gyp_closed * 2 * (1 + MgCa_gyp + SrCa_gyp)
    psi =  2 * (XCa_gyp_closed / XSO4_gyp_closed)
    return (sum_cation_SO4 + psi/phi * c_cation_open_last) / (1 + psi/phi)

# def calculate_c_SO4_from_sum(sum_cation_SO4, c_cation):
    # return (c_cation - sum_cation_SO4) / 2
def calculate_c_SO4_from_sum(c_cation, c_cation_open_last, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, XSO4_gyp_closed,   MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil):
    # phi = XCa_carb_closed * 2 * (1 + MgCa_carb + SrCa_carb) + XCa_sil_closed * (2 * (1 + MgCa_sil + SrCa_sil) + NaCa_sil + KCa_sil)
    phi = XCa_carb_closed * 2 * (1 + MgCa_carb + SrCa_carb) + XCa_sil_closed * (2 * (1 + MgCa_sil + SrCa_sil) + NaCa_sil + KCa_sil) + XCa_gyp_closed * 2 * (1 + MgCa_gyp + SrCa_gyp)
    psi =  2 * (XCa_gyp_closed / XSO4_gyp_closed)
    return ((c_cation - c_cation_open_last) / phi ) * 0.5*psi

def calculate_c_cation_dis_groudwater(c_cation, SrCa_rock, MgCa_rock, NaCa_rock, KCa_rock): 
    c_Ca = c_cation / (2 + 2*MgCa_rock + 2*SrCa_rock + NaCa_rock + KCa_rock)
    c_Sr = SrCa_rock * c_Ca
    c_Mg = MgCa_rock * c_Ca
    c_Na =  NaCa_rock * c_Ca
    c_K = KCa_rock * c_Ca
    
    return c_Ca, c_Mg, c_Sr, c_Na, c_K

def calculate_c_Ca_p(c_Ca_1, c_CO3, gamma_Ca, gamma_CO3, Ksp):
    c_Ca_p = symbols('c_Ca_p')
    equation = Eq((c_Ca_1 - c_Ca_p) * (c_CO3 - c_Ca_p) * gamma_Ca * gamma_CO3, Ksp)
    solutions = solve(equation, c_Ca_p)
    
    c_Ca_p_value = float(solutions[0].evalf()) 
    return c_Ca_p_value

def calculate_c_Ca_prec(c_Ca_p, c_Ca, c_CO3, gamma_Ca, gamma_CO3, Ksp):
    return (c_Ca - c_Ca_p) * (c_CO3 - c_Ca_p) * gamma_Ca * gamma_CO3 - Ksp

def Q_calculate(gamma_Ca, c_Ca, gamma_CO3, c_CO3):
    return gamma_Ca * c_Ca * gamma_CO3 * c_CO3

def SI_calculate(Q, Ksp):
    return np.log10(Q / Ksp)

def omega_calculate(Q,Ksp):
    return Q / Ksp 

def calculate_P_Ca(c_CO3, c_Ca, F_CO3):
    return (c_CO3/c_Ca) * F_CO3

def calculate_c_SO4_pyr(c_SO4_gyp, XSO4_gyp_closed, XSO4_pyr_closed):
    return c_SO4_gyp * (XSO4_pyr_closed / XSO4_gyp_closed) 

#%% calcite dissolution simulation
# systems = ['open', 'closed', 'inter']


results = []
current_results = {k: [] for k in ['label', 'pH', 'c_H', 'c_OH', 
                                   'c_DIC', 'c_CO2_aq', 'p_CO2_aq_pcatm', 'c_HCO3', 'c_CO3', 
                                   'dC_DIC', 'pMC_DIC', 'dC_CO2_aq', 'pMC_CO2_aq', 'dC_HCO3', 'pMC_HCO3', 'dC_CO3', 'pMC_CO3', 
                                   'c_cation', 'c_Ca', 'c_Mg', 'c_Sr', 'c_Na', 'c_K', 'c_SO4', 'c_Cl_rain', 'I',
                                   'd42Ca', 'dSr', 'omega', 'XCa_carb_open', 'XCa_sil_open', 'XCa_carb_closed', 'XCa_sil_closed', 'XCa_gyp_closed', 'XSO4_gyp_closed', 'XSO4_pyr_closed']}
c_Ca_sam = df['Ca'].mean() / 1e6 # convert from umol/L to mol/L
c_Na_sam = df['Na'].mean() / 1e6 
c_K_sam = df['K'].mean() / 1e6 
c_Mg_sam = df['Mg'].mean() / 1e6 
c_Sr_sam = df['Sr_uM'].mean() / 1e6
c_NO3_sam = df['NO3'].mean() / 1e6 
c_Cl_sam = df['Cl'].mean() / 1e6
c_SO4_sam = df['SO4'].mean() / 1e6
# pH_values = [4.9, 4.775, 4.65, 4.525, 4.4, 4.275, 4.15]
# p_CO2_soil_values = [7000, 12000, 20000, 40000, 70000, 120000, 200000]
pH = 4.9
T = 283.15 # in Kelvin, 10 degree celsius
R = 0.0821
# pH_value = {'min': 4.15, 'max':4.9}
# pH = np.random.uniform(4.15, 4.9)
# p_CO2_soil_value = {'min': 7000, 'max':200000}
# p_CO2_soil = np.random.uniform(7000, 200000)
# x_groundwater  = 0.001 # the fraction of open system in inter system calculation
# x_groundwater = np.random.uniform(0, 0.01)
# x_degas = 0.5 # the stop point for degassing simulation

c_CO2_aq_end = calculate_p_CO2(p_CO2_air_ppm, 'ppm-mol', R, T) # unit of mol
p_CO2_aq_end = 100 * calculate_p_CO2(p_CO2_air_ppm, 'ppm-atm', R, T) # unit of %mol
dC_CO2_aq_end = -8.4 + aC_CO2aq_CO2g
pMC_CO2_aq_end = 106 + apMC_CO2aq_CO2g
#%% leachate composition
# Calculate all ratios and isotopes
# NaCl_rain = np.random.normal(4.17, 1) # Na/Cl of 4.17, Zhang et al., 2003
NaCl_rain = 1

ratios = {}
ratios.update(calculate_ED_ratios(df, 'Sr', 'Ca', ['Acetic', 'Residual']))
ratios.update(calculate_ED_ratios(df, 'Mg', 'Ca', ['Acetic', 'Residual']))
ratios.update(calculate_ED_ratios(df, 'Na', 'Ca', ['Acetic', 'Residual']))
ratios.update(calculate_ED_ratios(df, 'K', 'Ca', ['Acetic', 'Residual']))
ratios.update(calculate_ED_ratios(df, 'Ca', 'Sr', ['Acetic', 'Residual']))
ratios.update(calculate_ED_ratios(df, 'Ca', 'Mg', ['Acetic', 'Residual']))
ratios.update(calculate_ED_ratios(df, 'Ca', 'Na', ['Acetic', 'Residual']))
ratios.update(calculate_ED_ratios(df, 'Ca', 'K', ['Acetic', 'Residual']))

# SrCa_carb = np.random.uniform(ratios['SrCa_acetic']['min'], ratios['SrCa_acetic']['max'])
# MgCa_carb = np.random.uniform(ratios['MgCa_acetic']['min'], ratios['MgCa_acetic']['max'])
SrCa_carb = np.mean([ratios['SrCa_acetic']['min'], ratios['SrCa_acetic']['max']])
MgCa_carb = np.mean([ratios['MgCa_acetic']['min'], ratios['MgCa_acetic']['max']])

NaCa_carb = 0
KCa_carb = 0

# SrCa_sil = np.random.uniform(ratios['SrCa_residual']['min'], ratios['SrCa_residual']['max'])
# MgCa_sil = np.random.uniform(ratios['MgCa_residual']['min'], ratios['MgCa_residual']['max'])
# NaCa_sil = np.random.uniform(ratios['NaCa_residual']['min'], ratios['NaCa_residual']['max'])
# KCa_sil = np.random.uniform(ratios['KCa_residual']['min'], ratios['KCa_residual']['max'])
SrCa_sil = np.mean([ratios['SrCa_residual']['min'], ratios['SrCa_residual']['max']])
MgCa_sil = np.mean([ratios['MgCa_residual']['min'], ratios['MgCa_residual']['max']])
NaCa_sil = np.mean([ratios['NaCa_residual']['min'], ratios['NaCa_residual']['max']])
KCa_sil = np.mean([ratios['KCa_residual']['min'], ratios['KCa_residual']['max']])

# MgCa_gyp = np.random.uniform(0.002, 0.025)
# SrCa_gyp = np.random.uniform(0.0007, 0.0036)

MgCa_gyp = np.mean([0.002, 0.025])
SrCa_gyp = np.mean([0.0007, 0.0036]) # Fei et al., 2022

CaSO4_gyp = 1
CaSO4_pyr = 0

# Fractions of silicate dissolution in groundwater, in open system, which is correspond to the top layer, more silicate. Vice verse.
# XCa_carb_open = np.random.uniform(0.1, 0.2)
#%% Figure configuration
XCa_carb_open = 0.5
XCa_sil_open = 1 - XCa_carb_open 

XCa_sil_closed = 0
XCa_gyp_closed =0.3
XCa_carb_closed = 1 - XCa_sil_closed - XCa_gyp_closed

XSO4_gyp_closed = 0.5
XSO4_pyr_closed = 1 - XSO4_gyp_closed

n_gw_scp = 5000
#%% random values
# XCa_carb_open = 0.5
# XCa_sil_open = 1 - XCa_carb_open

# XCa_carb_closed = np.random.uniform(0.3, 0.5)
# XCa_carb_closed = 0.5
# XCa_gyp_closed = np.random.uniform(0.3, 0.5)
# XCa_gyp_closed =0.2
# XCa_sil_closed = 1 - XCa_carb_closed - XCa_gyp_closed
# XSO4_gyp_closed = np.random.uniform(0.4, 0.6)
# XSO4_gyp_closed = 0.5
# XSO4_pyr_closed = 1 - XSO4_gyp_closed

SrCa_rock_open = calculate_EDs_open(SrCa_carb, SrCa_sil, XCa_carb_open, XCa_sil_open)
MgCa_rock_open = calculate_EDs_open(MgCa_carb, MgCa_sil, XCa_carb_open, XCa_sil_open)
NaCa_rock_open = calculate_EDs_open(NaCa_carb, NaCa_sil,XCa_carb_open, XCa_sil_open)
KCa_rock_open = calculate_EDs_open(KCa_carb, KCa_sil,XCa_carb_open, XCa_sil_open)

SrCa_rock_closed = calculate_EDs_closed(SrCa_carb, SrCa_sil, SrCa_gyp, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed)
MgCa_rock_closed = calculate_EDs_closed(MgCa_carb, MgCa_sil, MgCa_gyp, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed)
NaCa_rock_closed = calculate_EDs_closed(0, NaCa_carb, NaCa_sil, 0, XCa_carb_closed, XCa_sil_closed)
KCa_rock_closed = calculate_EDs_closed(0, KCa_carb, KCa_sil, 0, XCa_carb_closed, XCa_sil_closed)
CaSO4_rock_closed = calculate_EDs_closed(0, CaSO4_gyp, CaSO4_pyr, 0, XSO4_gyp_closed, XSO4_pyr_closed)

# Isotopic calculation
d42Ca_leachate = calculate_ED_isotopes(df, 'Ca', ['d42/44_ace', 'd42/44_res'], inverse=True)
# d42Ca_carb = np.random.uniform(d42Ca_leachate['d42/44_ace_ca']['min'], d42Ca_leachate['d42/44_ace_ca']['max'])
# d42Ca_sil = np.random.uniform(d42Ca_leachate['d42/44_res_ca']['min'], d42Ca_leachate['d42/44_res_ca']['max'])
# d42Ca_gyp = np.random.uniform(-0.39, -0.04) # Gussone2016, d44/40Ca_915a of -1.5~-0.8, P104. d44/42Ca_915b = (d44/40Ca_915a+0.72) / 2 
d42Ca_carb = np.mean([d42Ca_leachate['d42/44_ace_ca']['min'], d42Ca_leachate['d42/44_ace_ca']['max']])
d42Ca_sil = np.mean([d42Ca_leachate['d42/44_res_ca']['min'], d42Ca_leachate['d42/44_res_ca']['max']])
d42Ca_gyp = np.mean([-0.39, -0.04])  

d42Ca_rock_open = calculate_EDs_open(d42Ca_carb, d42Ca_sil, XCa_carb_open, XCa_sil_open)
d42Ca_rock_closed = calculate_EDs_closed(d42Ca_carb, d42Ca_sil, d42Ca_gyp, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed)

dSr_leachate = calculate_ED_isotopes(df, 'Sr', ['87/86Sr_ace', '87/86Sr_res'], inverse=False)
# dSr_carb = np.random.uniform(dSr_leachate['87/86Sr_ace_sr']['min'], dSr_leachate['87/86Sr_ace_sr']['max'])
# dSr_sil = np.random.uniform(dSr_leachate['87/86Sr_res_sr']['min'], dSr_leachate['87/86Sr_res_sr']['max'])
# dSr_gyp = np.random.uniform(0.707531, 0.708163) #Fei 2022
dSr_carb = np.mean([dSr_leachate['87/86Sr_ace_sr']['min'], dSr_leachate['87/86Sr_ace_sr']['max']])
dSr_sil = np.mean([dSr_leachate['87/86Sr_res_sr']['min'], dSr_leachate['87/86Sr_res_sr']['max']])
dSr_gyp = np.mean([0.707531, 0.708163]) #Fei 2022

dSr_rock_open = (SrCa_carb * dSr_carb * XCa_carb_open + SrCa_sil* dSr_sil * XCa_sil_open) / SrCa_rock_open
dSr_rock_closed = (SrCa_carb * dSr_carb * XCa_carb_closed + SrCa_sil * dSr_sil * XCa_sil_closed + SrCa_gyp * dSr_gyp * XCa_gyp_closed) / SrCa_rock_closed

# F_CO3 = np.random.uniform(0, 1)  
F_CO3 = 0.5
# P_CO2 = np.random.uniform(0,0.05)
F_CO2 = 0.05
# DC_Sr = np.random.uniform(0.02, 0.40) # Isotope fractionation factor for Sr, Enzo 1999
# DC_Mg = np.random.uniform(0.01, 0.124)
# DC_Na = np.random.uniform(2e-4, 0.006)
# DC_K = np.random.uniform(5e-5, 0.002)
DC_Sr = np.mean([0.02, 0.40]) 
DC_Mg = np.mean([0.01, 0.124]) 
DC_Na = np.mean([2e-4, 0.006])  
DC_K = np.mean([5e-5, 0.002]) 

# a42Ca_calcite_prec = np.random.uniform(0.9993, 1) # p76 (Gussone et al., Calcium Stable Isotope Geochemistry), delta_44/40Ca_Cc_Ca2+ = -1.5 to 0‰, alpha_44/40 = 0.9985 to 1... alpha_44/42=(alpha_44/40)^beta, beta = 0.4763, so delta_44/42Ca = 0.9993 to 1
# a42Ca_calcite_Ca = np.random.uniform(-0.8, 0) #  p76 (Gussone et al., Calcium Stable Isotope Geochemistry), delta_44/40Ca_Cc_Ca2+ = -1.5 to 0‰
a42Ca_calcite_prec = np.mean([0.9993, 1])
a42Ca_calcite_Ca = np.mean([-0.8, 0])

CaSr_carb = 1/SrCa_carb
CaSr_sil = 1/SrCa_sil

d34S_gyp = np.random.uniform(15.3, 20.8) # Fei et al., 2022
d34S_pyr = np.random.uniform(-15, -5) # river water data
# slope = (d42Ca_sil - d42Ca_carb) / (CaSr_sil - CaSr_carb)
# intercept = d42Ca_carb - slope * CaSr_carb

# def line_equation(x):
#     return slope * (x - CaSr_carb) + d42Ca_carb
# equation_text = f'y = {slope:.6f}x  + {intercept:.2f}'
#%% MAIN
systems = ['open_closed_unsaturated_degas/prep', # SI<0 in the whole groundwater and degassing
           'open_closed_unsaturated_degas', # SI<0 in only the groundwater
           'open_closed_saturated_diss/prep_degas/prep', # SI>0 in the groundwater
           'open_closed_saturated_diss/prep_degas',
           'open_degas', # SI<0 in the whole groundwater and degassing, pure open system
           'open_degas/prep' # SI<0 in the groundwater, pure open system
           ]

# system = 'open_closed_unsaturated_degas/prep'
system = 'open_closed_saturated_diss/prep_degas/prep'
# system = 'open_closed_saturated_diss/prep_degas'
go_to_precipitation = False
# for system in systems:
    # for initial_pH, p_CO2_soil in zip(pH_values, p_CO2_soil_values):
        # label= f'{system}: pH={initial_pH}, pCO2={p_CO2_soil}ppm'
c_H = 10 ** (-pH)


# Initial soil water, OPEN system
phase = "soil water"
initial_guess_closed = [0, 0]
finished = False
          
# the transition point from open to closed system, it must be smaller than omega_degas
omega_transition = 0.2 
if system == 'open_closed_unsaturated_degas':
    p_CO2_soil = 1000       
else:
    p_CO2_soil = 100000  
p_CO2_soil_atm = calculate_p_CO2(p_CO2_soil, 'ppm-atm', R, T)
# the transition point from groundwater to riverwater
if system in ['open_closed_unsaturated_degas/prep', 'open_closed_unsaturated_degas']:
    omega_degas = 0.5  # omega_degas = 1
else:
    omega_degas = 1
    
    
while not finished:
    # 1. initial solution that comprises only CO2 + H2O, to get initial_CO2_aq
    if phase == "soil water":
        label = 'soil water'
        gamma_converged = False
        while not gamma_converged:

            prev_gamma_H = gamma_H
            prev_gamma_OH = gamma_OH
            prev_gamma_HCO3 = gamma_HCO3
            prev_gamma_CO3 = gamma_CO3

            c_OH = calculate_c_OH(c_H, Kw, gamma_H, gamma_OH)
            c_CO2_aq = calculate_CO2_aq_open(p_CO2_soil_atm, KCO2)
            p_CO2_aq_pcatm = 100 * calculate_p_CO2(c_CO2_aq, 'mol-atm', R, T)
            c_HCO3 = calculate_c_HCO3(c_H, c_CO2_aq, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
            c_CO3 = calculate_c_CO3(c_H, c_CO2_aq, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
            c_DIC = calculate_c_DIC(c_CO2_aq, c_HCO3, c_CO3)
            dC_DIC = calculate_dC_DIC_open(c_CO2_aq, c_HCO3, c_CO3, dC_CO2_aq_open, dC_HCO3_open, dC_CO3_open)
            pMC_DIC = calculate_dC_DIC_open(c_CO2_aq, c_HCO3, c_CO3, pMC_CO2_aq_open, pMC_HCO3_open, pMC_CO3_open)
            c_Ca = 0
            c_Mg = 0
            c_Sr = 0
            c_K = 0
            c_Na = 0
            c_SO4 = 0
            c_cation = 0
            c_Cl_rain = c_Cl_sam
            c_Na_rain = c_Cl_rain * NaCl_rain
            
            
            dC_CO2_aq = dC_CO2_aq_open
            dC_HCO3 = dC_HCO3_open
            dC_CO3 = dC_CO3_open
            pMC_CO2_aq = pMC_CO2_aq_open
            pMC_HCO3 = pMC_HCO3_open
            pMC_CO3 = pMC_CO3_open

            concentrations = [c_H, c_OH, c_HCO3, c_CO3, c_Na_rain, c_Cl_rain]
            charges = [1, -1, -1, -2, 1, -1]
            I = ionic_strength(concentrations, charges)

            gamma_H = calculate_activity_coefficient(A, B, 1, I, a_H)
            gamma_OH = calculate_activity_coefficient(A, B, -1, I, a_OH)
            gamma_HCO3 = calculate_activity_coefficient(A, B, -1, I, a_HCO3)
            gamma_CO3 = calculate_activity_coefficient(A, B, -2, I, a_CO3)

            if abs(gamma_H - prev_gamma_H) < 1e-6 and abs(gamma_OH - prev_gamma_OH) < 1e-6 and \
                abs(gamma_HCO3 - prev_gamma_HCO3) < 1e-6 and abs(gamma_CO3 - prev_gamma_CO3) < 1e-6:
                gamma_converged = True
            phase = "groundwater_dissolution_open"
            sigma_DIC_initial = calculate_sigma_DIC(c_CO2_aq, c_HCO3, c_CO3)
        

        for key, value in zip(current_results.keys(), [label, pH, c_H, c_OH,\
                                                       c_DIC, c_CO2_aq, p_CO2_aq_pcatm, c_HCO3,  c_CO3,\
                                                       dC_DIC, pMC_DIC, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3,\
                                                       c_cation, c_Ca, c_Mg, c_Sr, c_Na_rain, c_K, c_SO4, c_Cl_rain, I,\
                                                       None, None, None, XCa_carb_open, XCa_sil_open, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, XSO4_gyp_closed, XSO4_pyr_closed]):
            current_results[key].append(value)  
        
        
        SI_initial = -1
        SI = SI_initial
        omega = 0
        iteration_calcite = 0 
    
    # 2. the initial solution is balanced with the soil atmosphere, then minerals dissolve by pH increment
    elif phase == "groundwater_dissolution_open":             
        # while SI < 0: # this is for calcite dissolution upto SI = 0
        label = 'groundwater_dissolution_open'
        while omega < omega_transition: # open system
        # while x < x_degas: # this is for degas

            pH += pH_increment
            c_H = 10 ** (-pH)
            gamma_converge = False

            while not gamma_converge:
                prev_gamma_H = gamma_H
                prev_gamma_OH = gamma_OH
                prev_gamma_HCO3 = gamma_HCO3
                prev_gamma_CO3 = gamma_CO3
                prev_gamma_Ca = gamma_Ca
                
                c_CO2_aq_gamma = calculate_CO2_aq_open(p_CO2_soil_atm, KCO2)
                c_OH_gamma = calculate_c_OH(c_H, Kw, gamma_H, gamma_OH)
                c_HCO3_gamma = calculate_c_HCO3(c_H, c_CO2_aq_gamma, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
                c_CO3_gamma = calculate_c_CO3(c_H, c_CO2_aq_gamma, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)

                c_cation_gamma = calculate_c_cation(c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_H)
                if c_cation_gamma < 0:
                    c_cation_gamma = 1e-20
                c_Ca_gamma,  c_Mg_gamma, c_Sr_gamma, c_Na_gamma, c_K_gamma  = calculate_c_cation_dis_groudwater(c_cation_gamma, SrCa_rock_open, MgCa_rock_open, NaCa_rock_open, KCa_rock_open)
                c_SO4_gamma = 0
                # c_cation_gamma = c_cation_gamma_1
                concentrations = [c_H, c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_Ca_gamma, c_Mg_gamma, c_Sr_gamma, c_Na_gamma, c_K_gamma, c_Cl_rain, c_SO4_gamma]
                charges = [1, -1, -1, -2, 2, 2, 2, 1, 1, -1, -2]
                I = ionic_strength(concentrations, charges)

                gamma_H = calculate_activity_coefficient(A, B, 1, I, a_H)
                gamma_OH = calculate_activity_coefficient(A, B, -1, I, a_OH)
                gamma_HCO3 = calculate_activity_coefficient(A, B, -1, I, a_HCO3)
                gamma_CO3 = calculate_activity_coefficient(A, B, -2, I, a_CO3)
                gamma_Ca = calculate_activity_coefficient(A, B, 2, I, a_Ca)

                if abs(gamma_H - prev_gamma_H) < 1e-6 and abs(gamma_Ca - prev_gamma_Ca) < 1e-6 and abs(gamma_OH - prev_gamma_OH) < 1e-6 and \
                    abs(gamma_HCO3 - prev_gamma_HCO3) < 1e-6 and abs(gamma_CO3 - prev_gamma_CO3) < 1e-6:
                    gamma_converge = True    
                
            c_CO2_aq_1 = calculate_CO2_aq_open(p_CO2_soil_atm, KCO2)
            p_CO2_aq_pcatm = 100 * calculate_p_CO2(c_CO2_aq_1, 'mol-atm', R, T)
            c_OH_1 = calculate_c_OH(c_H, Kw, gamma_H, gamma_OH)
            c_HCO3_1 = calculate_c_HCO3(c_H, c_CO2_aq_1, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
            c_CO3_1 = calculate_c_CO3(c_H, c_CO2_aq_1, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
                                
            c_DIC_1 = calculate_c_DIC(c_CO2_aq_1, c_HCO3_1, c_CO3_1)
            c_alk_1 = calculate_c_alk(c_HCO3_1, c_CO3_1, c_OH_1, c_H)                    

            c_cation_1 = calculate_c_cation(c_OH_1, c_HCO3_1, c_CO3_1, c_H)
            if c_cation_1 < 0:
                c_cation_1 = 1e-20
            c_Ca_1,  c_Mg_1, c_Sr_1, c_Na_1, c_K_1  = calculate_c_cation_dis_groudwater(c_cation_1, SrCa_rock_open, MgCa_rock_open, NaCa_rock_open, KCa_rock_open)
            c_SO4_1 = 0
            c_SO4_gyp_1 = c_SO4_1 * XSO4_gyp_closed
            c_SO4_pyr_1 = c_SO4_1 * XSO4_pyr_closed
            
            dC_DIC_1 = calculate_dC_DIC_open(c_CO2_aq_1, c_HCO3_1, c_CO3_1, dC_CO2_aq_open, dC_HCO3_open, dC_CO3_open)
            pMC_DIC_1 = calculate_dC_DIC_open(c_CO2_aq_1, c_HCO3_1, c_CO3_1, pMC_CO2_aq_open, pMC_HCO3_open, pMC_CO3_open)
            
            d42Ca = d42Ca_rock_open
            dSr = dSr_rock_open    
                
            dC_CO2_aq_1 = calculate_dC_CO2_aq(c_CO2_aq_1, c_HCO3_1, c_CO3_1, dC_DIC_1, aC_CO2aq_HCO3, aC_CO2aq_CO3)
            dC_HCO3_1 = calculate_dC_HCO3(dC_CO2_aq_1, aC_CO2aq_HCO3)
            dC_CO3_1 = calculate_dC_CO3(dC_CO2_aq_1, aC_CO2aq_CO3)
            
            pMC_CO2_aq_1 = calculate_dC_CO2_aq(c_CO2_aq_1, c_HCO3_1, c_CO3_1, pMC_DIC_1, apMC_CO2aq_HCO3, apMC_CO2aq_CO3)
            pMC_HCO3_1 = calculate_dC_HCO3(pMC_CO2_aq_1, apMC_CO2aq_HCO3)
            pMC_CO3_1 = calculate_dC_CO3(pMC_CO2_aq_1, apMC_CO2aq_CO3)
            
            c_CO2_aq, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3, c_OH, c_HCO3, c_CO3, c_Ca, c_Mg, c_Sr, c_Na, c_K, c_SO4, c_SO4_gyp, c_SO4_pyr, c_cation, c_DIC, c_alk, dC_DIC, pMC_DIC = \
                c_CO2_aq_1, dC_CO2_aq_1, pMC_CO2_aq_1, dC_HCO3_1, pMC_HCO3_1, dC_CO3_1, pMC_CO3_1, c_OH_1, c_HCO3_1, c_CO3_1, c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1, c_K_1, c_SO4_1, c_SO4_gyp_1, c_SO4_pyr_1, c_cation_1, c_DIC_1, c_alk_1, dC_DIC_1, pMC_DIC_1  

            Q = Q_calculate(gamma_Ca, c_Ca, gamma_CO3, c_CO3)
            omega = omega_calculate(Q, Ksp)

            c_DIC_initial, c_cation_initial, c_Ca_initial, c_Mg_initial, c_Sr_initial, c_Na_initial, c_K_initial, c_SO4_initial  = c_DIC, c_cation, c_Ca, c_Mg, c_Sr, c_Na, c_K, c_SO4

            initial_guess_closed = [c_cation, c_SO4]                  
            for key, value in zip(current_results.keys(), [label, pH, c_H, c_OH,\
                                                           c_DIC, c_CO2_aq, p_CO2_aq_pcatm, c_HCO3,  c_CO3,\
                                                           dC_DIC, pMC_DIC, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3,\
                                                           c_cation, c_Ca, c_Mg, c_Sr, c_Na_rain+c_Na, c_K, c_SO4, c_Cl_rain, I,\
                                                           d42Ca, dSr, omega, None, None, None, None, None, None, None]):
                current_results[key].append(value)  
        if system in ('open_closed_unsaturated_degas/prep', 'open_closed_unsaturated_degas', 'open_closed_saturated_diss/prep_degas/prep', 'open_closed_saturated_diss/prep_degas'):
            phase = 'groundwater_dissolution_closed'

        elif system in ('open_degas', 'open_degas/prep'):
            phase = 'precipitation_dissolution'
                
    elif phase == "groundwater_dissolution_closed": 
        label = 'groundwater_dissolution_closed'
        while omega <= omega_degas : 
            pH += pH_increment
            c_H = 10 ** (-pH)
            gamma_converge = False            
            
            while not gamma_converge:
                prev_gamma_H = gamma_H
                prev_gamma_OH = gamma_OH
                prev_gamma_HCO3 = gamma_HCO3
                prev_gamma_CO3 = gamma_CO3
                prev_gamma_Ca = gamma_Ca
                
                c_CO2_aq_gamma = calculate_CO2_aq_closed_inter(Kw, K1, K2, c_H, gamma_H, gamma_OH, gamma_CO2_aq, gamma_HCO3, gamma_CO3, c_DIC_initial, c_cation_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp, XSO4_gyp_closed)
                c_OH_gamma = calculate_c_OH(c_H, Kw, gamma_H, gamma_OH)
                c_HCO3_gamma = calculate_c_HCO3(c_H, c_CO2_aq_gamma, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
                c_CO3_gamma = calculate_c_CO3(c_H, c_CO2_aq_gamma, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
                
                sum_cation_SO4_gamma = calculate_c_cation(c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_H)
                c_Ca_gamma = calculate_c_Ca_inter_closed(sum_cation_SO4_gamma, c_cation_initial, c_Ca_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, XSO4_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp)
                c_SO4_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca_initial, XCa_gyp_closed, c_SO4_initial)
                c_Mg_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca_initial, MgCa_rock_closed, c_Mg_initial)
                c_Sr_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca_initial, SrCa_rock_closed, c_Sr_initial)
                c_Na_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca_initial, NaCa_rock_closed, c_Na_initial)
                c_K_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca_initial, KCa_rock_closed, c_K_initial)
                
                concentrations = [c_H, c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_Ca_gamma, c_Mg_gamma, c_Sr_gamma, c_Na_gamma, c_K_gamma, c_Cl_rain, c_SO4_gamma]
                charges = [1, -1, -1, -2, 2, 2, 2, 1, 1, -1, -2]
                I = ionic_strength(concentrations, charges)

                gamma_H = calculate_activity_coefficient(A, B, 1, I, a_H)
                gamma_OH = calculate_activity_coefficient(A, B, -1, I, a_OH)
                gamma_HCO3 = calculate_activity_coefficient(A, B, -1, I, a_HCO3)
                gamma_CO3 = calculate_activity_coefficient(A, B, -2, I, a_CO3)
                gamma_Ca = calculate_activity_coefficient(A, B, 2, I, a_Ca)

                if abs(gamma_H - prev_gamma_H) < 1e-6 and abs(gamma_Ca - prev_gamma_Ca) < 1e-6 and abs(gamma_OH - prev_gamma_OH) < 1e-6 and \
                    abs(gamma_HCO3 - prev_gamma_HCO3) < 1e-6 and abs(gamma_CO3 - prev_gamma_CO3) < 1e-6:
                    gamma_converge = True    

            c_CO2_aq_1 = calculate_CO2_aq_closed_inter(Kw, K1, K2, c_H, gamma_H, gamma_OH, gamma_CO2_aq, gamma_HCO3, gamma_CO3, c_DIC_initial, c_cation_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp, XSO4_gyp_closed)
            p_CO2_aq_pcatm = 100 * calculate_p_CO2(c_CO2_aq_1, 'mol-atm', R, T)
            c_OH_1 = calculate_c_OH(c_H, Kw, gamma_H, gamma_OH)
            c_HCO3_1 = calculate_c_HCO3(c_H, c_CO2_aq_1, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
            c_CO3_1 = calculate_c_CO3(c_H, c_CO2_aq_1, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
            
            c_DIC_1 = calculate_c_DIC(c_CO2_aq_1, c_HCO3_1, c_CO3_1)
            c_alk_1 = calculate_c_alk(c_HCO3_1, c_CO3_1, c_OH_1, c_H)                    
                                    
            sum_cation_SO4_1 = calculate_c_cation(c_OH_1, c_HCO3_1, c_CO3_1, c_H) 
            
            c_Ca_1 = calculate_c_Ca_inter_closed(sum_cation_SO4_1, c_cation_initial, c_Ca_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, XSO4_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp)
            c_SO4_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca_initial, XCa_gyp_closed, c_SO4_initial)
            c_Mg_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca_initial, MgCa_rock_closed, c_Mg_initial)
            c_Sr_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca_initial, SrCa_rock_closed, c_Sr_initial)
            c_Na_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca_initial, NaCa_rock_closed, c_Na_initial)
            c_K_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca_initial, KCa_rock_closed, c_K_initial)
            
            dC_DIC_1 = calculate_dC_DIC_dis(dC_DIC, c_DIC, c_DIC_1, dC_calcite, c_Ca, c_Ca_1, XCa_carb_closed)
            pMC_DIC_1 = calculate_dC_DIC_dis(pMC_DIC, c_DIC, c_DIC_1, pMC_calcite,  c_Ca, c_Ca_1, XCa_carb_closed)
            
            # d42Ca = d42Ca_rock_closed
            # dSr = dSr_rock_closed
            d42Ca_1 = (d42Ca * c_Ca + (c_Ca_1-c_Ca) * d42Ca_rock_closed) / c_Ca_1
            dSr_1 = (dSr * c_Sr + (c_Sr_1-c_Sr) * dSr_rock_closed) / c_Sr_1
                
            dC_CO2_aq_1 = calculate_dC_CO2_aq(c_CO2_aq_1, c_HCO3_1, c_CO3_1, dC_DIC_1, aC_CO2aq_HCO3, aC_CO2aq_CO3)
            dC_HCO3_1 = calculate_dC_HCO3(dC_CO2_aq_1, aC_CO2aq_HCO3)
            dC_CO3_1 = calculate_dC_CO3(dC_CO2_aq_1, aC_CO2aq_CO3)
            
            pMC_CO2_aq_1 = calculate_dC_CO2_aq(c_CO2_aq_1, c_HCO3_1, c_CO3_1, pMC_DIC_1, apMC_CO2aq_HCO3, apMC_CO2aq_CO3)
            pMC_HCO3_1 = calculate_dC_HCO3(pMC_CO2_aq_1, apMC_CO2aq_HCO3)
            pMC_CO3_1 = calculate_dC_CO3(pMC_CO2_aq_1, apMC_CO2aq_CO3)
            
            c_CO2_aq, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3, c_OH, c_HCO3, c_CO3, c_Ca, c_Mg, c_Sr, c_Na, c_K, c_SO4, c_SO4_gyp, c_SO4_pyr, c_cation, c_DIC, c_alk, dC_DIC, pMC_DIC, d42Ca, dSr = \
                c_CO2_aq_1, dC_CO2_aq_1, pMC_CO2_aq_1, dC_HCO3_1, pMC_HCO3_1, dC_CO3_1, pMC_CO3_1, c_OH_1, c_HCO3_1, c_CO3_1, c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1, c_K_1, c_SO4_1, c_SO4_gyp_1, c_SO4_pyr_1, c_cation_1, c_DIC_1, c_alk_1, dC_DIC_1, pMC_DIC_1 , d42Ca_1, dSr_1

            Q = Q_calculate(gamma_Ca, c_Ca, gamma_CO3, c_CO3)
            omega = omega_calculate(Q, Ksp)
            print(f'phase = {phase}')
            
            initial_guess_closed = [c_cation, c_SO4]                  
            for key, value in zip(current_results.keys(), [label, pH, c_H, c_OH,\
                                                           c_DIC, c_CO2_aq, p_CO2_aq_pcatm, c_HCO3,  c_CO3,\
                                                           dC_DIC, pMC_DIC, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3,\
                                                           c_cation, c_Ca, c_Mg, c_Sr, c_Na_rain+c_Na, c_K, c_SO4, c_Cl_rain, I,\
                                                           d42Ca, dSr, omega, None, None, None, None, None, None, None]):
                current_results[key].append(value)  
        if system in ('open_closed_unsaturated_degas/prep', 'open_closed_unsaturated_degas'):
                phase = 'river_degassing'
                
        else:
                phase = "groundwater_precipitation"
                
    # elif phase == "groundwater_precipitation":
    #     label = 'groundwater_precipitation'
    #     initial_guess = [c_H, c_CO2_aq]
                                   
    #     c_CO3_prec = c_CO3 * F_CO3
    #     c_Ca_prec = c_CO3_prec
        
    #     F_Ca = calculate_P_Ca(c_CO3, c_Ca, F_CO3)  
    #     c_Ca_1 = c_Ca - c_Ca_prec
    #     c_Sr_1 = c_Sr * (1 - F_Ca * DC_Sr)
    #     c_Mg_1 = c_Mg * (1 - F_Ca * DC_Mg)
    #     c_Na_1 = c_Na * (1 - F_Ca * DC_Na)
    #     c_K_1 = c_K * (1 - F_Ca * DC_K)
    #     c_CO3_1 = c_CO3 - c_CO3_prec
    #     c_alk_1 = c_alk - 2 * c_CO3_prec
    #     c_DIC_1 = c_DIC - c_CO3_prec
            
    #     dC_CO3_prec = dC_CO3 + aC_prec_calcite_CO3
    #     dC_CO3_1 = (c_CO3 * dC_CO3 - c_CO3_prec * dC_CO3_prec)/c_CO3_1
    #     dC_DIC_1 = calculate_dC_DIC_closed(c_CO2_aq, c_HCO3, c_CO3_1, dC_CO2_aq, dC_HCO3, dC_CO3_1)

    #     pMC_CO3_prec = pMC_CO3 + apMC_prec_calcite_CO3
    #     pMC_CO3_1 = (c_CO3 * pMC_CO3 - c_CO3_prec * pMC_CO3_prec)/c_CO3_1
    #     pMC_DIC_1 = calculate_dC_DIC_closed(c_CO2_aq, c_HCO3, c_CO3_1, pMC_CO2_aq, pMC_HCO3, pMC_CO3_1)

    #     gamma_converge = False
    #     iteration_gamma = 0
    #     while not gamma_converge:
    #         iteration_gamma += 1
    #         prev_gamma_H, prev_gamma_OH, prev_gamma_HCO3, prev_gamma_CO3 = gamma_H, gamma_OH, gamma_HCO3, gamma_CO3
    #         c_H_CO2_solution_gamma = fsolve(residual_c_H, initial_guess, args=(c_alk_1, c_DIC_1, Kw, K1, K2, gamma_H, gamma_OH, gamma_HCO3, gamma_CO3, gamma_CO2_aq))
    #         c_H_gamma, c_CO2_aq_gamma = c_H_CO2_solution_gamma
        
    #         c_HCO3_gamma = calculate_c_HCO3(c_H_gamma, c_CO2_aq_gamma, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
    #         c_CO3_gamma = calculate_c_CO3(c_H_gamma, c_CO2_aq_gamma, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
    #         c_OH_gamma = calculate_c_OH(c_H_gamma, Kw, gamma_H, gamma_OH)
                                        
    #         concentrations = [c_H_gamma, c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1+c_Na_rain, c_K_1, c_Cl_rain, c_SO4]
    #         charges = [1, -1, -1, -2, 2, 2, 2, 1, 1, -1, -2]
            
    #         I = ionic_strength(concentrations, charges)
        
    #         gamma_H = calculate_activity_coefficient(A, B, 1, I, a_H)
    #         gamma_OH = calculate_activity_coefficient(A, B, -1, I, a_OH)
    #         gamma_HCO3 = calculate_activity_coefficient(A, B, -1, I, a_HCO3)
    #         gamma_CO3 = calculate_activity_coefficient(A, B, -2, I, a_CO3)
    #         gamma_Ca = calculate_activity_coefficient(A, B, 2, I, a_Ca)

    #         gamma_converge = abs(gamma_H - prev_gamma_H) < 1e-6 and abs(gamma_OH - prev_gamma_OH) < 1e-6 and \
    #             abs(gamma_HCO3 - prev_gamma_HCO3) < 1e-6 and abs(gamma_CO3 - prev_gamma_CO3) < 1e-6

    #     c_H_2, c_CO2_aq_2 = fsolve(residual_c_H, initial_guess, args=(c_alk_1, c_DIC_1, Kw, K1, K2, gamma_H, gamma_OH, gamma_HCO3, gamma_CO3, gamma_CO2_aq))
    #     pH = -np.log10(c_H_2)
    #     p_CO2_aq_pcatm = 100 * calculate_p_CO2(c_CO2_aq_2, 'mol-atm', R, T)
    #     c_OH_2 = calculate_c_OH(c_H_2, Kw, gamma_H, gamma_OH)
    #     c_CO3_2 = calculate_c_CO3(c_H_2, c_CO2_aq_2, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
    #     c_HCO3_2 = calculate_c_HCO3(c_H_2, c_CO2_aq_2, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)

    #     dC_CO3_2 = (c_DIC_1 * dC_DIC_1 - c_CO2_aq_2 * aC_CO2aq_CO3 - c_HCO3_2 * aC_HCO3_CO3) / c_DIC_1
    #     # Principle: dC_CO3_old * c_CO3_old - dC_CO3_drop * c_CO3_drop = dC_CO3_new * c_CO3_new
    #     # Principle: dC_CO3_drop * c_CO3_drop = (c_CO2_new - c_CO2_old) * aC_CO2aq_CO3 + (c_HCO3_new - c_HCO3_old) * aC_CO2aq_HCO3
    #     dC_CO2_aq_2 = dC_CO3_2 + aC_CO2aq_CO3
    #     dC_HCO3_2 = dC_CO3_2 + aC_HCO3_CO3
        
    #     pMC_CO3_2 = (c_DIC_1 * pMC_DIC_1 - c_CO2_aq_2 * apMC_CO2aq_CO3 - c_HCO3_2 * apMC_HCO3_CO3) / c_DIC_1
    #     pMC_CO2_aq_2 = pMC_CO3_2 + apMC_CO2aq_CO3
    #     pMC_HCO3_2 = pMC_CO3_2 + apMC_HCO3_CO3

    #     # d42Ca_1 = ((d42Ca/1000 +1) * ((1-P_Ca) / (1-P_Ca/a42Ca_calcite_prec)) -1)*1000  
    #     d42Ca_1 = (d42Ca * c_Ca - (d42Ca + a42Ca_calcite_Ca) * c_Ca_prec) / c_Ca_1
        
    #     c_Ca, c_Mg, c_Sr, c_Na, c_K, c_HCO3, c_H, c_CO2_aq, c_CO3, c_OH, c_DIC, c_alk, dC_HCO3, dC_CO2_aq, dC_CO3, dC_DIC, pMC_HCO3, pMC_CO2_aq, pMC_CO3, pMC_DIC, d42Ca = \
    #         c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1, c_K_1, c_HCO3_2, c_H_2, c_CO2_aq_2, c_CO3_2, c_OH_2, c_DIC_1, c_alk_1, dC_HCO3_2, dC_CO2_aq_2, dC_CO3_2, dC_DIC_1, pMC_HCO3_2, pMC_CO2_aq_2, pMC_CO3_2, pMC_DIC_1, d42Ca_1
            
    #     Q = Q_calculate(gamma_Ca, c_Ca, gamma_CO3, c_CO3)
    #     omega = omega_calculate(Q, Ksp)

    #     for key, value in zip(current_results.keys(), [label, pH, c_H, c_OH,\
    #                                                    c_DIC, c_CO2_aq, p_CO2_aq_pcatm, c_HCO3,  c_CO3,\
    #                                                    dC_DIC, pMC_DIC, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3,\
    #                                                    c_cation, c_Ca, c_Mg, c_Sr, c_Na_rain+c_Na, c_K, c_SO4, c_Cl_rain, I,\
    #                                                    d42Ca, dSr, omega, None, None, None, None, None, None, None]):
    #         current_results[key].append(value)  
    #     phase = "groundwater_precipitation_dissolution"  

    # elif phase == "groundwater_precipitation_dissolution":
    #     iteration_calcite += 1
    #     initial_guess = [c_cation, c_SO4]
    #     label = 'groundwater_precipitation_dissolution'
    #     while omega <= 1: # this is for calcite dissolution upto SI = 0/ omega = 1
            
    #    # while x < x_degas: # this is for degas

    #         pH += pH_increment
    #         c_H = 10 ** (-pH)
            
    #         gamma_converge = False

    #         while not gamma_converge:
    #             iteration_gamma += 1

    #             prev_gamma_H = gamma_H
    #             prev_gamma_OH = gamma_OH
    #             prev_gamma_HCO3 = gamma_HCO3
    #             prev_gamma_CO3 = gamma_CO3
    #             prev_gamma_Ca = gamma_Ca

    #             c_CO2_aq_gamma = calculate_CO2_aq_closed_inter(Kw, K1, K2, c_H, gamma_H, gamma_OH, gamma_CO2_aq, gamma_HCO3, gamma_CO3, c_DIC_initial, c_cation_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp, XSO4_gyp_closed)

    #             c_OH_gamma = calculate_c_OH(c_H, Kw, gamma_H, gamma_OH)
    #             c_HCO3_gamma = calculate_c_HCO3(c_H, c_CO2_aq_gamma, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
    #             c_CO3_gamma = calculate_c_CO3(c_H, c_CO2_aq_gamma, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
                
    #             sum_cation_SO4_gamma = calculate_c_cation(c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_H)
    #             c_Ca_gamma = calculate_c_Ca_inter_closed(sum_cation_SO4_gamma, c_cation_initial, c_Ca_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, XSO4_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp)
    #             c_SO4_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca, XCa_gyp_closed, c_SO4)
    #             c_Mg_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca, MgCa_rock_closed, c_Mg)
    #             c_Sr_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca, SrCa_rock_closed, c_Sr)
    #             c_Na_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca, NaCa_rock_closed, c_Na)
    #             c_K_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca, KCa_rock_closed, c_K)
                        
    #             concentrations = [c_H, c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_Ca_gamma, c_Mg_gamma, c_Sr_gamma, c_Na_gamma, c_K_gamma, c_Cl_rain, c_SO4_gamma]
    #             charges = [1, -1, -1, -2, 2, 2, 2, 1, 1, -1, -2]
    #             I = ionic_strength(concentrations, charges)
                 
    #             gamma_H = calculate_activity_coefficient(A, B, 1, I, a_H)
    #             gamma_OH = calculate_activity_coefficient(A, B, -1, I, a_OH)
    #             gamma_HCO3 = calculate_activity_coefficient(A, B, -1, I, a_HCO3)
    #             gamma_CO3 = calculate_activity_coefficient(A, B, -2, I, a_CO3)
    #             gamma_Ca = calculate_activity_coefficient(A, B, 2, I, a_Ca)
 
    #             if abs(gamma_H - prev_gamma_H) < 1e-6 and abs(gamma_Ca - prev_gamma_Ca) < 1e-6 and abs(gamma_OH - prev_gamma_OH) < 1e-6 and \
    #                 abs(gamma_HCO3 - prev_gamma_HCO3) < 1e-6 and abs(gamma_CO3 - prev_gamma_CO3) < 1e-6:
    #                 gamma_converge = True                    
               

    #         c_CO2_aq_1 = calculate_CO2_aq_closed_inter(Kw, K1, K2, c_H, gamma_H, gamma_OH, gamma_CO2_aq, gamma_HCO3, gamma_CO3, c_DIC_initial, c_cation_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp, XSO4_gyp_closed)
    #         p_CO2_aq_pcatm = 100 * calculate_p_CO2(c_CO2_aq_1, 'mol-atm', R, T)
    #         c_OH_1 = calculate_c_OH(c_H, Kw, gamma_H, gamma_OH)
    #         c_HCO3_1 = calculate_c_HCO3(c_H, c_CO2_aq_1, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
    #         c_CO3_1 = calculate_c_CO3(c_H, c_CO2_aq_1, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
        
    #         c_DIC_1 = calculate_c_DIC(c_CO2_aq_1, c_HCO3_1, c_CO3_1)
    #         c_alk_1 = calculate_c_alk(c_HCO3_1, c_CO3_1, c_OH_1, c_H) 
            
    #         sum_cation_SO4_1 = calculate_c_cation(c_OH_1, c_HCO3_1, c_CO3_1,c_H)
           
    #         c_Ca_1 = calculate_c_Ca_inter_closed(sum_cation_SO4_1, c_cation_initial, c_Ca_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, XSO4_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp)
    #         c_SO4_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca, XCa_gyp_closed, c_SO4)
    #         c_Mg_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca, MgCa_rock_closed, c_Mg)
    #         c_Sr_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca, SrCa_rock_closed, c_Sr)
    #         c_Na_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca, NaCa_rock_closed, c_Na)
    #         c_K_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca, KCa_rock_closed, c_K)     

    #         c_SO4_gyp_1 = c_SO4_1 * XSO4_gyp_closed
    #         c_SO4_pyr_1 = c_SO4_1 * XSO4_pyr_closed
  
    #         dC_DIC_1 = calculate_dC_DIC_dis(dC_DIC, c_DIC, c_DIC_1, dC_calcite,  c_Ca, c_Ca_1, XCa_carb_closed)
    #         pMC_DIC_1 = calculate_dC_DIC_dis(pMC_DIC, c_DIC, c_DIC_1, pMC_calcite,  c_Ca, c_Ca_1, XCa_carb_closed)
   
    #         dC_CO2_aq_1 = calculate_dC_CO2_aq(c_CO2_aq_1, c_HCO3_1, c_CO3_1, dC_DIC_1, aC_CO2aq_HCO3, aC_CO2aq_CO3)
    #         dC_HCO3_1 = calculate_dC_HCO3(dC_CO2_aq_1, aC_CO2aq_HCO3)
    #         dC_CO3_1 = calculate_dC_CO3(dC_CO2_aq_1, aC_CO2aq_CO3)
           
    #         pMC_CO2_aq_1 = calculate_dC_CO2_aq(c_CO2_aq_1, c_HCO3_1, c_CO3_1, pMC_DIC_1, apMC_CO2aq_HCO3, apMC_CO2aq_CO3)
    #         pMC_HCO3_1 = calculate_dC_HCO3(pMC_CO2_aq_1, apMC_CO2aq_HCO3)
    #         pMC_CO3_1 = calculate_dC_CO3(pMC_CO2_aq_1, apMC_CO2aq_CO3) 
            
    #         d42Ca_1 = (d42Ca * c_Ca + (c_Ca_1-c_Ca) * d42Ca_rock_closed) / c_Ca_1
    #         dSr_1 = (dSr * c_Sr + (c_Sr_1-c_Sr) * dSr_rock_closed) / c_Sr_1
    #         # d42Ca_1 = (((d42Ca/1000 +1) + (d42Ca_rock_closed/1000+1) * ((P_Ca/(1-P_Ca)) * (c_Ca_1/c_Sr_1) * SrCa_rock_closed)) / (1 + (P_Ca/(1-P_Ca)) * (c_Ca_1/c_Sr_1) * SrCa_rock_closed) - 1) * 1000
            
    #         # dSr_1 = (dSr+ dSr_carb * ((P_Ca/(1-P_Ca)) * (c_Ca_1/c_Sr_1) * SrCa_carb)) / (1 + (P_Ca/(1-P_Ca)) * (c_Ca_1/c_Sr_1) * SrCa_carb) 

    #         # c_Ca, c_Mg, c_Sr, c_Na, c_K, c_HCO3, c_H, c_CO2_aq, c_CO3, c_OH, c_SO4, c_DIC, c_alk, dC_HCO3, dC_CO2_aq, dC_CO3, dC_DIC, pMC_HCO3, pMC_CO2_aq, pMC_CO3, pMC_DIC, d42Ca, dSr = \
    #         #     c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1, c_K_1, c_HCO3_1, c_H_1, c_CO2_aq_1, c_CO3_1, c_OH_1, c_SO4_1, c_DIC_1, c_alk_1, dC_HCO3_1, dC_CO2_aq_1, dC_CO3_1, dC_DIC_1, pMC_HCO3_1, pMC_CO2_aq_1, pMC_CO3_1, pMC_DIC_1, d42Ca_1, dSr_1
    #         c_CO2_aq, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3, c_OH, c_HCO3, c_CO3, c_Ca, c_Mg, c_Sr, c_Na, c_K, c_SO4, c_SO4_gyp, c_SO4_pyr, c_cation, c_DIC, c_alk, dC_DIC, pMC_DIC, d42Ca, dSr = \
    #             c_CO2_aq_1, dC_CO2_aq_1, pMC_CO2_aq_1, dC_HCO3_1, pMC_HCO3_1, dC_CO3_1, pMC_CO3_1, c_OH_1, c_HCO3_1, c_CO3_1, c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1, c_K_1, c_SO4_1, c_SO4_gyp_1, c_SO4_pyr_1, c_cation_1, c_DIC_1, c_alk_1, dC_DIC_1, pMC_DIC_1, d42Ca_1, dSr_1 

    #         Q = Q_calculate(gamma_Ca, c_Ca, gamma_CO3, c_CO3)
    #         omega = omega_calculate(Q, Ksp)

    #         for key, value in zip(current_results.keys(), [label, pH, c_H, c_OH,\
    #                                                        c_DIC, c_CO2_aq, p_CO2_aq_pcatm, c_HCO3,  c_CO3,\
    #                                                        dC_DIC, pMC_DIC, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3,\
    #                                                        c_cation, c_Ca, c_Mg, c_Sr, c_Na_rain+c_Na, c_K, c_SO4, c_Cl_rain, I,\
    #                                                        d42Ca, dSr, omega, None, None, None, None, None, None, None]):
    #             current_results[key].append(value)  
    #     if c_Ca < 0: 
    #             finished = True
                
    #     if iteration_calcite <= n_gw_scp:
    #         phase = 'groundwater_precipitation'
            
    #     else:
    #         if omega<=1.0001:
    #             phase = "river_degassing"  
    #             go_to_precipitation = False    
    #         # else:
    #         elif omega>1.0001:
    #             phase = "river_precipitation"
    #             go_to_precipitation = True
        # print('after gw SI =' + str(SI))
    elif phase == "groundwater_precipitation":
        while iteration_calcite < n_gw_scp:
            iteration_calcite += 1
            label = 'groundwater_precipitation'
            initial_guess = [c_H, c_CO2_aq]
                                       
            c_CO3_prec = c_CO3 * F_CO3
            c_Ca_prec = c_CO3_prec
            
            F_Ca = calculate_P_Ca(c_CO3, c_Ca, F_CO3)  
            c_Ca_1 = c_Ca - c_Ca_prec
            c_Sr_1 = c_Sr * (1 - F_Ca * DC_Sr)
            c_Mg_1 = c_Mg * (1 - F_Ca * DC_Mg)
            c_Na_1 = c_Na * (1 - F_Ca * DC_Na)
            c_K_1 = c_K * (1 - F_Ca * DC_K)
            c_CO3_1 = c_CO3 - c_CO3_prec
            c_alk_1 = c_alk - 2 * c_CO3_prec
            c_DIC_1 = c_DIC - c_CO3_prec
                
            dC_CO3_prec = dC_CO3 + aC_prec_calcite_CO3
            dC_CO3_1 = (c_CO3 * dC_CO3 - c_CO3_prec * dC_CO3_prec)/c_CO3_1
            dC_DIC_1 = calculate_dC_DIC_closed(c_CO2_aq, c_HCO3, c_CO3_1, dC_CO2_aq, dC_HCO3, dC_CO3_1)
    
            pMC_CO3_prec = pMC_CO3 + apMC_prec_calcite_CO3
            pMC_CO3_1 = (c_CO3 * pMC_CO3 - c_CO3_prec * pMC_CO3_prec)/c_CO3_1
            pMC_DIC_1 = calculate_dC_DIC_closed(c_CO2_aq, c_HCO3, c_CO3_1, pMC_CO2_aq, pMC_HCO3, pMC_CO3_1)
    
            gamma_converge = False
            iteration_gamma = 0
            while not gamma_converge:
                iteration_gamma += 1
                prev_gamma_H, prev_gamma_OH, prev_gamma_HCO3, prev_gamma_CO3 = gamma_H, gamma_OH, gamma_HCO3, gamma_CO3
                c_H_CO2_solution_gamma = fsolve(residual_c_H, initial_guess, args=(c_alk_1, c_DIC_1, Kw, K1, K2, gamma_H, gamma_OH, gamma_HCO3, gamma_CO3, gamma_CO2_aq))
                c_H_gamma, c_CO2_aq_gamma = c_H_CO2_solution_gamma
            
                c_HCO3_gamma = calculate_c_HCO3(c_H_gamma, c_CO2_aq_gamma, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
                c_CO3_gamma = calculate_c_CO3(c_H_gamma, c_CO2_aq_gamma, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
                c_OH_gamma = calculate_c_OH(c_H_gamma, Kw, gamma_H, gamma_OH)
                                            
                concentrations = [c_H_gamma, c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1+c_Na_rain, c_K_1, c_Cl_rain, c_SO4]
                charges = [1, -1, -1, -2, 2, 2, 2, 1, 1, -1, -2]
                
                I = ionic_strength(concentrations, charges)
                
                gamma_H = calculate_activity_coefficient(A, B, 1, I, a_H)
                gamma_OH = calculate_activity_coefficient(A, B, -1, I, a_OH)
                gamma_HCO3 = calculate_activity_coefficient(A, B, -1, I, a_HCO3)
                gamma_CO3 = calculate_activity_coefficient(A, B, -2, I, a_CO3)
                gamma_Ca = calculate_activity_coefficient(A, B, 2, I, a_Ca)
    
                gamma_converge = abs(gamma_H - prev_gamma_H) < 1e-6 and abs(gamma_OH - prev_gamma_OH) < 1e-6 and \
                    abs(gamma_HCO3 - prev_gamma_HCO3) < 1e-6 and abs(gamma_CO3 - prev_gamma_CO3) < 1e-6
    
            c_H_2, c_CO2_aq_2 = fsolve(residual_c_H, initial_guess, args=(c_alk_1, c_DIC_1, Kw, K1, K2, gamma_H, gamma_OH, gamma_HCO3, gamma_CO3, gamma_CO2_aq))
            pH = -np.log10(c_H_2)
            p_CO2_aq_pcatm = 100 * calculate_p_CO2(c_CO2_aq_2, 'mol-atm', R, T)
            c_OH_2 = calculate_c_OH(c_H_2, Kw, gamma_H, gamma_OH)
            c_CO3_2 = calculate_c_CO3(c_H_2, c_CO2_aq_2, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
            c_HCO3_2 = calculate_c_HCO3(c_H_2, c_CO2_aq_2, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
    
            dC_CO3_2 = (c_DIC_1 * dC_DIC_1 - c_CO2_aq_2 * aC_CO2aq_CO3 - c_HCO3_2 * aC_HCO3_CO3) / c_DIC_1
            # Principle: dC_CO3_old * c_CO3_old - dC_CO3_drop * c_CO3_drop = dC_CO3_new * c_CO3_new
            # Principle: dC_CO3_drop * c_CO3_drop = (c_CO2_new - c_CO2_old) * aC_CO2aq_CO3 + (c_HCO3_new - c_HCO3_old) * aC_CO2aq_HCO3
            dC_CO2_aq_2 = dC_CO3_2 + aC_CO2aq_CO3
            dC_HCO3_2 = dC_CO3_2 + aC_HCO3_CO3
            
            pMC_CO3_2 = (c_DIC_1 * pMC_DIC_1 - c_CO2_aq_2 * apMC_CO2aq_CO3 - c_HCO3_2 * apMC_HCO3_CO3) / c_DIC_1
            pMC_CO2_aq_2 = pMC_CO3_2 + apMC_CO2aq_CO3
            pMC_HCO3_2 = pMC_CO3_2 + apMC_HCO3_CO3
    
            # d42Ca_1 = ((d42Ca/1000 +1) * ((1-P_Ca) / (1-P_Ca/a42Ca_calcite_prec)) -1)*1000  
            d42Ca_1 = (d42Ca * c_Ca - (d42Ca + a42Ca_calcite_Ca) * c_Ca_prec) / c_Ca_1
            
            c_Ca, c_Mg, c_Sr, c_Na, c_K, c_HCO3, c_H, c_CO2_aq, c_CO3, c_OH, c_DIC, c_alk, dC_HCO3, dC_CO2_aq, dC_CO3, dC_DIC, pMC_HCO3, pMC_CO2_aq, pMC_CO3, pMC_DIC, d42Ca = \
                c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1, c_K_1, c_HCO3_2, c_H_2, c_CO2_aq_2, c_CO3_2, c_OH_2, c_DIC_1, c_alk_1, dC_HCO3_2, dC_CO2_aq_2, dC_CO3_2, dC_DIC_1, pMC_HCO3_2, pMC_CO2_aq_2, pMC_CO3_2, pMC_DIC_1, d42Ca_1
                
            Q = Q_calculate(gamma_Ca, c_Ca, gamma_CO3, c_CO3)
            omega = omega_calculate(Q, Ksp)
    
            for key, value in zip(current_results.keys(), [label, pH, c_H, c_OH,\
                                                           c_DIC, c_CO2_aq, p_CO2_aq_pcatm, c_HCO3,  c_CO3,\
                                                           dC_DIC, pMC_DIC, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3,\
                                                           c_cation, c_Ca, c_Mg, c_Sr, c_Na_rain+c_Na, c_K, c_SO4, c_Cl_rain, I,\
                                                           d42Ca, dSr, omega, None, None, None, None, None, None, None]):
                current_results[key].append(value)  
            
            phase = "groundwater_precipitation_dissolution"  
    
        # elif phase == "groundwater_precipitation_dissolution":
            # iteration_calcite += 1
            # initial_guess = [c_cation, c_SO4]
            label = 'groundwater_precipitation_dissolution'
            while omega < 1: # this is for calcite dissolution upto SI = 0/ omega = 1
                
           # while x < x_degas: # this is for degas
    
                pH += pH_increment
                c_H = 10 ** (-pH)
                
                gamma_converge = False
    
                while not gamma_converge:
                    iteration_gamma += 1
    
                    prev_gamma_H = gamma_H
                    prev_gamma_OH = gamma_OH
                    prev_gamma_HCO3 = gamma_HCO3
                    prev_gamma_CO3 = gamma_CO3
                    prev_gamma_Ca = gamma_Ca
    
                    c_CO2_aq_gamma = calculate_CO2_aq_closed_inter(Kw, K1, K2, c_H, gamma_H, gamma_OH, gamma_CO2_aq, gamma_HCO3, gamma_CO3, c_DIC_initial, c_cation_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp, XSO4_gyp_closed)
    
                    c_OH_gamma = calculate_c_OH(c_H, Kw, gamma_H, gamma_OH)
                    c_HCO3_gamma = calculate_c_HCO3(c_H, c_CO2_aq_gamma, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
                    c_CO3_gamma = calculate_c_CO3(c_H, c_CO2_aq_gamma, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
                    
                    sum_cation_SO4_gamma = calculate_c_cation(c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_H)
                    c_Ca_gamma = calculate_c_Ca_inter_closed(sum_cation_SO4_gamma, c_cation_initial, c_Ca_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, XSO4_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp)
                    c_SO4_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca, XCa_gyp_closed, c_SO4)
                    c_Mg_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca, MgCa_rock_closed, c_Mg)
                    c_Sr_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca, SrCa_rock_closed, c_Sr)
                    c_Na_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca, NaCa_rock_closed, c_Na)
                    c_K_gamma = calculate_c_ion_inter_closed(c_Ca_gamma, c_Ca, KCa_rock_closed, c_K)
                            
                    concentrations = [c_H, c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_Ca_gamma, c_Mg_gamma, c_Sr_gamma, c_Na_gamma, c_K_gamma, c_Cl_rain, c_SO4_gamma]
                    charges = [1, -1, -1, -2, 2, 2, 2, 1, 1, -1, -2]
                    I = ionic_strength(concentrations, charges)
                    
                    gamma_H = calculate_activity_coefficient(A, B, 1, I, a_H)
                    gamma_OH = calculate_activity_coefficient(A, B, -1, I, a_OH)
                    gamma_HCO3 = calculate_activity_coefficient(A, B, -1, I, a_HCO3)
                    gamma_CO3 = calculate_activity_coefficient(A, B, -2, I, a_CO3)
                    gamma_Ca = calculate_activity_coefficient(A, B, 2, I, a_Ca)
     
                    if abs(gamma_H - prev_gamma_H) < 1e-6 and abs(gamma_Ca - prev_gamma_Ca) < 1e-6 and abs(gamma_OH - prev_gamma_OH) < 1e-6 and \
                        abs(gamma_HCO3 - prev_gamma_HCO3) < 1e-6 and abs(gamma_CO3 - prev_gamma_CO3) < 1e-6:
                        gamma_converge = True                    
                   
    
                c_CO2_aq_1 = calculate_CO2_aq_closed_inter(Kw, K1, K2, c_H, gamma_H, gamma_OH, gamma_CO2_aq, gamma_HCO3, gamma_CO3, c_DIC_initial, c_cation_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp, XSO4_gyp_closed)
                p_CO2_aq_pcatm = 100 * calculate_p_CO2(c_CO2_aq_1, 'mol-atm', R, T)
                c_OH_1 = calculate_c_OH(c_H, Kw, gamma_H, gamma_OH)
                c_HCO3_1 = calculate_c_HCO3(c_H, c_CO2_aq_1, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
                c_CO3_1 = calculate_c_CO3(c_H, c_CO2_aq_1, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
            
                c_DIC_1 = calculate_c_DIC(c_CO2_aq_1, c_HCO3_1, c_CO3_1)
                c_alk_1 = calculate_c_alk(c_HCO3_1, c_CO3_1, c_OH_1, c_H) 
                
                sum_cation_SO4_1 = calculate_c_cation(c_OH_1, c_HCO3_1, c_CO3_1,c_H)
               
                c_Ca_1 = calculate_c_Ca_inter_closed(sum_cation_SO4_1, c_cation_initial, c_Ca_initial, XCa_carb_closed, XCa_sil_closed, XCa_gyp_closed, XSO4_gyp_closed, MgCa_carb, SrCa_carb, MgCa_sil, SrCa_sil, NaCa_sil, KCa_sil, MgCa_gyp, SrCa_gyp)
                c_SO4_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca, XCa_gyp_closed, c_SO4)
                c_Mg_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca, MgCa_rock_closed, c_Mg)
                c_Sr_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca, SrCa_rock_closed, c_Sr)
                c_Na_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca, NaCa_rock_closed, c_Na)
                c_K_1 = calculate_c_ion_inter_closed(c_Ca_1, c_Ca, KCa_rock_closed, c_K)     
    
                c_SO4_gyp_1 = c_SO4_1 * XSO4_gyp_closed
                c_SO4_pyr_1 = c_SO4_1 * XSO4_pyr_closed
      
                dC_DIC_1 = calculate_dC_DIC_dis(dC_DIC, c_DIC, c_DIC_1, dC_calcite,  c_Ca, c_Ca_1, XCa_carb_closed)
                pMC_DIC_1 = calculate_dC_DIC_dis(pMC_DIC, c_DIC, c_DIC_1, pMC_calcite,  c_Ca, c_Ca_1, XCa_carb_closed)
       
                dC_CO2_aq_1 = calculate_dC_CO2_aq(c_CO2_aq_1, c_HCO3_1, c_CO3_1, dC_DIC_1, aC_CO2aq_HCO3, aC_CO2aq_CO3)
                dC_HCO3_1 = calculate_dC_HCO3(dC_CO2_aq_1, aC_CO2aq_HCO3)
                dC_CO3_1 = calculate_dC_CO3(dC_CO2_aq_1, aC_CO2aq_CO3)
               
                pMC_CO2_aq_1 = calculate_dC_CO2_aq(c_CO2_aq_1, c_HCO3_1, c_CO3_1, pMC_DIC_1, apMC_CO2aq_HCO3, apMC_CO2aq_CO3)
                pMC_HCO3_1 = calculate_dC_HCO3(pMC_CO2_aq_1, apMC_CO2aq_HCO3)
                pMC_CO3_1 = calculate_dC_CO3(pMC_CO2_aq_1, apMC_CO2aq_CO3) 
                
                d42Ca_1 = (d42Ca * c_Ca + (c_Ca_1-c_Ca) * d42Ca_rock_closed) / c_Ca_1
                dSr_1 = (dSr * c_Sr + (c_Sr_1-c_Sr) * dSr_rock_closed) / c_Sr_1
                # d42Ca_1 = (((d42Ca/1000 +1) + (d42Ca_rock_closed/1000+1) * ((P_Ca/(1-P_Ca)) * (c_Ca_1/c_Sr_1) * SrCa_rock_closed)) / (1 + (P_Ca/(1-P_Ca)) * (c_Ca_1/c_Sr_1) * SrCa_rock_closed) - 1) * 1000
                
                # dSr_1 = (dSr+ dSr_carb * ((P_Ca/(1-P_Ca)) * (c_Ca_1/c_Sr_1) * SrCa_carb)) / (1 + (P_Ca/(1-P_Ca)) * (c_Ca_1/c_Sr_1) * SrCa_carb) 
    
                # c_Ca, c_Mg, c_Sr, c_Na, c_K, c_HCO3, c_H, c_CO2_aq, c_CO3, c_OH, c_SO4, c_DIC, c_alk, dC_HCO3, dC_CO2_aq, dC_CO3, dC_DIC, pMC_HCO3, pMC_CO2_aq, pMC_CO3, pMC_DIC, d42Ca, dSr = \
                #     c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1, c_K_1, c_HCO3_1, c_H_1, c_CO2_aq_1, c_CO3_1, c_OH_1, c_SO4_1, c_DIC_1, c_alk_1, dC_HCO3_1, dC_CO2_aq_1, dC_CO3_1, dC_DIC_1, pMC_HCO3_1, pMC_CO2_aq_1, pMC_CO3_1, pMC_DIC_1, d42Ca_1, dSr_1
                c_CO2_aq, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3, c_OH, c_HCO3, c_CO3, c_Ca, c_Mg, c_Sr, c_Na, c_K, c_SO4, c_SO4_gyp, c_SO4_pyr, c_cation, c_DIC, c_alk, dC_DIC, pMC_DIC, d42Ca, dSr = \
                    c_CO2_aq_1, dC_CO2_aq_1, pMC_CO2_aq_1, dC_HCO3_1, pMC_HCO3_1, dC_CO3_1, pMC_CO3_1, c_OH_1, c_HCO3_1, c_CO3_1, c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1, c_K_1, c_SO4_1, c_SO4_gyp_1, c_SO4_pyr_1, c_cation_1, c_DIC_1, c_alk_1, dC_DIC_1, pMC_DIC_1, d42Ca_1, dSr_1 
    
                Q = Q_calculate(gamma_Ca, c_Ca, gamma_CO3, c_CO3)
                omega = omega_calculate(Q, Ksp)
    
                for key, value in zip(current_results.keys(), [label, pH, c_H, c_OH,\
                                                               c_DIC, c_CO2_aq, p_CO2_aq_pcatm, c_HCO3,  c_CO3,\
                                                               dC_DIC, pMC_DIC, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3,\
                                                               c_cation, c_Ca, c_Mg, c_Sr, c_Na_rain+c_Na, c_K, c_SO4, c_Cl_rain, I,\
                                                               d42Ca, dSr, omega, None, None, None, None, None, None, None]):
                    current_results[key].append(value)  
            # if c_Ca < 0: 
            #         finished = True
                    
            # if iteration_calcite <= n_gw_scp:
            #     phase = 'groundwater_precipitation'
        if omega<=1.0001:
            phase = "river_degassing"  
            go_to_precipitation = False    
        # else:
        elif omega>1.0001:
            phase = "river_precipitation"
            go_to_precipitation = True   
    elif phase == "river_degassing":
        # CO2 degas, while only c_CO2_aq decreases, c_alk keep constant, meanwhile c_DIC decreases the same amount of c_CO2_aq dropdown                
        label = 'river_degassing'   
        # while omega < 1.0001:
        while go_to_precipitation == False:
            # nwhile +=1    
            c_CO2_aq_degas = c_CO2_aq * F_CO2
            c_CO2_aq_1 = c_CO2_aq - c_CO2_aq_degas
            p_CO2_aq_pcatm = 100 * calculate_p_CO2(c_CO2_aq_1, 'mol-atm', R, T)
        
            dC_CO2_aq_1 = dC_CO2_aq + (dC_CO2_aq_end - dC_CO2_aq) * F_CO2
            # dC_CO2_aq_1 = (dC_CO2_aq * c_CO2_aq - dC_CO2_aq_end * c_CO2_aq_degas) / c_CO2_aq_1
            pMC_CO2_aq_1 = pMC_CO2_aq + (pMC_CO2_aq_end - pMC_CO2_aq) * F_CO2
            # pMC_CO2_aq_1 = (pMC_CO2_aq * c_CO2_aq - pMC_CO2_aq_end * c_CO2_aq_degas) / c_CO2_aq_1
            c_DIC_1 = c_DIC - c_CO2_aq_degas
            gamma_converge = False
            iteration_gamma = 0  # reset iteration_gamma back to 0 in every calcite iteration

            while not gamma_converge:
                iteration_gamma += 1
                prev_gamma_H = gamma_H
                prev_gamma_OH = gamma_OH
                prev_gamma_HCO3 = gamma_HCO3
                prev_gamma_CO3 = gamma_CO3
                prev_gamma_Ca = gamma_Ca
            
                c_H_solution_gamma = fsolve(residual_c_H_single, c_H, args=(c_CO2_aq_1, c_alk, Kw, K1, K2, gamma_H, gamma_OH, gamma_HCO3, gamma_CO3, gamma_CO2_aq))
                c_H_gamma = c_H_solution_gamma[0]
                c_HCO3_gamma = calculate_c_HCO3(c_H_gamma, c_CO2_aq_1, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
                c_CO3_gamma = calculate_c_CO3(c_H_gamma, c_CO2_aq_1, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
                c_OH_gamma = calculate_c_OH(c_H_gamma, Kw, gamma_H, gamma_OH)
                
                concentrations = [c_H_gamma, c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1+c_Na_rain, c_K_1, c_Cl_rain, c_SO4]
                charges = [1, -1, -1, -2, 2, 2, 2, 1, 1, -1, -2]
                I = ionic_strength(concentrations, charges)

                gamma_H = calculate_activity_coefficient(A, B, 1, I, a_H)
                gamma_OH = calculate_activity_coefficient(A, B, -1, I, a_OH)
                gamma_HCO3 = calculate_activity_coefficient(A, B, -1, I, a_HCO3)
                gamma_CO3 = calculate_activity_coefficient(A, B, -2, I, a_CO3)
                gamma_Ca = calculate_activity_coefficient(A, B, 2, I, a_Ca)
                    
                if abs(gamma_H - prev_gamma_H) < 1e-6 and abs(gamma_Ca - prev_gamma_Ca) < 1e-6 and abs(gamma_OH - prev_gamma_OH) < 1e-6 and \
                    abs(gamma_HCO3 - prev_gamma_HCO3) < 1e-6 and abs(gamma_CO3 - prev_gamma_CO3) < 1e-6:
                    gamma_converge = True
                    
            c_H_solution_1 = fsolve(residual_c_H_single, c_H_gamma, args=(c_CO2_aq_1, c_alk, Kw, K1, K2, gamma_H, gamma_OH, gamma_HCO3, gamma_CO3, gamma_CO2_aq))
            c_H_1 = c_H_solution_1[0]
            pH = -math.log10(c_H_1)
            c_HCO3_1 = calculate_c_HCO3(c_H_1, c_CO2_aq_1, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
            c_CO3_1 = calculate_c_CO3(c_H_1, c_CO2_aq_1, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
            c_OH_1 = calculate_c_OH(c_H_1, Kw, gamma_H, gamma_OH)
            
            # dC_CO3 and dC_HCO3 are in derived from dC_CO2_aq in a equilibrium system
            dC_CO3_1 = dC_CO2_aq_1 - aC_CO2aq_CO3
            dC_HCO3_1 = dC_CO2_aq_1 - aC_CO2aq_HCO3
            pMC_CO3_1 = pMC_CO2_aq_1 - apMC_CO2aq_CO3
            pMC_HCO3_1 = pMC_CO2_aq_1 - apMC_CO2aq_HCO3
        
            dC_DIC_1 = calculate_dC_DIC_open(c_CO2_aq_1, c_HCO3_1, c_CO3_1, dC_CO2_aq_1, dC_HCO3_1, dC_CO3_1)
            pMC_DIC_1 = calculate_dC_DIC_open(c_CO2_aq_1, c_HCO3_1, c_CO3_1, pMC_CO2_aq_1, pMC_HCO3_1, pMC_CO3_1)
                
            c_CO2_aq, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3, c_H, c_OH, c_HCO3, c_CO3, c_DIC, dC_DIC, pMC_DIC = \
                c_CO2_aq_1, dC_CO2_aq_1, pMC_CO2_aq_1, dC_HCO3_1, pMC_HCO3_1, dC_CO3_1, pMC_CO3_1, c_H_1, c_OH_1, c_HCO3_1, c_CO3_1, c_DIC_1, dC_DIC_1, pMC_DIC_1  
            Q = Q_calculate(gamma_Ca, c_Ca, gamma_CO3, c_CO3)
            omega = omega_calculate(Q, Ksp)

            for key, value in zip(current_results.keys(), [label, pH, c_H, c_OH,\
                                                           c_DIC, c_CO2_aq, p_CO2_aq_pcatm, c_HCO3,  c_CO3,\
                                                           dC_DIC, pMC_DIC, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3,\
                                                           c_cation, c_Ca, c_Mg, c_Sr, c_Na_rain+c_Na, c_K, c_SO4, c_Cl_rain, I,\
                                                           d42Ca, dSr, omega, None, None, None, None, None, None, None]):
                current_results[key].append(value)  
            if c_CO2_aq <= c_CO2_aq_end:
                finished = True
                break
            elif  omega >= 1.0001:
                if system == 'open_closed_saturated_diss/prep_degas':
                    phase == "river_degassing"
                else:
                        
                    phase = "river_precipitation"
                    go_to_precipitation =  True
    elif phase == "river_precipitation":                    
        # precipitation_started = True
        label = 'river_precipitation'
        
        initial_guess = [c_H, c_CO2_aq]
        p_initial_guess = c_Ca * 0.001
        
           
        p_solution = fsolve(calculate_c_Ca_prec, p_initial_guess, args=(c_Ca, c_CO3, gamma_Ca, gamma_CO3, Ksp))
        # solution = calculate_c_Ca_p(c_Ca_p, c_Ca, c_CO3, gamma_Ca, gamma_CO3, Ksp)
           
        c_Ca_prec = p_solution[0]
        c_CO3_prec = c_Ca_prec
        F_Ca = c_Ca_prec /c_Ca
        c_Ca_1 = c_Ca - c_Ca_prec
        c_Sr_1 = c_Sr * (1 - F_Ca * DC_Sr)
        c_Mg_1 = c_Mg * (1 - F_Ca * DC_Mg)
        c_CO3_1 = c_CO3 - c_CO3_prec
        c_alk_1 = c_alk - 2 * c_CO3_prec
        c_DIC_1 = c_DIC - c_CO3_prec
            
        dC_CO3_prec = dC_CO3 + aC_prec_calcite_CO3
        dC_CO3_1 = (c_CO3 * dC_CO3 - c_CO3_prec * dC_CO3_prec)/c_CO3_1
        dC_DIC_1 = calculate_dC_DIC_closed(c_CO2_aq, c_HCO3, c_CO3_1, dC_CO2_aq, dC_HCO3, dC_CO3_1)

        pMC_CO3_prec = pMC_CO3 + apMC_prec_calcite_CO3
        pMC_CO3_1 = (c_CO3 * pMC_CO3 - c_CO3_prec * pMC_CO3_prec)/c_CO3_1
        pMC_DIC_1 = calculate_dC_DIC_closed(c_CO2_aq, c_HCO3, c_CO3_1, pMC_CO2_aq, pMC_HCO3, pMC_CO3_1)
        # while omega>=1.0001:
        #     initial_guess = [c_H, c_CO2_aq]

        #     c_CO3_prec = c_CO3 * F_CO3
        #     c_Ca_prec = c_CO3_prec
        #     # while iteration_calcite < 100:
        #     P_Ca = calculate_P_Ca(c_CO3, c_Ca, F_CO3)  

        #     c_Ca_1 = c_Ca - c_Ca_prec
        #     c_Sr_1 = c_Sr * (1 - P_Ca * DC_Sr)
        #     c_Mg_1 = c_Mg * (1 - P_Ca * DC_Mg)
        #     c_Na_1 = c_Na * (1 - P_Ca * DC_Na)
        #     c_K_1 = c_K * (1 - P_Ca * DC_K)
        #     c_CO3_1 = c_CO3 - c_CO3_prec
        #     c_alk_1 = c_alk - 2 * c_CO3_prec
        #     c_DIC_1 = c_DIC - c_CO3_prec
                
        #     dC_CO3_prec = dC_CO3 + aC_prec_calcite_CO3
        #     dC_CO3_1 = (c_CO3 * dC_CO3 - c_CO3_prec * dC_CO3_prec)/c_CO3_1
        #     dC_DIC_1 = calculate_dC_DIC_closed(c_CO2_aq, c_HCO3, c_CO3_1, dC_CO2_aq, dC_HCO3, dC_CO3_1)
    
        #     pMC_CO3_prec = pMC_CO3 + apMC_prec_calcite_CO3
        #     pMC_CO3_1 = (c_CO3 * pMC_CO3 - c_CO3_prec * pMC_CO3_prec)/c_CO3_1
        #     pMC_DIC_1 = calculate_dC_DIC_closed(c_CO2_aq, c_HCO3, c_CO3_1, pMC_CO2_aq, pMC_HCO3, pMC_CO3_1)

        gamma_converge = False
        iteration_gamma = 0
        while not gamma_converge:
            iteration_gamma += 1
            prev_gamma_H, prev_gamma_OH, prev_gamma_HCO3, prev_gamma_CO3 = gamma_H, gamma_OH, gamma_HCO3, gamma_CO3
            c_H_CO2_solution_gamma = fsolve(residual_c_H, initial_guess, args=(c_alk_1, c_DIC_1, Kw, K1, K2, gamma_H, gamma_OH, gamma_HCO3, gamma_CO3, gamma_CO2_aq))
            c_H_gamma, c_CO2_aq_gamma = c_H_CO2_solution_gamma
        
            c_HCO3_gamma = calculate_c_HCO3(c_H_gamma, c_CO2_aq_gamma, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
            c_CO3_gamma = calculate_c_CO3(c_H_gamma, c_CO2_aq_gamma, K1, K2, gamma_H, gamma_CO3, gamma_CO2_aq)
            c_OH_gamma = calculate_c_OH(c_H_gamma, Kw, gamma_H, gamma_OH)
            
            concentrations = [c_H_gamma, c_OH_gamma, c_HCO3_gamma, c_CO3_gamma, c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1+c_Na_rain, c_K_1, c_Cl_rain, c_SO4]
            charges = [1, -1, -1, -2, 2, 2, 2, 1, 1, -1, -2]
            I = ionic_strength(concentrations, charges)
        
            gamma_H = calculate_activity_coefficient(A, B, 1, I, a_H)
            gamma_OH = calculate_activity_coefficient(A, B, -1, I, a_OH)
            gamma_HCO3 = calculate_activity_coefficient(A, B, -1, I, a_HCO3)
            gamma_CO3 = calculate_activity_coefficient(A, B, -2, I, a_CO3)
            gamma_Ca = calculate_activity_coefficient(A, B, 2, I, a_Ca)

            gamma_converge = abs(gamma_H - prev_gamma_H) < 1e-6 and abs(gamma_OH - prev_gamma_OH) < 1e-6 and \
                abs(gamma_HCO3 - prev_gamma_HCO3) < 1e-6 and abs(gamma_CO3 - prev_gamma_CO3) < 1e-6

        c_H_2, c_CO2_aq_2 = fsolve(residual_c_H, initial_guess, args=(c_alk_1, c_DIC_1, Kw, K1, K2, gamma_H, gamma_OH, gamma_HCO3, gamma_CO3, gamma_CO2_aq))
        pH = -np.log10(c_H_2)
        p_CO2_aq_pcatm = 100 * calculate_p_CO2(c_CO2_aq_2, 'mol-atm', R, T)
        c_OH_2 = calculate_c_OH(c_H_2, Kw, gamma_H, gamma_OH)
        c_HCO3_2 = calculate_c_HCO3(c_H_2, c_CO2_aq_2, K1, gamma_H, gamma_HCO3, gamma_CO2_aq)
        
        dC_CO3_2 = (c_DIC_1 * dC_DIC_1 - c_CO2_aq_2 * aC_CO2aq_CO3 - c_HCO3_2 * aC_HCO3_CO3) / c_DIC_1
        # Principle: dC_CO3_old * c_CO3_old - dC_CO3_drop * c_CO3_drop = dC_CO3_new * c_CO3_new
        # Principle: dC_CO3_drop * c_CO3_drop = (c_CO2_new - c_CO2_old) * aC_CO2aq_CO3 + (c_HCO3_new - c_HCO3_old) * aC_CO2aq_HCO3
        dC_CO2_aq_2 = dC_CO3_2 + aC_CO2aq_CO3
        dC_HCO3_2 = dC_CO3_2 + aC_HCO3_CO3
        
        pMC_CO3_2 = (c_DIC_1 * pMC_DIC_1 - c_CO2_aq_2 * apMC_CO2aq_CO3 - c_HCO3_2 * apMC_HCO3_CO3) / c_DIC_1
        pMC_CO2_aq_2 = pMC_CO3_2 + apMC_CO2aq_CO3
        pMC_HCO3_2 = pMC_CO3_2 + apMC_HCO3_CO3
        
        # d42Ca_1 = ((d42Ca/1000 +1) * ((1-P_Ca) / (1-P_Ca/a42Ca_calcite_prec)) -1)*1000  
        d42Ca_2 = (d42Ca * c_Ca - (d42Ca + a42Ca_calcite_Ca) * c_Ca_prec) / c_Ca_1

        c_Ca, c_Mg, c_Sr, c_Na, c_K, c_HCO3, c_H, c_CO2_aq, c_CO3, c_OH, c_DIC, c_alk, dC_HCO3, dC_CO2_aq, dC_CO3, dC_DIC, pMC_HCO3, pMC_CO2_aq, pMC_CO3, pMC_DIC, d42Ca = \
            c_Ca_1, c_Mg_1, c_Sr_1, c_Na_1, c_K_1, c_HCO3_2, c_H_2, c_CO2_aq_2, c_CO3_1, c_OH_2, c_DIC_1, c_alk_1, dC_HCO3_2, dC_CO2_aq_2, dC_CO3_2, dC_DIC_1, pMC_HCO3_2, pMC_CO2_aq_2, pMC_CO3_2, pMC_DIC_1, d42Ca_2
            
        Q = Q_calculate(gamma_Ca, c_Ca, gamma_CO3, c_CO3)
        omega = omega_calculate(Q, Ksp)

        for key, value in zip(current_results.keys(), [label, pH, c_H, c_OH,\
                                                       c_DIC, c_CO2_aq, p_CO2_aq_pcatm, c_HCO3,  c_CO3,\
                                                       dC_DIC, pMC_DIC, dC_CO2_aq, pMC_CO2_aq, dC_HCO3, pMC_HCO3, dC_CO3, pMC_CO3,\
                                                       c_cation, c_Ca, c_Mg, c_Sr, c_Na_rain+c_Na, c_K, c_SO4, c_Cl_rain, I,\
                                                       d42Ca, dSr, omega, None, None, None, None, None, None, None]):
            current_results[key].append(value)  
    
        phase = "river_degassing" 
        go_to_precipitation = False                      
results.append(current_results) 
#%% save data into csv file
data_to_save = []

for result in results:
    for values in zip(
        result['label'], result['pH'], result['c_H'], result['c_OH'], 
        result['c_DIC'], result['c_CO2_aq'], result['p_CO2_aq_pcatm'], result['c_HCO3'], result['c_CO3'], 
        result['dC_DIC'], result['pMC_DIC'], result['dC_CO2_aq'], result['pMC_CO2_aq'], result['dC_HCO3'], result['pMC_HCO3'], result['dC_CO3'], result['pMC_CO3'],
        result['c_cation'], result['c_Ca'], result['c_Mg'], result['c_Sr'], result['c_Na'], result['c_K'], result['c_SO4'], result['c_Cl_rain'], result['I'], 
        result['d42Ca'], result['dSr'], result['omega'], result['XCa_carb_open'], result['XCa_sil_open'], result['XCa_carb_closed'], result['XCa_sil_closed'], result['XCa_gyp_closed'], result['XSO4_gyp_closed'], result['XSO4_pyr_closed']
    ):
        data_to_save.append(dict(zip(current_results.keys(), values)))


df_results = pd.DataFrame(data_to_save)
output_csv_path = os.path.join(save_path, "comb.csv")  
# df_results.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
df_results.to_csv(output_csv_path, index=False, encoding='utf-8-sig', sep=',')

#%% group data to make plots
df = pd.read_csv(output_csv_path, sep=',', header='infer')

style_map = {
    'soil water': {'color': 'black', 'linestyle': '', 'label': 'soil water'},
    'groundwater_dissolution_open': {'color': 'black', 'linestyle': '-', 'label': 'groundwater_dissolution_open'},
    'groundwater_dissolution_closed': {'color': 'black', 'linestyle': '--', 'label': 'groundwater_dissolution_closed'},
    'groundwater_precipitation_grouped': {'color': '#1a80bb', 'linestyle': '-', 'label': 'groundwater_precipitation_dissolution'},
    'river_degassing_precipitation': {'color': '#ea801c', 'linestyle': '-', 'label': 'river_degassing'}
}

df['label_grouped'] = df['label'].replace({
    'groundwater_precipitation_dissolution': 'groundwater_precipitation_grouped',
    'groundwater_precipitation': 'groundwater_precipitation_grouped',
    'river_degassing': 'river_degassing_precipitation',
    'river_precipitation': 'river_degassing_precipitation'
})

#%% basic
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(wspace=0.3, hspace=0.4)

added_labels = set()

for label in df['label_grouped'].unique():
    df_label = df[df['label_grouped'] == label]
    
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)

    axs[0, 0].plot(df_label['p_CO2_aq_pcatm'], df_label['pH'], label=label_to_add, color=color, linestyle=linestyle)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlabel(r'$p_{CO_2,aq}$ (% atm)', fontsize=14)
    axs[0, 0].set_ylabel(r'$pH$', fontsize=14)

    axs[0, 1].plot(df_label['pH'], df_label['c_HCO3'], label=label_to_add, color=color, linestyle=linestyle)
    axs[0, 1].set_xlabel(r'$pH$', fontsize=14)
    axs[0, 1].set_ylabel(r'$[HCO_3^-]$', fontsize=14)

    axs[1, 0].plot(df_label['p_CO2_aq_pcatm'], df_label['c_CO3'], label=label_to_add, color=color, linestyle=linestyle)
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_xlabel(r'$p_{CO_2,aq}$ (% atm)', fontsize=14)
    axs[1, 0].set_ylabel(r'$[CO_3^{2-}]$', fontsize=14)

    axs[1, 1].plot(df_label['pH'], df_label['omega'], label=label_to_add, color=color, linestyle=linestyle)
    axs[1, 1].set_xlabel(r'$pH$', fontsize=14)
    axs[1, 1].set_ylabel(r'$omega$', fontsize=14)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

handles, labels = axs[0, 0].get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
axs[0, 0].legend(unique_labels.values(), unique_labels.keys())

plt.show()



#%% C isotopes for CO2_aq
dC_CO2_aq = df['dC_CO2_aq'].to_numpy()
pMC_CO2_aq = df['pMC_CO2_aq'].to_numpy()

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(wspace=0.3, hspace=0.4)

added_labels = set()

for label in df['label_grouped'].unique():
    df_label = df[df['label_grouped'] == label]
    
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)

    axs[0, 0].plot(df_label['p_CO2_aq_pcatm'], df_label['dC_CO2_aq'], color=color, linestyle=linestyle, label=label_to_add)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlabel(r'$p_{CO_2,aq}$ (% atm)', fontsize=14)
    axs[0, 0].set_ylabel(r'$\delta^{13}C_{CO2_{aq}}$', fontsize=14)

    axs[1, 0].plot(df_label['p_CO2_aq_pcatm'], df_label['pMC_CO2_aq'], color=color, linestyle=linestyle, label=label_to_add)
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_xlabel(r'$p_{CO_2,aq}$ (% atm)', fontsize=14)
    axs[1, 0].set_ylabel(r'$pMC_{CO2_{aq}}$', fontsize=14)

    axs[0, 1].plot(df_label['pH'], df_label['dC_CO2_aq'], color=color, linestyle=linestyle, label=label_to_add)
    axs[0, 1].set_xlabel(r'$pH$', fontsize=14)
    axs[0, 1].set_ylabel(r'$\delta^{13}C_{CO2_{aq}}$', fontsize=14)

    axs[1, 1].plot(df_label['pH'], df_label['pMC_CO2_aq'], color=color, linestyle=linestyle, label=label_to_add)
    axs[1, 1].set_xlabel(r'$pH$', fontsize=14)
    axs[1, 1].set_ylabel(r'$pMC_{CO2_{aq}}$', fontsize=14)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

for ax in axs.flat:
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()

#%% C isotopes for HCO3
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(wspace=0.3, hspace=0.4)

added_labels = set()

for label in df['label_grouped'].unique():
    df_label = df[df['label_grouped'] == label]
    
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)

    axs[0, 0].plot(df_label['c_HCO3'], df_label['dC_HCO3'], color=color, linestyle=linestyle, label=label_to_add)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlabel(r'$c_{HCO3}$', fontsize=14)
    axs[0, 0].set_ylabel(r'$\delta^{13}C_{HCO3}$', fontsize=14)

    axs[0, 1].plot(df_label['pH'], df_label['dC_HCO3'], color=color, linestyle=linestyle, label=label_to_add)
    axs[0, 1].set_xlabel(r'$pH$', fontsize=14)
    axs[0, 1].set_ylabel(r'$\delta^{13}C_{HCO3}$', fontsize=14)

    axs[1, 0].plot(df_label['c_HCO3'], df_label['pMC_HCO3'], color=color, linestyle=linestyle, label=label_to_add)
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_xlabel(r'$c_{HCO3}$', fontsize=14)
    axs[1, 0].set_ylabel(r'$pMC_{HCO3}$', fontsize=14)

    axs[1, 1].plot(df_label['pH'], df_label['pMC_HCO3'], color=color, linestyle=linestyle, label=label_to_add)
    axs[1, 1].set_xlabel(r'$pH$', fontsize=14)
    axs[1, 1].set_ylabel(r'$pMC_{HCO3}$', fontsize=14)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

for ax in axs.flat:
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()

#%% C isotopes for CO3

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(wspace=0.3, hspace=0.4, top=0.95)

added_labels = set()

for label in df['label_grouped'].unique():
    df_label = df[df['label_grouped'] == label]
    
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)

    axs[0, 0].plot(df_label['c_CO3'], df_label['dC_CO3'], color=color, linestyle=linestyle, label=label_to_add)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlabel(r'$c_{CO3}$', fontsize=14)
    axs[0, 0].set_ylabel(r'$\delta^{13}C_{CO3}$', fontsize=14)

    axs[0, 1].plot(df_label['pH'], df_label['dC_CO3'], color=color, linestyle=linestyle, label=label_to_add)
    axs[0, 1].set_xlabel(r'$pH$', fontsize=14)
    axs[0, 1].set_ylabel(r'$\delta^{13}C_{CO3}$', fontsize=14)

    axs[1, 0].plot(df_label['c_CO3'], df_label['pMC_CO3'], color=color, linestyle=linestyle, label=label_to_add)
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_xlabel(r'$c_{CO3}$', fontsize=14)
    axs[1, 0].set_ylabel(r'$pMC_{CO3}$', fontsize=14)

    axs[1, 1].plot(df_label['pH'], df_label['pMC_CO3'], color=color, linestyle=linestyle, label=label_to_add)
    axs[1, 1].set_xlabel(r'$pH$', fontsize=14)
    axs[1, 1].set_ylabel(r'$pMC_{CO3}$', fontsize=14)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

for ax in axs.flat:
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()

#%% cation
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(wspace=0.3, hspace=0.4)

added_labels = set()

for label in df['label_grouped'].unique():
    df_label = df[df['label_grouped'] == label]
    
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)

    axs[0, 0].plot(df_label['p_CO2_aq_pcatm'], df_label['dC_CO2_aq'], color=color, linestyle=linestyle, label=label_to_add)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlabel(r'$p_{CO_2,aq}$ (% atm)', fontsize=14)
    axs[0, 0].set_ylabel(r'$\delta^{13}C_{CO2_{aq}}$', fontsize=14)

    axs[0, 1].plot(df_label['pH'], df_label['dC_CO2_aq'], color=color, linestyle=linestyle, label=label_to_add)
    axs[0, 1].set_xlabel(r'$pH$', fontsize=14)
    axs[0, 1].set_ylabel(r'$\delta^{13}C_{CO2_{aq}}$', fontsize=14)

    axs[1, 0].plot(df_label['p_CO2_aq_pcatm'], df_label['pMC_CO2_aq'], color=color, linestyle=linestyle, label=label_to_add)
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_xlabel(r'$p_{CO_2,aq}$ (% atm)', fontsize=14)
    axs[1, 0].set_ylabel(r'$pMC_{CO2_{aq}}$', fontsize=14)

    axs[1, 1].plot(df_label['pH'], df_label['pMC_CO2_aq'], color=color, linestyle=linestyle, label=label_to_add)
    axs[1, 1].set_xlabel(r'$pH$', fontsize=14)
    axs[1, 1].set_ylabel(r'$pMC_{CO2_{aq}}$', fontsize=14)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

for ax in axs.flat:
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()

#%% Ca/Sr - d42Ca
fig, axs = plt.subplots(1, 2, figsize=(9, 6))
fig.subplots_adjust(wspace=0.3, hspace=0.4)

added_labels = set()

for label in df['label_grouped'].unique():
    df_label = df[df['label_grouped'] == label]
    
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)

    axs[0].plot(df_label['c_Ca']/df_label['c_Sr'], df_label['d42Ca'], color=color, linestyle=linestyle, label=label_to_add)
    axs[0].set_xlabel(r'$Ca/Sr$', fontsize=14)
    axs[0].set_ylabel(r'$\delta^{44/42}Ca$', fontsize=14)

    axs[1].plot(df_label['c_Ca']/df_label['c_Sr'], df_label['dSr'], color=color, linestyle=linestyle, label=label_to_add)
    axs[1].set_xlabel(r'$Ca/Sr$', fontsize=14)
    axs[1].set_ylabel(r'$^{87/86}Sr$', fontsize=14)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

for ax in axs.flat:
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()


#%% Ca
fig, axs = plt.subplots(1, 2, figsize=(9, 6))
fig.subplots_adjust(wspace=0.3, hspace=0.4)

added_labels = set()

for label in df['label_grouped'].unique():
    df_label = df[df['label_grouped'] == label]
    
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)

    axs[0].plot(df_label['c_CO2_aq'], df_label['c_Ca'], color=color, linestyle=linestyle, label=label_to_add)
    axs[0].set_xlabel(r'$p_{CO_2,aq}$ (% atm)', fontsize=14)
    axs[0].set_ylabel(r'$c_{Ca}$', fontsize=14)

    axs[1].plot(df_label['pH'], df_label['c_Ca'], color=color, linestyle=linestyle, label=label_to_add)
    axs[1].set_xlabel(r'$pH$', fontsize=14)
    axs[1].set_ylabel(r'$c_{Ca}$', fontsize=14)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

for ax in axs.flat:
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()

#%% SO4
fig, axs = plt.subplots(1, 2, figsize=(12, 9))
fig.subplots_adjust(wspace=0.3, hspace=0.4, top=0.95)

added_labels = set()

for label in df['label_grouped'].unique():
    df_label = df[df['label_grouped'] == label]
    
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)

    axs[0].plot(df_label['c_CO2_aq'], df_label['c_SO4'], color=color, linestyle=linestyle, label=label_to_add)
    axs[0].set_xlabel(r'$p_{CO_2,aq}$ (% atm)', fontsize=14)
    axs[0].set_ylabel(r'$c_{SO_4}$', fontsize=14)

    axs[1].plot(df_label['pH'], df_label['c_SO4'], color=color, linestyle=linestyle, label=label_to_add)
    axs[1].set_xlabel(r'$pH$', fontsize=14)
    axs[1].set_ylabel(r'$c_{SO_4}$', fontsize=14)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

for ax in axs.flat:
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()

#%% Fig 2 & 3. pH, pCO2 -- 13C, 14C, XCasil = 0 (Folhmesiter) / 0.5

# selected_labels = {'soil water', 'groundwater_dissolution_open'}
selected_labels = {
    'soil water',
    'groundwater_dissolution_open'
}
fig, axs = plt.subplots(2, 2, figsize=(12,9), sharex='col', sharey='row') 
fig.subplots_adjust(wspace=0.05, hspace=0.05,top=0.95)

added_labels = set()

for label in df['label_grouped'].unique():
    if label not in selected_labels: 
        continue  

    df_label = df[df['label_grouped'] == label]
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)
    if label == 'soil water':
        axs[0, 0].scatter(df_label['pH'], df_label['dC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[0, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 0].scatter(df_label['pH'], df_label['pMC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], label=label_to_add, color=color, marker='o', s=10)
    else:
        axs[0, 0].plot(df_label['pH'], df_label['dC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel(r'$\delta^{13}C_{DIC}\ (‰)$', fontsize=14)
        rect1 = patches.Rectangle((4.6, dC_CO2_aq_open), 1, 0.8, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')  # Graven 2019
        axs[0, 0].add_patch(rect1)
        axs[0, 0].text(4.7, -24.8, r'$CO_{2(aq)}$', fontsize=12, color='green', ha='left', va='center')
        rect2 = patches.Rectangle((6.5, dC_HCO3_open-0.5), 1, 0.9, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')  # Graven 2019
        axs[0, 0].add_patch(rect2)
        axs[0, 0].text(6.4, -16.5, r'$HCO_3^-/CO_3^{2-}$', fontsize=12, color='green', ha='left', va='center')
        axs[0, 0].set_ylim(-27, -15) 
    
        axs[0, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 1].set_xscale('log')
        axs[0, 1].set_xlabel('')
        axs[0, 1].set_ylabel('')
        
        axs[1, 0].plot(df_label['pH'], df_label['pMC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 0].set_xlabel(r'$pH$', fontsize=14)
        axs[1, 0].set_ylabel(r'$pMC_{DIC}\ (\%)$', fontsize=14) 
        rect3 = patches.Rectangle((4.6, pMC_CO2_aq_open), 1, 0.3, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')  # Graven 2019
        axs[1, 0].add_patch(rect3)
        axs[1, 0].text(4.7, 106.2, r'$CO_{2(aq)}$', fontsize=12, color='green', ha='left', va='center')
        rect4 = patches.Rectangle((6.5, pMC_HCO3_open-0.5), 1, 0.3, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')  # Graven 2019
        axs[1, 0].add_patch(rect4)
        axs[1, 0].text(5.9, 107.8, r'$HCO_3^-/CO_3^{2-}$', fontsize=12, color='green', ha='left', va='center')
        axs[1, 0].set_xlim(4.6, 7)
        axs[1, 0].set_ylim(105, 108)
        
        axs[1, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 1].set_xscale('log')
        axs[1, 1].set_xlabel(r'$p_{CO_2,aq}\ (\% atm)$', fontsize=14)
        axs[1, 1].set_ylabel('')
        axs[1, 1].set_xlim(2, 1e2)
        
for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, top=True, right=True)
    ax.tick_params(length=4, width=1) 
    
handles, labels = axs[0, 0].get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
axs[0, 0].legend(unique_labels.values(), unique_labels.keys())

plt.show()

# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_2_XCa_sil_open_0.eps", format='eps')
# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_3_XCa_sil_open_0.5.eps", format='eps')

#%% Fig 4 & 5. pH, pCO2 -- 13C, 14C, XCasil = 0 (Folhmesiter) / 0.5

selected_labels = {
    'soil water',
    'groundwater_dissolution_open',
    'groundwater_dissolution_closed'
}
fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex='col', sharey='row') 
fig.subplots_adjust(wspace=0.05, hspace=0.05, top = 0.95)

added_labels = set()

for label in df['label_grouped'].unique():
    if label not in selected_labels: 
        continue  

    df_label = df[df['label_grouped'] == label]
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)
    if label == 'soil water':
        axs[0, 0].scatter(df_label['pH'], df_label['dC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[0, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 0].scatter(df_label['pH'], df_label['pMC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], label=label_to_add, color=color, marker='o', s=10)
    else:
        axs[0, 0].plot(df_label['pH'], df_label['dC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel(r'$\delta^{13}C_{DIC}\ (‰)$', fontsize=14)
        axs[0, 0].set_ylim(-27, -15) 
    
        axs[0, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 1].set_xscale('log')
        axs[0, 1].set_xlabel('')
        axs[0, 1].set_ylabel('')
        
        axs[1, 0].plot(df_label['pH'], df_label['pMC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 0].set_xlabel(r'$pH$', fontsize=14)
        axs[1, 0].set_ylabel(r'$pMC_{DIC}\ (\%)$', fontsize=14) 
        axs[1, 0].set_xlim(4.6, 7)
        axs[1, 0].set_ylim(95, 110)
        
        axs[1, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 1].set_xscale('log')
        axs[1, 1].set_xlabel(r'$p_{CO_2,aq}\ (\% atm)$', fontsize=14)
        axs[1, 1].set_ylabel('')
        axs[1, 1].set_xlim(2, 1e2)
                
for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, top=True, right=True)
    ax.tick_params(length=4, width=1) 
    
handles, labels = axs[0, 0].get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
axs[0, 0].legend(unique_labels.values(), unique_labels.keys())

plt.show()

# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_4_XCa_sil_closed_0_XCa_gyp_closed_0_XSO4_gyp_1.eps", format='eps')
# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_5_XCa_sil_closed_0_XCa_gyp_closed_0_XSO4_gyp_0.5.eps", format='eps')
#%% Fig 6. pH, pCO2 -- 13C, 14C, XCasil = 0 (Folhmesiter) / 0.5

selected_labels = {'groundwater_precipitation_grouped'}
fig, axs = plt.subplots(2, 2, figsize=(12,9), sharex='col', sharey='row') 
fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95)

added_labels = set()

for label in df['label_grouped'].unique():
    if label not in selected_labels: 
        continue   
    df_label = df[df['label_grouped'] == label]
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)
    if label == 'soil water':
        axs[0, 0].scatter(df_label['pH'], df_label['dC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[0, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 0].scatter(df_label['pH'], df_label['pMC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], label=label_to_add, color=color, marker='o', s=10)
    else:
        axs[0, 0].plot(df_label['pH'], df_label['dC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel(r'$\delta^{13}C_{DIC}\ (‰)$', fontsize=14)
        axs[0, 0].set_ylim(-19, -5) 
    
        axs[0, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 1].set_xlabel('')
        axs[0, 1].set_ylabel('')
        
        axs[1, 0].plot(df_label['pH'], df_label['pMC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 0].set_xlabel(r'$pH$', fontsize=14)
        axs[1, 0].set_ylabel(r'$pMC_{DIC}\ (\%)$', fontsize=14) 
        axs[1, 0].set_xlim(6.87, 6.879)
        axs[1, 0].set_ylim(18, 105)
        
        axs[1, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 1].set_xlabel(r'$p_{CO_2,aq}\ (\% atm)$', fontsize=14)
        axs[1, 1].set_ylabel('')
        axs[1, 1].set_xlim(8.6, 10.1)

ax_inset1 = inset_axes(axs[0, 0], width=1.2, height=1.2, bbox_to_anchor=(0.98, 0.6, 0, 0), bbox_transform=axs[0, 0].transAxes)
ax_inset1.set_xlim(6.8729, 6.8749)  
ax_inset1.set_ylim(-16.571, -16.5695)  
ax_inset1.plot(df_label['pH'], df_label['dC_DIC'], color=color, linestyle=linestyle)
ax_inset1.ticklabel_format(useOffset=False)
mark_inset(axs[0, 0], ax_inset1, loc1=2, loc2=4, fc="none", ec="gray", lw=1, linestyle= '--')

 
ax_inset2 = inset_axes(axs[0, 1], width=1.2, height=1.2, bbox_to_anchor=(0.55, 0.6, 0, 0), bbox_transform=axs[0, 1].transAxes)
ax_inset2.set_xlim(9.99, 10.01)  
ax_inset2.set_ylim(-16.571, -16.5695)  
ax_inset2.ticklabel_format(useOffset=False)
ax_inset2.tick_params(axis='x', rotation=45, labelsize=10)
ax_inset2.plot(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], color=color, linestyle=linestyle)
mark_inset(axs[0, 1], ax_inset2, loc1=1, loc2=3, fc="none", ec="gray", lw=1, linestyle= '--')


ax_inset3 = inset_axes(axs[1, 0], width=1.2, height=1.2, bbox_to_anchor=(0.98, 0.8, 0, 0), bbox_transform=axs[1, 0].transAxes)  
ax_inset3.set_xlim(6.8729, 6.8749)  
ax_inset3.set_ylim(97.569, 97.573)  
ax_inset3.ticklabel_format(useOffset=False)
ax_inset3.plot(df_label['pH'], df_label['pMC_DIC'], color=color, linestyle=linestyle)
mark_inset(axs[1, 0], ax_inset3, loc1=1, loc2=3, fc="none", ec="gray", lw=1, linestyle= '--')

ax_inset4 = inset_axes(axs[1, 1], width=1.2, height=1.2, bbox_to_anchor=(0.45, 0.8, 0, 0), bbox_transform=axs[1, 1].transAxes)  
ax_inset4.set_xlim(9.99, 10.01)  
ax_inset4.set_ylim(97.569, 97.573)  
ax_inset4.ticklabel_format(useOffset=False)
ax_inset4.tick_params(axis='x', rotation=45, labelsize=10)
ax_inset4.plot(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], color=color, linestyle=linestyle)
mark_inset(axs[1, 1], ax_inset4, loc1=2, loc2=4, fc="none", ec="gray", lw=1, linestyle= '--')
          
for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, top=True, right=True)
    ax.tick_params(length=4, width=1) 
handles, labels = axs[0, 0].get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
axs[0, 0].legend(unique_labels.values(), unique_labels.keys())

plt.show()

# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_6_GW_prec_diss.eps", format='eps')


#%% Fig 7. Ca/Sr, c_CO3 -- 87Sr, 44/42Ca, XCasil = 0 (Folhmesiter) / 0.5

selected_labels = {'groundwater_precipitation_grouped'}
fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex='col', sharey='row') 
fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95)

added_labels = set()

for label in df['label_grouped'].unique():
    if label not in selected_labels: 
        continue  
    df_label = df[df['label_grouped'] == label]
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)
    if label == 'soil water':
        axs[0, 0].scatter(df_label['c_Ca']/df_label['c_Sr'], df_label['dSr'], label=label_to_add, color=color, marker='o', s=10)
        axs[0, 1].scatter(df_label['c_CO3'], df_label['dSr'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 0].scatter(df_label['c_Ca']/df_label['c_Sr'], df_label['d42Ca'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 1].scatter(df_label['c_CO3'], df_label['d42Ca'], label=label_to_add, color=color, marker='o', s=10)
    else:
        axs[0, 0].plot(df_label['c_Ca']/df_label['c_Sr'], df_label['dSr'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 0].set_xlabel(r'$Ca/Sr$', fontsize=14)
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel(r'$^{87/86}Sr$', fontsize=14)
        axs[0, 0].set_ylim(0.713, 0.730) 
        # rect1. closed system bedrock end-member
        rect1 = patches.Rectangle((1/SrCa_rock_closed-40,  dSr_rock_closed-0.0005), 100, 0.001, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')  # Graven 2019
        axs[0, 0].add_patch(rect1)
        axs[0, 0].text(1/SrCa_rock_closed -70, dSr_rock_closed+0.0015, r'$bedrock_{closed}$', fontsize=12, color='green', ha='left', va='center')
        # rect2. open system bedrock end-member
        rect2 = patches.Rectangle((1/SrCa_rock_open-50,  dSr_rock_open+0.00005), 100, 0.001, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')  # Graven 2019
        axs[0, 0].add_patch(rect2)
        axs[0, 0].text(1/SrCa_rock_open -50, dSr_rock_open-0.001, r'$bedrock_{open}$', fontsize=12, color='green', ha='left', va='center')

            
        axs[0, 1].plot(df_label['c_CO3'], df_label['dSr'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 1].set_xlabel('')
        axs[0, 1].set_ylabel('')
        
        axs[1, 0].plot(df_label['c_Ca']/df_label['c_Sr'], df_label['d42Ca'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 0].set_xlabel(r'$Ca/Sr$', fontsize=14)
        axs[1, 0].set_ylabel(r'$\delta^{44/42}Ca\ (‰)$', fontsize=14) 
        axs[1, 0].set_xlim(0, 700)
        axs[1, 0].set_ylim(-0.15, 0.4)
        # rect3. closed system bedrock end-member
        rect3 = patches.Rectangle((1/SrCa_rock_closed-50,  d42Ca_rock_closed-0.02), 100, 0.04, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')  # Graven 2019
        axs[1, 0].add_patch(rect3)
        axs[1, 0].text(1/SrCa_rock_closed-70, d42Ca_rock_closed+0.04, r'$bedrock_{closed}$', fontsize=12, color='green', ha='left', va='center')
        # rect4. open system bedrock end-member
        rect4 = patches.Rectangle((1/SrCa_rock_open-50, d42Ca_rock_open-0.02), 100, 0.04, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')  # Graven 2019
        axs[1, 0].add_patch(rect4)
        axs[1, 0].text(1/SrCa_rock_open-50, d42Ca_rock_open+0.04, r'$bedrock_{open}$', fontsize=12, color='green', ha='left', va='center')

        
        axs[1, 1].plot(df_label['c_CO3'], df_label['d42Ca'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 1].set_xlabel(r'$[CO_3^{2-}]$', fontsize=14)
        axs[1, 1].set_ylabel('')
        axs[1, 1].set_xlim(8.1e-6, 1.5e-5)
        axs[1, 1].ticklabel_format(useOffset=False)

ax_inset1 = inset_axes(axs[0, 0], width=1.2, height=1.2, bbox_to_anchor=(0.9, 0.8, 0, 0), bbox_transform=axs[0, 0].transAxes)
ax_inset1.set_xlim(260, 261.4)  
ax_inset1.set_ylim(0.723529, 0.723533)  
ax_inset1.plot(df_label['c_Ca']/df_label['c_Sr'], df_label['dSr'], color=color, linestyle=linestyle)
ax_inset1.ticklabel_format(useOffset=False)
mark_inset(axs[0, 0], ax_inset1, loc1=2, loc2=3, fc="none", ec="gray", lw=1, linestyle= '--')

 
ax_inset2 = inset_axes(axs[0, 1], width=1.2, height=1.2, bbox_to_anchor=(0.9, 0.8, 0, 0), bbox_transform=axs[0, 1].transAxes)
ax_inset2.set_xlim(8.79e-6, 8.83e-6)  
ax_inset2.set_ylim(0.723529, 0.723533)  
ax_inset2.ticklabel_format(useOffset=False)
# ax_inset2.tick_params(axis='x', rotation=45, labelsize=10)
ax_inset2.plot(df_label['c_CO3'], df_label['dSr'], color=color, linestyle=linestyle)
mark_inset(axs[0, 1], ax_inset2, loc1=2, loc2=3, fc="none", ec="gray", lw=1, linestyle= '--')


ax_inset3 = inset_axes(axs[1, 0], width=1.2, height=1.2, bbox_to_anchor=(0.9, 0.8, 0, 0), bbox_transform=axs[1, 0].transAxes)  
ax_inset3.set_xlim(260.6, 261.15)  
ax_inset3.set_ylim(-0.085, -0.084)  
ax_inset3.ticklabel_format(useOffset=False)
ax_inset3.tick_params(axis='x', rotation=45, labelsize=10)
ax_inset3.plot(df_label['c_Ca']/df_label['c_Sr'], df_label['d42Ca'], color=color, linestyle=linestyle)
mark_inset(axs[1, 0], ax_inset3, loc1=3, loc2=4, fc="none", ec="gray", lw=1, linestyle= '--')

ax_inset4 = inset_axes(axs[1, 1], width=1.2, height=1.2, bbox_to_anchor=(0.9, 0.7, 0, 0), bbox_transform=axs[1, 1].transAxes)  
ax_inset4.set_xlim(8.79e-6, 8.83e-6)  
ax_inset4.set_ylim(-0.085, -0.084)  
ax_inset4.ticklabel_format(useOffset=False)
ax_inset4.plot(df_label['c_CO3'], df_label['d42Ca'], color=color, linestyle=linestyle)
mark_inset(axs[1, 1], ax_inset4, loc1=2, loc2=4, fc="none", ec="gray", lw=1, linestyle= '--')

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, top=True, right=True)
    ax.tick_params(length=4, width=1) 
handles, labels = axs[0, 0].get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
axs[0, 0].legend(unique_labels.values(), unique_labels.keys())

plt.show()
# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_7_GW_prec_diss_Ca_Sr.eps", format='eps')
#%% Fig 8. RW_degas_prec pH, pCO2 -- 13C, 14C

selected_labels = {'river_degassing_precipitation'}
fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex='col', sharey='row')  
fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95) 
added_labels = set()

for label in df['label_grouped'].unique():
    if label not in selected_labels: 
        continue  
    df_label = df[df['label_grouped'] == label]
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)
    if label == 'soil water':
        axs[0, 0].scatter(df_label['pH'], df_label['dC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[0, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 0].scatter(df_label['pH'], df_label['pMC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], label=label_to_add, color=color, marker='o', s=10)
    else:
        axs[0, 0].plot(df_label['pH'], df_label['dC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel(r'$\delta^{13}C_{DIC}\ (‰)$', fontsize=14)
        axs[0, 0].set_ylim(-10, 2) 
    
        axs[0, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 1].set_xscale('log')
        axs[0, 1].set_xlabel('')
        axs[0, 1].set_ylabel('')
        rect1 = patches.Rectangle((p_CO2_aq_end*0.99, dC_CO2_aq_end), 0.02, 4.5, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')  # Graven 2019
        axs[0, 1].add_patch(rect1)
        axs[0, 1].text(p_CO2_aq_end*0.7, -4.5, r'atmospheric $\mathrm{CO}_{2}$', fontsize=12, color='green', ha='left', va='center')
        
        axs[1, 0].plot(df_label['pH'], df_label['pMC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 0].set_xlabel(r'$pH$', fontsize=14)
        axs[1, 0].set_ylabel(r'$pMC_{DIC}\ (\%)$', fontsize=14) 
        axs[1, 0].set_xlim(6.6,9.2)
        axs[1, 0].set_ylim(20, 114)
        
        axs[1, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 1].set_xscale('log')
        axs[1, 1].set_xlabel(r'$p_{CO_2,aq}\ (\% atm)$', fontsize=14)
        axs[1, 1].set_ylabel('')
        rect2 = patches.Rectangle((p_CO2_aq_end*0.99, pMC_CO2_aq_end), 0.02, 4.5, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')  # Graven 2019
        axs[1, 1].add_patch(rect2)
        axs[1, 1].text(p_CO2_aq_end*0.7, 100, r'atmospheric $\mathrm{CO}_{2}$', fontsize=12, color='green', ha='left', va='center')
        axs[1, 1].set_xlim(2e-2, 10)

ax_inset1 = inset_axes(axs[0, 0], width=1.2, height=1.2, bbox_to_anchor=(0.8, 0.7, 0, 0), bbox_transform=axs[0, 0].transAxes)
ax_inset1.set_xlim(7.35, 7.4)  
ax_inset1.set_ylim(-0.8, -0.5)  
ax_inset1.plot(df_label['pH'], df_label['dC_DIC'], color=color, linestyle=linestyle)
ax_inset1.ticklabel_format(useOffset=False)
mark_inset(axs[0, 0], ax_inset1, loc1=1, loc2=3, fc="none", ec="gray", lw=1, linestyle= '--')

ax_inset2 = inset_axes(axs[0, 1], width=1.2, height=1.2, bbox_to_anchor=(0.8, 0.7, 0, 0), bbox_transform=axs[0, 1].transAxes)
ax_inset2.set_xlim(2.5, 2.9)  
ax_inset2.set_ylim(-0.8, -0.5)  
ax_inset2.ticklabel_format(useOffset=False)
# ax_inset2.tick_params(axis='x', rotation=45, labelsize=10)
ax_inset2.plot(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], color=color, linestyle=linestyle)
mark_inset(axs[0, 1], ax_inset2, loc1=2, loc2=4, fc="none", ec="gray", lw=1, linestyle= '--')

ax_inset3 = inset_axes(axs[1, 0], width=1.2, height=1.2, bbox_to_anchor=(0.8, 0.7, 0, 0), bbox_transform=axs[1, 0].transAxes)  
ax_inset3.set_xlim(7.35, 7.4)  
ax_inset3.set_ylim(86, 90.5)  
# ax_inset3.ticklabel_format(useOffset=False)
# ax_inset3.tick_params(axis='x', rotation=45, labelsize=10)
ax_inset3.plot(df_label['pH'], df_label['pMC_DIC'], color=color, linestyle=linestyle)
mark_inset(axs[1, 0], ax_inset3, loc1=1, loc2=3, fc="none", ec="gray", lw=1, linestyle= '--')

ax_inset4 = inset_axes(axs[1, 1], width=1.2, height=1.2, bbox_to_anchor=(0.8, 0.7, 0, 0), bbox_transform=axs[1, 1].transAxes)  
ax_inset4.set_xlim(2.5, 2.9)  
ax_inset4.set_ylim(86, 90.5)  
ax_inset4.ticklabel_format(useOffset=False)
# ax_inset4.tick_params(axis='x', rotation=45, labelsize=10)
ax_inset4.plot(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], color=color, linestyle=linestyle)
mark_inset(axs[1, 1], ax_inset4, loc1=2, loc2=4, fc="none", ec="gray", lw=1, linestyle= '--')
                
for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, top=True, right=True)
    ax.tick_params(length=4, width=1)  
    
handles, labels = axs[0, 0].get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
axs[0, 0].legend(unique_labels.values(), unique_labels.keys())

plt.show()
# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_8_RW_degas_C.eps", format='eps')
#%% Fig 9. RW_degas_prec pH, pCO2 -- 42Ca， Ca/Sr
selected_labels = {'river_degassing_precipitation'}
fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex='col', sharey='row') 
fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95) 

added_labels = set()

# for ax in axs.flat:
for label in df['label_grouped'].unique():
    if label not in selected_labels: 
        continue  

    df_label = df[df['label_grouped'] == label]
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)
    if label == 'soil water':
        axs[0, 0].scatter(df_label['pH'], df_label['d42Ca'], label=label_to_add, color=color, marker='o', s=10)
        axs[0, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['d42Ca'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 0].scatter(df_label['pH'], df_label['c_Ca']/df_label['c_Sr'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['c_Ca']/df_label['c_Sr'], label=label_to_add, color=color, marker='o', s=10)
    else:
        axs[0, 0].plot(df_label['pH'], df_label['d42Ca'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel(r'$\delta^{44/42}Ca\ (‰)$', fontsize=14)
        axs[0, 0].set_ylim(-0.5, 2.1) 
    
        axs[0, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['d42Ca'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 1].set_xscale('log')
        axs[0, 1].set_xlabel('')
        axs[0, 1].set_ylabel('')
        
        axs[1, 0].plot(df_label['pH'], df_label['c_Ca']/df_label['c_Sr'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 0].set_xlabel(r'$pH$', fontsize=14)
        axs[1, 0].set_ylabel(r'$Ca/Sr$', fontsize=14) 
        axs[1, 0].set_xlim(6.5,9.2)
        axs[1, 0].set_ylim(-5, 190)
        
        axs[1, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['c_Ca']/df_label['c_Sr'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 1].set_xscale('log')
        axs[1, 1].set_xlabel(r'$p_{CO_2,aq}\ (\% atm)$', fontsize=14)
        axs[1, 1].set_ylabel('')
        axs[1, 1].set_xlim(2e-2, 10)

ax_inset1 = inset_axes(axs[0, 0], width=1.2, height=1.2, bbox_to_anchor=(0.9, 0.5, 0, 0), bbox_transform=axs[0, 0].transAxes)
ax_inset1.set_xlim(7.55, 7.58)  
ax_inset1.set_ylim(0.59, 0.62)  
ax_inset1.plot(df_label['pH'], df_label['d42Ca'], color=color, linestyle=linestyle)
ax_inset1.ticklabel_format(useOffset=False)
mark_inset(axs[0, 0], ax_inset1, loc1=2, loc2=3, fc="none", ec="gray", lw=1, linestyle= '--')

 
ax_inset2 = inset_axes(axs[0, 1], width=1.2, height=1.2, bbox_to_anchor=(0.5, 0.5, 0, 0), bbox_transform=axs[0, 1].transAxes)
ax_inset2.set_xlim(1.42,1.55)  
ax_inset2.set_ylim(0.59, 0.62)  
ax_inset2.ticklabel_format(useOffset=False)
ax_inset2.plot(df_label['p_CO2_aq_pcatm'], df_label['d42Ca'], color=color, linestyle=linestyle)
mark_inset(axs[0, 1], ax_inset2, loc1=1, loc2=4, fc="none", ec="gray", lw=1, linestyle= '--')


ax_inset3 = inset_axes(axs[1, 0], width=1.2, height=1.2, bbox_to_anchor=(0.9, 0.7, 0, 0), bbox_transform=axs[1, 0].transAxes)  
ax_inset3.set_xlim(7.55, 7.58)  
ax_inset3.set_ylim(74, 77)  
ax_inset3.plot(df_label['pH'], df_label['c_Ca']/df_label['c_Sr'], color=color, linestyle=linestyle)
mark_inset(axs[1, 0], ax_inset3, loc1=2, loc2=3, fc="none", ec="gray", lw=1, linestyle= '--')

ax_inset4 = inset_axes(axs[1, 1], width=1.2, height=1.2, bbox_to_anchor=(0.5, 0.7, 0, 0), bbox_transform=axs[1, 1].transAxes)  
ax_inset4.set_xlim(1.42,1.55)  
ax_inset4.set_ylim(74, 77)  
ax_inset4.ticklabel_format(useOffset=False)
ax_inset4.plot(df_label['p_CO2_aq_pcatm'], df_label['c_Ca']/df_label['c_Sr'], color=color, linestyle=linestyle)
mark_inset(axs[1, 1], ax_inset4, loc1=1, loc2=4, fc="none", ec="gray", lw=1, linestyle= '--')
              
for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, top=True, right=True)
    ax.tick_params(length=4, width=1)  
    
handles, labels = axs[0, 0].get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
axs[0, 0].legend(unique_labels.values(), unique_labels.keys())

plt.show()
# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_9_RW_degas_prec_Ca.eps", format='eps')
#%% Fig. 10/12  pH, p_CO2 - 13C, 14C
fig, axs = plt.subplots(2, 2, figsize=(12,9), sharex='col', sharey='row') 
fig.subplots_adjust(wspace=0.05, hspace=0.05,top=0.95)

added_labels = set()

for label in df['label_grouped'].unique():

    df_label = df[df['label_grouped'] == label]
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)
    if label == 'soil water':
        axs[0, 0].scatter(df_label['pH'], df_label['dC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[0, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 0].scatter(df_label['pH'], df_label['pMC_DIC'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], label=label_to_add, color=color, marker='o', s=10)
    else:
        axs[0, 0].plot(df_label['pH'], df_label['dC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel(r'$\delta^{13}C_{DIC}\ (‰)$', fontsize=14)
        axs[0, 0].set_ylim(-28, 5) 
        
        axs[0, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['dC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 1].set_xscale('log')
        axs[0, 1].set_xlabel('')
        axs[0, 1].set_ylabel('')
        
        axs[1, 0].plot(df_label['pH'], df_label['pMC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 0].set_xlabel(r'$pH$', fontsize=14)
        axs[1, 0].set_ylabel(r'$pMC_{DIC}\ (\%)$', fontsize=14)
        axs[1, 0].set_xlim(4.5,9.2)
        axs[1, 0].set_ylim(10, 120)
        
        axs[1, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['pMC_DIC'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 1].set_xscale('log')
        axs[1, 1].set_xlabel(r'$p_{CO_2,aq}\ (\% atm)$', fontsize=14)
        axs[1, 1].set_ylabel('')
        axs[1, 1].set_xlim(2e-2,3e1)
        
for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, top=True, right=True)
    ax.tick_params(length=4, width=1) 

handles, labels = axs[0, 1].get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
axs[0, 1].legend(unique_labels.values(), unique_labels.keys())

plt.show()

# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_10_overview_n_5000_pH_CO2_13C_pMC.eps", format='eps')
# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_12_overview_n_100_pH_CO2_13C_pMC.eps", format='eps')
# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_14_overview_no_gwprec_pH_CO2_13C_pMC.eps", format='eps')
# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_16_overview_no_rwprec_pH_CO2_13C_pMC.eps", format='eps')
#%% Fig. 11/13 - pH, p_CO2 - 42Ca， Ca/Sr
selected_labels = {'groundwater_dissolution_open',
                   'groundwater_dissolution_closed',
                   'groundwater_precipitation_grouped',
                   'river_degassing_precipitation'}

fig, axs = plt.subplots(2, 2, figsize=(12,9), sharex='col', sharey='row') 
fig.subplots_adjust(wspace=0.05, hspace=0.05,top=0.95)

added_labels = set()

for label in df['label_grouped'].unique():
    if label not in selected_labels: 
        continue  
    df_label = df[df['label_grouped'] == label]
    color = style_map[label]['color']
    linestyle = style_map[label]['linestyle']

    label_to_add = style_map[label]['label'] if label not in added_labels else ""
    added_labels.add(label)
    if label == 'soil water':
        axs[0, 0].scatter(df_label['pH'], df_label['d42Ca'], label=label_to_add, color=color, marker='o', s=10)
        axs[0, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['d42Ca'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 0].scatter(df_label['pH'], df_label['c_Ca']/df_label['c_Sr'], label=label_to_add, color=color, marker='o', s=10)
        axs[1, 1].scatter(df_label['p_CO2_aq_pcatm'], df_label['c_Ca']/df_label['c_Sr'], label=label_to_add, color=color, marker='o', s=10)
    else:
        axs[0, 0].plot(df_label['pH'], df_label['d42Ca'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel(r'$\delta^{44/42}Ca\ (‰)$', fontsize=14)
        axs[0, 0].set_ylim(-0.3, 2.5) 
        
        axs[0, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['d42Ca'], label=label_to_add, color=color, linestyle=linestyle)
        axs[0, 1].set_xscale('log')
        axs[0, 1].set_xlabel('')
        axs[0, 1].set_ylabel('')
        
        axs[1, 0].plot(df_label['pH'], df_label['c_Ca']/df_label['c_Sr'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 0].set_xlabel(r'$pH$', fontsize=14)
        axs[1, 0].set_ylabel(r'$Ca/Sr$', fontsize=14)
        axs[1, 0].set_xlim(4.5,9.2)
        axs[1, 0].set_ylim(-10, 280)
        
        axs[1, 1].plot(df_label['p_CO2_aq_pcatm'], df_label['c_Ca']/df_label['c_Sr'], label=label_to_add, color=color, linestyle=linestyle)
        axs[1, 1].set_xscale('log')
        axs[1, 1].set_xlabel(r'$p_{CO_2,aq}\ (\% atm)$', fontsize=14)
        axs[1, 1].set_ylabel('')
        axs[1, 1].set_xlim(2e-2,3e1)
        
for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, top=True, right=True)
    ax.tick_params(length=4, width=1) 

handles, labels = axs[0, 1].get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
axs[0, 1].legend(unique_labels.values(), unique_labels.keys())

plt.show()

# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_11_overview_n_5000_pH_CO2_42Ca_CaSr.eps", format='eps')
# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_13_overview_n_100_pH_CO2_42Ca_CaSr.eps", format='eps')
# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_15_overview_no_gwprec_pH_CO2_42Ca_CaSr.eps", format='eps')
# plt.savefig("E:\Doctorat\Codes\C isotopes\comb\Figure_17_overview_no_rwprec_pH_CO2_42Ca_CaSr.eps", format='eps')
