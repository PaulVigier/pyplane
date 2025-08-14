"""Function to generate plots for Air Data System Calibration (ADSC) report.

WARNING : Graphs plotted should be updated to better fit Erb's requirements !
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from libs import constants as cst
from libs import atmos_1976 as atm
from labellines import labelLines
from scipy import stats

# Constants for tower fly-by
GRID_DELTA_H_BY_DIV=31.4 #ft/div

#MIL-P-26292C
MIL26292C_A_MACH=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
MIL26292C_A_MIN=[-0.015,-0.015,-0.015,-0.015,-0.012,-0.008,-0.005,-0.003,-0.002,-0.002]
MIL26292C_A_MAX=[0.02,0.02,0.02,0.017,0.012,0.008,0.005,0.003,0.002,0.002]

MIL26292C_B_MACH = np.array([0.3, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0, 1.0355, 1.15])
MIL26292C_B_MAX = np.array([0.02, 0.02, 0.0205, 0.0215, 0.02269, 0.0246, 0.02707, 0.0305, 0.03649, 0.0445, 0.01, 0.01])
MIL26292C_B_MIN = -np.array([0.015, 0.015, 0.01539, 0.01637, 0.01775, 0.01952, 0.022, 0.025, 0.03073, 0.04, 0.01, 0.01])

def ADSC_computation(data: pd.DataFrame, export: bool = False) -> None:  # noqa: C901
    """Perform Air Data System Calibration (ADSC) computations for tower fly-by report.

    Generates plots for position error, altitude correction,
    airspeed correction, Mach correction, and temperature recovery factor for flight test data.
    WARNING : Graphs plotted should be updated to better fit Erb's requirements !

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing flight test data. Required columns include:
        - 'Tower_Temperature'
        - 'Tower_Pressure'
        - 'Grid_Reading'
        - 'Ti' (impact temperature)
        - 'Vic' (instrument corrected airspeed)
        - 'Hic' (instrument corrected altitude)
        - 'Mic' (instrument corrected Mach number)
        - 'YAPS' (indicator for YAPS boom presence)
    export : bool, optional
        If True, saves generated plots and output CSV files to disk.

    Notes
    -----
    - Computes derived air data quantities and error corrections using flight test data and constants.
    - Performs polynomial and linear regression for position error and recovery factor analysis.
    - Generates and optionally exports plots for:
        - Static position error pressure coefficient (ΔPp/qcic)
        - Comparison with MIL P 26292C curves
        - Altitude position correction (ΔHpc)
        - Airspeed position correction (ΔVpc)
        - Mach position correction (ΔMpc)
        - Temperature probe recovery factor (Ti/Ta-1)
    - Exports processed data and trend data to CSV files if `export` is True.

    """
    #inputs needed : Tower Temperature, Tower Pressure, Grid_Reading, Ti (impact temp), Vic, Hic, Mic
    data['Delta_Hc'] = (data['Tower_Temperature'] + cst.C_TO_K_OFFSET) / cst.T_SL_K * data['Grid_Reading'] * GRID_DELTA_H_BY_DIV   #pressure alt corrected for temperature
    data['Hc'] = data['Tower_Pressure'] + data['Delta_Hc']
    data['Pa'] = data['Hc'].apply(atm.pressure)
    data['Ps'] = data['Hic'].apply(atm.pressure)
    data['DeltaPp/Ps'] = (data['Ps'] - data['Pa']) / data['Ps']
    data['Mic'] = np.sqrt(5 * (((cst.P_SL_PSF / data['Ps']) * ((1 + 0.2 * (data['Vic'] / cst.A_SL_KT) ** 2) ** (7 / 2) - 1) + 1) ** (2 / 7) - 1))  #erb's equation C118

    data['qcic/Ps'] = cst.P_SL_PSF / data['Ps'] * ((1 + 0.2 * (data['Vic'] / cst.A_SL_KT) ** 2) ** (7 / 2) - 1)  #Erb's equation C106/delta

    data['DeltaPp/qcic'] = data['DeltaPp/Ps'] / data['qcic/Ps']  #position error pressure coefficient
    data['qc/Pa'] = (data['qcic/Ps'] + 1) / (1 - data['DeltaPp/Ps']) - 1

    data['Vc'] = cst.A_SL_KT * np.sqrt(5 * ((data['Pa'] / cst.P_SL_PSF * data['qc/Pa'] + 1) ** (2 / 7) - 1))  #erb's C126

    data['Mc'] = np.sqrt(5 * ((data['qc/Pa'] + 1) ** (2 / 7) - 1))
    data['DeltaHpc'] = data['Hc'] - data['Hic']
    data['DeltaMpc'] = data['Mc'] - data['Mic']
    data['DeltaVpc'] = data['Vc'] - data['Vic']

    data['Recovery_factor'] = ((data['Ti'] + cst.C_TO_K_OFFSET) / (data['Tower_Temperature'] + cst.C_TO_K_OFFSET) - 1)  # total air temperature probe recovery factor
    data['Reduced_Mach'] = 0.2 * data['Mc'] ** 2

    data_no_yaps = data[data['YAPS'] == 0]
    data_yaps = data[data['YAPS'] == 1]  # keep only the points with yaps boom

    #----------------------------------------------------------
    #Additional plots for position error
    #poly regression for the position error
    pos_error_coef = np.polyfit(data_yaps['Mic'], data_yaps['DeltaPp/qcic'], 2)  # get 2nd order polynom
    pos_error_fn = np.poly1d(pos_error_coef)  # create the reg function

    #std deviation
    nb_data = len(data_yaps['Mic'])
    std_dev_pos_error = np.std(data_yaps['DeltaPp/qcic'] - pos_error_fn(data_yaps['Mic']), ddof=1)

    # t-value for 95% confidence interval (two-tailed)
    t_value_pos_error = stats.t.ppf(0.975, nb_data - 1)

    # Calculate the margin of error
    interval_pos_error = t_value_pos_error * std_dev_pos_error

    #----------------------------------------------------------
    #Additional plots for recovery Factor

    #linear regression
    rec_factor_coef = np.polyfit(data_yaps['Reduced_Mach'], data_yaps['Recovery_factor'], 1)  # get the lin reg coeff
    rec_factor_fn = np.poly1d(rec_factor_coef)  # create the lin reg function

    #std deviation
    nb_data = len(data_yaps['Recovery_factor'])
    std_dev_rec_factor = np.std(data_yaps['Recovery_factor'] - rec_factor_fn(data_yaps['Reduced_Mach']), ddof=1)

    # t-value for 95% confidence interval (two-tailed)
    t_value = stats.t.ppf(0.975, nb_data - 1)

    # Calculate the margin of error
    interval_rec_factor = t_value * std_dev_rec_factor

    #-------------------------------------------------------------
    # trend extrapolated to Hpc, Vpc and Mpc graphs
    x = np.linspace(0.281, 0.938, 100)
    data_trend = pd.DataFrame({'Mic': x, 'DeltaPp/qcic': pos_error_fn(x)})

    for Hc_interpolation in [2300, 10000, 20000, 30000]:

        # Hc_interpolation=0
        Pa_interpolation = atm.pressure(Hc_interpolation)

        data_trend['qcic/Ps'] = (1 + 0.2 * data_trend['Mic'] ** 2) ** (7 / 2) - 1  # Erb's equation C70
        data_trend['DeltaPp/Ps'] = data_trend['DeltaPp/qcic'] * data_trend['qcic/Ps']
        data_trend['qc/Pa'] = (data_trend['qcic/Ps'] + 1) / (1 - data_trend['DeltaPp/Ps']) - 1

        data_trend['Ps_'+str(Hc_interpolation)+'ft'] = Pa_interpolation / (1 - data_trend['DeltaPp/Ps'])

        #calculate Mach error correction
        data_trend['Mc_'+str(Hc_interpolation)+'ft'] = np.sqrt(5 * ((data_trend['qc/Pa'] + 1) ** (2 / 7) - 1))  #Erb's equation C117
        data_trend['DeltaMpc_'+str(Hc_interpolation)+'ft'] = data_trend['Mc_'+str(Hc_interpolation)+'ft'] - data_trend['Mic']

        #calculate altitude error correction
        data_trend['Hic_'+str(Hc_interpolation)+'ft'] = (1 - (data_trend['Ps_'+str(Hc_interpolation)+'ft'] / cst.P_SL_PSF) ** (1 / 5.2559)) / (6.87559 * 10 ** -6)
        data_trend['DeltaHpc_'+str(Hc_interpolation)+'ft'] = Hc_interpolation - data_trend['Hic_'+str(Hc_interpolation)+'ft']

        #calculate airspeed error correction
        data_trend['Vc_'+str(Hc_interpolation)+'ft'] = cst.A_SL_KT * np.sqrt(5 * ((Pa_interpolation / cst.P_SL_PSF * data_trend['qc/Pa'] + 1) ** (2 / 7) - 1))
        data_trend['Vic_'+str(Hc_interpolation)+'ft'] = cst.A_SL_KT * np.sqrt(5 * ((data_trend['Ps_'+str(Hc_interpolation)+'ft'] / cst.P_SL_PSF * data_trend['qcic/Ps'] + 1) ** (2 / 7) - 1))
        data_trend['DeltaVpc_'+str(Hc_interpolation)+'ft'] = data_trend['Vc_'+str(Hc_interpolation)+'ft'] - data_trend['Vic_'+str(Hc_interpolation)+'ft']

    #-------------------------------------------------------------
    #Plot ΔPc/qcic + interval
    #-------------------------------------------------------------
    plt.figure(100, figsize=(10,7.5), dpi=100)
    ax=plt.subplot()

    trend, =ax.plot(data_yaps['Mic'],pos_error_fn(data_yaps['Mic']),'k', label='Flight test data fairing')

    ax.scatter(data_yaps['Mic'], data_yaps['DeltaPp/qcic'], marker='+', color='black', s=30, label='YAPS')
    ax.scatter(data_no_yaps['Mic'], data_no_yaps['DeltaPp/qcic'], marker='.', color='black', s=30, label='No YAPS')
    ax.plot(data_yaps['Mic'], pos_error_fn(data_yaps['Mic']) + interval_pos_error, ':k', label='95% interval confidence')
    ax.plot(data_yaps['Mic'], pos_error_fn(data_yaps['Mic']) - interval_pos_error, ':k')

    ax.set_xlabel('Instrument Corrected Mach Number, Mic', size=14, fontweight='bold')
    ax.set_ylabel('Static Position Error Pressure Coefficient, ΔPp/qcic', size=14, fontweight='bold')
    ax.set_title('Northrop T-38C Talon Position Error', pad=100, size=22, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.045, 0.045)
    ax.legend()

    ax.axhline(0, color='black', linewidth=2)   #thick line for y=0

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.tick_params(axis='both', labelsize=12, length=5)

    plt.subplots_adjust(top=0.85)

    if export:
        plt.savefig('data/TFB/Position_error.png', dpi=300, bbox_inches='tight')

    #-------------------------------------------------------------
    #Plot ΔPc/qcic + MIL_P_26292C
    #-------------------------------------------------------------
    plt.figure(101, figsize=(10,7.5), dpi=100)
    ax=plt.subplot()

    ax.plot(MIL26292C_A_MACH,MIL26292C_A_MIN, 'k--', linewidth=1, label='MIL P 26292C curve A')
    ax.plot(MIL26292C_A_MACH,MIL26292C_A_MAX, 'k--', linewidth=1, label='MIL P 26292C curve A')
    ax.plot(MIL26292C_B_MACH,MIL26292C_B_MIN, 'k:', linewidth=1, label='MIL P 26292C curve B')
    ax.plot(MIL26292C_B_MACH,MIL26292C_B_MAX, 'k:', linewidth=1, label='MIL P 26292C curve B')
    labelLines(plt.gca().get_lines(), ha='left', xvals=[0.62,0.62,0.32,0.32], yoffsets=[0.0017,-0.0017,0.0017,-0.0017], fontsize=8)

    trend, =ax.plot(data_yaps['Mic'],pos_error_fn(data_yaps['Mic']),'k', label='Flight test data fairing')

    ax.set_xlabel('Instrument Corrected Mach Number, Mic', size=14, fontweight='bold')
    ax.set_ylabel('Static Position Error Pressure Coefficient, ΔPp/qcic', size=14, fontweight='bold')
    ax.set_title('Northrop T-38C Talon Position Error Mil Spec Comparison', pad=100, size=22, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.045, 0.045)
    ax.legend(handles=[trend], labels=['Flight test data fairing'])

    ax.axhline(0, color='black', linewidth=2)   #thick line for y=0

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.tick_params(axis='both', labelsize=12, length=5)
    ax.grid()

    plt.subplots_adjust(top=0.85)

    if export:
        plt.savefig('data/TFB/Position_error_MIL_P_26292C.png', dpi=300, bbox_inches='tight')

    #-------------------------------------------------------------
    #Plot ΔHpc
    #-------------------------------------------------------------
    plt.figure(102, figsize=(10, 7.5), dpi=100)
    ax = plt.subplot()
    scatter1 = ax.scatter(data_yaps['Vic'], data_yaps['DeltaHpc'], marker='+', color='black', s=30, label='YAPS equipped aircrafts')                                 #<----------------DATA HERE
    ax.set_xlabel('Instrument Corrected Airspeed, Vic', size=14, fontweight='bold')
    ax.set_ylabel('Altitude Position Correction, ΔHpc (ft)', size=14, fontweight='bold')
    ax.set_title('Northrop T-38C Talon Altitude Position Correction', pad=100, size=22, fontweight='bold')            #leave some space below the title
    ax.set_xlim(0,661)
    ax.set_ylim(-40,100)


    styles = [':', '--', '-.', '-']
    for i, Hc_interpolation in enumerate([2300, 10000, 20000, 30000]):
        ax.plot(data_trend['Vic_'+str(Hc_interpolation)+'ft'],data_trend['DeltaHpc_'+str(Hc_interpolation)+'ft'],styles[i]+'k', label=f'{Hc_interpolation:,}'+'ft')

    labelLines(plt.gca().get_lines(), ha='left', xvals=[560,470,410,320], yoffsets=[0.00,0.00,0.00,0.00], fontsize=8)


    ax.axhline(0, color='black', linewidth=2)   #thick line for y=0
    ax.legend(handles=[scatter1], labels=['YAPS equipped aircrafts'])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis='both', labelsize=12, length=5)
    ax.grid()

    plt.subplots_adjust(top=0.85)

    if export:
        plt.savefig('data/TFB/Altitude_error.png', dpi=300, bbox_inches='tight')

    #-------------------------------------------------------------
    #Plot ΔVpc
    #-------------------------------------------------------------
    plt.figure(103, figsize=(10, 7.5), dpi=100)
    ax = plt.subplot()
    scatter1 = ax.scatter(data_yaps['Vic'], data_yaps['DeltaVpc'], marker='+', color='black', s=30, label='YAPS equipped aircrafts')                        #<----------------DATA HERE
    #ax.scatter(data_no_yaps['Vic'],data_no_yaps['DeltaVpc'],marker='.', color='black', s=30, label='No YAPS')                                 #<----------------DATA HERE
    ax.set_xlabel('Instrument Corrected Airspeed, Vic', size=14, fontweight='bold')
    ax.set_ylabel('Airspeed Position Correction, ΔVpc (kt)', size=14, fontweight='bold')
    ax.set_title('Northrop T-38C Talon Airspeed Position Correction', pad=100, size=22, fontweight='bold')            #leave some space below the title
    ax.set_xlim(0,661)
    ax.set_ylim(-0.5,1)

    for i, Hc_interpolation in enumerate([2300, 10000, 20000, 30000]):
        ax.plot(data_trend['Vic_'+str(Hc_interpolation)+'ft'],data_trend['DeltaVpc_'+str(Hc_interpolation)+'ft'],styles[i]+'k', label=f'{Hc_interpolation:,}'+'ft')

    labelLines(plt.gca().get_lines(), ha='left', xvals=[560,470,410,150], yoffsets=[0.00,0.00,0.00,0.00], fontsize=8)


    ax.axhline(0, color='black', linewidth=2)   #thick line for y=0
    ax.legend(handles=[scatter1], labels=['YAPS equipped aircrafts'])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis='both', labelsize=12, length=5)
    ax.grid()

    plt.subplots_adjust(top=0.85)

    if export:
        plt.savefig('data/TFB/Airspeed_error.png', dpi=300, bbox_inches='tight')


    #-------------------------------------------------------------
    #Plot ΔMpc
    #-------------------------------------------------------------
    plt.figure(104, figsize=(10, 7.5), dpi=100)
    ax = plt.subplot()
    scatter1 = ax.scatter(data_yaps['Mic'], data_yaps['DeltaMpc'], marker='+', color='black', s=30, label='YAPS equipped aircrafts')
    ax.set_ylabel('Mach Position Correction, ΔMpc', size=14, fontweight='bold')
    ax.set_title('Northrop T-38C Talon Mach Position Correction', pad=100, size=22, fontweight='bold')            #leave some space below the title
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.001, 0.005)

    for i, Hc_interpolation in enumerate([2300, 10000, 20000, 30000]):
        ax.plot(data_trend['Mic'], data_trend['DeltaMpc_' + str(Hc_interpolation) + 'ft'], styles[i] + 'k', label='Trend extrapolated to Hc=' + str(Hc_interpolation) + 'ft')


    ax.axhline(0, color='black', linewidth=2)   #thick line for y=0
    ax.legend(handles=[scatter1], labels=['YAPS equipped aircrafts'])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.tick_params(axis='both', labelsize=12, length=5)
    ax.grid()

    if export:
        plt.savefig('data/TFB/Mach_error.png', dpi=300, bbox_inches='tight')

    #-------------------------------------------------------------
    #Plot Ti/Ta-1
    #-------------------------------------------------------------
    plt.figure(105, figsize=(10, 7.5), dpi=100)
    ax = plt.subplot()

    ax.scatter(data_yaps['Reduced_Mach'], data_yaps['Recovery_factor'], marker='+', color='black', s=30)
    ax.plot(data_yaps['Reduced_Mach'], rec_factor_fn(data_yaps['Reduced_Mach']), '--k', label='Flight test data linear regression')
    ax.plot(data_yaps['Reduced_Mach'], rec_factor_fn(data_yaps['Reduced_Mach']) + interval_rec_factor, ':k', label='95% interval convidence')
    ax.plot(data_yaps['Reduced_Mach'], rec_factor_fn(data_yaps['Reduced_Mach']) - interval_rec_factor, ':k')

    ax.set_xlabel('Reduced Mach Number, Mred=0.2*M²', size=14, fontweight='bold')
    ax.set_ylabel('Reduced Recovery Factor, Ti/Ta-1', size=14, fontweight='bold')
    ax.set_title('Northrop T-38C Talon Temperature Calibration', pad=100, size=22, fontweight='bold')            #leave some space below the title
    ax.text(0.05, 0.85, 'Least square curve fit\nK='+str(round(rec_factor_coef[0],3))+'\nBiais = '+str(round(rec_factor_coef[1],3)), transform=plt.gca().transAxes,
        horizontalalignment='left', verticalalignment='top', fontsize=10, color='black', backgroundcolor='white', bbox=dict(facecolor='#dddddd', edgecolor='black', boxstyle='square,pad=0.5'))

    ax.set_xlim(0, 0.2)
    ax.set_ylim(-0.02, 0.175)

    ax.axhline(0, color='black', linewidth=2)   #thick line for y=0

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.025))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.005))
    ax.tick_params(axis='both', labelsize=12, length=5)
    ax.grid()
    ax.legend()

    if export:
        plt.savefig('data/TFB/Recovery_factor.png', dpi=300, bbox_inches='tight')
        data.to_csv(r"data/TFB/Output.csv", quoting=csv.QUOTE_NONE)
        data_trend.to_csv(r"data/TFB/Output_trend.csv", quoting=csv.QUOTE_NONE)
        print('CSV file generated in : data/TFB/Output.csv')

    plt.show()
