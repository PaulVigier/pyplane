"""Functions for reading DAS files from T-38 or C-12 and aeromodeling computations.

Functions:
----------
das_read(file_path, aircraft):
    Reads and processes flight data from a CSV file for the specified aircraft type ('T-38' or 'C-12').
    Returns a pandas DataFrame with standardized columns and calculated fields.
to_test_point_lite(times):
    Converts time segment lists to a DataFrame with begin and end timestamps.
to_test_point_full(times):
    Converts time segment lists to a DataFrame with begin and end timestamps, including additional metadata (for C-12
    aero modeling).
C12_CL_theo(current_CT, current_AOA, conf='Cruise'):
    Calculates the theoretical lift coefficient (CL) for the C-12 aircraft given thrust coefficient (CT),
    angle of attack (AOA), and configuration ('Cruise', 'Approach', 'Landing').
    Uses 2D interpolation and root finding based on excel C-12 model.
C12_CD_theo(current_CL, current_CT, conf='Cruise'):
    Calculates the theoretical drag coefficient (CD) for the C-12 aircraft given lift coefficient (CL),
    thrust coefficient (CT), and configuration.
    Uses 2D interpolation based on excel C-12 model.
C12_Thrust_coeff(current_adv_ratio, current_blade_angle):
    Returns the thrust coefficient for the C-12 propeller given advance ratio and blade angle.
    Uses 2D interpolation based on excel C-12 model.
C12_prop_eff(current_adv_ratio, current_power_coeff):
    Returns the propeller efficiency for the C-12 given advance ratio and power coefficient.
    Uses 2D interpolation based on excel C-12 model.
C12_blade_angle(current_adv_ratio, current_power_coeff):
    Calculates the blade angle for the C-12 propeller given advance ratio and power coefficient using root finding
    and interpolation.
C12_prop_param(RPM, TQ, BARO_ALT, TAS, SAT):
    Computes propeller parameters: advance ratio, power coefficient, and thrust for the C-12 given RPM, torque,
    barometric altitude, true airspeed, and static air temperature.
C12_pitot_static(IAS, BARO_ALT, YAPS=True):
    Computes corrected pressure altitude, calibrated airspeed, and Mach number for the C-12 from indicated airspeed
    and pressure altitude.
    Returns Hpc (ft), Vpc (kts), Mpc (Mach).
C12_corrrectedAOA(AOA, PITCH_RATE, TAS, Lxa):
    Computes corrected angle of attack for the C-12 using pitch rate, true airspeed, and reference length.
das_to_tacview(data):
    Converts processed flight data to Tacview-compatible CSV format for visualization.
    Filters repeated values and exports the file.
plot_traj(data):
    Plots the flight trajectory on a map and displays altitude, airspeed, and event data over time using Plotly.
deltaHDG(time, hdg):
    Calculates the total heading change over a time series, accounting for discontinuities when crossing north heading.

Dependencies:
-------------
- numpy
- pandas
- plotly.graph_objects
- scipy.optimize
- scipy.interpolate
- warnings
- .atmos_1976 (custom atmospheric calculations)
- .constants (custom constants)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import atmos_1976 as atm
from . import constants as cst
from scipy.optimize import fsolve, root_scalar
from scipy.interpolate import RegularGridInterpolator

#################################################################
# CONSTANTS
#################################################################


def das_read(file_path: str, aircraft: str) -> tuple[pd.DataFrame, dict]:
    """Read a DAS file and returns a pandas DataFrame containing relevant data for the specified aircraft type.

    Parameters
    ----------
    file_path : str
        Path to the DAS file to be read.
    aircraft : str
        Aircraft type. Supported values are 'T-38' and 'C-12'.

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing processed flight data for the specified aircraft, with standardized columns and calculated fields.
        Returns None if the aircraft type is not supported.

    Notes
    -----
    - If an unsupported aircraft type is provided, a warning is issued and None is returned.

    """
    if aircraft == 'T-38':
        data = pd.read_csv(file_path, sep=',')
        data['Aircraft'] = 'T-38'
        data[['IRIG_DAYS', 'IRIG_HOURS', 'IRIG_MINUTES', 'IRIG_SECONDS']] = (
            data['IRIG_TIME'].str.split(':', expand=True)
        )
        data['Time_sec'] = (
            cst.EPOCH_OFFSET +
            (data['IRIG_DAYS'].astype('int') - 1) * 86400 +
            data['IRIG_HOURS'].astype('int') * 3600 +
            data['IRIG_MINUTES'].astype('int') * 60 +
            data['IRIG_SECONDS'].astype('float').round(3)
        )
        data['Time_ISO'] = pd.to_datetime(data['Time_sec'], unit='s')
        data['Time_ISO'] = data['Time_ISO'].dt.round('ms')
        data['Time'] = data['Time_ISO'].dt.time
        data['Formatted_Time'] = data['Time_ISO'].dt.strftime('%H:%M:%S:%f').str[:-4]
        data.set_index('Time_ISO', inplace=True)
        data['LON'] = -2 * (data['GPS_LONG_DIRECT'] - 0.5) * (
            data['GPS_LONG_DEG'] + data['GPS_LONG_MIN'] / 60)
        data['LAT'] = 2 * (data['GPS_LAT_DIRECT'] - 0.5) * (
            data['GPS_LAT_DEG'] + data['GPS_LAT_MIN'] / 60)
        data['FUEL_TOTAL'] = data['EED_RIGHT_FUEL_QTY'] + data['EED_LEFT_FUEL_QTY']
        data = data.rename(columns={'AOA': 'AOA_YAPS'})
        data = data.rename(columns={
            'GPS_ALTITUDE': 'GPS_ALT',
            'ADC_AMBIENT_AIR_TEMP': 'SAT',
            'ADC_TOTAL_AIR_TEMP': 'TAT',
            'ADC_PRESSURE_ALTITUDE': 'BARO_ALT',
            'EGI_TRUE_HEADING': 'YAW',
            'EGI_PITCH_ANGLE': 'PITCH',
            'EGI_ROLL_ANGLE': 'ROLL',
            'ADC_TRUE_AIRSPEED': 'TAS',
            'ADC_COMPUTED_AIRSPEED': 'CAS',
            'ADC_AOA_NORMALIZED': 'AOA',
            'ADC_MACH': 'MACH',
            'NX_LONG_ACCEL': 'LON_ACCEL',
            'NY_LATERAL_ACCEL': 'LAT_ACCEL',
            'NZ_NORMAL_ACCEL': 'NORM_ACCEL',
            'EGI_ROLL_RATE_P': 'ROLL_RATE',
            'EGI_PITCH_RATE_Q': 'PITCH_RATE',
            'EGI_YAW_RATE_R': 'YAW_RATE',
            'RUD_PED_POS': 'DELTA_RUD',
            'LON_SP': 'DELTA_STICK_LON',
            'LAT_SP': 'DELTA_STICK_LAT',
            'EED_RIGHT_FUEL_QTY':'RIGHT_FUEL',
            'EED_LEFT_FUEL_QTY':'LEFT_FUEL',
            'EED_RIGHT_ENGINE_RPM': 'RIGHT_N1',
            'EED_LEFT_ENGINE_RPM': 'LEFT_N1',
            # 'EVENT': 'EVENT',
            # 'LEFT_FUEL_FLOW': 'LEFT_FUEL_FLOW',
            # 'RIGHT_FUEL_FLOW': 'RIGHT_FUEL_FLOW',
        })

    elif aircraft == 'C-12':
        data = pd.read_csv(file_path, sep=',')
        data['Aircraft'] = 'C-12'
        data[['IRIG_DAYS', 'IRIG_HOURS', 'IRIG_MINUTES', 'IRIG_SECONDS']] = (
            data['IRIG_TIME'].str.split(':', expand=True)
        )
        data['Time_sec'] = (
            cst.EPOCH_OFFSET +
            (data['IRIG_DAYS'].astype('int') - 1) * 86400 +
            data['IRIG_HOURS'].astype('int') * 3600 +
            data['IRIG_MINUTES'].astype('int') * 60 +
            data['IRIG_SECONDS'].astype('float').round(3)
        )
        data['Time_ISO'] = pd.to_datetime(data['Time_sec'], unit='s')
        data['Time'] = data['Time_ISO'].dt.time
        data['Formatted_Time'] = data['Time_ISO'].dt.strftime('%H:%M:%S:%f').str[:-4]
        data.set_index('Time_ISO', inplace=True)
        data['YAW'] = data['AHRS_MAGH'] + cst.DECLINATION_KEDW
        data['LON'] = -2 * (data['GPS_P_LOND'] - 0.5) * (
            data['GPS_P_LON1'] + data['GPS_P_LON2'] / 60)
        data['LAT'] = 2 * (data['GPS_P_LATD'] - 0.5) * (
            data['GPS_P_LAT1'] + data['GPS_P_LAT2'] / 60)
        data['ADC_ALT_29'] = data['ADC_ALT_29'].astype('float')
        data = data.rename(columns={
            'GPS_P_ALT': 'GPS_ALT',
            'ADC_SAT': 'SAT',
            'ADC_TAT': 'TAT',
            'ADC_ALT_29': 'BARO_ALT',
            # 'YAW': 'YAW',
            'AHRS_MAGH': 'MAG_HDG',
            'AHRS_PA': 'PITCH',
            'AHRS_RA': 'ROLL',
            'ADC_TAS': 'TAS',
            'ADC_IAS': 'CAS',
            'POS_AOA': 'AOA',
            'POS_AOSS': 'AOSS',
            'ADC_MACH': 'MACH',
            'ICU_EVNT_CNT': 'EVENT',
            'FF_MASS_LE': 'LEFT_FUEL_FLOW',
            'FF_MASS_RE': 'RIGHT_FUEL_FLOW',
            'TORQ_LE': 'LEFT_TQ',
            'TORQ_RE': 'RIGHT_TQ',
            'RPM_PROP_LE': 'LEFT_RPM',
            'RPM_PROP_RE': 'RIGHT_RPM',
            'AHRS_BPRT': 'PITCH_RATE',
            'AHRS_BRRT': 'ROLL_RATE',
            'AHRS_BYRT': 'YAW_RATE',
            'AHRS_BLGA': 'LON_ACCEL',
            'AHRS_BLTA': 'LAT_ACCEL',
            'AHRS_BNMA': 'NORM_ACCEL',
            'ADC_VS': 'VS',
        })

    else:
        raise ValueError(f"Unsupported aircraft type: '{aircraft}'. Supported types are 'T-38' and 'C-12'.")

    #get a table with events and their timestamp
    first_occurrences = data.drop_duplicates(subset='EVENT', keep='first')
    event_list=dict(zip(first_occurrences['EVENT'], first_occurrences.index,strict=True))
    return data,event_list


def to_test_point_lite(event: list,event_list:dict) -> pd.DataFrame:
    """Convert time segment lists to a DataFrame with begin and end timestamps.

    If an element is an integer, replace it with the time of the event.
    """
    processed_event = []
    for begin, end in event:
        begin_val = pd.to_datetime(begin, format='mixed') if isinstance(begin, str) else event_list.get(begin)
        end_val = pd.to_datetime(end, format='mixed') if isinstance(end, str) else event_list.get(end)
        processed_event.append([begin_val, end_val])

    timestamps = pd.DataFrame(processed_event, columns=['Begin', 'End'])
    return timestamps

def to_test_point_full(event: list, event_list: dict) -> pd.DataFrame:
    """Convert time segment lists to a DataFrame with begin and end timestamps and other metadata.

    - If Begin/End is a string, interpret it as hh:mm:ss on the date of the first event in event_list.
    - If an integer, use event_list lookup.
    """
    processed_event = []

    # Use the date of the first datetime in event_list
    first_dt = next(v for v in event_list.values() if isinstance(v, pd.Timestamp))
    base_date = first_dt.normalize()  # midnight of that date

    for begin, end, fuel, conf, label in event:
        if isinstance(begin, str):
            begin_time = pd.to_datetime(begin).time()
            begin_val = pd.Timestamp.combine(base_date, begin_time)
        else:
            begin_val = event_list.get(begin)

        if isinstance(end, str):
            end_time = pd.to_datetime(end).time()
            end_val = pd.Timestamp.combine(base_date, end_time)
        else:
            end_val = event_list.get(end)

        processed_event.append([begin_val, end_val, float(fuel), conf, label])

    return pd.DataFrame(processed_event, columns=['Begin', 'End', 'Fuel', 'Conf', 'Label'])

def to_turn_test_point(event: list, event_list: dict) -> pd.DataFrame:
    """Convert time segment lists to a DataFrame with begin, end timestamps, 2 intermediate timestamps and other metadata.

    - If events are a string, interpret them as hh:mm:ss on the date of the first event in event_list.
    - If an integer, use event_list lookup.
    """
    processed_event = []

    # Use the date of the first datetime in event_list
    first_dt = next(v for v in event_list.values() if isinstance(v, pd.Timestamp))
    base_date = first_dt.normalize()  # midnight of that date

    for begin, end_turn, start_descent, end, conf, label in event:
        if isinstance(begin, str):
            begin_time = pd.to_datetime(begin).time()
            begin_val = pd.Timestamp.combine(base_date, begin_time)
        else:
            begin_val = event_list.get(begin)

        if isinstance(end_turn, str):
            end_turn_time = pd.to_datetime(end_turn).time()
            end_turn_val = pd.Timestamp.combine(base_date, end_turn_time)
        else:
            end_turn_val = event_list.get(end_turn)

        if isinstance(start_descent, str):
            start_descent_time = pd.to_datetime(start_descent).time()
            start_descent_val = pd.Timestamp.combine(base_date, start_descent_time)
        else:
            start_descent_val = event_list.get(start_descent)

        if isinstance(end, str):
            end_time = pd.to_datetime(end).time()
            end_val = pd.Timestamp.combine(base_date, end_time)
        else:
            end_val = event_list.get(end)

        processed_event.append([begin_val, end_turn_val, start_descent_val, end_val, conf, label])

    return pd.DataFrame(processed_event, columns=['Begin', 'End_Turn', 'Start_Descent', 'End', 'Conf', 'Label'])


def C12_CL_theo(current_CT: float, current_AOA: float, conf: str = 'Cruise') -> float:
    """Return the theoretical lift coefficient (CL) based on C-12 aerodynamic excel model.

    Parameters
    ----------
    current_CT : float
        The current thrust coefficient.
    current_AOA : float
        The current angle of attack in degrees. Must be <= 20.
    conf : str, optional
        Aircraft configuration. Must be one of 'Cruise', 'Approach', or 'Landing'.
        - 'Cruise': gear up, flaps up
        - 'Approach': gear down, flaps 40%
        - 'Landing': gear down, flaps 100%
        Default is 'Cruise'.

    Returns
    -------
    float
        Theoretical lift coefficient (CL) corresponding to the given CT and AOA.

    Raises
    ------
    ValueError
        If current_AOA > 20.
        If conf is not one of 'Cruise', 'Approach', or 'Landing'.
        If root finding does not converge.

    """
    if current_AOA > 20:
        raise ValueError("CL not available for AOA>20deg.")
    if conf == 'Cruise':
        CT = np.array([0, 0.15, 0.3, 0.45, 0.6])
        CL_theo = np.array([
            -1.4, -0.7, 0, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.125,
            1.15, 1.20, 1.225, 1.25, 1.30, 1.325, 1.35, 1.40
        ])  # -1.4 and -0.7 are extrapolated values
        AOA = np.array([
            [-12.499, -11.458, -10.549, -9.753, -9.045],  # extrapolated
            [-7.291, -6.790, -6.352, -5.969, -5.628],     # extrapolated
            [-2.083, -2.122, -2.155, -2.185, -2.211],
            [5.208, 4.668, 4.197, 3.784, 3.417],
            [5.880, 5.280, 4.760, 4.280, 3.840],
            [6.710, 6.010, 5.380, 4.820, 4.310],
            [7.750, 6.900, 6.070, 5.360, 4.810],
            [8.820, 7.900, 6.920, 6.070, 5.400],
            [10.180, 9.060, 7.860, 6.920, 6.020],
            [11.680, 10.390, 8.960, 7.820, 6.760],
            [13.500, 11.820, 10.140, 8.800, 7.520],
            [16.040, 13.450, 11.450, 9.900, 8.390],
            [19.980, 14.420, 12.100, 10.490, 8.890],
            [20.000, 15.730, 12.990, 11.200, 9.400],
            [10**9, 19.850, 15.380, 12.810, 10.600],
            [10**9, 20.000, 16.900, 13.780, 11.300],
            [10**9, 10**9, 19.450, 14.810, 12.000],
            [10**9, 10**9, 21.000, 17.520, 13.980],
            [10**9, 10**9, 10**9, 20.030, 15.200],
            [10**9, 10**9, 10**9, 21.000, 16.980],
            [10**9, 10**9, 10**9, 10**9, 21.000],
        ])
    elif conf == 'Approach':
        CT = np.array([0, 0.15, 0.3, 0.45, 0.6])
        CL_theo = np.array([0, 0.700, 1.000, 1.050, 1.100, 1.150, 1.200, 1.250, 1.300, 1.350,
                            1.400, 1.450, 1.500, 1.525, 1.550, 1.600, 1.632, 1.650, 1.700, 1.725])
        AOA = np.array([
            [-4.462, -4.675, -4.862, -5.027, -5.175],
            [2.637, 1.970, 1.383, 0.862, 0.398],
            [5.660, 4.817, 4.010, 3.387, 2.800],
            [6.190, 5.310, 4.440, 3.820, 3.190],
            [6.820, 5.860, 4.920, 4.260, 3.600],
            [7.580, 6.430, 5.370, 4.690, 3.970],
            [8.410, 7.030, 5.870, 5.110, 4.360],
            [9.620, 7.740, 6.420, 5.560, 4.760],
            [11.600, 8.600, 7.000, 6.000, 5.140],
            [18.950, 9.500, 7.720, 6.550, 5.540],
            [20.000, 10.950, 8.660, 7.190, 5.980],
            [10**9, 19.000, 9.690, 7.950, 6.500],
            [10**9, 20.000, 11.400, 8.920, 7.180],
            [10**9, 10**9, 14.580, 9.600, 7.540],
            [10**9, 10**9, 19.000, 10.430, 8.020],
            [10**9, 10**9, 21.000, 12.720, 9.100],
            [10**9, 10**9, 10**9, 19.000, 9.820],
            [10**9, 10**9, 10**9, 21.000, 10.490],
            [10**9, 10**9, 10**9, 10**9, 13.200],
            [10**9, 10**9, 10**9, 10**9, 19.000],])
    elif conf == 'Landing':
        CT = np.array([0.000, 0.700, 1.300, 1.350, 1.400, 1.450, 1.500, 1.550, 1.600, 1.650, 1.700,
                       1.750, 1.800, 1.850, 1.900, 1.950, 2.000, 2.050, 2.100, 2.150, 2.200, 2.250,
                       2.300, 2.350])
        AOA = np.array([
            [-7.451, -7.885, -8.268, -8.609, -8.915],
            [-0.588, -1.448, -2.208, -2.288, -3.488],
            [5.220, 4.069, 2.900, 2.025, 1.190],
            [5.750, 4.580, 3.360, 2.490, 1.590],
            [6.260, 5.040, 3.790, 2.850, 1.980],
            [6.910, 5.530, 4.200, 3.290, 2.350],
            [7.610, 6.110, 4.730, 3.690, 2.700],
            [8.410, 6.600, 5.200, 4.090, 3.100],
            [9.400, 7.150, 5.640, 4.490, 3.460],
            [10.600, 7.800, 6.150, 4.930, 3.880],
            [12.280, 8.640, 6.690, 5.420, 4.220],
            [15.400, 9.520, 7.310, 5.860, 4.620],
            [20.000, 10.680, 7.960, 6.370, 5.010],
            [10**9, 12.270, 8.780, 6.970, 5.380],
            [10**9, 15.750, 9.800, 7.640, 5.840],
            [10**9, 20.000, 11.020, 8.300, 6.370],
            [10**9, 10**9, 13.120, 9.200, 6.970],
            [10**9, 10**9, 19.000, 10.400, 7.610],
            [10**9, 10**9, 20.000, 12.180, 8.330],
            [10**9, 10**9, 10**9, 15.000, 9.220],
            [10**9, 10**9, 10**9, 19.040, 10.310],
            [10**9, 10**9, 10**9, 21.000, 11.800],
            [10**9, 10**9, 10**9, 10**9, 14.100],
            [10**9, 10**9, 10**9, 10**9, 19.000],])
    else:
        raise ValueError("Configuration not available. Use 'Cruise', 'Approach' or 'Landing'.")
    # Create a 2D interpolator (note: axis order matters)
    interpolator = RegularGridInterpolator(
        (CL_theo, CT),
        AOA,
        bounds_error=False,
        fill_value=None
    )

    # Root-finding function: given a blade_angle, compute error from target power_coeff
    def func_to_solve(CL_theo_val: float) -> float:
        point = np.array([[CL_theo_val, current_CT]])
        return interpolator(point)[0] - current_AOA

    # Define bounds for blade angle based on input
    lower, upper = np.min(CL_theo), np.max(CL_theo)

    # Use scalar root finder to find the correct blade angle
    sol = root_scalar(func_to_solve, bracket=[lower, upper], method='brentq')

    if sol.converged:
        return sol.root
    else:
        raise ValueError("Root finding did not converge.")


def C12_CD_theo(current_CL: float, current_CT: float, conf: str = 'Cruise') -> float:
    """Calculate the theoretical drag coefficient (CD)  based on C-12 aerodynamic excel model.

    The function interpolates the drag coefficient from predefined tables for three configurations: 'Cruise',
    'Approach', and 'Landing'. For 'Landing', CT is ignored and set to zero.

    Parameters
    ----------
    current_CL : float
        The current lift coefficient. Must be greater than or equal to -0.5.
    current_CT : float
        The current thrust coefficient. Ignored for 'Landing' configuration.
    conf : str, optional
        The aircraft configuration. Must be one of 'Cruise', 'Approach', or 'Landing'. Default is 'Cruise'.

    Returns
    -------
    float
        The interpolated drag coefficient (CD) for the specified CL, CT, and configuration.

    Raises
    ------
    ValueError
        If current_CL is less than -0.5.
        If conf is not one of 'Cruise', 'Approach', or 'Landing'.
        If interpolation is attempted outside the bounds of the CL or CT tables.

    Examples
    --------
    >>> C12_CD_theo(0.5, 0.15, 'Cruise')
    0.04443
    >>> C12_CD_theo(1.0, 0.0, 'Landing')
    0.11295

    """
    if current_CL < -0.5:
        raise ValueError("CL_theo out of bounds (<-0.5).")
    CL = np.array([-0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                   0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                   1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00])
    CT = np.array([0, 0.15, 0.3, 0.45, 0.6])
    if conf == 'Cruise':
        CD = np.array([
            [0.03288, 0.04443, 0.05598, 0.06753, 0.07908],
            [0.02900, 0.04055, 0.05210, 0.06365, 0.07520],
            [0.02598, 0.03753, 0.04908, 0.06063, 0.07218],
            [0.02382, 0.03537, 0.04692, 0.05847, 0.07002],
            [0.02253, 0.03408, 0.04563, 0.05718, 0.06873],
            [0.02210, 0.03365, 0.04520, 0.05675, 0.06830],
            [0.02221, 0.03376, 0.04531, 0.05686, 0.06841],
            [0.02253, 0.03408, 0.04563, 0.05718, 0.06873],
            [0.02307, 0.03462, 0.04617, 0.05777, 0.06927],
            [0.02382, 0.03537, 0.04692, 0.05847, 0.07002],
            [0.02479, 0.03634, 0.04789, 0.05944, 0.07099],
            [0.02598, 0.03753, 0.04908, 0.06063, 0.07218],
            [0.02738, 0.03893, 0.05048, 0.06203, 0.07358],
            [0.02900, 0.04055, 0.05210, 0.06365, 0.07520],
            [0.03083, 0.04238, 0.05393, 0.06548, 0.07703],
            [0.03288, 0.04443, 0.05598, 0.06753, 0.07908],
            [0.03514, 0.04669, 0.05824, 0.06979, 0.08134],
            [0.03762, 0.04917, 0.06072, 0.07227, 0.08382],
            [0.04031, 0.05186, 0.06341, 0.07496, 0.08651],
            [0.04322, 0.05477, 0.06632, 0.07787, 0.08942],
            [0.04634, 0.05789, 0.06944, 0.08099, 0.09254],
            [0.04968, 0.06123, 0.07278, 0.08443, 0.09588],
            [0.05324, 0.06479, 0.07634, 0.08789, 0.09944],
            [0.05701, 0.06856, 0.08011, 0.09166, 0.10321],
            [0.06100, 0.07255, 0.08410, 0.09565, 0.10720],
            [0.06520, 0.07675, 0.08830, 0.09985, 0.11140],
            [0.07425, 0.08580, 0.09735, 0.10890, 0.12045],
            [0.08416, 0.09571, 0.10726, 0.11881, 0.13036],
            [0.09494, 0.10649, 0.11804, 0.12959, 0.14114],
            [0.10658, 0.11813, 0.12968, 0.14123, 0.15278],
            [0.11908, 0.13063, 0.14218, 0.15373, 0.16528],
            [0.13244, 0.14399, 0.15554, 0.16709, 0.17864],
            [0.14666, 0.15821, 0.16976, 0.18131, 0.19286],
            [0.16174, 0.17329, 0.18484, 0.19639, 0.20794],
            [0.17769, 0.18924, 0.20079, 0.21234, 0.22389],
            [0.19450, 0.20605, 0.21760, 0.22915, 0.24070],])
    elif conf == 'Approach':
        CD = np.array([
            [0.05875, 0.07432, 0.08989, 0.10546, 0.12103],
            [0.05423, 0.06980, 0.08537, 0.10094, 0.11651],
            [0.05072, 0.06629, 0.08186, 0.09743, 0.11300],
            [0.04821, 0.06378, 0.07935, 0.09492, 0.11049],
            [0.04670, 0.06227, 0.07784, 0.09341, 0.10998],
            [0.04620, 0.06177, 0.07734, 0.09291, 0.10848],
            [0.04633, 0.06190, 0.07747, 0.09304, 0.10861],
            [0.04670, 0.06227, 0.07784, 0.09341, 0.10898],
            [0.04733, 0.06290, 0.07847, 0.09404, 0.10961],
            [0.04821, 0.06378, 0.07935, 0.09492, 0.11049],
            [0.04934, 0.06491, 0.08048, 0.09605, 0.11162],
            [0.05072, 0.06629, 0.08186, 0.09743, 0.11300],
            [0.05235, 0.06792, 0.08349, 0.09906, 0.11463],
            [0.05423, 0.06980, 0.08537, 0.10094, 0.11651],
            [0.05637, 0.07194, 0.08751, 0.10308, 0.11865],
            [0.05875, 0.07432, 0.08989, 0.10546, 0.12103],
            [0.06139, 0.07696, 0.09253, 0.10810, 0.12367],
            [0.06428, 0.07985, 0.09542, 0.11099, 0.12356],
            [0.06741, 0.08298, 0.09855, 0.11412, 0.12969],
            [0.07080, 0.08637, 0.10194, 0.11751, 0.13308],
            [0.07444, 0.09001, 0.10558, 0.12115, 0.13672],
            [0.07833, 0.09390, 0.10947, 0.12504, 0.14061],
            [0.08248, 0.09805, 0.11362, 0.12919, 0.14476],
            [0.08687, 0.10244, 0.11801, 0.13358, 0.14915],
            [0.09151, 0.10708, 0.12265, 0.13822, 0.15379],
            [0.09641, 0.11198, 0.12755, 0.14312, 0.15869],
            [0.10695, 0.12252, 0.13809, 0.15366, 0.16923],
            [0.11850, 0.13407, 0.14964, 0.16521, 0.18078],
            [0.13105, 0.14662, 0.16219, 0.17776, 0.19333],
            [0.14461, 0.16018, 0.17575, 0.19132, 0.20689],
            [0.15917, 0.17474, 0.19031, 0.20588, 0.22145],
            [0.17474, 0.19031, 0.20588, 0.22145, 0.23702],
            [0.19131, 0.20688, 0.22245, 0.23802, 0.25359],
            [0.20888, 0.22445, 0.24002, 0.25559, 0.27116],
            [0.22746, 0.24303, 0.25860, 0.27417, 0.28974],
            [0.24704, 0.26261, 0.27818, 0.29375, 0.30932],])
    elif conf == 'Landing':
        CT = np.array([0])  # landing configuration is independant of CT
        current_CT = 0
        CD = np.array([[0.07628],
                      [0.07122],
                      [0.06729],
                      [0.06448],
                      [0.06279],
                      [0.06223],
                      [0.06237],
                      [0.06279],
                      [0.06349],
                      [0.06448],
                      [0.06574],
                      [0.06729],
                      [0.06911],
                      [0.07122],
                      [0.07361],
                      [0.07628],
                      [0.07923],
                      [0.08246],
                      [0.08597],
                      [0.08977],
                      [0.09384],
                      [0.09820],
                      [0.10283],
                      [0.10775],
                      [0.11295],
                      [0.11843],
                      [0.13023],
                      [0.14316],
                      [0.15721],
                      [0.17238],
                      [0.18868],
                      [0.20610],
                      [0.22465],
                      [0.24432],
                      [0.26511],
                      [0.28703],])
    else:
        raise ValueError("Configuration not available. Use 'Cruise', 'Approach' or 'Landing'.")
    interp = RegularGridInterpolator((CL, CT), CD, bounds_error=True)

    return interp([[current_CL, current_CT]])[0]


def C12_Thrust_coeff(current_adv_ratio: float, current_blade_angle: float) -> float:
    """Calculate the C12 thrust coefficient from advance ration and blade angle based on the C12 aero excel model.

    Parameters
    ----------
    current_adv_ratio : float
        The current advance ratio (J) of the propeller.
    current_blade_angle : float
        The current blade angle (degrees) of the propeller.

    Returns
    -------
    float
        The interpolated thrust coefficient value. Returns 0.0 if the input values are out of bounds.

    Raises
    ------
    ValueError
        If the provided advance ratio or blade angle is outside the bounds of the interpolation grid.

    """
    blade_angle = np.linspace(10, 55, num=10)
    adv_ratio = np.linspace(-0.1, 3.95, num=82)
    thrust_coeff = np.array([
        [0.115075, 0.152802, 0.167018, 0.129259, 0.145351, 0.157170, 0.164473, 0.166687, 0.163336, 0.154362],
        [0.109173, 0.147628, 0.162760, 0.126942, 0.143304, 0.155439, 0.163188, 0.165652, 0.162656, 0.154016],
        [0.103700, 0.141439, 0.170239, 0.124397, 0.141038, 0.153550, 0.161666, 0.164520, 0.161910, 0.153635],
        [0.100744, 0.142067, 0.164807, 0.139786, 0.138539, 0.151500, 0.160003, 0.163283, 0.161094, 0.153215],
        [0.093555, 0.135137, 0.169601, 0.154771, 0.135860, 0.149286, 0.158191, 0.162008, 0.160203, 0.152754],
        [0.089631, 0.133192, 0.162708, 0.150362, 0.133014, 0.146955, 0.156264, 0.160586, 0.159288, 0.152257],
        [0.082818, 0.130460, 0.164416, 0.162207, 0.129969, 0.144352, 0.154326, 0.159047, 0.158266, 0.151718],
        [0.076877, 0.125909, 0.156354, 0.171404, 0.126708, 0.141639, 0.152094, 0.157463, 0.157194, 0.151128],
        [0.066407, 0.121844, 0.156447, 0.164928, 0.123259, 0.138764, 0.149779, 0.155736, 0.156039, 0.150510],
        [0.055530, 0.111434, 0.159038, 0.170642, 0.119642, 0.135733, 0.147406, 0.153881, 0.154828, 0.149832],
        [0.043484, 0.099822, 0.159596, 0.174943, 0.115791, 0.132553, 0.144835, 0.152035, 0.153537, 0.149126],
        [0.031181, 0.087736, 0.148792, 0.176990, 0.130956, 0.129227, 0.142202, 0.150004, 0.152198, 0.148367],
        [0.018024, 0.075423, 0.136841, 0.183674, 0.145112, 0.125761, 0.139428, 0.147962, 0.150765, 0.147559],
        [0.004449, 0.062669, 0.124251, 0.188626, 0.173298, 0.122222, 0.136561, 0.145837, 0.149306, 0.146715],
        [-0.009732, 0.049132, 0.111943, 0.177249, 0.181131, 0.118433, 0.133615, 0.143547, 0.147776, 0.145821],
        [0, 0.035451, 0.098462, 0.164803, 0.197033, 0.114578, 0.130515, 0.141250, 0.146148, 0.144870],
        [0, 0.021165, 0.085447, 0.151959, 0.220754, 0.110631, 0.127340, 0.138880, 0.144504, 0.143888],
        [0, 0.006468, 0.071336, 0.139191, 0.209239, 0.128251, 0.124069, 0.136363, 0.142793, 0.142857],
        [0, -0.008552, 0.057341, 0.125993, 0.196219, 0.164513, 0.120734, 0.133813, 0.141016, 0.141776],
        [0, 0, 0.042900, 0.112406, 0.183927, 0.191930, 0.117238, 0.131197, 0.139176, 0.140646],
        [0, 0, 0.028198, 0.098659, 0.170771, 0.245334, 0.114939, 0.128514, 0.137248, 0.139456],
        [0, 0, 0.013206, 0.084396, 0.157924, 0.232797, 0.111244, 0.125704, 0.135294, 0.138233],
        [0, 0, -0.002093, 0.070326, 0.144497, 0.220396, 0.107477, 0.122848, 0.133276, 0.136960],
        [0, 0, 0, 0.055849, 0.130915, 0.208025, 0.105526, 0.119916, 0.131195, 0.135636],
        [0, 0, 0, 0.041081, 0.117483, 0.195309, 0.207318, 0.116906, 0.129019, 0.134262],
        [0, 0, 0, 0.026129, 0.103529, 0.182474, 0.262983, 0.115421, 0.126808, 0.132836],
        [0, 0, 0, 0.011008, 0.089614, 0.169660, 0.251231, 0.112242, 0.124535, 0.131358],
        [0, 0, 0, -0.004295, 0.075452, 0.156546, 0.239155, 0.111451, 0.122202, 0.129829],
        [0, 0, 0, 0, 0.061170, 0.143323, 0.227036, 0.108013, 0.119807, 0.128233],
        [0, 0, 0, 0, 0.046630, 0.130253, 0.214836, 0.108097, 0.117347, 0.126603],
        [0, 0, 0, 0, 0.032015, 0.116658, 0.202844, 0.289678, 0.114820, 0.124920],
        [0, 0, 0, 0, 0.017230, 0.103084, 0.190255, 0.278428, 0.112227, 0.123183],
        [0, 0, 0, 0, 0.002333, 0.089534, 0.177795, 0.267149, 0.111703, 0.121392],
        [0, 0, 0, 0, -0.012699, 0.075774, 0.165260, 0.255821, 0.108964, 0.119547],
        [0, 0, 0, 0, 0, 0.061860, 0.152821, 0.244442, 0.109307, 0.117647],
        [0, 0, 0, 0, 0, 0.047930, 0.140029, 0.233236, 0.110862, 0.115692],
        [0, 0, 0, 0, 0, 0.033863, 0.127260, 0.221570, 0.316785, 0.113664],
        [0, 0, 0, 0, 0, 0.019696, 0.114422, 0.210036, 0.306282, 0.111597],
        [0, 0, 0, 0, 0, 0.005418, 0.101513, 0.198455, 0.295989, 0.109473],
        [0, 0, 0, 0, 0, -0.008941, 0.088617, 0.186828, 0.285671, 0.107293],
        [0, 0, 0, 0, 0, 0, 0.075542, 0.175297, 0.275420, 0.105056],
        [0, 0, 0, 0, 0, 0, 0.062415, 0.163462, 0.265144, 0.102761],
        [0, 0, 0, 0, 0, 0, 0.049221, 0.151688, 0.254579, 0.100408],
        [0, 0, 0, 0, 0, 0, 0.036006, 0.139868, 0.244155, 0.097998],
        [0, 0, 0, 0, 0, 0, 0.022682, 0.128002, 0.233706, 0.339574],
        [0, 0, 0, 0, 0, 0, 0.009288, 0.116089, 0.223230, 0.330485],
        [0, 0, 0, 0, 0, 0, -0.004178, 0.104181, 0.212766, 0.321427],
        [0, 0, 0, 0, 0, 0, 0, 0.092151, 0.202304, 0.312373],
        [0, 0, 0, 0, 0, 0, 0, 0.080093, 0.191653, 0.303301],
        [0, 0, 0, 0, 0, 0, 0, 0.067988, 0.181064, 0.294046],
        [0, 0, 0, 0, 0, 0, 0, 0.055836, 0.170446, 0.284908],
        [0, 0, 0, 0, 0, 0, 0, 0.043635, 0.159800, 0.275759],
        [0, 0, 0, 0, 0, 0, 0, 0.031412, 0.149124, 0.266597],
        [0, 0, 0, 0, 0, 0, 0, 0.019117, 0.138418, 0.257422],
        [0, 0, 0, 0, 0, 0, 0, 0.006773, 0.127682, 0.248260],
        [0, 0, 0, 0, 0, 0, 0, -0.005623, 0.116970, 0.239091],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.106128, 0.229902],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.095295, 0.220583],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.084431, 0.211332],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.073533, 0.202063],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.062603, 0.192778],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.051640, 0.183474],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.040643, 0.174151],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.029612, 0.164810],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.018559, 0.155448],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.007459, 0.146085],
        [0, 0, 0, 0, 0, 0, 0, 0, -0.003677, 0.136704],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.127247],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.117802],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.108336],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.098847],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.089336],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.079803],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.070246],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.060667],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.051063],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.041436],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.031785],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.022110],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.012415],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002691],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -0.007059],])

    interp = RegularGridInterpolator((adv_ratio, blade_angle), thrust_coeff, bounds_error=True)
    try:
        return interp([[current_adv_ratio, current_blade_angle]])[0]
    except ValueError:
        print(f"ValueError: adv ratio={current_adv_ratio} or "
              f"blade angle={current_blade_angle} out of bounds for prop efficiency.")
        return 0.0


def C12_prop_eff(current_adv_ratio: float, current_power_coeff: float) -> float:
    """Return the C-12 propeller efficiency based on advance ratio and power coefficient from the C-12 aero excel model.

    Parameters
    ----------
    current_adv_ratio : float
        The advance ratio for which to estimate the propeller efficiency.
    current_power_coeff : float
        The power coefficient for which to estimate the propeller efficiency.

    Returns
    -------
    float
        The interpolated propeller efficiency value. Returns 0.0 if the input values are out of bounds.

    Raises
    ------
    ValueError
        If the input advance ratio or power coefficient is outside the bounds of the predefined grid.

    """
    adv_ratio = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                          1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8])
    power_coeff = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
                            0.16, 0.17, 0.18, 0.19, 0.2, 0.3, 0.4])
    prop_eff = np.array([
        [0.57400, 0.56900, 0.54000, 0.52500, 0.50300, 0.48100, 0.46100, 0.43900, 0.41700,
         0.40000, 0.38300, 0.36800, 0.35300, 0.33850, 0.32400, 0.31150, 0.29900, 0.20400, 0.13500],
        [0.66200, 0.66200, 0.65000, 0.63500, 0.61500, 0.59800, 0.57400, 0.55400, 0.53400,
         0.51600, 0.49800, 0.47950, 0.46100, 0.44500, 0.42900, 0.41450, 0.40000, 0.27500, 0.13600],
        [0.68300, 0.71500, 0.72500, 0.71000, 0.69800, 0.68100, 0.66600, 0.64750, 0.62900,
         0.60850, 0.58800, 0.57100, 0.55400, 0.53650, 0.51900, 0.50300, 0.48700, 0.34300, 0.21000],
        [0.70500, 0.75800, 0.76600, 0.76300, 0.75300, 0.74000, 0.72700, 0.71250, 0.69800,
         0.68150, 0.66500, 0.65050, 0.63600, 0.61900, 0.60200, 0.58600, 0.57000, 0.41100, 0.25000],
        [0.70900, 0.77700, 0.79400, 0.79400, 0.79200, 0.78400, 0.77700, 0.76350, 0.75000, 0.73650,
         0.72300, 0.71100, 0.69900, 0.68450, 0.67000, 0.65500, 0.64000, 0.48900, 0.32500],
        [0.70800, 0.78100, 0.80900, 0.82100, 0.82300, 0.82000, 0.81000, 0.79850, 0.78700, 0.77850,
         0.77000, 0.75800, 0.74600, 0.73400, 0.72200, 0.70850, 0.69500, 0.55200, 0.39200],
        [0.70700, 0.78400, 0.81400, 0.82400, 0.83200, 0.83100, 0.82700, 0.82050, 0.81400, 0.80550,
         0.79700, 0.78850, 0.78000, 0.77100, 0.76200, 0.75100, 0.74000, 0.61200, 0.46000],
        [0.70600, 0.78400, 0.81400, 0.83400, 0.84400, 0.84500, 0.84200, 0.83750, 0.83300, 0.82700,
         0.82100, 0.81350, 0.80600, 0.79800, 0.79000, 0.78200, 0.74400, 0.66700, 0.53300],
        [0.70500, 0.77800, 0.81200, 0.83100, 0.84700, 0.85200, 0.85300, 0.85000, 0.84700, 0.84200,
         0.83700, 0.83200, 0.82700, 0.81950, 0.81200, 0.80550, 0.79900, 0.71000, 0.60000],
        [0.69900, 0.77000, 0.81000, 0.83100, 0.84800, 0.85300, 0.85700, 0.85700, 0.85700, 0.85350,
         0.85000, 0.83700, 0.82400, 0.82700, 0.83000, 0.82500, 0.82000, 0.74300, 0.64000],
        [0.68900, 0.75000, 0.78600, 0.82400, 0.84300, 0.85200, 0.85900, 0.86050, 0.86200, 0.86050,
         0.85900, 0.85550, 0.85200, 0.84800, 0.84400, 0.83950, 0.83500, 0.77300, 0.69800],
        [0.67400, 0.73000, 0.75800, 0.80900, 0.83500, 0.84800, 0.85900, 0.86150, 0.86400, 0.86400,
         0.86400, 0.86100, 0.85800, 0.85550, 0.85300, 0.84900, 0.84500, 0.79200, 0.72800],
        [0.66300, 0.70550, 0.72950, 0.78400, 0.82100, 0.83700, 0.85000, 0.85550, 0.86100, 0.86250,
         0.86400, 0.86325, 0.86250, 0.86050, 0.85850, 0.85525, 0.85200, 0.80800, 0.75250],
        [0.65200, 0.68100, 0.70100, 0.75900, 0.80700, 0.82600, 0.84100, 0.84950, 0.85800, 0.86100,
         0.86400, 0.86550, 0.86700, 0.86550, 0.86400, 0.86150, 0.85900, 0.82400, 0.77700],
        [0.62000, 0.63950, 0.66400, 0.72900, 0.77800, 0.80600, 0.82900, 0.83975, 0.85050, 0.85500,
         0.85950, 0.86225, 0.86500, 0.86475, 0.86450, 0.86325, 0.86200, 0.84050, 0.79300],
        [0.58800, 0.59800, 0.62700, 0.69900, 0.74900, 0.78600, 0.81700, 0.83000, 0.84300, 0.84900,
         0.85500, 0.85900, 0.86300, 0.86400, 0.86500, 0.86500, 0.86500, 0.85700, 0.80900],
        [0, 0, 0, 0.63175, 0.69425, 0.75075, 0.79250, 0.81063, 0.82875, 0.83725, 0.84575, 0.85138,
         0.85700, 0.85900, 0.86100, 0.86175, 0.86250, 0.85600, 0.81675],
        [0, 0, 0, 0.56450, 0.63950, 0.71550, 0.76800, 0.79125, 0.81450, 0.82550, 0.83650, 0.84375,
         0.85100, 0.85400, 0.85700, 0.85850, 0.86000, 0.85500, 0.82450],
        [0, 0, 0, 0.43000, 0.53000, 0.64500, 0.71900, 0.75250, 0.78600, 0.80200, 0.81800, 0.82850,
         0.83900, 0.84400, 0.84900, 0.85200, 0.85500, 0.85300, 0.84000],
        [0, 0, 0, 0, 0.3700, 0.5250, 0.6400, 0.6830, 0.7260, 0.7525, 0.7790, 0.7950, 0.8110,
         0.8210, 0.8310, 0.8370, 0.8430, 0.8530, 0.8450],
        [0, 0, 0, 0, 0, 0.36500, 0.54000, 0.59500, 0.6500, 0.6895, 0.7290, 0.7500, 0.7710, 0.7855,
         0.80000, 0.81000, 0.82000, 0.85000, 0.84800],
        [0, 0, 0, 0, 0, 0.14000, 0.4200, 0.4900, 0.5600, 0.6125, 0.6650, 0.6955, 0.7260, 0.7430,
         0.76000, 0.77150, 0.78300, 0.84000, 0.84500],])

    interp = RegularGridInterpolator((adv_ratio, power_coeff), prop_eff, bounds_error=True)
    try:
        return interp([[current_adv_ratio, current_power_coeff]])[0]
    except ValueError:
        print(f"ValueError: adv ratio={current_adv_ratio} or power coeff={current_power_coeff} "
              f"out of bounds for prop efficiency.")
        return 0.0


def C12_blade_angle(current_adv_ratio: float, current_power_coeff: float) -> float:
    """Return C-12 blade angle based on advance ratio and power coefficient from the C-12 aero excel model.

    Parameters
    ----------
    current_adv_ratio : float
        The advance ratio (dimensionless).
    current_power_coeff : float
        The power coefficient (dimensionless).

    Returns
    -------
    float
        The blade angle (in degrees) that matches the specified power coefficient at the given advance ratio.

    Raises
    ------
    ValueError
        If the root-finding algorithm does not converge.

    Notes
    -----
    - The function relies on `RegularGridInterpolator` and `root_scalar` from SciPy.
    - The interpolation grid is defined by fixed arrays for advance ratio, blade angle, and power coefficient.
    - The blade angle is searched within the range [10, 55] degrees.

    """
    blade_angle = np.linspace(10, 55, num=10)
    advance_ratio = np.linspace(-0.1, 3.95, num=82)
    power_coeff = np.array([
        [0.038313942, 0.064798668, 0.099820566, 0.159067791, 0.2104677, 0.267577571, 0.329148457,
         0.393285171, 0.457714495, 0.520414783],
        [0.038184766, 0.064918291, 0.099789283, 0.154772765, 0.208434053, 0.265514052, 0.32729812,
         0.39143527, 0.456098091, 0.51909351],
        [0.037907534, 0.064867752, 0.098915023, 0.152976319, 0.206181125, 0.263272016, 0.325126009,
         0.389445945, 0.454379928, 0.517713471],
        [0.037380509, 0.064801389, 0.098856602, 0.146298381, 0.203692025, 0.260845485, 0.322776133,
         0.387313008, 0.452561009, 0.516281391],
        [0.036612119, 0.064399797, 0.098921779, 0.141884875, 0.201015858, 0.258230923, 0.320238363,
         0.385156831, 0.450642034, 0.514804501],
        [0.035889916, 0.06425516, 0.098614014, 0.140776555, 0.198164081, 0.255483743, 0.317564207,
         0.382795271, 0.448743906, 0.513322613],
        [0.034582794, 0.064189559, 0.099060769, 0.138887954, 0.195097635, 0.252417449, 0.314896632,
         0.380288782, 0.446699535, 0.511832982],
        [0.033274825, 0.063841088, 0.098215863, 0.13861598, 0.191793921, 0.249220219, 0.311845742,
         0.377757899, 0.444636083, 0.510326671],
        [0.030581887, 0.063629375, 0.098783465, 0.137625145, 0.188273923, 0.245826686, 0.308699757,
         0.375043962, 0.442496111, 0.50887458],
        [0.027243864, 0.061120563, 0.100660579, 0.138341384, 0.184550428, 0.242236919, 0.305489009,
         0.372176829, 0.440334236, 0.507412958],
        [0.022903271, 0.057710612, 0.102765542, 0.139855441, 0.180999505, 0.238452454, 0.302018409,
         0.36936378, 0.438110094, 0.506019547],
        [0.01776995, 0.05347318, 0.099993256, 0.141570348, 0.177914872, 0.234470533, 0.298468894,
         0.366307068, 0.435881356, 0.504646544],
        [0.01149635, 0.048432194, 0.096252439, 0.145964354, 0.177330809, 0.230287598, 0.294723541,
         0.363263331, 0.433566512, 0.503303919],
        [0.004174621, 0.04243823, 0.091541202, 0.151030514, 0.183441651, 0.225977659, 0.290840973,
         0.360118291, 0.431275209, 0.502012524],
        [-0.004393344, 0.035214353, 0.086167256, 0.147464344, 0.186868643, 0.22131318, 0.28683243,
         0.356743644, 0.428926423, 0.500745886],
        [0, 0.027011695, 0.079406564, 0.142791333, 0.195620777, 0.216509786, 0.282584188,
         0.3533665, 0.426477026, 0.499491063],
        [0, 0.01747815, 0.072009525, 0.137117527, 0.21275151, 0.211526267, 0.278197703,
         0.349877379,
         0.424038732, 0.498274974],
        [0, 0.006640526, 0.063021686, 0.130615147, 0.208714282, 0.216618951, 0.273630919,
         0.346159414, 0.421527431, 0.497063354],
        [0, -0.005512365, 0.053111765, 0.122986228, 0.203268997, 0.230123148, 0.268920027,
         0.342370845,  0.418936446, 0.49584744],
        [0, 0, 0.041846933, 0.114165286, 0.197264241, 0.245745501, 0.263920154, 0.338451922,
         0.41625749, 0.494617139],
        [0, 0, 0.029295583, 0.104238826, 0.189901022, 0.286677144, 0.259577982, 0.334392042,
         0.41344405,
         0.493350926],
        [0, 0, 0.015373321, 0.09287328, 0.181771608, 0.281293054, 0.254214176, 0.33008996,
         0.410577497, 0.492065395],
        [0, 0, 4.65785E-07, 0.080597056, 0.172278172, 0.275050469, 0.24865461, 0.325658713,
         0.407590949, 0.490730243],
        [0, 0, -0.016806843, 0.066862114, 0.161634661, 0.26790599, 0.250952066, 0.321041818,
         0.404474079, 0.489333753],
        [0, 0, 0, 0.051698927, 0.150076117, 0.259600369, 0.320700053, 0.316229117, 0.401167583,
         0.487864144],
        [0, 0, 0, 0.035165289, 0.136980866, 0.250223099, 0.374887937, 0.312492983, 0.397754355,
         0.486309705],
        [0, 0, 0, 0.017238142, 0.122816124, 0.239859174, 0.368463155, 0.30730837, 0.394180628,
         0.484658887],
        [0, 0, 0, -0.002133644, 0.107266813, 0.228211488, 0.360923668, 0.304088287, 0.390441545,
         0.482900382],
        [0, 0, 0, 0, 0.090428924, 0.215397806, 0.352395024, 0.298399814, 0.386526863, 0.481005862],
        [0, 0, 0, 0, 0.072097298, 0.201674375, 0.342827379, 0.295997081, 0.382419878, 0.479003182],
        [0, 0, 0, 0, 0.052464085, 0.186283655, 0.332457572, 0.490160118, 0.378111316, 0.476860684],
        [0, 0, 0, 0, 0.031373944, 0.169779098, 0.320537272, 0.483337863, 0.373592215, 0.474568436],
        [0, 0, 0, 0, 0.008883384, 0.15217138, 0.307690219, 0.47557915, 0.370929665, 0.472116878],
        [0, 0, 0, 0, -0.015068876, 0.133135422, 0.293710615, 0.46685123, 0.366007145, 0.469496826],
        [0, 0, 0, 0, 0, 0.112703534, 0.278787948, 0.457134713, 0.364297128, 0.466699469],
        [0, 0, 0, 0, 0, 0.091061437, 0.262351947, 0.446628499, 0.364147914, 0.463716376],
        [0, 0, 0, 0, 0, 0.068003548, 0.244840146, 0.434697582, 0.637499393, 0.460511208],
        [0, 0, 0, 0, 0, 0.043566779, 0.226120089, 0.4219002, 0.63065472, 0.457130092],
        [0, 0, 0, 0, 0, 0.017711212, 0.206174831, 0.408045736, 0.623066882, 0.453540208],
        [0, 0, 0, 0, 0, -0.009530237, 0.185126296, 0.393118961, 0.614578547, 0.449734536],
        [0, 0, 0, 0, 0, 0, 0.162641265, 0.377309348, 0.605263477, 0.445706392],
        [0, 0, 0, 0, 0, 0, 0.138913633, 0.360036215, 0.595038575, 0.441449407],
        [0, 0, 0, 0, 0, 0, 0.113901791, 0.341800572, 0.583592685, 0.436957521],
        [0, 0, 0, 0, 0, 0, 0.087685734, 0.322438856, 0.571367996, 0.432226371],
        [0, 0, 0, 0, 0, 0, 0.060081234, 0.301938955, 0.55818031, 0.828593791],
        [0, 0, 0, 0, 0, 0, 0.031151821, 0.280289277, 0.544018692, 0.822112218],
        [0, 0, 0, 0, 0, 0, 0.000876676, 0.257578092, 0.528930659, 0.8148586],
        [0, 0, 0, 0, 0, 0, -0.030749662, 0.233554382, 0.512901563, 0.80680843],
        [0, 0, 0, 0, 0, 0, 0, 0.208383781, 0.495610624, 0.797936154],
        [0, 0, 0, 0, 0, 0, 0, 0.182022148, 0.477449164, 0.788045973],
        [0, 0, 0, 0, 0, 0, 0, 0.154460062, 0.45826568, 0.777445576],
        [0, 0, 0, 0, 0, 0, 0, 0.125688475, 0.43805164, 0.765996198],
        [0, 0, 0, 0, 0, 0, 0, 0.095760689, 0.416798842, 0.753690111],
        [0, 0, 0, 0, 0, 0, 0, 0.064550501, 0.394499397, 0.740519881],
        [0, 0, 0, 0, 0, 0, 0, 0.032104914, 0.371145718, 0.726519843],
        [0, 0, 0, 0, 0, 0, 0, -0.001594832, 0.346860259, 0.71165903],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.321276561, 0.695913021],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.294714239, 0.679067536],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.267070223, 0.661472985],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.238338213, 0.642975201],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.208512134, 0.623568466],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.177586127, 0.603247261],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.145554537, 0.58200626],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.11241191, 0.559840324],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.078191319, 0.536744494],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.042813022, 0.512762167],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.006303525, 0.487850704],
        [0, 0, 0, 0, 0, 0, 0, 0, -0.031332312, 0.461844521],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.434981462],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.407166194],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.37839468],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.34866302],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.317967441],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.286304293],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.253670048],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.220061291],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.185474717],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.149907128],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.113355431],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.075836424],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.037308217],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -0.002214551],
        ])

    # Create a 2D interpolator (note: axis order matters)
    interpolator = RegularGridInterpolator(
        (advance_ratio, blade_angle),
        power_coeff,
        bounds_error=False,
        fill_value=None
    )

    # Root-finding function: given a blade_angle, compute error from target power_coeff
    def func_to_solve(blade_angle_val: float) -> float:
        point = np.array([[current_adv_ratio, blade_angle_val]])
        return interpolator(point)[0] - current_power_coeff

    # Define bounds for blade angle based on input
    lower, upper = np.min(blade_angle), np.max(blade_angle)

    # Use scalar root finder to find the correct blade angle
    sol = root_scalar(func_to_solve, bracket=[lower, upper], method='brentq')

    # Check if the root-finding was successful
    if sol.converged:
        return sol.root
    else:
        raise ValueError("Root finding did not converge.")


def C12_prop_param(RPM: float, TQ: float, BARO_ALT: float, TAS: float, SAT: float) -> tuple:
    """Calculate propeller parameters for the C12 aircraft.

    Parameters
    ----------
    RPM : float
        Propeller revolutions per minute.
    TQ : float
        Engine torque (units assumed to be consistent with calculation).
    BARO_ALT : float
        Pressure altitude in feet.
    TAS : float
        True airspeed in knots.
    SAT : float
        Static air temperature in Celsius.

    Returns
    -------
    tuple: A tuple containing:
        - adv_ratio (float): Advance ratio of the propeller.
        - power_coeff (float): Power coefficient of the propeller.
        - thrust (float): Estimated propeller thrust in pounds.

    Notes
    -----
    - Uses interpolation of empirical data for pitot-static corrections.
    - Requires external modules/constants: atm, cst, np.

    """
    delta = atm.pressure_ratio(BARO_ALT, units='ft') / ((SAT + cst.C_TO_K_OFFSET) / cst.T_SL_K)
    rho = delta * cst.RHO_SL_SCF
    torque = TQ * 2230 / 100
    SHP = torque * RPM * 2 * np.pi / 60 * 1 / 550  # HP

    adv_ratio = 101.4 * TAS / (RPM * cst.C12_PROP_DIAMETER)
    power_coeff = SHP * 550 / (rho * (RPM / 60) ** 3 * cst.C12_PROP_DIAMETER ** 5)
    prop_eff = C12_prop_eff(adv_ratio, power_coeff)

    thrust = prop_eff * 550 * SHP / (TAS * cst.KT_TO_FPS) * (1 + 0.15 / 0.85)

    return adv_ratio, power_coeff, thrust


def C12_pitot_static(IAS: float, BARO_ALT: float, YAPS: bool = True) -> tuple:
    """Calculate corrected pressure altitude, calibrated airspeed, and Mach number for C-12 with or without YAPS boom.

    Parameters
    ----------
    IAS : float
        Indicated Airspeed in knots.
    BARO_ALT : float
        Pressure altitude in feet.
    YAPS : bool, optional
        If True, applies YAPS correction (default is True).

    Returns
    -------
    Hpc : float
        Corrected pressure altitude in feet.
    Vpc : float
        Corrected calibrated airspeed in knots.
    Mpc : float
        Corrected Mach number.

    Notes
    -----
    - Uses interpolation of empirical data for pitot-static corrections.
    - Requires external modules/constants: atm, cst, np.

    """
    Mic_non_yaps = [0.157967378, 0.170702635, 0.183877039, 0.200125471, 0.22076537, 0.250188206,
                    0.300250941, 0.350752823, 0.392910916, 0.6]
    deltaPp_Qcic_non_yaps = [0.05433526, 0.038439306, 0.027456647, 0.016473988, 0.007803468,
                             0.001156069, -0.005491329, -0.010982659, -0.013583815, -0.013583815]

    Mic_yaps = [0.157967378, 0.17685069, 0.200564617, 0.220326223, 0.236135508, 0.282685069,
                0.315621079, 0.345483061, 0.361731493, 0.6]
    deltaPp_Qcic_yaps = [0.05433526, 0.05982659, 0.069364162, 0.076878613, 0.080924855,
                         0.088150289, 0.089017341, 0.082947977, 0.07716763, 0.07716763]

    Ps = atm.pressure(BARO_ALT)
    Mic = np.sqrt(
        5 * (
            (
                (cst.P_SL_PSF / Ps)
                * ((1 + 0.2 * (IAS / cst.A_SL_KT) ** 2) ** (7 / 2) - 1)
                + 1
            ) ** (2 / 7)
            - 1
        )
    )  # erb's equation C118

    deltaPp_Qcic = (
        np.interp(Mic, Mic_yaps, deltaPp_Qcic_yaps)
        if YAPS
        else np.interp(Mic, Mic_non_yaps, deltaPp_Qcic_non_yaps)
    )  # interpolation of the data from annexe E

    # get the corrected altitude
    Qcic = ((1 + 0.2 * (IAS / cst.A_SL_KT) ** 2) ** (7 / 2) - 1) * cst.P_SL_PSF
    Pa = Ps - deltaPp_Qcic * Qcic
    Hpc = atm.pressure_alt(Pa / cst.P_SL_PSF)  # get the corrected altitude in feet

    # get the corrected Mach number
    Qc_Pa = (Qcic / Ps + 1) / (1 - deltaPp_Qcic * Qcic / Ps) - 1
    Mpc = np.sqrt(5 * ((Qc_Pa + 1) ** (2 / 7) - 1))

    # get the corrected calibrated airspeed
    Vpc = cst.A_SL_KT * np.sqrt(5 * ((Pa / cst.P_SL_PSF * Qc_Pa + 1) ** (2 / 7) - 1))

    return Hpc, Vpc, Mpc


def C12_correctedAOA(AOA: float, PITCH_RATE: float, TAS: float, Lxa: float) -> float:
    """Correct the AOA from the YAPS boom for pitch rate.

    Parameters
    ----------
    AOA : float
        Measured angle of attack in degrees.
    PITCH_RATE : float
        Aircraft pitch rate in degrees per second.
    TAS : float
        True airspeed in knots.
    Lxa : float
        Lever arm distance in feet between CG and AOA vane on the YAPS boom.

    Returns
    -------
    float
        Corrected angle of attack in degrees.

    """
    def equation(x: float) -> float:
        res = (
            np.tan(x)
            - np.tan(np.deg2rad(AOA))
            - (PITCH_RATE * Lxa) / (TAS * cst.KT_TO_FPS * np.cos(x))
        )
        return res
    AOA_T = fsolve(equation, 0)
    return float(np.rad2deg(AOA_T))


def das_to_tacview(data: pd.DataFrame) -> None:
    """Convert a DAS-formatted pandas DataFrame to a Tacview-compatible CSV file.

    Processed data are exported to a CSV file named '<Aircraft_Type> Tacview.csv' in the 'output' directory.

    Args:
        data (pd.DataFrame): Input DataFrame containing DAS flight data (as returned by das_read function).

    Returns:
        None

    """
    tacview_data = pd.DataFrame({
                                 'Time': data['Time_sec'],
                                'Longitude': data['LON'],
                                'Latitude': data['LAT'],
                                'Altitude': data['GPS_ALT']*0.3048,
                                'Yaw': data['YAW'],
                                'Pitch': data['PITCH'],
                                'Roll': data['ROLL'],
                                'TAS': data['TAS'],  # in KTAS
                                'CAS': data['CAS'],  # in KCAS
                                'AOA': data['AOA'],  # in degrees
                                'Mach': data['MACH'],
                                'Event': data['EVENT'],
                                })

    # formatting the data
    tacview_data['Longitude'] = tacview_data['Longitude'].round(6)
    tacview_data['Latitude'] = tacview_data['Latitude'].round(6)
    tacview_data['Altitude'] = tacview_data['Altitude'].round(2)
    tacview_data['Yaw'] = tacview_data['Yaw'].round(2)
    tacview_data['Pitch'] = tacview_data['Pitch'].round(2)
    tacview_data['Roll'] = tacview_data['Roll'].round(2)
    tacview_data['TAS'] = tacview_data['TAS'].round(2)
    tacview_data['CAS'] = tacview_data['CAS'].round(2)
    tacview_data['AOA'] = tacview_data['AOA'].round(2)
    tacview_data['Mach'] = tacview_data['Mach'].round(3)

    Aircraft_Type = data['Aircraft'].iloc[0]

    # Replace repeated values with ''
    tacview_data_filtered = tacview_data.where(tacview_data.ne(tacview_data.shift()), '')

    # Optionally, keep the first row unchanged (non-empty)
    tacview_data_filtered.iloc[0] = tacview_data.iloc[0]

    tacview_data_filtered.to_csv('output\\'+Aircraft_Type+' Tacview.csv', index=False)
    print('Tacview file exported')


def plot_traj(data: pd.DataFrame) -> None:
    """Plot trajectory for flight overview.

    This function generates two interactive plots using Plotly:
    1. A map displaying the trajectory based on latitude and longitude, with markers colored by barometric altitude.
       - Hovering over points shows formatted time, altitude, computed airspeed (CAS), and event information.
    2. A multi-panel plot showing altitude, CAS, and event as functions of time.
    Additionally, prints a summary of each time the event value changes, including the timestamp and event name.

    Parameters
    ----------
    data : pd.DataFrame
        as returned by das_read function

    Returns
    -------
    None

    """
    graph = go.Scattermap(
        lat=data['LAT'],
        lon=data['LON'],
        mode='markers+lines',
        line=dict(
            color='black',
        ),
        marker=dict(
            color=data['BARO_ALT'],  # Color by GPS_ALTITUDE
            colorscale='HSV',  # Color scale
            size=6,  # Size of markers
            showscale=True,  # Show color scale
            cmin=0,  # Set minimum value of the color scale
            cmax=40000,  # Set maximum value of the color scale
            colorbar=dict(
                title='Altitude',
                tickvals=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
                ticktext=['0', '5k', '10k', '15k', '20k', '25k', '30k', '35k', '40k'],
            ),
        ),
        customdata=data[['Formatted_Time', 'BARO_ALT', 'CAS', 'EVENT']],
        hovertemplate='<b>%{customdata[0]}</b><br>' +  # Index (or time if you prefer)
        'Altitude: %{customdata[1]:.0f} ft<br>' +  # Rounded Altitude
        'CAS: %{customdata[2]:.1f} kt<br>' +  # CAS (Computed Airspeed)
        'Event: %{customdata[3]}' +
        '<extra></extra>',
        )

    layout = go.Layout(
        map=dict(
            style="satellite",  # Change style to open-street-map (no token required)
            center=dict(lat=data['LAT'].mean(), lon=data['LON'].mean()),  # Center the map
            zoom=10,  # Adjust zoom level as needed
        ),
        width=1300,
        height=800,
    )

    fig = go.Figure(data=[graph], layout=layout)
    fig.show()

    # altitude and CAS as a function of time
    layout = dict(
        hoversubplots="axis",
        title="Altitude and Airspeed",
        hovermode="x unified",
        grid=dict(rows=3, columns=1),
        xaxis=dict(showgrid=True,
                   hoverformat='<b>%H:%M:%S.%f</b>'),
        yaxis=dict(showgrid=True, title='Altitude (ft)'),
        yaxis2=dict(showgrid=True, title='CAS (kt)'),
        yaxis3=dict(showgrid=True, title='Event'),
        width=1300,
        height=800,
        )

    graph = [
        go.Scatter(
            x=data.index, y=data["BARO_ALT"], xaxis="x", yaxis="y",
            name="Altitude",
            hovertemplate='Altitude: %{y:.0f} ft<extra></extra>'
        ),
        go.Scatter(
            x=data.index, y=data["CAS"], xaxis="x", yaxis="y2",
            name="CAS",
            hovertemplate='CAS: %{y:.1f} KCAS<extra></extra>'
        ),
        go.Scatter(
            x=data.index, y=data["EVENT"], xaxis="x", yaxis="y3",
            name="Event",
            hovertemplate='Event: %{y} <extra></extra>'
        ),
    ]

    fig2 = go.Figure(data=graph, layout=layout)

    fig2.show()

    # print each time event is incremented
    new_event = data['EVENT'] != data['EVENT'].shift()
    formatted_data = data[new_event][['EVENT']].copy()
    formatted_data.index = formatted_data.index.strftime('%Y-%m-%d %H:%M:%S')
    print(formatted_data)


def deltaHDG(time: pd.Series, hdg: pd.Series) -> float:
    """Calculate the heading change for over a time serie.

    Args:
        time (pd.Series): A pandas Series representing the time steps.
        hdg (pd.Series): A pandas Series representing the heading values at each time step.

    Returns:
        float: The total heading change over the given time series, with corrections for large jumps (>60 degrees).

    """
    total_hdg_change = 0
    previous_delta_hdg = 0

    # Calculate the delta HDG for each time step
    for i in range(1, len(time)):
        delta_hdg = hdg.iloc[i] - hdg.iloc[i-1]
        # print('delta_hdg',delta_hdg)
        if np.abs(delta_hdg) > 60:  # avoids jumps when crossing north heading
            delta_hdg = previous_delta_hdg
        else:
            previous_delta_hdg = delta_hdg
        total_hdg_change += delta_hdg
    # print('Total heading change:', total_hdg_change)
    return total_hdg_change
