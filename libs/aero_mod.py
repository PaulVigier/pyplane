"""aero_mod.py.

This module provides functions for aerodynamic analysis of flight data,
including calculation of lift and drag coefficients, theoretical and measured values,
and plotting of results. It uses aircraft and atmospheric constants, and processes flight
data segments for the C-12 aircraft. Results can be visualized using Plotly and exported to CSV.

Functions:
- to_timestamps: Converts time segment lists to a DataFrame with proper types.
- lift_coeff: Computes aerodynamic coefficients for each segment, optionally plots
and saves results.

Dependencies:
- pandas, numpy, plotly, libs.constants, libs.das, libs.atmos_1976
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from libs import constants as cst
from libs import das
from libs import atmos_1976 as atm


def to_timestamps(times: list) -> pd.DataFrame:
    """Convert a list of time segments to a timestamps DataFrame."""
    timestamps = pd.DataFrame(times, columns=['Begin', 'End', 'Fuel', 'Conf', 'Label'])
    timestamps['Begin'] = pd.to_datetime(timestamps['Begin'], format='mixed')
    timestamps['End'] = pd.to_datetime(timestamps['End'], format='mixed')
    timestamps['Fuel'] = timestamps['Fuel'].astype(float)
    return timestamps


def lift_coeff(data: pd.DataFrame, times: list, smooth: str, ZF_CG: float, RAMP_CG: float, ZFW: float, Max_fuel: float, YAPS: bool = True, graphs: bool = False) -> pd.DataFrame:
    """Calculate lift and drag coefficients for flight data segments.

    Args:
        data (pd.DataFrame): Flight data indexed by time.
        times (list): List of segment times and configuration.
        smooth (str): Resample frequency for averaging.
        ZF_CG (float): Zero fuel center of gravity.
        RAMP_CG (float): Ramp center of gravity.
        ZFW (float): Zero fuel weight.
        Max_fuel (float): Maximum fuel weight.
        YAPS (bool): Use YAPS pitot static correction.
        graphs (bool): If True, plot results.

    Returns:
        pd.DataFrame: DataFrame of computed aerodynamic coefficients.

    """
    timestamps = to_timestamps(times)

    # Correct the timestamp to the date of the flight
    flight_date = data.index[0].date()
    timestamps_date = timestamps.Begin[0].date()
    date_offset = timestamps_date - flight_date
    timestamps['Begin'] = timestamps['Begin'] - date_offset
    timestamps['End'] = timestamps['End'] - date_offset

    lift_list = []

    nb_subplots = len(timestamps)
    titles = []
    for i in range(nb_subplots):
        titles.extend([
            f"Altitude {i + 1}",
            f"Cl vs Cl_theo {i + 1}",
            f"Cd vs Cd_theo {i + 1}"
        ])

    fig = make_subplots(
        rows=nb_subplots * 3,
        cols=1,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=titles,
        specs=[[{"secondary_y": True}] for _ in range(nb_subplots * 3)]
    )

    for lift_index, row in enumerate(timestamps.itertuples()):
        conf = row.Conf
        fuel = row.Fuel
        label = row.Label
        weight = ZFW + fuel
        Lxa = cst.C12_L_AOA + cst.C12_MAC * (ZF_CG + fuel * (RAMP_CG - ZF_CG) / Max_fuel)
        filtered_df = data.loc[row.Begin:row.End]
        filtered_df_avg = filtered_df[
            ['BARO_ALT', 'GPS_ALT', 'CAS', 'TAS', 'SAT', 'PITCH_RATE', 'LON_ACCEL', 'NORM_ACCEL',
             'AOA', 'LEFT_RPM', 'RIGHT_RPM', 'LEFT_TQ', 'RIGHT_TQ', 'AOSS']
        ].resample(smooth).mean()

        filtered_df_avg[['Hpc', 'Vpc', 'Mpc']] = filtered_df_avg.apply(
            lambda rows: pd.Series(das.C12_pitot_static(rows['CAS'], rows['BARO_ALT'], YAPS=YAPS)),
            axis=1
        )

        for i in range(len(filtered_df_avg) - 1):
            data_row = filtered_df_avg.iloc[i]
            Time_ISO = data_row.name
            Hpc = data_row['Hpc']
            CAS = data_row['CAS']
            Vpc = data_row['Vpc']
            TAS = data_row['TAS']
            Mpc = data_row['Mpc']
            AOSS = data_row['AOSS']

            Vtpc = Mpc * np.sqrt(cst.GAMMA * cst.R_SI *
                                 (data_row['SAT'] + cst.C_TO_K_OFFSET)) * cst.MS_TO_KT

            delta = atm.pressure_ratio(data_row['BARO_ALT'], units='ft') / (
                (data_row['SAT'] + cst.C_TO_K_OFFSET) / cst.T_SL_K
            )
            rho = delta * cst.RHO_SL_SCF
            AOA = data_row['AOA'] - 0.7
            AOA_true = das.C12_corrrectedAOA(AOA, np.deg2rad(data_row['PITCH_RATE']), Vtpc, Lxa)

            left_adv_ratio, left_power_coeff, left_thrust = das.C12_prop_param(
                data_row['LEFT_RPM'], data_row['LEFT_TQ'], Hpc, Vpc, data_row['SAT']
            )
            right_adv_ratio, right_power_coeff, right_thrust = das.C12_prop_param(
                data_row['RIGHT_RPM'], data_row['RIGHT_TQ'], Hpc, Vpc, data_row['SAT']
            )
            thrust = left_thrust + right_thrust

            Lift = weight * (
                data_row['LON_ACCEL'] * np.sin(np.deg2rad(AOA_true)) +
                data_row['NORM_ACCEL'] * np.cos(np.deg2rad(AOA_true))
            ) - thrust * np.sin(np.deg2rad(AOA_true))

            CL = 2 * Lift / (rho * (Vtpc * cst.KT_TO_FPS) ** 2 * cst.C12_WING_AREA)

            adv_ratio = (left_adv_ratio + right_adv_ratio) / 2
            power_coeff = (left_power_coeff + right_power_coeff) / 2

            left_blade_angle = das.C12_blade_angle(left_adv_ratio, left_power_coeff)
            right_blade_angle = das.C12_blade_angle(right_adv_ratio, right_power_coeff)
            blade_angle = (left_blade_angle + right_blade_angle) / 2
            CT = das.C12_Thrust_coeff(adv_ratio, blade_angle)
            CL_theo = das.C12_CL_theo(CT, AOA_true, conf=conf)
            Lift_theo = 0.5 * rho * (Vtpc * cst.KT_TO_FPS) ** 2 * CL_theo * cst.C12_WING_AREA

            Drag = -weight * (
                data_row['LON_ACCEL'] * np.cos(np.deg2rad(AOA_true)) -
                data_row['NORM_ACCEL'] * np.sin(np.deg2rad(AOA_true))
            ) + thrust * np.cos(np.deg2rad(AOA_true))
            CD = 2 * Drag / (rho * (Vtpc * cst.KT_TO_FPS) ** 2 * cst.C12_WING_AREA)
            CD_theo = das.C12_CD_theo(CL, CT, conf=conf)

            if 115 < CAS < 125:
                Airspeed_cat = '120'
            elif 145 < CAS < 155:
                Airspeed_cat = '150'
            elif 175 < CAS < 185:
                Airspeed_cat = '180'
            else:
                Airspeed_cat = 'out of tolerance'

            torque = (data_row['LEFT_TQ'] + data_row['RIGHT_TQ']) / 2
            deltaT = (filtered_df_avg.index[i + 1] - filtered_df_avg.index[i]).total_seconds()
            lift_list.append([
                Time_ISO, lift_index, Hpc, data_row['BARO_ALT'], data_row['GPS_ALT'], CAS,
                Vpc, TAS, Mpc, Vtpc, deltaT, data_row['AOA'], AOA_true, thrust, Lift, Lift_theo,
                CL, CL_theo, adv_ratio, power_coeff, blade_angle,
                CT, Drag, CD, CD_theo, conf, label, torque, AOSS, Airspeed_cat
            ])

    lift_data = pd.DataFrame(lift_list, columns=[
        'Time_ISO', 'ID', 'Hpc', 'BARO_ALT', 'GPS_ALT', 'CAS',
        'Vpc', 'TAS', 'Mpc', 'Vtpc', 'deltaT', 'AOA', 'AOA_true', 'Thrust', 'Lift', 'Lift_theo',
        'CL', 'CL_theo', 'adv_ratio', 'Power_coeff', 'Blade_angle',
        'CT', 'Drag', 'CD', 'CD_theo', 'Conf', 'Label', 'Torque', 'AOSS', 'Airspeed cat'
    ])
    lift_data.set_index('Time_ISO', inplace=True)

    if graphs:
        fig.update_layout(
            height=1200 * nb_subplots,
            width=1600,
            title_text="Aerodynamic Coefficients",
            showlegend=False,
        )

    for idx, group in lift_data.groupby('ID'):
        fig.add_trace(
            go.Scatter(
                x=group.index, y=group['Hpc'], yaxis='y', name=label,
                line=dict(color='blue')
            ),
            row=idx * 3 + 1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=group.index, y=group['Vpc'], yaxis='y2', name=label,
                line=dict(color='red')
            ),
            row=idx * 3 + 1, col=1, secondary_y=True
        )
        fig.update_yaxes(
            title_text="Pressure Altitude Hpc (ft)",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue"),
            row=idx * 3 + 1, col=1
        )
        fig.update_yaxes(
            title_text="Calibrated Airspeed corrected (kt)",
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            row=idx * 3 + 1, col=1, secondary_y=True
        )
        fig.update_xaxes(
            title_text="Time (UTC)",
            row=idx + 1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=group.index, y=group['CL'], yaxis='y', name=label,
                line=dict(color='green')
            ),
            row=idx * 3 + 2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=group.index, y=group['CL_theo'], yaxis='y', name=label,
                line=dict(color='green', dash='dash')
            ),
            row=idx * 3 + 2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=group.index, y=group['CD'], yaxis='y', name=label,
                line=dict(color='orange')
            ),
            row=idx * 3 + 3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=group.index, y=group['CD_theo'], yaxis='y', name=label,
                line=dict(color='orange', dash='dash')
            ),
            row=idx * 3 + 3, col=1
        )

    fig.show()
    lift_data.to_csv("data\\Aeromod\\" + flight_date.strftime('%Y-%m-%d') + "_lift_coeff.csv")
    return lift_data
