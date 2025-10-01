"""Library for Have dream TMP."""

import numpy as np
import pandas as pd
from libs import atmos_1976 as atm
from libs import constants as cst
from libs import das
from libs import geo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from typing import Tuple, Dict


def sawtooth_descent(data: pd.DataFrame, event_list: dict, times: list, ZFW: float, flight: int, smooth: str='200ms',  graphs: bool=False
                   ) -> pd.DataFrame:
    """Post process data for T-38 sawtooth descent for L/D regression.

    Workflow:
        - TO BE COMPLETED
        - Return the processed DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing flight data indexed by timestamp as return by das_read function
        event_list (dict): Dictionary containing time for each event.
        times (list): List specifying the start and end timestamps for each level acceleration segment.
        smooth (string): rolling average to smooth the data (default is '200ms').
        ZFW (float): Zero fuel weight in pounds.
        flight (int): Flight number.
        graphs (bool, optional): If True, generate and display plots for each sawtooth climb segment
        (default is False).

    Returns:
        pd.DataFrame: DataFrame containing processed sawtooth climb data, including calculated energy states,
        fitted curves, and performance metrics.

    """
    sawtooth_data=das.to_test_point_full(times,event_list)

    #correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date=data.index[0].date()
    timestamps_date=sawtooth_data.Begin[0].date()
    date_offset=timestamps_date-flight_date
    sawtooth_data['Begin']=sawtooth_data['Begin']-date_offset
    sawtooth_data['End']=sawtooth_data['End']-date_offset

    #prepare plots
    i = len(sawtooth_data)
    cols = 3
    rows = (i + cols - 1) // cols  # This calculates the number of rows needed

    # Create a subplot grid
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=False,  # Optional: you can customize axes sharing
        shared_yaxes=False,  # We will be using individual y-axes for each plot
        vertical_spacing=0.05,
        horizontal_spacing=0.1,
        subplot_titles=[f"Sawtooth Descent {plot_index+1}" for plot_index in range(i)],
        # Define a secondary y-axis for each subplot
        specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
    )

    results=[]

    print(f"{'Segment':>7} | {'Roll':>10} | {'CAS':>10} | {'N1':>10} | {'Pitch':>10}")
    print("-"*60)

    #loop for pairs of timestamps
    for climb_index,row in enumerate(sawtooth_data.itertuples()):

        filtered_df=data.loc[row.Begin:row.End]
        # print(f"Processing sawtooth descent {climb_index+1} from {row.Begin.time()} to {row.End.time()} with {len(filtered_df)} samples")
        filtered_df_avg=filtered_df[['BARO_ALT','GPS_ALT','CAS','TAS','SAT', 'RIGHT_N1', 'LEFT_N1','FUEL_TOTAL','ROLL','PITCH','NORM_ACCEL']].resample(smooth).mean() #average the data

        deltaT=(filtered_df_avg.index[-1]-filtered_df_avg.index[0]).total_seconds()
        deltaH=filtered_df_avg['BARO_ALT'].max()-filtered_df_avg['BARO_ALT'].min()
        BARO_ALT=(filtered_df_avg['BARO_ALT'].max()+filtered_df_avg['BARO_ALT'].min())/2
        GPS_ALT=(filtered_df_avg['GPS_ALT'].max()+filtered_df_avg['GPS_ALT'].min())/2
        CAS=filtered_df_avg['CAS'].mean()
        TAS=filtered_df_avg['TAS'].mean()
        SAT=filtered_df_avg['SAT'].mean()
        ROD_corr=deltaH*cst.FT_TO_M/deltaT*(SAT+cst.C_TO_K_OFFSET)/atm.temperature(BARO_ALT*cst.FT_TO_M,units='m') #m/s
        deltaV=filtered_df_avg['TAS'].iloc[-1]-filtered_df_avg['TAS'].iloc[0]
        density=atm.density(BARO_ALT*cst.FT_TO_M,units='m') # Update to take temp into account ?
        gamma=-np.degrees(np.arcsin(ROD_corr/(TAS*cst.KT_TO_MS))) #negative in a descent
        l_over_d=-1/np.tan(np.radians(gamma)) #L/D >0 in a descent
        weight_lb=ZFW+filtered_df_avg['FUEL_TOTAL'].mean()
        weight_n=weight_lb*cst.LB_TO_KG*cst.G_METRIC

        #validity checks
        roll_valid=filtered_df_avg['ROLL'].abs().max()<5
        CAS_valid=filtered_df_avg['CAS'].std()<1
        N1_valid=(filtered_df_avg['RIGHT_N1'].std()<1) & (filtered_df_avg['LEFT_N1'].std()<1)
        pitch_valid=filtered_df_avg['PITCH'].std()<2

        # Store results for later assignment
        results.append(
            dict(
                deltaT=deltaT,
                deltaH=deltaH,
                BARO_ALT=BARO_ALT,
                GPS_ALT=GPS_ALT,
                CAS=CAS,
                TAS=TAS,
                deltaV=deltaV,
                SAT=SAT,
                ROD_corr=ROD_corr,
                density=density,
                gamma=gamma,
                l_over_d=l_over_d,
                weight_lb=weight_lb,
                weight_n=weight_n,
                roll_valid=roll_valid,
                CAS_valid=CAS_valid,
                N1_valid=N1_valid,
                pitch_valid=pitch_valid,
                valid=roll_valid & CAS_valid & N1_valid & pitch_valid
                )
        )
        print(f"{climb_index+1:>7} | {str(roll_valid):>10} | {str(CAS_valid):>10} | {str(N1_valid):>10} | {str(pitch_valid):>10}")

        row = climb_index // cols + 1
        col = climb_index % cols + 1

        # Creating the plots
        trace1 = go.Scatter(x=filtered_df_avg.index, y=filtered_df_avg['BARO_ALT'], mode='lines', name='Pressure Altitude (ft)', line=dict(color='blue'),)

        trace2 = go.Scatter(x=filtered_df_avg.index, y=filtered_df_avg['CAS'], mode='lines', name='Calibrated airspeed (kt)', line=dict(color='red'), )

        # Add traces to the subplot
        fig.add_trace(trace1, row=row, col=col)
        fig.add_trace(trace2, row=row, col=col, secondary_y=True)

        # Update axis labels and titles
        fig.update_yaxes(title_text="Pressure Altitude (ft)", title_font=dict(color="blue"),  tickfont=dict(color="blue"), row=row, col=col)
        fig.update_yaxes(title_text="Calibrated Airspeed (kt)", title_font=dict(color="red"),  range=[CAS-3, CAS+3], tickfont=dict(color="red"), row=row, col=col, secondary_y=True)

     # Assign results in one go
    results_df = pd.DataFrame(results, index=sawtooth_data.index)
    sawtooth_data.drop(columns="Fuel", inplace=True)
    sawtooth_data = pd.concat([sawtooth_data, results_df], axis=1)

    if graphs:
        # Update layout for better presentation
        fig.update_layout(
            height=350*rows,  # Adjust height based on number of rows
            width=1500,  # You can adjust the width
            title_text="Sawtooth climbs",
            showlegend=False,
        )

        # Show the figure
        fig.show()


    sawtooth_data.to_csv("data\\Have dream\\"+flight_date.strftime('%Y-%m-%d')+"_sawtooth_flight_"+str(flight)+".csv")

    return sawtooth_data

def ld_2coeff(TAS: float, Mass_kg: float, rho: float, S: float, Cd0: float, k: float) -> float:
    q = 0.5 * rho * TAS**2
    CL = Mass_kg*cst.G_METRIC / (q * S)
    CD = Cd0 + k * CL**2
    return CL / CD

def ld_3coeff(TAS: float, Mass_kg: float, rho: float, S: float, Cd0: float, k: float, kl: float) -> float:
    q = 0.5 * rho * TAS**2
    CL = Mass_kg*cst.G_METRIC / (q * S)
    CD = Cd0 + k * CL**2 + kl * CL
    return CL / CD

def fit_drag_models(df: pd.DataFrame, S: float) -> Dict[str, dict]:
    """
    Fit Cd0, k (2-coeff model) and Cd0, k, kl (3-coeff model).
    
    Returns:
        dict with results for both models, each containing:
        - coeffs (fitted parameters)
        - cov (covariance matrix)
        - std (standard deviations)
    """
    # Prepare inputs
    V = df["TAS"].values * cst.KT_TO_MS
    W = df["weight_lb"].values * cst.LB_TO_KG
    rho = df["density"].values
    L_over_D = df["l_over_d"].values

    # --- 2-coefficient model ---
    def model_2(x: tuple, Cd0: float, k: float):
        V, W, rho = x
        return ld_2coeff(V, W, rho, S, Cd0, k)

    p0_2 = [0.12, 0.016]
    popt2, pcov2 = curve_fit(
        model_2, (V, W, rho), L_over_D, p0=p0_2, 
        bounds=([0, 0], [1, 10]),
        maxfev=5000
    )

    # --- 3-coefficient model ---
    def model_3(x: tuple, Cd0: float, k: float, kl: float):
        V, W, rho = x
        return ld_3coeff(V, W, rho, S, Cd0, k, kl)

    p0_3 = [0.2, 0.24, 0.0]
    popt3, pcov3 = curve_fit(
        model_3, (V, W, rho), L_over_D, p0=p0_3,
        bounds=([0, 0, -np.inf], [1, 10, np.inf]),
        maxfev=5000
    )

    return {
        "2coeff": {
            "coeffs": popt2,
            "cov": pcov2,
            "std": np.sqrt(np.diag(pcov2)),
        },
        "3coeff": {
            "coeffs": popt3,
            "cov": pcov3,
            "std": np.sqrt(np.diag(pcov3)),
        },
    }

def plot_ld_surface_models(
    df: pd.DataFrame, 
    coeffs2: Tuple[float, float], 
    coeffs3: Tuple[float, float, float], 
    S: float
) -> None:
    """Plot smooth L/D surfaces from both 2- and 3-coefficient models, 
    overlay measured data, and display RMSE for each fit.

    Args:
        df: DataFrame containing 'TAS', 'weight_lb', 'density', 'l_over_d', 'conf'
        coeffs2: (Cd0, k) from 2-coefficient model
        coeffs3: (Cd0, k, kl) from 3-coefficient model
        S: Wing reference area [m^2]
    """
    # Extract data
    V = df["TAS"].values       # TAS in knots
    W = df["weight_lb"].values # Weight in pounds
    rho = df["density"].values
    L_over_D = df["l_over_d"].values
    conf = df["Conf"].iloc[0]

    # Build regular grid
    V_lin = np.linspace(V.min(), V.max(), 50)
    W_lin = np.linspace(W.min(), W.max(), 50)
    VV, WW = np.meshgrid(V_lin, W_lin)

    # Interpolate rho onto grid
    rho_grid = griddata(
        points=(V, W),
        values=rho,
        xi=(VV, WW),
        method="linear"
    )
    if np.isnan(rho_grid).any():
        rho_grid[np.isnan(rho_grid)] = griddata(
            (V, W), rho, (VV, WW), method="nearest"
        )[np.isnan(rho_grid)]

    # Convert to SI before model evaluation
    V_ms = V * cst.KT_TO_MS
    W_kg = W * cst.LB_TO_KG
    VV_ms = VV * cst.KT_TO_MS
    WW_kg = WW * cst.LB_TO_KG

    # Unpack coefficients
    Cd0_2, k_2 = coeffs2
    Cd0_3, k_3, kl_3 = coeffs3

    # Compute smooth surfaces
    ZZ2 = ld_2coeff(VV_ms, WW_kg, rho_grid, S, Cd0_2, k_2)
    ZZ3 = ld_3coeff(VV_ms, WW_kg, rho_grid, S, Cd0_3, k_3, kl_3)

    # --- RMSE calculation on measured data ---
    pred2 = ld_2coeff(V_ms, W_kg, rho, S, Cd0_2, k_2)
    pred3 = ld_3coeff(V_ms, W_kg, rho, S, Cd0_3, k_3, kl_3)

    rmse2 = np.sqrt(np.mean((L_over_D - pred2) ** 2))
    rmse3 = np.sqrt(np.mean((L_over_D - pred3) ** 2))

    # --- Plot ---
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=VV, y=WW, z=ZZ2,
        colorscale="Greys",
        opacity=0.6,
        name=f"2-coeff model (RMSE={rmse2:.2f})",
        showscale=False
    ))

    fig.add_trace(go.Surface(
        x=VV, y=WW, z=ZZ3,
        colorscale="Rainbow",
        opacity=0.6,
        name=f"3-coeff model (RMSE={rmse3:.2f})",
        showscale=False
    ))

    # --- Measured data with validity coloring ---
    if "valid" in df.columns:
        fig.add_trace(go.Scatter3d(
            x=V,
            y=W,
            z=L_over_D,
            mode="markers",
            marker=dict(
                size=6,
                color=df["valid"].map({True: "green", False: "red"}),
                symbol="circle"
            ),
            name="Measured data"
        ))

    fig.update_layout(
        title=f"L/D surfaces for {conf} configuration<br>"
              f"2-coeff RMSE={rmse2:.2f} | 3-coeff RMSE={rmse3:.2f}",
        scene=dict(
            xaxis_title="TAS [kt]",
            yaxis_title="Weight [lb]",
            zaxis_title="L/D"
        ),
        width=1300,
        height=1000,
        legend=dict(x=0.02, y=0.98)
    )
    fig.show()


def ld_max_vs_weight(df: pd.DataFrame, coeffs: Tuple[float, float], S: float, step: int = 500):
    """
    For each 500 lb weight step, compute the max L/D and the speed to reach it
    using the 2-coefficient model.
    
    Args:
        df: DataFrame containing density values (rho)
        coeffs: (Cd0, k) from 2-coeff model
        S: Wing reference area [m^2]
        step: Weight increment in pounds (default = 500)
    
    Returns:
        DataFrame with columns [Weight_lb, Max_LD, TAS_kt]
    """
    Cd0, k = coeffs

    # Weight range
    W_min = int(df["weight_lb"].min() // step * step)
    W_max = int(df["weight_lb"].max() // step * step) + step
    weights = np.arange(W_min, W_max + 1, step)

    # Representative density (mean of dataset)
    rho = df["density"].mean()

    results = []
    for W_lb in weights:
        W_kg = W_lb * cst.LB_TO_KG

        # Speed sweep
        V_lin = np.linspace(df["TAS"].min(), df["TAS"].max(), 200)  # knots
        V_ms = V_lin * cst.KT_TO_MS

        # Compute L/D
        L_over_D = ld_2coeff(V_ms, W_kg, rho, S, Cd0, k)

        # Max L/D
        idx_max = np.argmax(L_over_D)
        max_ld = L_over_D[idx_max]
        opt_speed = V_lin[idx_max]

        results.append({"Weight_lb": W_lb, "Max_LD": max_ld, "TAS_kt": opt_speed})

    return pd.DataFrame(results)

def incremental_drag(data: pd.DataFrame, event_list: dict, times: list, ZFW: float, Sref: float, flight: int, smooth: str='200ms',  graphs: bool=False
                   ) -> pd.DataFrame:
    """Post process data for T-38 incremental drag descent.

    Workflow:
        - TO BE COMPLETED
        - Return the processed DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing flight data indexed by timestamp as return by das_read function
        event_list (dict): Dictionary containing time for each event.
        times (list): List specifying the start and end timestamps for each level acceleration segment.
        smooth (string): rolling average to smooth the data (default is '200ms').
        ZFW (float): Zero fuel weight in pounds.
        graphs (bool, optional): If True, generate and display plots for each sawtooth climb segment
        (default is False).

    Returns:
        pd.DataFrame: DataFrame containing processed incremental drag descent data, including calculated energy states,
        fitted curves, and performance metrics.

    """
    event_data=das.to_test_point_full(times,event_list)

    #correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date=data.index[0].date()
    timestamps_date=event_data.Begin[0].date()
    date_offset=timestamps_date-flight_date
    event_data['Begin']=event_data['Begin']-date_offset
    event_data['End']=event_data['End']-date_offset

    #prepare plots
    i = len(event_data)
    cols = 1
    rows = (i + cols - 1) // cols  # This calculates the number of rows needed

    # Create a subplot grid
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=False,  # Optional: you can customize axes sharing
        shared_yaxes=False,  # We will be using individual y-axes for each plot
        vertical_spacing=0.05,
        horizontal_spacing=0.1,
        subplot_titles=[f"Incremental drag descent {plot_index+1}" for plot_index in range(i)],
        # Define a secondary y-axis for each subplot
        specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
    )

    results=[]

    print(f"{'Segment':>7} | {'Roll':>10} | {'CAS':>10} | {'N1':>10} | {'Pitch':>10}")
    print("-"*60)

    for climb_index,row in enumerate(event_data.itertuples()):

        filtered_df=data.loc[row.Begin:row.End]
        # print(f"Processing incremental drag descent {climb_index+1} from {row.Begin.time()} to {row.End.time()} with {len(filtered_df)} samples")
        filtered_df_avg=filtered_df[['BARO_ALT','GPS_ALT','CAS','TAS','SAT', 'RIGHT_N1', 'LEFT_N1','FUEL_TOTAL','ROLL','PITCH','NORM_ACCEL']].resample(smooth).mean() #average the data

        deltaT=(filtered_df_avg.index[-1]-filtered_df_avg.index[0]).total_seconds()
        deltaH=filtered_df_avg['BARO_ALT'].max()-filtered_df_avg['BARO_ALT'].min()
        BARO_ALT=(filtered_df_avg['BARO_ALT'].max()+filtered_df_avg['BARO_ALT'].min())/2
        GPS_ALT=(filtered_df_avg['GPS_ALT'].max()+filtered_df_avg['GPS_ALT'].min())/2
        CAS=filtered_df_avg['CAS'].mean()
        TAS=filtered_df_avg['TAS'].mean()
        SAT=filtered_df_avg['SAT'].mean()
        ROD_corr=deltaH*cst.FT_TO_M/deltaT*(SAT+cst.C_TO_K_OFFSET)/atm.temperature(BARO_ALT*cst.FT_TO_M,units='m') #m/s
        deltaV=filtered_df_avg['TAS'].iloc[-1]-filtered_df_avg['TAS'].iloc[0]
        density=atm.density(BARO_ALT*cst.FT_TO_M,units='m') # Update to take temp into account ?
        gamma=-np.degrees(np.arcsin(ROD_corr/(TAS*cst.KT_TO_MS))) #negative in a descent
        weight_lb=ZFW+filtered_df_avg['FUEL_TOTAL'].mean()
        weight_n=weight_lb*cst.LB_TO_KG*cst.G_METRIC

        deltaDrag=-weight_n*np.sin(np.radians(gamma))
        deltaDC=deltaDrag/(0.5*density*(TAS*cst.KT_TO_MS)**2*Sref)

        #validity checks
        roll_valid=filtered_df_avg['ROLL'].abs().max()<5
        CAS_valid=filtered_df_avg['CAS'].std()<1
        N1_valid=(filtered_df_avg['RIGHT_N1'].std()<1) & (filtered_df_avg['LEFT_N1'].std()<1)
        pitch_valid=filtered_df_avg['PITCH'].std()<2

        # Store results for later assignment
        results.append(
            dict(
                deltaT=deltaT,
                deltaH=deltaH,
                BARO_ALT=BARO_ALT,
                GPS_ALT=GPS_ALT,
                CAS=CAS,
                TAS=TAS,
                deltaV=deltaV,
                SAT=SAT,
                ROD_corr=ROD_corr,
                density=density,
                gamma=gamma,
                weight_lb=weight_lb,
                weight_n=weight_n,
                deltaDrag=deltaDrag,
                deltaDC=deltaDC,
                roll_valid=roll_valid,
                CAS_valid=CAS_valid,
                N1_valid=N1_valid,
                pitch_valid=pitch_valid,
                valid=roll_valid & CAS_valid & N1_valid & pitch_valid
                )
        )
        print(f"{climb_index+1:>7} | {str(roll_valid):>10} | {str(CAS_valid):>10} | {str(N1_valid):>10} | {str(pitch_valid):>10}")

        row = climb_index // cols + 1
        col = climb_index % cols + 1

        # Creating the plots
        trace1 = go.Scatter(x=filtered_df_avg.index, y=filtered_df_avg['BARO_ALT'], mode='lines', name='Pressure Altitude (ft)', line=dict(color='blue'),)

        trace2 = go.Scatter(x=filtered_df_avg.index, y=filtered_df_avg['CAS'], mode='lines', name='Calibrated airspeed (kt)', line=dict(color='red'), )

        # Add traces to the subplot
        fig.add_trace(trace1, row=row, col=col)
        fig.add_trace(trace2, row=row, col=col, secondary_y=True)

        # Update axis labels and titles
        fig.update_yaxes(title_text="Pressure Altitude (ft)", title_font=dict(color="blue"),  tickfont=dict(color="blue"), row=row, col=col)
        fig.update_yaxes(title_text="Calibrated Airspeed (kt)", title_font=dict(color="red"),  range=[CAS-3, CAS+3], tickfont=dict(color="red"), row=row, col=col, secondary_y=True)

    # Assign results in one go
    results_df = pd.DataFrame(results, index=event_data.index)
    event_data.drop(columns="Fuel", inplace=True)
    event_data = pd.concat([event_data, results_df], axis=1)

    if graphs:
        # Update layout for better presentation
        fig.update_layout(
            height=350*rows,  # Adjust height based on number of rows
            width=1500,  # You can adjust the width
            title_text="Incremental drag descents",
            showlegend=False,
        )

        # Show the figure
        fig.show()


    event_data.to_csv("data\\Have dream\\"+flight_date.strftime('%Y-%m-%d')+"_inc_drag_flight_"+str(flight)+".csv")

    return event_data

def turn_3_steps(data: pd.DataFrame, event_list: dict, times: list, ZFW: float, flight: int, smooth: str='200ms',  graphs: bool=False
                   ) -> pd.DataFrame:
    """Post process data for T-38 incremental drag descent.

    Workflow:
        - TO BE COMPLETED
        - Return the processed DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing flight data indexed by timestamp as return by das_read function
        event_list (dict): Dictionary containing time for each event.
        times (list): List specifying the start and end timestamps for each level acceleration segment.
        smooth (string): rolling average to smooth the data (default is '200ms').
        ZFW (float): Zero fuel weight in pounds.
        graphs (bool, optional): If True, generate and display plots for each sawtooth climb segment
        (default is False).

    Returns:
        pd.DataFrame: DataFrame containing processed incremental drag descent data, including calculated energy states,
        fitted curves, and performance metrics.

    """
    event_data=das.to_turn_test_point(times,event_list)

    #correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date=data.index[0].date()
    timestamps_date=event_data.Begin[0].date()
    date_offset=timestamps_date-flight_date
    event_data['Begin']=event_data['Begin']-date_offset
    event_data['End_Turn']=event_data['End_Turn']-date_offset
    event_data['Start_Descent']=event_data['Start_Descent']-date_offset
    event_data['End']=event_data['End']-date_offset


    results=[]

    for turn_index,row in enumerate(event_data.itertuples()):
        df=data.loc[row.Begin:row.End]
        avg_df = df[['BARO_ALT','GPS_ALT','CAS','TAS','SAT', 'RIGHT_N1', 'LEFT_N1','FUEL_TOTAL','ROLL','PITCH','YAW','NORM_ACCEL']].resample(smooth).mean() #average the data
        avg_df['Rel_HDG']=geo.normalize_angle(avg_df['YAW']-avg_df['YAW'].iloc[0])
        turn_df=avg_df.loc[row.Begin:row.End_Turn]
        level_df=avg_df.loc[row.End_Turn:row.Start_Descent]
        descent_df=avg_df.loc[row.Start_Descent:row.End]

        weight_lb=ZFW+turn_df['FUEL_TOTAL'].mean()
        weight_n=weight_lb*cst.LB_TO_KG*cst.G_METRIC

        ## Turn analysis
        turn_deltaH=turn_df['BARO_ALT'].iloc[-1]-turn_df['BARO_ALT'].iloc[0]
        turn_CAS=turn_df['CAS'].mean()
        turn_TAS=turn_df['TAS'].mean()
        turn_SAT=turn_df['SAT'].mean()
        turn_BARO_ALT=turn_df['BARO_ALT'].mean()
        turn_deltaT=(turn_df.index[-1]-turn_df.index[0]).total_seconds()
        turn_ROD_corr=turn_deltaH*cst.FT_TO_M/turn_deltaT*(turn_SAT+cst.C_TO_K_OFFSET)/atm.temperature(turn_BARO_ALT*cst.FT_TO_M,units='m') #m/s
        density=atm.density(turn_BARO_ALT*cst.FT_TO_M,units='m') # Update to take temp into account ?
        turn_gamma=-np.degrees(np.arcsin(turn_ROD_corr/(turn_TAS*cst.KT_TO_MS))) #negative in a descent
        turn_dist = (turn_df['TAS'] * turn_df.index.diff().total_seconds()).sum() #in Nm
        deltaHDG=geo.normalize_angle(turn_df['YAW'].iloc[-1]-turn_df['YAW'].iloc[0])

        ## Level analysis
        if row.End_Turn==row.Start_Descent: #no level segment
            level_deltaT=0
            level_BARO_ALT=0
            level_dist=0
            level_CAS_start=turn_df['CAS'].iloc[-1]
            level_CAS_end=level_CAS_start
        else:
            level_deltaT=(level_df.index[-1]-level_df.index[0]).total_seconds()
            level_BARO_ALT=level_df['BARO_ALT'].mean()
            level_dist = (level_df['TAS'] * level_df.index.diff().total_seconds()).sum() #in Nm
            level_CAS_start=level_df['CAS'].iloc[0]
            level_CAS_end=level_df['CAS'].iloc[-1]

        ##descent analysis
        if row.Start_Descent==row.End: #no descent segment
            descent_deltaT=0
            descent_deltaH=0
            descent_TAS=0
            descent_ROD_corr=0
            descent_gamma=0
            descent_dist=0
            descent_start_alt=turn_df['BARO_ALT'].iloc[-1]
            descent_end_alt=turn_df['BARO_ALT'].iloc[-1]
        else:
            descent_deltaH=descent_df['BARO_ALT'].iloc[-1]-descent_df['BARO_ALT'].iloc[0]
            descent_deltaT=(descent_df.index[-1]-descent_df.index[0]).total_seconds()
            descent_TAS=descent_df['TAS'].mean()
            descent_SAT=descent_df['SAT'].mean()
            descent_BARO_ALT=descent_df['BARO_ALT'].mean()
            descent_ROD_corr=descent_deltaH*cst.FT_TO_M/descent_deltaT*(descent_SAT+cst.C_TO_K_OFFSET)/atm.temperature(descent_BARO_ALT*cst.FT_TO_M,units='m') #m/s
            descent_gamma=-np.degrees(np.arcsin(descent_ROD_corr/(descent_TAS*cst.KT_TO_MS))) #negative in a descent
            descent_dist = (descent_df['TAS'] * descent_df.index.diff().total_seconds()).sum() #in Nm
            descent_start_alt=descent_df['BARO_ALT'].iloc[0],
            descent_end_alt=descent_df['BARO_ALT'].iloc[-1],

        # Store results for later assignment
        results.append(
            dict(
                weight_lb=weight_lb,
                weight_n=weight_n,
                turn_duration=turn_deltaT,
                deltaH=turn_deltaH,
                deltaHDG=deltaHDG,
                Start_alt=turn_df['BARO_ALT'].iloc[0],
                End_alt=turn_df['BARO_ALT'].iloc[-1],
                turn_roll=turn_df['ROLL'].abs().mean(),
                turn_CAS=turn_CAS,
                turn_TAS=turn_TAS,
                turn_ROD_corr=turn_ROD_corr,
                turn_gamma=turn_gamma,
                turn_dist=turn_dist,
                level_duration=level_deltaT,
                Start_CAS=level_CAS_start,
                End_CAS=level_CAS_end,
                level_BARO_ALT=level_BARO_ALT,
                level_dist=level_dist,
                descent_duration=descent_deltaT,
                descent_deltaH=descent_deltaH,
                descent_start_alt=descent_start_alt,
                descent_end_alt=descent_end_alt,
                descent_TAS=descent_TAS,
                descent_ROD_corr=descent_ROD_corr,
                descent_gamma=descent_gamma,
                descent_dist=descent_dist,
                )
        )

        if graphs:
                    layout = dict(
                        hoversubplots="axis",
                        title=row.Label,
                        hovermode="x unified",
                        grid=dict(rows=5, columns=1),
                        xaxis=dict(showgrid=True,
                                hoverformat='<b>%H:%M:%S.%f</b>'),
                        yaxis=dict(showgrid=True, title='Altitude (ft)'),
                        yaxis2=dict(showgrid=True, title='CAS (kt)'),
                        yaxis3=dict(showgrid=True, title='Roll (deg)'),
                        yaxis4=dict(showgrid=True, title='Relative heading (deg)'),
                        yaxis5=dict(showgrid=True, title='Pitch (deg)'),
                        width=1300,
                        height=800,
                        )

                    graph = [
                        go.Scatter(
                            x=avg_df.index, y=avg_df["BARO_ALT"], xaxis="x", yaxis="y",
                            name="Altitude",
                            hovertemplate='Altitude: %{y:.0f} ft<extra></extra>'
                        ),
                        go.Scatter(
                            x=avg_df.index, y=avg_df["CAS"], xaxis="x", yaxis="y2",
                            name="CAS",
                            hovertemplate='CAS: %{y:.1f} KCAS<extra></extra>'
                        ),
                        go.Scatter(
                            x=avg_df.index, y=avg_df["ROLL"], xaxis="x", yaxis="y3",
                            name="Roll",
                            hovertemplate='Roll: %{y:.1f} deg<extra></extra>'
                        ),
                        go.Scatter(
                            x=avg_df.index, y=avg_df["Rel_HDG"], xaxis="x", yaxis="y4",
                            name="Relative heading",
                            hovertemplate='Relative heading: %{y:.1f} deg<extra></extra>'
                        ),
                        go.Scatter(
                            x=avg_df.index, y=avg_df["PITCH"], xaxis="x", yaxis="y5",
                            name="Pitch",
                            hovertemplate='Pitch: %{y:.1f} deg<extra></extra>'
                        ),
                    ]

                    fig= go.Figure(data=graph, layout=layout)

                    # --- End of turn line + label ---
                    end_turn_time = pd.to_datetime(row.End_Turn)

                    fig.add_shape(
                        type="line",
                        x0=end_turn_time,
                        x1=end_turn_time,
                        y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(color="black", width=2)
                    )

                    fig.add_annotation(
                        x=end_turn_time,
                        y=0.5,                   # middle of the figure
                        xref="x",
                        yref="paper",
                        text="End of turn",
                        showarrow=False,
                        textangle=-90,           # vertical text
                        xanchor="right",         # place to the left of the line
                        font=dict(color="black", size=12)
                    )

                    # --- Start descent line + label ---
                    start_descent_time = pd.to_datetime(row.Start_Descent)

                    fig.add_shape(
                        type="line",
                        x0=start_descent_time,
                        x1=start_descent_time,
                        y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(color="black", width=2)
                    )

                    fig.add_annotation(
                        x=start_descent_time,
                        y=0.5,
                        xref="x",
                        yref="paper",
                        text="Start descent",
                        showarrow=False,
                        textangle=-90,            # vertical text, opposite direction
                        xanchor="left",          # place to the right of the line
                        font=dict(color="black", size=12)
                    )

                    fig.show()

    # Assign results in one go
    results_df = pd.DataFrame(results, index=event_data.index)
    event_data = pd.concat([event_data, results_df], axis=1)

    


    event_data.to_csv("data\\Have dream\\"+flight_date.strftime('%Y-%m-%d')+"_turns_"+str(flight)+".csv")

    return event_data