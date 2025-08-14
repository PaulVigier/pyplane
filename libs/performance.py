"""Post-processing functions for C-12 aircraft performance data."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from libs import constants as cst
from libs import das

# savgol filter parameters
window_length = 101
polyorder = 2

def level_accel(data: pd.DataFrame, times: np.ndarray, smooth: str='1s', anchor: float=0, YAPS: bool=True,
                graphs: bool=False)-> pd.DataFrame:
    """Post process data for C-12 level accelerations.

    Workflow:
        - Correct for the pitot static error (with or without YAPS boom).
        - Calculate the specific energy height (Es). Fit a polynomial curve.
        - Derive the polynomial fit to get the specific energy height (Ps).
        - Optionally plot the results to check data quality.
        - Export the results in a CSV file (for report plots).
        - Return the processed DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing flight data indexed by timestamp as return by das_read function
        times (np.ndarray): Array specifying the start and end timestamps for each level acceleration segment.
        smooth (str, optional): Resampling interval or smoothing the flight data. Default is 1s.
        anchor (float, optional): Anchor value at the end of level acceleration (Ps=0).
        YAPS (bool, optional): If True, corrected pitot static data using YAPS coefficients (default is True).
        graphs (bool, optional): If True, generate and display plots for each level acceleration segment
        (default is False).

    Returns:
        pd.DataFrame: DataFrame containing processed level acceleration data, including calculated energy states,
        fitted curves, and performance metrics.

    """
    timestamps=das.to_test_point_lite(times)

    #correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date=data.index[0].date()
    timestamps_date=timestamps.Begin[0].date()
    date_offset=timestamps_date-flight_date
    timestamps['Begin']=timestamps['Begin']-date_offset
    timestamps['End']=timestamps['End']-date_offset

    #create series for the results
    accel_list=[] #list of all points within a level accel

    #prepare plots
    nb_subplots = len(timestamps)

    # Create a subplot grid
    fig = make_subplots(
        rows=nb_subplots,
        cols=1,
        shared_xaxes=False,  # Optional: you can customize axes sharing
        shared_yaxes=False,  # We will be using individual y-axes for each plot
        vertical_spacing=0.15,
        subplot_titles=[f"Level accel {plot_index+1}" for plot_index in range(nb_subplots)],
        # Define a secondary y-axis for each subplot
        specs=[[{"secondary_y": True} for _ in range(1)] for _ in range(nb_subplots)]
    )


    for accel_index,row in enumerate(timestamps.itertuples()):

        filtered_df=data.loc[row.Begin:row.End].copy()

        filtered_df_avg = filtered_df[['BARO_ALT','GPS_ALT','CAS','TAS','SAT']].resample(smooth).first() #average the data

        filtered_df_avg[['Hpc', 'Vpc', 'Mpc']] = filtered_df_avg.apply(lambda row: pd.Series(das.C12_pitot_static(row['CAS'], row['BARO_ALT'],YAPS=YAPS)), axis=1)


        for i in range(len(filtered_df_avg)-1):
            Time_ISO=filtered_df_avg.index[i]
            Time_rel=(Time_ISO-row.Begin).total_seconds()       #Time since the beginning of the accel
            BARO_ALT=filtered_df_avg['BARO_ALT'].iloc[i]
            Hpc=filtered_df_avg['Hpc'].iloc[i]
            GPS_ALT=filtered_df_avg['GPS_ALT'].iloc[i]
            CAS=filtered_df_avg['CAS'].iloc[i]
            Vpc=filtered_df_avg['Vpc'].iloc[i]
            TAS=filtered_df_avg['TAS'].iloc[i]
            Mpc=filtered_df_avg['Mpc'].iloc[i]
            SAT=filtered_df_avg['SAT'].iloc[i]
            Vtpc=Mpc*np.sqrt(cst.GAMMA*cst.R_SI*(filtered_df_avg['SAT'].iloc[0]+cst.C_TO_K_OFFSET))*cst.MS_TO_KT  #true airspeed at pressure altitude
            deltaT=(filtered_df_avg.index[i+1]-filtered_df_avg.index[i]).total_seconds()
            deltaH=filtered_df_avg['BARO_ALT'].iloc[i+1]-filtered_df_avg['BARO_ALT'].iloc[i]
            deltaV=filtered_df_avg['TAS'].iloc[i+1]-filtered_df_avg['TAS'].iloc[i]
            Es=BARO_ALT+(TAS*cst.KT_TO_FPS)**2/(2*cst.G_IMPERIAL)
            #Ps=deltaH/deltaT+(TAS*cst.KT_TO_FPS)*(deltaV*cst.KT_TO_FPS/deltaT)/(cst.G_IMPERIAL*deltaT)
            accel_list.append([Time_ISO,Time_rel,accel_index,Hpc,BARO_ALT,GPS_ALT,SAT,CAS,Vpc,TAS,Mpc,Vtpc,deltaT,deltaH,deltaV,Es])



    accel_data=pd.DataFrame(accel_list, columns=['Time_ISO','Time_rel','ID','Hpc','BARO_ALT','GPS_ALT','SAT','CAS','Vpc','TAS','Mpc','Vtpc','deltaT','deltaH','deltaV','Es'])
    accel_data.set_index('Time_ISO', inplace=True)
    #accel_data['Ps_smooth'] = accel_data.groupby('ID')['Ps'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

    accel_data['Ps'] = 0.0  # Initialize Ps column
    accel_data['Es_fit'] = 0.0  # Initialize Es_fit column

    for _, group in accel_data.groupby('ID'):

        LA_Es=group['Es'].to_numpy()
        LA_time= group['Time_rel'].to_numpy()

        coeffs = np.polyfit(LA_time, LA_Es, 4)

        # Create a polynomial function from the coefficients
        poly_eq = np.poly1d(coeffs)

        group = group.copy()  # avoid SettingWithCopyWarning
        # Generate fitted values
        group['Es_fit'] = group.apply(lambda row, poly_eq=poly_eq: pd.Series(poly_eq(row['Time_rel'])), axis=1)

        #Generate Ps values differentiating the Es_fit values
        group['Ps']=group['Es_fit'].diff() / group.index.to_series().diff().dt.total_seconds()
        group.iloc[0, group.columns.get_loc('Ps')] = group['Ps'].iloc[1]  # fix first value

        # Create a form fitting curve for Ps
        LA_Ps=group['Ps'].to_numpy()
        LA_TAS= group['TAS'].to_numpy()
        coeffs = np.polyfit(LA_TAS, LA_Ps, 4)

        # Create a polynomial function from the coefficients
        poly_eq = np.poly1d(coeffs)
        group['Ps_fit']=poly_eq(group['TAS'])

        # Assign back to accel_data
        accel_data.loc[group.index, ['Es_fit', 'Ps','Ps_fit']] = group[['Es_fit', 'Ps','Ps_fit']]

    if graphs:
        # Update layout for better presentation
        fig.update_layout(
            height=350*nb_subplots,  # Adjust height based on number of rows
            width=1200,  # You can adjust the width
            title_text="Level accels",
            showlegend=False,
        )

    for id, group in accel_data.groupby('ID'):
        fig.add_trace(go.Scatter(x=group.index, y=group['Es'],yaxis='y', name=f'Level accel {id}', mode='markers'),row=id+1,col=1)
        fig.add_trace(go.Scatter(x=group.index, y=group['Es_fit'],yaxis='y', name=f'Level accel {id}', line=dict(color='red')),row=id+1,col=1)

        # Update axis labels and titles
        fig.update_yaxes(title_text="Pressure Altitude (ft)", title_font=dict(color="blue"),  tickfont=dict(color="blue"), row=id+1, col=1)
        fig.update_yaxes(title_text="Calibrated Airspeed (kt)", title_font=dict(color="red"),  tickfont=dict(color="red"),row=id+1, col=1, secondary_y=True)

        # Show the figure
    fig.show()

    accel_data.to_csv("data\\Perf\\"+flight_date.strftime('%Y-%m-%d')+"_level_accel.csv")

    return accel_data


def sawtooth_climb(data: pd.DataFrame, times: np.ndarray, smooth: str='1s', YAPS: bool=True, graphs: bool=False
                   ) -> pd.DataFrame:
    """Post process data for C-12 sawtooth climbs.

    Workflow:
        - Correct for the pitot static error (with or without YAPS boom).
        - Average data during each descent
        - Calculate the specific energy height (Es) and specific energy height (Ps).
        - Optionally plot the results to check data quality.
        - Export the results in a CSV file (for report plots).
        - Return the processed DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing flight data indexed by timestamp as return by das_read function
        times (np.ndarray): Array specifying the start and end timestamps for each level acceleration segment.
        smooth (str, optional): Resampling interval or smoothing the flight data. Default is 1s.
        YAPS (bool, optional): If True, corrected pitot static data using YAPS coefficients (default is True).
        graphs (bool, optional): If True, generate and display plots for each sawtooth climb segment
        (default is False).

    Returns:
        pd.DataFrame: DataFrame containing processed sawtooth climb data, including calculated energy states,
        fitted curves, and performance metrics.

    """
    sawtooth_data=das.to_test_point_lite(times)

    #correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date=data.index[0].date()
    timestamps_date=sawtooth_data.Begin[0].date()
    date_offset=timestamps_date-flight_date
    sawtooth_data['Begin']=sawtooth_data['Begin']-date_offset
    sawtooth_data['End']=sawtooth_data['End']-date_offset

    #create series for the results
    sawtooth_data['deltaT']=0.0
    sawtooth_data['deltaH']=0.0
    sawtooth_data['BARO_ALT']=0.0
    sawtooth_data['Hpc']=0.0
    sawtooth_data['GPS_ALT']=0.0
    sawtooth_data['CAS']=0.0
    sawtooth_data['Vpc']=0.0
    sawtooth_data['TAS']=0.0
    sawtooth_data['Mpc']=0.0
    sawtooth_data['Vtpc']=0.0
    sawtooth_data['Es']=0.0
    sawtooth_data['Ps']=0.0


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
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
        subplot_titles=[f"Sawtooth climb {plot_index+1}" for plot_index in range(i)],
        # Define a secondary y-axis for each subplot
        specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
    )

    #loop for pairs of timestamps
    for climb_index,row in enumerate(sawtooth_data.itertuples()):

        filtered_df=data.loc[row.Begin:row.End]

        filtered_df_avg=filtered_df[['BARO_ALT','GPS_ALT','CAS','TAS','SAT']].resample(smooth).mean() #average the data
        filtered_df_avg[['Hpc', 'Vpc', 'Mpc']] = filtered_df_avg.apply(lambda row: pd.Series(das.C12_pitot_static(row['CAS'], row['BARO_ALT'],YAPS=YAPS)), axis=1)


        deltaT=(filtered_df_avg.index[-1]-filtered_df_avg.index[0]).total_seconds()

        deltaH=filtered_df_avg['BARO_ALT'].max()-filtered_df_avg['BARO_ALT'].min()
        BARO_ALT=(filtered_df_avg['BARO_ALT'].max()+filtered_df_avg['BARO_ALT'].min())/2
        Hpc=(filtered_df_avg['Hpc'].max()-filtered_df_avg['Hpc'].min())/2   #difference in pressure altitude
        GPS_ALT=(filtered_df_avg['GPS_ALT'].max()+filtered_df_avg['GPS_ALT'].min())/2
        CAS=filtered_df_avg['CAS'].mean()
        Vpc=filtered_df_avg['Vpc'].mean()
        TAS=filtered_df_avg['TAS'].mean()
        Mpc=filtered_df_avg['Mpc'].mean()
        Vtpc=Mpc*np.sqrt(cst.GAMMA*cst.R_SI*(filtered_df_avg['SAT'].iloc[0]+cst.C_TO_K_OFFSET))*cst.MS_TO_KT  #true airspeed at pressure altitude
        deltaV=filtered_df_avg['TAS'].iloc[-1]-filtered_df_avg['TAS'].iloc[0]
        Es=deltaH+(TAS*cst.KT_TO_FPS)**2/(2*cst.G_IMPERIAL)
        Ps=deltaH/deltaT+(TAS*cst.KT_TO_FPS)*(deltaV*cst.KT_TO_FPS)/(cst.G_IMPERIAL*deltaT)

        #assign values to dataframe
        sawtooth_data.loc[climb_index,'deltaT']=deltaT
        sawtooth_data.loc[climb_index,'deltaH']=deltaH
        sawtooth_data.loc[climb_index,'BARO_ALT']=BARO_ALT
        sawtooth_data.loc[climb_index,'Hpc']=Hpc
        sawtooth_data.loc[climb_index,'GPS_ALT']=GPS_ALT
        sawtooth_data.loc[climb_index,'CAS']=CAS
        sawtooth_data.loc[climb_index,'Vpc']=Vpc
        sawtooth_data.loc[climb_index,'TAS']=TAS
        sawtooth_data.loc[climb_index,'Mpc']=Mpc
        sawtooth_data.loc[climb_index,'Vtpc']=Vtpc
        sawtooth_data.loc[climb_index,'Es']=Es
        sawtooth_data.loc[climb_index,'Ps']=Ps

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
        fig.update_yaxes(title_text="Calibrated Airspeed (kt)", title_font=dict(color="red"),  tickfont=dict(color="red"),row=row, col=col, secondary_y=True)

    if graphs:
        # Update layout for better presentation
        fig.update_layout(
            height=350*rows,  # Adjust height based on number of rows
            width=1200,  # You can adjust the width
            title_text="Sawtooth climbs",
            showlegend=False,
        )

        # Show the figure
        fig.show()


    sawtooth_data.to_csv("data\\Perf\\"+flight_date.strftime('%Y-%m-%d')+"_sawtooth.csv")

    return sawtooth_data


def cruise_climb(data: pd.DataFrame, times: pd.DataFrame, smooth: str, YAPS: bool = True, graphs: bool = False
                 ) -> pd.DataFrame:
    """Post process data for C-12 climbs.

    Workflow:
        - Correct for the pitot static error (with or without YAPS boom).
        - Get fuel consumption from integrated fuel flow data.
        - Get distance traveled integrating TAS.
        - Optionally plot the results to check data quality.
        - Export the results in a CSV file (for report plots).
        - Return the processed DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing flight data indexed by timestamp as return by das_read function
        times (np.ndarray): Array specifying the start and end timestamps for each climb segment.
        smooth (str, optional): Resampling interval or smoothing the flight data. Default is 1s.
        YAPS (bool, optional): If True, corrected pitot static data using YAPS coefficients (default is True).
        graphs (bool, optional): If True, generate and display plots for each climb segment
        (default is False).

    Returns:
        pd.DataFrame: DataFrame containing processed climb data, including calculated energy states,
        fitted curves, and performance metrics.

    """
    timestamps = das.to_test_point_lite(times)

    #correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date=data.index[0].date()
    timestamps_date=timestamps.Begin[0].date()
    date_offset=timestamps_date-flight_date
    timestamps['Begin']=timestamps['Begin']-date_offset
    timestamps['End']=timestamps['End']-date_offset

    #create series for the results
    climb_list=[] #list of all points within a level accel

    #prepare plots
    nb_subplots = len(timestamps)


    # Create a subplot grid
    fig = make_subplots(
        rows=nb_subplots,
        cols=1,
        shared_xaxes=False,  # Optional: you can customize axes sharing
        shared_yaxes=False,  # We will be using individual y-axes for each plot
        vertical_spacing=0.15,
        subplot_titles=[f"Cruise climb {plot_index+1}" for plot_index in range(nb_subplots)],
        # Define a secondary y-axis for each subplot
        specs=[[{"secondary_y": True} for _ in range(1)] for _ in range(nb_subplots)]
    )

    #loop for pairs of timestamps
    for climb_index,row in enumerate(timestamps.itertuples()):

        filtered_df=data.loc[row.Begin:row.End]
        filtered_df_avg=filtered_df[['BARO_ALT','GPS_ALT','CAS','TAS','LEFT_FUEL_FLOW',
                                     'RIGHT_FUEL_FLOW','SAT']].resample(smooth).mean()
        filtered_df_avg[['Hpc', 'Vpc', 'Mpc']] = filtered_df_avg.apply(lambda row:
                                        pd.Series(das.C12_pitot_static(row['CAS'], row['BARO_ALT'],YAPS=YAPS)), axis=1)

        for i in range(len(filtered_df_avg)-1):
            Time_ISO=filtered_df_avg.index[i]
            Time_rel=(Time_ISO-row.Begin).total_seconds()       #Time since the beginning of the accel
            BARO_ALT=filtered_df_avg['BARO_ALT'].iloc[i]
            Hpc=filtered_df_avg['Hpc'].iloc[i]
            GPS_ALT=filtered_df_avg['GPS_ALT'].iloc[i]
            CAS=filtered_df_avg['CAS'].iloc[i]
            Vpc=filtered_df_avg['Vpc'].iloc[i]
            TAS=filtered_df_avg['TAS'].iloc[i]
            Mpc=filtered_df_avg['Mpc'].iloc[i]
            Vtpc=Mpc*np.sqrt(cst.GAMMA*cst.R_SI*(filtered_df_avg['SAT'].iloc[0]+cst.C_TO_K_OFFSET))*cst.MS_TO_KT
            deltaT=(filtered_df_avg.index[i+1]-filtered_df_avg.index[i]).total_seconds()
            FF=(filtered_df_avg['LEFT_FUEL_FLOW'].iloc[i]+filtered_df_avg['RIGHT_FUEL_FLOW'].iloc[i])/3600
            Fuel = climb_list[-1][13] + FF*deltaT if i > 0 else FF*deltaT   #fuel burned
            Distance=climb_list[-1][14] + TAS/3600*deltaT if i > 0 else TAS/3600*deltaT  #distance traveled

            climb_list.append([Time_ISO,Time_rel,climb_index,BARO_ALT,Hpc,GPS_ALT,CAS,Vpc,TAS,Mpc,Vtpc,deltaT,FF,Fuel,Distance])

    climb_data=pd.DataFrame(climb_list, columns=['Time_ISO','Time_rel','ID','BARO_ALT','Hpc','GPS_ALT','CAS','Vpc',
                                                 'TAS','Mpc','Vtpc','deltaT','FF','Fuel','Distance'])
    climb_data.set_index('Time_ISO', inplace=True)

    if graphs:
        # Update layout for better presentation
        fig.update_layout(
            height=350*nb_subplots,  # Adjust height based on number of rows
            width=1200,  # You can adjust the width
            title_text="Cruise climb",
            showlegend=False,
        )

    for id, group in climb_data.groupby('ID'):
        fig.add_trace(
            go.Scatter(
            x=group.index,
            y=group['BARO_ALT'],
            yaxis='y',
            name=f'Alt Level accel {id}',
            line=dict(color='blue')
            ),
            row=id+1, col=1
        )
        fig.add_trace(
            go.Scatter(
            x=group.index,
            y=group['CAS'],
            yaxis='y2',
            name=f'TAS Level accel {id}',
            line=dict(color='red')
            ),
            row=id+1, col=1, secondary_y=True
        )

        fig.update_yaxes(
            title_text="Pressure Altitude(ft)",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue"),
            row=id+1, col=1
        )
        fig.update_yaxes(
            title_text="Calibrated Airspeed (kt)",
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            row=id+1, col=1, secondary_y=True
        )

    fig.show()

    climb_data.to_csv("data\\Perf\\"+flight_date.strftime('%Y-%m-%d')+"_cruise_climb.csv")

    return climb_data


def FM_descent(data: pd.DataFrame, times: pd.DataFrame, smooth: str, YAPS: bool = True, graphs: bool = False
               ) -> pd.DataFrame:
    """Post process data for C-12 descent.

    Workflow:
        - Correct for the pitot static error (with or without YAPS boom).
        - Get fuel consumption from integrated fuel flow data.
        - Get distance traveled integrating TAS.
        - Optionally plot the results to check data quality.
        - Export the results in a CSV file (for report plots).
        - Return the processed DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing flight data indexed by timestamp as return by das_read function
        times (np.ndarray): Array specifying the start and end timestamps for each climb segment.
        smooth (str, optional): Resampling interval or smoothing the flight data. Default is 1s.
        YAPS (bool, optional): If True, corrected pitot static data using YAPS coefficients (default is True).
        graphs (bool, optional): If True, generate and display plots for each descent segment
        (default is False).

    Returns:
        pd.DataFrame: DataFrame containing processed descent data, including calculated energy states,
        fitted curves, and performance metrics.

    """
    timestamps = das.to_test_point_lite(times)
    #correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date=data.index[0].date()
    timestamps_date=timestamps.Begin[0].date()
    date_offset=timestamps_date-flight_date
    timestamps['Begin']=timestamps['Begin']-date_offset
    timestamps['End']=timestamps['End']-date_offset

    #create series for the results
    descent_list=[] #list of all points within a level accel

    #prepare plots
    nb_subplots = len(timestamps)


    # Create a subplot grid
    fig = make_subplots(
        rows=nb_subplots,
        cols=1,
        shared_xaxes=False,  # Optional: you can customize axes sharing
        shared_yaxes=False,  # We will be using individual y-axes for each plot
        vertical_spacing=0.15,
        subplot_titles=[f"Flight manual descent {plot_index+1}" for plot_index in range(nb_subplots)],
        # Define a secondary y-axis for each subplot
        specs=[[{"secondary_y": True} for _ in range(1)] for _ in range(nb_subplots)]
    )

    #loop for pairs of timestamps
    for descent_index,row in enumerate(timestamps.itertuples()):

        filtered_df=data.loc[row.Begin:row.End]
        filtered_df_avg=filtered_df[['BARO_ALT','GPS_ALT','CAS','TAS','LEFT_FUEL_FLOW','RIGHT_FUEL_FLOW',
                                     'SAT']].resample(smooth).mean()
        filtered_df_avg[['Hpc', 'Vpc', 'Mpc']] = filtered_df_avg.apply(lambda row:
                                        pd.Series(das.C12_pitot_static(row['CAS'], row['BARO_ALT'],YAPS=YAPS)), axis=1)

        for i in range(len(filtered_df_avg)-1):
            Time_ISO=filtered_df_avg.index[i]
            Time_rel=(Time_ISO-row.Begin).total_seconds()       #Time since the beginning of the accel
            BARO_ALT=filtered_df_avg['BARO_ALT'].iloc[i]
            Hpc=filtered_df_avg['Hpc'].iloc[i]
            GPS_ALT=filtered_df_avg['GPS_ALT'].iloc[i]
            CAS=filtered_df_avg['CAS'].iloc[i]
            Vpc=filtered_df_avg['Vpc'].iloc[i]
            TAS=filtered_df_avg['TAS'].iloc[i]
            Mpc=filtered_df_avg['Mpc'].iloc[i]
            Vtpc=Mpc*np.sqrt(cst.GAMMA*cst.R_SI*(filtered_df_avg['SAT'].iloc[0]+cst.C_TO_K_OFFSET))*cst.MS_TO_KT
            deltaT=(filtered_df_avg.index[i+1]-filtered_df_avg.index[i]).total_seconds()
            FF=(filtered_df_avg['LEFT_FUEL_FLOW'].iloc[i]+filtered_df_avg['RIGHT_FUEL_FLOW'].iloc[i])/3600
            Fuel = descent_list[-1][13] + FF*deltaT if i > 0 else FF*deltaT   #fuel burned
            Distance=descent_list[-1][14] + TAS/3600*deltaT if i > 0 else TAS/3600*deltaT  #distance traveled

            descent_list.append([Time_ISO,Time_rel,descent_index,BARO_ALT,Hpc,GPS_ALT,CAS,Vpc,
                                 TAS,Mpc,Vtpc,deltaT,FF,Fuel,Distance])

    descent_data=pd.DataFrame(descent_list, columns=['Time_ISO','Time_rel','ID','BARO_ALT','Hpc','GPS_ALT','CAS','Vpc',
                                                     'TAS','Mpc','Vtpc','deltaT','FF','Fuel','Distance'])
    descent_data.set_index('Time_ISO', inplace=True)

    if graphs:
        # Update layout for better presentation
        fig.update_layout(
            height=350*nb_subplots,  # Adjust height based on number of rows
            width=1200,  # You can adjust the width
            title_text="Flight manual descent",
            showlegend=False,
        )

    for id, group in descent_data.groupby('ID'):
        fig.add_trace(
            go.Scatter(
            x=group.index, y=group['BARO_ALT'],
            yaxis='y', name='Altitude',
            line=dict(color='blue')
            ),
            row=id+1, col=1
        )
        fig.add_trace(
            go.Scatter(
            x=group.index, y=group['CAS'],
            yaxis='y2', name='TAS',
            line=dict(color='red')
            ),
            row=id+1, col=1, secondary_y=True
        )

        fig.update_yaxes(
            title_text="Pressure Altitude (ft)",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue"),
            row=id+1, col=1
        )
        fig.update_yaxes(
            title_text="Calibrated Airspeed (kt)",
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            row=id+1, col=1, secondary_y=True
        )

        # Show the figure
    fig.show()

    descent_data.to_csv("data\\Perf\\"+flight_date.strftime('%Y-%m-%d')+"_FM_descent.csv")

    return descent_data


def Emer_descent(data: pd.DataFrame, times: pd.DataFrame, smooth: str, YAPS: bool = True,
                 graphs: bool = False) -> pd.DataFrame:
    """Post process data for C-12 emergency descent. Identical to FM_descent except for titles and export files name."""
    timestamps=das.to_test_point_lite(times)

    #correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date=data.index[0].date()
    timestamps_date=timestamps.Begin[0].date()
    date_offset=timestamps_date-flight_date
    timestamps['Begin']=timestamps['Begin']-date_offset
    timestamps['End']=timestamps['End']-date_offset

    #create series for the results
    descent_list=[]

    nb_subplots = len(timestamps)
    # Create a subplot grid
    fig = make_subplots(
        rows=nb_subplots,
        cols=1,
        shared_yaxes=False,
        vertical_spacing=0.15,
        subplot_titles=[f"Emergency descent {plot_index+1}" for plot_index in range(nb_subplots)],
        # Define a secondary y-axis for each subplot
        specs=[[{"secondary_y": True} for _ in range(1)] for _ in range(nb_subplots)]
    )

    #loop for pairs of timestamps
    for descent_index,row in enumerate(timestamps.itertuples()):

        filtered_df=data.loc[row.Begin:row.End]
        filtered_df_avg=filtered_df[['BARO_ALT','GPS_ALT','CAS','TAS','LEFT_FUEL_FLOW','RIGHT_FUEL_FLOW','SAT',
                                     'VS']].resample(smooth).mean() #average the data
        filtered_df_avg[['Hpc', 'Vpc', 'Mpc']] = filtered_df_avg.apply(lambda row:
                                        pd.Series(das.C12_pitot_static(row['CAS'], row['BARO_ALT'],YAPS=YAPS)), axis=1)

        for i in range(len(filtered_df_avg)-1):
            Time_ISO=filtered_df_avg.index[i]
            Time_rel=(Time_ISO-row.Begin).total_seconds()       #Time since the beginning of the accel
            BARO_ALT=filtered_df_avg['BARO_ALT'].iloc[i]
            Hpc=filtered_df_avg['Hpc'].iloc[i]
            GPS_ALT=filtered_df_avg['GPS_ALT'].iloc[i]
            CAS=filtered_df_avg['CAS'].iloc[i]
            Vpc=filtered_df_avg['Vpc'].iloc[i]
            TAS=filtered_df_avg['TAS'].iloc[i]
            Mpc=filtered_df_avg['Mpc'].iloc[i]
            VS=filtered_df_avg['VS'].iloc[i]
            Vtpc=Mpc*np.sqrt(cst.GAMMA*cst.R_SI*(filtered_df_avg['SAT'].iloc[0]+cst.C_TO_K_OFFSET))*cst.MS_TO_KT
            deltaT=(filtered_df_avg.index[i+1]-filtered_df_avg.index[i]).total_seconds()
            FF=(filtered_df_avg['LEFT_FUEL_FLOW'].iloc[i]+filtered_df_avg['RIGHT_FUEL_FLOW'].iloc[i])/3600
            Fuel = descent_list[-1][14] + FF*deltaT if i > 0 else FF*deltaT  # fuel burned
            Distance=descent_list[-1][15] + TAS/3600*deltaT if i > 0 else TAS/3600*deltaT  # distance traveled

            descent_list.append([Time_ISO,Time_rel,descent_index,BARO_ALT,Hpc,GPS_ALT,CAS,Vpc,
                                 TAS,Mpc,Vtpc,VS,deltaT,FF,Fuel,Distance])

    descent_data=pd.DataFrame(descent_list, columns=['Time_ISO','Time_rel','ID','BARO_ALT','Hpc','GPS_ALT','CAS','Vpc',
                                                     'TAS','Mpc','Vtpc','VS','deltaT','FF','Fuel','Distance'])
    descent_data.set_index('Time_ISO', inplace=True)

    if graphs:
        # Update layout for better presentation
        fig.update_layout(
            height=350*nb_subplots,  # Adjust height based on number of rows
            width=1200,  # You can adjust the width
            title_text="Flight manual descent",
            showlegend=False,
        )

    for id, group in descent_data.groupby('ID'):
        fig.add_trace(
            go.Scatter(x=group.index, y=group['BARO_ALT'], name='Altitude', line=dict(color='blue')),
            row=id+1, col=1
        )
        fig.add_trace(
            go.Scatter(x=group.index, y=group['CAS'], name='TAS', line=dict(color='red')),
            row=id+1, col=1, secondary_y=True
        )
        fig.update_yaxes(
            title_text="Pressure Altitude (ft)",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue"),
            row=id+1, col=1
        )
        fig.update_yaxes(
            title_text="Calibrated Airspeed (kt)",
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            row=id+1, col=1, secondary_y=True
        )

    fig.show()

    descent_data.to_csv("data\\Perf\\"+flight_date.strftime('%Y-%m-%d')+"_emer_descent.csv")

    return descent_data


def level_turn(data: pd.DataFrame, times: pd.DataFrame, smooth: int, YAPS: bool = True,
               graphs: bool = False) -> pd.DataFrame:
    """Post process data for C-12 level turns.

    Workflow:
        - Correct for the pitot static error (with or without YAPS boom).
        - Calculate the turn radius.
        - Optionally plot the results to check data quality.
        - Export the results in a CSV file (for report plots).
        - Return the processed DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing flight data indexed by timestamp as return by das_read function
        times (np.ndarray): Array specifying the start and end timestamps for each climb segment.
        smooth (str, optional): Resampling interval or smoothing the flight data. Default is 1s.
        YAPS (bool, optional): If True, corrected pitot static data using YAPS coefficients (default is True).
        graphs (bool, optional): If True, generate and display plots for each descent segment
        (default is False).

    Returns:
        pd.DataFrame: DataFrame containing processed descent data, including calculated energy states,
        fitted curves, and performance metrics.

    """
    level_turn_data = das.to_test_point_lite(times)

    #correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date=data.index[0].date()
    timestamps_date=level_turn_data.Begin[0].date()
    date_offset=timestamps_date-flight_date
    level_turn_data['Begin']=level_turn_data['Begin']-date_offset
    level_turn_data['End']=level_turn_data['End']-date_offset


   #prepare plots
    i = len(level_turn_data)
    cols = 3
    rows = i+1 # This calculates the number of rows needed

    # Create a subplot grid
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=False,  # Optional: you can customize axes sharing
        shared_yaxes=False,  # We will be using individual y-axes for each plot
        vertical_spacing=0.1,
        horizontal_spacing=0.15,
        subplot_titles=[f"Level turn {(plot_index+3)//3}" if plot_index % 3 == 0 else "" for plot_index in range(3*i)],
        # Define a secondary y-axis for each subplot
        specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
    )

    #loop for pairs of timestamps
    for turn_index,row in enumerate(level_turn_data.itertuples()):

        filtered_df=data.loc[row.Begin:row.End]

        filtered_df_avg=filtered_df[['BARO_ALT','GPS_ALT','CAS','TAS','ROLL','MAG_HDG','SAT']].resample(smooth).mean()
        filtered_df_avg[['Hpc', 'Vpc', 'Mpc']] = filtered_df_avg.apply(lambda row:
                                        pd.Series(das.C12_pitot_static(row['CAS'], row['BARO_ALT'],YAPS=YAPS)), axis=1)
        filtered_df_avg['Nz']=1/np.cos(np.deg2rad(filtered_df_avg['ROLL']))

        deltaT=(filtered_df_avg.index[-1]-filtered_df_avg.index[0]).total_seconds()
        deltaHDG=np.abs(das.deltaHDG(filtered_df_avg.index,filtered_df_avg['MAG_HDG']))  #in deg
        ROLL=np.abs(filtered_df_avg['ROLL'].mean())
        Nz=filtered_df_avg['Nz'].mean()

        TAS=filtered_df_avg['TAS'].mean()
        TURN_RATE=deltaHDG/deltaT*np.pi/180 if ROLL<60 else cst.G_IMPERIAL*np.sqrt(Nz**2-1)/(TAS*cst.KT_TO_FPS)
        TURN_RATE=TURN_RATE*180/np.pi  #in deg/s
        BARO_ALT=(filtered_df_avg['BARO_ALT'].max()+filtered_df_avg['BARO_ALT'].min())/2
        Hpc=(filtered_df_avg['Hpc'].max()+filtered_df_avg['Hpc'].min())/2
        GPS_ALT=(filtered_df_avg['GPS_ALT'].max()+filtered_df_avg['GPS_ALT'].min())/2
        CAS=filtered_df_avg['CAS'].mean()
        Vpc=filtered_df_avg['Vpc'].mean()
        Mpc=filtered_df_avg['Mpc'].mean()
        Vptc=Mpc*np.sqrt(cst.GAMMA*cst.R_SI*(filtered_df_avg['SAT'].iloc[0]+cst.C_TO_K_OFFSET))*cst.MS_TO_KT
        TURN_RADIUS=Vptc*cst.KT_TO_FPS/np.deg2rad(TURN_RATE)  #in ft

        #assign values to dataframe
        level_turn_data.loc[turn_index,'deltaT']=deltaT
        level_turn_data.loc[turn_index,'BARO_ALT']=BARO_ALT
        level_turn_data.loc[turn_index,'Hpc']=Hpc
        level_turn_data.loc[turn_index,'GPS_ALT']=GPS_ALT
        level_turn_data.loc[turn_index,'CAS']=CAS
        level_turn_data.loc[turn_index,'Vpc']=Vpc
        level_turn_data.loc[turn_index,'TAS']=TAS
        level_turn_data.loc[turn_index,'Mpc']=Mpc
        level_turn_data.loc[turn_index,'Vtpc']=Vptc
        level_turn_data.loc[turn_index,'TURN_RATE']=TURN_RATE
        level_turn_data.loc[turn_index,'Nz']=Nz
        level_turn_data.loc[turn_index,'TURN_RADIUS']=TURN_RADIUS
        level_turn_data.loc[turn_index,'ROLL']=ROLL
        level_turn_data.loc[turn_index,'CATEGORY']=int(np.round(Hpc/1000,0)*1000)  #in ft

        row = turn_index + 1
        # Creating the plots
        trace1 = go.Scatter(
            x=filtered_df_avg.index, y=filtered_df_avg['BARO_ALT'],
            mode='lines', name='Pressure Altitude', line=dict(color='blue')
        )
        trace2 = go.Scatter(
            x=filtered_df_avg.index, y=filtered_df_avg['CAS'],
            mode='lines', name='CAS', line=dict(color='red')
        )
        trace3 = go.Scatter(
            x=filtered_df_avg.index, y=filtered_df_avg['ROLL'],
            mode='lines', name='Bank angle', line=dict(color='purple')
        )
        trace4 = go.Scatter(
            x=filtered_df_avg.index, y=filtered_df_avg['Nz'],
            mode='lines', name='Load factor', line=dict(color='green')
        )
        trace5 = go.Scatter(
            x=filtered_df_avg.index, y=filtered_df_avg['MAG_HDG'],
            mode='lines', name='Mag heading', line=dict(color='orange')
        )

        # Add traces to the subplot
        fig.add_trace(trace1, row=row, col=1)
        fig.add_trace(trace2, row=row, col=1, secondary_y=True)
        fig.add_trace(trace3, row=row, col=2)
        fig.add_trace(trace4, row=row, col=2, secondary_y=True)
        fig.add_trace(trace5, row=row, col=3)

    if graphs:
        # Update layout for better presentation
        fig.update_layout(
            height=300*rows,  # Adjust height based on number of rows
            width=1200,  # You can adjust the width
            title_text="Level turn",
            showlegend=False,
        )

        fig.show()

    level_turn_data.to_csv("data\\Perf\\"+flight_date.strftime('%Y-%m-%d')+"_level_turn.csv")

    return level_turn_data


##############################  ARCHIVES ##############################################################
# def level_accel(data,times,smooth, YAPS=True,graphs=False):   #smooth is the number of values to consider for rolling average
#     timestamps=to_test_point_lite(times)

#     #correct the timestamp to the date of the flight (to make time comparison easier)
#     flight_date=data.index[0].date()
#     timestamps_date=timestamps.Begin[0].date()
#     date_offset=timestamps_date-flight_date
#     timestamps['Begin']=timestamps['Begin']-date_offset
#     timestamps['End']=timestamps['End']-date_offset

#     #create series for the results
#     accel_list=[] #list of all points within a level accel

#     #prepare plots
#     nb_subplots = len(timestamps)

#     # Create a subplot grid
#     fig = make_subplots(
#         rows=nb_subplots,
#         cols=1,
#         shared_xaxes=False,  # Optional: you can customize axes sharing
#         shared_yaxes=False,  # We will be using individual y-axes for each plot
#         vertical_spacing=0.15,
#         subplot_titles=[f"Level accel {plot_index+1}" for plot_index in range(nb_subplots)],
#         # Define a secondary y-axis for each subplot
#         specs=[[{"secondary_y": True} for _ in range(1)] for _ in range(nb_subplots)]
#     )


#     for accel_index,row in enumerate(timestamps.itertuples()):

#         filtered_df=data.loc[row.Begin:row.End].copy()

#         columns_to_smooth = ['BARO_ALT','GPS_ALT','CAS','TAS','SAT']

#         for col in columns_to_smooth:
#             filtered_values = signal.savgol_filter(filtered_df[col].astype(float).to_numpy(),
#                                            window_length=window_length,
#                                            polyorder=polyorder)
#             filtered_df.loc[:,col] = filtered_values

#         filtered_df_avg = filtered_df[['BARO_ALT','GPS_ALT','CAS','TAS','SAT']].resample(smooth).first() #average the data

#         filtered_df_avg[['Hpc', 'Vpc', 'Mpc']] = filtered_df_avg.apply(lambda row: pd.Series(das.C12_pitot_static(row['CAS'], row['BARO_ALT'],YAPS=YAPS)), axis=1)


#         for i in range(len(filtered_df_avg)-1):
#             Time_ISO=filtered_df_avg.index[i]
#             Time_rel=(Time_ISO-row.Begin).total_seconds()       #Time since the beginning of the accel
#             BARO_ALT=filtered_df_avg['BARO_ALT'].iloc[i]
#             Hpc=filtered_df_avg['Hpc'].iloc[i]
#             GPS_ALT=filtered_df_avg['GPS_ALT'].iloc[i]
#             CAS=filtered_df_avg['CAS'].iloc[i]
#             Vpc=filtered_df_avg['Vpc'].iloc[i]
#             TAS=filtered_df_avg['TAS'].iloc[i]
#             Mpc=filtered_df_avg['Mpc'].iloc[i]
#             Vtpc=Mpc*np.sqrt(cst.GAMMA*cst.R_SI*(filtered_df_avg['SAT'].iloc[0]+cst.C_TO_K_OFFSET))*cst.MS_TO_KT  #true airspeed at pressure altitude
#             deltaT=(filtered_df_avg.index[i+1]-filtered_df_avg.index[i]).total_seconds()
#             deltaH=filtered_df_avg['BARO_ALT'].iloc[i+1]-filtered_df_avg['BARO_ALT'].iloc[i]
#             deltaV=filtered_df_avg['TAS'].iloc[i+1]-filtered_df_avg['TAS'].iloc[i]
#             Es=BARO_ALT+(TAS*cst.KT_TO_FPS)**2/(2*cst.G_IMPERIAL)
#             Ps=deltaH/deltaT+(TAS*cst.KT_TO_FPS)*(deltaV*cst.KT_TO_FPS/deltaT)/(cst.G_IMPERIAL*deltaT)
#             accel_list.append([Time_ISO,Time_rel,accel_index,Hpc,BARO_ALT,GPS_ALT,CAS,Vpc,TAS,Mpc,Vtpc,deltaT,deltaH,deltaV,Es,Ps])

#     accel_data=pd.DataFrame(accel_list, columns=['Time_ISO','Time_rel','ID','Hpc','BARO_ALT','GPS_ALT','CAS','Vpc','TAS','Mpc','Vtpc','deltaT','deltaH','deltaV','Es','Ps'])
#     accel_data.set_index('Time_ISO', inplace=True)
#     #accel_data['Ps_smooth'] = accel_data.groupby('ID')['Ps'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())


#     if graphs:
#         # Update layout for better presentation
#         fig.update_layout(
#             height=350*nb_subplots,  # Adjust height based on number of rows
#             width=1200,  # You can adjust the width
#             title_text="Level accels",
#             showlegend=False,
#         )

#     for id, group in accel_data.groupby('ID'):
#         fig.add_trace(go.Scatter(x=group.index, y=group['BARO_ALT'],yaxis='y', name=f'Level accel {id}', line=dict(color='blue')),row=id+1,col=1)
#         fig.add_trace(go.Scatter(x=group.index, y=group['CAS'],yaxis='y2', name=f'Level accel {id}', line=dict(color='red')),row=id+1,col=1,secondary_y=True)

#         # Update axis labels and titles
#         fig.update_yaxes(title_text="Pressure Altitude (ft)", title_font=dict(color="blue"),  tickfont=dict(color="blue"), row=id+1, col=1)
#         fig.update_yaxes(title_text="Calibrated Airspeed (kt)", title_font=dict(color="red"),  tickfont=dict(color="red"),row=id+1, col=1, secondary_y=True)

#         # Show the figure
#     fig.show()

#     accel_data.to_csv("data\\Perf\\"+flight_date.strftime('%Y-%m-%d')+"_level_accel.csv")

#     return accel_data
