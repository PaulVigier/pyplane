"""Functions for the T-38 flying qualities report.

----------
to_timestamps(times):
    Converts a list or array of time intervals in HH:MM:SS format into a pandas DataFrame as 'Begin' and 'End'
    datetime objects.
plot_data(data, times, axis='pitch', graphs=True):
    Plots multiple flight parameters (stick position, pitch, pitch rate, altitude, airspeed, Nz, AoA) using subplots.
plot_force_per_g(data, times, axis='pitch', graphs=True):
    Plots stick force per g and related flight parameters over specified time intervals.
freq_sweep(data, times, axis='pitch', graphs=True):
    Performs frequency sweep analysis on flight data, computing and plotting FFT-based Bode magnitude, phase,
    and coherence between stick input and aircraft response.
matlab_template:
    A Plotly layout dictionary mimicking MATLAB-like appearance.
"""
import numpy as np
from scipy import signal
import pandas as pd
from .das import to_test_point_lite
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define a Plotly layout template that mimics MATLAB style
matlab_template = dict(
        layout=go.Layout(
                # plot layout
                width=1300,
                height=800,
                showlegend=True,
                plot_bgcolor="white",
                # title
                title_font=dict(size=24, color="black", weight='bold', family="Arial",),
                title_x=0.5,  # Center the title horizontally
                title_y=0.98,  # Position the title slightly lower vertically

                xaxis=dict(linecolor='black', showgrid=True, showline=True, gridcolor='lightgrey',
                           ticks='outside', tickcolor='black', tickwidth=2, ticklen=5,
                           title_font=dict(size=18, weight='bold', family='Arial'),
                           tickfont=dict(size=14, family='Arial')
                           ),
                yaxis=dict(linecolor='black', showgrid=True, showline=True, gridcolor='lightgrey',
                           ticks='outside', tickcolor='black', tickwidth=2, ticklen=5,
                           title_font=dict(size=18, weight='bold', family='Arial'),
                           tickfont=dict(size=14, family='Arial')
                           ),
                yaxis2=dict(linecolor='black', showgrid=False, showline=True,
                            gridcolor='lightgrey', ticks='outside', tickcolor='black',
                            tickwidth=2, ticklen=5,
                            title_font=dict(size=18, weight='bold', family='Arial'),
                            tickfont=dict(size=14, family='Arial'),
                            ),
                legend=dict(
                        x=1.1,                # Position the legend at the far-right of the plot
                        y=1.1,                # Position the legend at the top of the plot
                        xanchor='right',    # Anchor the legend on the right side
                        yanchor='bottom',      # Anchor the legend at the top
                        bordercolor='black',   # Border color around the legend (optional)
                        borderwidth=1,        # Border width around the legend (optional)
                        ),
            )
)


def plot_data(data: pd.DataFrame, times: list, axis: str = 'pitch', graphs: bool = True) -> None:
    """Plot Stick position, pitch, pitch rate, Altitude/Airspeed, Nz, AoA for FQ report.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing flight data as returned by das_read function.
    times : list
        List of time intervals (begin and end)o to plot. Each interval is processed to extract corresponding data
        from the DataFrame.
    axis : str, optional
        Axis to plot (default is 'pitch'). Yaw and roll not implemented.
    graphs : bool, optional
        If True, displays the generated plots (default is True).

    Returns
    -------
    None

    """
    timestamps = to_test_point_lite(times)
    nb_plot = 6
    # correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date = data.index[0].date()
    timestamps_date = timestamps.Begin[0].date()
    date_offset = timestamps_date - flight_date
    timestamps['Begin'] = timestamps['Begin'] - date_offset
    timestamps['End'] = timestamps['End'] - date_offset

    for _, row in enumerate(timestamps.itertuples()):
        # Extract the start and end times from the row
        start_time = row.Begin
        end_time = row.End

        filtered_data = data[(data.index >= start_time) & (data.index <= end_time)]

        fig = make_subplots(
            rows=nb_plot, cols=1,
            subplot_titles=("Long stick position (%)", "Pitch (deg)", "Pitch rate (deg/sec)",
                            "Altitude / Airspeed", 'Nz (g)', 'AoA (deg)'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": True} for _ in range(1)] for _ in range(nb_plot)]
            )

        fig.update_layout(template=matlab_template,
                          title='Time Evolution',
                          height=1500,
                          showlegend=True,
                          hovermode="x unified",
                          hoversubplots="axis",
                          )

        fig.update_xaxes(
            tickformat="%H:%M:%S", dtick=1000,
            hoverformat='<b>%H:%M:%S.%f</b>',
        )
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data['DELTA_STICK_LON_CORR'],
                xaxis='x',
                hovertemplate='Delta_es: %{y:.1f} %<extra></extra>',
                line=dict(color='purple')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['PITCH'],
                    hovertemplate='Pitch: %{y:.1f} deg<extra></extra>',
                    line=dict(color='red')
                    ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['PITCH_RATE'],
                    hovertemplate='Pitch rate: %{y:.1f} deg/sec<extra></extra>',
                    line=dict(color='green')
                    ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['BARO_ALT'],
                    hovertemplate='Altitude: %{y:.0f} ft<extra></extra>',
                    line=dict(color='blue')
                    ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['CAS'],
                    hovertemplate='CAS: %{y:.1f} KIAS<extra></extra>',
                    line=dict(color='red')
                    ),
            secondary_y=True, row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['NORM_ACCEL'],
                    hovertemplate='Nz: %{y:.1f} g<extra></extra>',
                    line=dict(color='orange')
                    ),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['AOA'] * 0.24 * 180 / np.pi,
                    hovertemplate='AoA: %{y:.1f} deg<extra></extra>',
                    line=dict(color='grey'),
                    opacity=0.6
                    ),
            row=6, col=1
        )

        fig.update_yaxes(
            title_text="Stick pos (%)",
            title_font=dict(color="purple"),
            tickfont=dict(color="purple"),
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Pitch (deg)",
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Pitch rate (deg/sec)",
            title_font=dict(color="green"),
            tickfont=dict(color="green"),
            row=3, col=1
        )
        fig.update_yaxes(
            title_text="Pressure Altitude (ft)",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue"),
            row=4, col=1
        )
        fig.update_yaxes(
            title_text="Calibrated Airspeed (kt)",
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            row=4, col=1,
            secondary_y=True
        )
        fig.update_yaxes(
            title_text="Nz (g)",
            title_font=dict(color="orange"),
            tickfont=dict(color="orange"),
            row=5, col=1
        )
        fig.update_yaxes(
            title_text="AoA (deg)",
            title_font=dict(color="grey"),
            tickfont=dict(color="grey"),
            row=6, col=1
        )

        fig.data[1].xaxis = 'x'
        fig.data[2].xaxis = 'x'
        fig.data[3].xaxis = 'x'
        fig.data[4].xaxis = 'x'
        fig.data[5].xaxis = 'x'
        fig.data[6].xaxis = 'x'

        fig.show()


def plot_force_per_g(data: pd.DataFrame, times: pd.DataFrame, axis: str = 'pitch', graphs: bool = True) -> None:
    """Plot Stick position, pitch, pitch rate, Altitude/Airspeed, Nz, AoA for FQ report.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing flight data as returned by das_read function.
    times : list
        List of time intervals (begin and end)o to plot. Each interval is processed to extract corresponding data
        from the DataFrame.
    axis : str, optional
        Axis to plot (default is 'pitch'). Yaw and roll not implemented.
    graphs : bool, optional
        If True, displays the generated plots (default is True).

    Returns
    -------
    None

    """
    timestamps = to_test_point_lite(times)
    nb_plot = 6
    # correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date = data.index[0].date()
    timestamps_date = timestamps.Begin[0].date()
    date_offset = timestamps_date - flight_date
    timestamps['Begin'] = timestamps['Begin'] - date_offset
    timestamps['End'] = timestamps['End'] - date_offset

    for _, row in enumerate(timestamps.itertuples()):
        # Extract the start and end times from the row
        start_time = row.Begin
        end_time = row.End

        filtered_data = data[(data.index >= start_time) & (data.index <= end_time)]

        fig = make_subplots(
            rows=nb_plot, cols=1,
            subplot_titles=("Long stick position (%)", "Pitch (deg)", "Pitch rate (deg/sec)",
                            "Altitude / Airspeed", 'Nz (g)', 'AoA (deg)'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": True} for _ in range(1)] for _ in range(nb_plot)]
            )

        fig.update_layout(template=matlab_template,
                          title='Time Evolution',
                          height=900,
                          showlegend=False,
                          hovermode="x unified",
                          hoversubplots="axis",
                          )

        fig.update_xaxes(tickformat="%H:%M:%S", dtick=1000, hoverformat='<b>%H:%M:%S.%f</b>',)
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['DELTA_STICK_LON_INCH'],
                    xaxis='x',
                    hovertemplate='Delta_es: %{y:.1f} %<extra></extra>',
                    line=dict(color='purple')
                    ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['PITCH'],
                    hovertemplate='Pitch: %{y:.1f} deg<extra></extra>',
                    line=dict(color='red')
                    ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['PITCH_RATE'],
                    hovertemplate='Pitch rate: %{y:.1f} deg/sec<extra></extra>',
                    line=dict(color='green')
                    ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['BARO_ALT'],
                    hovertemplate='Altitude: %{y:.0f} ft<extra></extra>',
                    line=dict(color='blue')
                    ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['CAS'],
                    hovertemplate='CAS: %{y:.1f} KIAS<extra></extra>',
                    line=dict(color='red')
                    ),
            secondary_y=True, row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['NORM_ACCEL'],
                    hovertemplate='Nz: %{y:.1f} g<extra></extra>',
                    line=dict(color='orange')
                    ),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['AOA'] * 0.24 * 180 / np.pi,
                    hovertemplate='AoA: %{y:.1f} deg<extra></extra>',
                    line=dict(color='grey')
                    ),
            row=6, col=1
        )

        fig.update_yaxes(
            title_text="Stick pos (inch)",
            title_font=dict(color="purple"),
            tickfont=dict(color="purple"),
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Pitch (deg)",
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Pitch rate (deg/sec)",
            title_font=dict(color="green"),
            tickfont=dict(color="green"),
            row=3, col=1
        )
        fig.update_yaxes(
            title_text="Pressure Altitude (ft)",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue"),
            row=4, col=1
        )
        fig.update_yaxes(
            title_text="Calibrated Airspeed (kt)",
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            row=4, col=1,
            secondary_y=True
        )
        fig.update_yaxes(
            title_text="Nz (g)",
            title_font=dict(color="orange"),
            tickfont=dict(color="orange"),
            row=5, col=1
        )
        fig.update_yaxes(
            title_text="AoA (deg)",
            title_font=dict(color="grey"),
            tickfont=dict(color="grey"),
            row=6, col=1
        )

        fig.data[1].xaxis = 'x'
        fig.data[2].xaxis = 'x'
        fig.data[3].xaxis = 'x'
        fig.data[4].xaxis = 'x'
        fig.data[5].xaxis = 'x'
        fig.data[6].xaxis = 'x'

        fig.show()


def freq_sweep(data: pd.DataFrame, times: pd.DataFrame, axis: str='pitch', graphs: bool=True) -> None:
    """Bode plot for frequency response analysis during pitch frequency sweeps.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing flight data as returned by das_read function.
    times : list
        List of time intervals (begin and end)o to plot. Each interval is processed to extract corresponding data
        from the DataFrame.
    axis : str, optional
        Axis to plot (default is 'pitch'). Yaw and roll not implemented.
    graphs : bool, optional
        If True, displays the generated plots (default is True).

    Returns
    -------
    None

    """
    timestamps = to_test_point_lite(times)

    # correct the timestamp to the date of the flight (to make time comparison easier)
    flight_date = data.index[0].date()
    timestamps_date = timestamps.Begin[0].date()
    date_offset = timestamps_date - flight_date
    timestamps['Begin'] = timestamps['Begin'] - date_offset
    timestamps['End'] = timestamps['End'] - date_offset

    for _, row in enumerate(timestamps.itertuples()):

        # Extract the start and end times from the row
        start_time = row.Begin
        end_time = row.End

        # Filter the data based on the time range
        mask = (data.index >= start_time) & (data.index <= end_time)
        filtered_data = data[mask]

        # --- Prepare Time and Signals ---
        if isinstance(filtered_data.index, pd.DatetimeIndex):
            time = (filtered_data.index - filtered_data.index[0]).total_seconds().to_numpy()
        else:
            time = filtered_data.index.to_numpy()

        u = filtered_data['DELTA_STICK_LON_CORR'].to_numpy()
        y = filtered_data['PITCH'].to_numpy()

        # Sampling info
        dt = np.mean(np.diff(time))
        fs = 1 / dt
        N = len(time)
        f = np.fft.rfftfreq(N, d=dt)

        # --- FFTs ---
        U = np.fft.rfft(u)
        Y = np.fft.rfft(y)

        # Avoid divide-by-zero (add small epsilon)
        H = Y / (U + 1e-12)

        # Bode Magnitude and Phase
        mag_db = 20 * np.log10(np.abs(H))
        phase_deg_raw = np.angle(H, deg=True)
        phase_deg = np.where(phase_deg_raw < 0, phase_deg_raw, phase_deg_raw - 360)

        coherence = signal.coherence(y, u, fs=fs)[1]

        # --- Plotting ---
        # Time Evolution
        fig_time = make_subplots(specs=[[{"secondary_y": True}]])

        fig_time.add_trace(
            go.Scatter(
                    x=time, y=u,
                    name='Stick longitudinal position (Input)',
                    line=dict(color='blue')
                    ),
            secondary_y=False
        )
        fig_time.add_trace(
            go.Scatter(
                    x=time, y=y,
                    name='PITCH (Output)',
                    line=dict(color='red')
                    ),
            secondary_y=True
        )

        fig_time.update_layout(
            title='Time Evolution',
            height=500,
            xaxis_title='Time (s)',
            showlegend=False
        )
        fig_time.update_yaxes(
            title_text='Stick longitudinal position (%)',
            title_font=dict(color="blue"),
            tickfont=dict(color="blue"),
            secondary_y=False
        )
        fig_time.update_yaxes(
            title_text='Pitch (deg)',
            title_font=dict(color="red"),
            tickfont=dict(color="red"),
            secondary_y=True
        )
        fig_time.show()

        # FFT-based Bode and Coherence
        fig_bode = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=(
                            "FFT Bode Magnitude",
                            "FFT Bode Phase",
                            "FFT Coherence"
                            ),
            vertical_spacing=0.1
        )

        fig_bode.add_trace(
            go.Scatter(x=f, y=mag_db, name='Magnitude (dB)'),
            row=1, col=1
        )
        fig_bode.add_trace(
            go.Scatter(
                    x=f, y=phase_deg, name='Phase (°)'
                    ),
            row=2, col=1
        )
        fig_bode.add_trace(
            go.Scatter(
                    x=f, y=coherence, name='Coherence',
                    line=dict(color='green')
                    ),
            row=3, col=1
        )

        fig_bode.update_layout(
            title='pitch response to longitudinal stick control input (FFT-based)',
            height=800,
            xaxis3_title='Frequency (Hz)',
            yaxis1_title='Magnitude (dB)',
            yaxis2_title='Phase (°)',
            yaxis3_title='Coherence',
            showlegend=False,
        )

        # Set frequency axis to log scale
        for i in range(1, 4):
            fig_bode.update_xaxes(type='log', row=i, col=1)

        fig_bode.update_yaxes(range=[0, 1.05], row=3, col=1)  # Coherence limits
        fig_bode.update_xaxes(range=[-1, 0.4], row=1, col=1)
        fig_bode.update_xaxes(range=[-1, 0.4], row=2, col=1)
        fig_bode.update_xaxes(range=[-1, 0.4], row=3, col=1)
        fig_bode.update_yaxes(tickvals=[-360, -270, -180, -90, 0],
                              ticktext=['-360°', '-270°', '-180°', '-90°', '0°'], row=2, col=1)

        fig_bode.show()
