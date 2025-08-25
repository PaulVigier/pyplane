"""Functions to export trajectory data to Tacview ACMI files.

Functions:
- traj2tacview: Write Tacview ACMI file from a list of Trajectory objects.
"""
import pandas as pd
from libs import constants as cst
from .trajectory import Trajectory
from collections import defaultdict


def das2tacview(filename: str, data: pd.DataFrame, params: dict=None) -> None:
    """Write Tacview ACMI file from a list of DAS data dictionaries."""
    default_params = {
        'Name': data['Aircraft'].iloc[0],
        'Type': 'Air+FixedWing',
        'Color': 'Blue'
    }
    init_time=data['Formatted_Time'].iloc[0]
    ID=1

    if params is None:
        params = {}
    # Update default parameters with any additional parameters
    default_params.update(params)


    if not filename.endswith(".acmi"):
        filename += ".acmi"

    #filter the dataframe to keep only the useful values
    tacview_data = pd.DataFrame({
                                 'Time': data['Time_sec']-data['Time_sec'].iloc[0],
                                'Longitude': data['LON'],
                                'Latitude': data['LAT'],
                                'Altitude': data['GPS_ALT']*cst.FT_TO_M,  # in meters
                                'Yaw': data['YAW'].round(2),
                                'Pitch': data['PITCH'].round(2),
                                'Roll': data['ROLL'].round(2),
                                'TAS': (data['TAS']*cst.KT_TO_MS).round(1),  # in KTAS
                                'CAS': (data['CAS']*cst.KT_TO_MS).round(1),  # in KCAS
                                'AOA': data['AOA'].round(2),  # in degrees
                                'Mach': data['MACH'].round(3),
                                'Event': data['EVENT'],
                                })

    # Replace repeated values with ''
    tacview_data_filtered = tacview_data.where(tacview_data.ne(tacview_data.shift()), '')

    custom_params=['TAS', 'CAS', 'AOA', 'Mach','Event']


    # Extract trajectory information from DAS data
    lines= []
    #1 ACMI file header
    lines.append("FileType=text/acmi/tacview")
    lines.append("FileVersion=2.2")
    lines.append(f"0,ReferenceTime={init_time}")
    lines.append("0,DataRecorder=Python Script")
    lines.append('#0')

    #Initial line with parameters
    params_str = ",".join([f"{key}={value}" for key, value in default_params.items()])
    lines.append(f"{ID},{params_str}")

    for _,row in tacview_data_filtered.iterrows():
        lines.append(f"#{row['Time']}")
        line = f"{ID},T={row['Longitude']}|{row['Latitude']}|{row['Altitude']}|{row['Roll']}|{row['Pitch']}|{row['Yaw']}"

        # Loop through the fields and add them to the line if they're not empty
        for field in custom_params:
            if row[field] != '':
                line += f",{field}={row[field]}"
        lines.append(line)

    # Write to file
    with open(filename, 'w') as f:
        f.write("\n".join(lines))

    print(f"Tacview file '{filename}' created.")




def traj2tacview(filename: str, trajectories: list[Trajectory]) -> None:
    """Write Tacview ACMI file from a list of Trajectory objects."""
    if not all(isinstance(traj, Trajectory) for traj in trajectories):
        raise TypeError("All trajectories must be instances of Trajectory")

    if not filename.endswith(".acmi"):
        filename += ".acmi"

    lines= []
    #1 ACMI file header
    lines.append("FileType=text/acmi/tacview")
    lines.append("FileVersion=2.2")
    lines.append("0,ReferenceTime=2025-01-01T00:00:00Z")
    lines.append("0,DataRecorder=Python Script")
    lines.append('#0')

    # 2. Object metadata
    for traj in trajectories:
        # You may adapt this depending on what traj.params is
        ID= traj.ID
        params=traj.params
        params_str = ",".join([f"{key}={value}" for key, value in params.items()])
        lines.append(f"{ID},{params_str}")

    # 3. Collect all time values across all trajectories
    time_map = defaultdict(list)  # time -> list of (id, coord)

    for traj in trajectories:
        times = traj.times
        coords = traj.coords  # assume shape (N, 3)
        ID=traj.ID

        for t, coord in zip(times, coords, strict=True):
            time_map[float(t)].append((ID, coord))

    # 4. Sort the times and write entries
    for t in sorted(time_map.keys()):
        if t== 0:
            t=0.0001  # Avoid duplicating the 0 time entry
        lines.append(f"#{t}")
        for ID, coord in time_map[t]:
            coord_str = f"T={coord.lon}|{coord.lat}|{coord.alt}|||{coord.yaw}"
            lines.append(f"{ID},{coord_str}")

    # 5. Write to file
    with open(filename, 'w') as f:
        f.write("\n".join(lines))

    print(f"Tacview file '{filename}' created with {len(trajectories)} trajectories.")
