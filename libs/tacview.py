"""Functions to export trajectory data to Tacview ACMI files.

Functions:
- traj2tacview: Write Tacview ACMI file from a list of Trajectory objects.
"""

from .trajectory import Trajectory
from collections import defaultdict

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
    lines.append("0,ReferenceTime=2025-09-01T05:00:00Z")
    lines.append("DataRecorder=Python Script")
    lines.append("")
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
