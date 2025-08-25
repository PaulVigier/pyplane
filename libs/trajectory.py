"""Define Trajectory class as sequences of Transform objects and their associated timestamps.

Classes:
    Trajectory: Represents a trajectory defined by coordinates (Transform objects), timestamps, and optional parameters.
Functions:
    merge_trajectories(trajectories: list[Trajectory]) -> Trajectory:
        Merges multiple Trajectory instances into a single trajectory, concatenating coordinates and timestamps, and merging parameters.
Usage:
    - Initialize a Trajectory with arrays of Transform objects and timestamps.
    - Set or update trajectory parameters.
    - Merge trajectories to combine their data.
    - Apply time offsets to all timestamps in a trajectory.

"""

import numpy as np
from libs.transform import Transform
from os import urandom

class Trajectory:
    """Represents a trajectory defined by a sequence of coordinates (Transform objects) and their corresponding timestamps.

    This class provides methods for initializing, merging, and manipulating trajectory data.

    Attributes:
        coords (np.ndarray): Array of Transform objects representing the trajectory's coordinates.
        times (np.ndarray): Array of timestamps corresponding to each coordinate.
        params (dict): Dictionary of additional parameters describing the trajectory.
        ID (int or str): Unique identifier for the trajectory.

    Methods:
        set_param(key, value): Set or update a parameter in the trajectory.
        merge(other): Merge another Trajectory instance into this one, combining coordinates, times, and parameters.
        time_offset(offset): Apply a numerical offset to all timestamps in the trajectory.

    """

    def __init__(self, coords:np.ndarray, times:np.ndarray, params:dict=None, ID:int=None)-> None:
        """Initialize a trajectory object with coordinates, timestamps, parameters, and an optional ID.

        Args:
            coords (np.ndarray): Array of Transform objects representing trajectory coordinates.
            times (np.ndarray): Array of timestamps corresponding to each coordinate.
            params (dict, optional): Additional parameters for the trajectory. Defaults to None.
            ID (int, optional): Unique identifier for the trajectory. If not provided, a random ID is generated.

        Raises:
            TypeError: If coords is not a numpy array or any element is not a Transform instance.
            TypeError: If times is not a numpy array.
            ValueError: If coords and times do not have the same length.

        """
        if not isinstance(coords, np.ndarray):
            raise TypeError("coords must be a numpy array")
        if not all(isinstance(c, Transform) for c in coords):
            raise TypeError("All elements of coords must be instances of Transform")

        if not isinstance(times, np.ndarray):
            raise TypeError("time must be a numpy array")

        if len(coords) != len(times):
            raise ValueError("coords and time must have the same length")

        self.coords = coords
        self.times = times
        self.params = params if params is not None else {}
        self.ID = ID if ID is not None else urandom(4).hex() # Generate a random ID if not provided

    def __str__(self)-> str:
        """Return a string representation of the trajectory."""
        return f"Traj({self.params})"

    def set_param(self, key: str, value: any) -> None:
        """Set or update a parameter."""
        self.params[key] = value

    def merge(self, other: "Trajectory") -> None:
        """Merge another Trajectory into this one.

        The other trajectory's coordinates and times are appended to this trajectory.
        """
        if not isinstance(other, Trajectory):
            raise TypeError("other must be an instance of Trajectory")

        self.coords = np.concatenate((self.coords, other.coords))
        self.times = np.concatenate((self.times, other.times))
        for key, value in other.params.items():
            self.set_param(key, value)

    def time_offset(self, offset: float) -> None:
        """Apply a time offset to the trajectory."""
        if not isinstance(offset, (int, float)):
            raise TypeError("offset must be a number")

        self.times += offset

def merge_trajectories(trajectories: list[Trajectory]) -> Trajectory:
    """Merge multiple Trajectory objects into one.

    The coordinates and times of all trajectories are concatenated.
    Params are merged if non-existent, otherwise they are ignored (priority is given to the first trajectory).
    """
    if not all(isinstance(t, Trajectory) for t in trajectories):
        raise TypeError("All elements must be instances of Trajectory")

    merged_coords = np.concatenate([t.coords for t in trajectories])
    merged_times = np.concatenate([t.times for t in trajectories])

    merged_params = {}
    for t in trajectories:
        for key, value in t.params.items():
            # print(key,value)
            if key not in merged_params:
                merged_params[key] = value

    return Trajectory(merged_coords, merged_times, merged_params)
