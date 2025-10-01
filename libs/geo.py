"""Collection of geographic and kinematic utility functions for aircraft maneuver calculations.

Includes coordinate transformations, distance calculations, turn geometry, intercept point computation, and
optimization routines for air combat scenarios.

Main Features:
--------------
- Angle normalization and heading utilities.
- 2D and 3D distance calculations using spherical and ellipsoidal Earth models.
- Turn radius and turn rate calculations for aircraft maneuvers.
- Antenna Train Angle (ATA) calculations in 2D and 3D.
- Collision point computation between two aircraft given positions, headings, and speeds.
- Coordinate transformations between local (body) and geodetic frames.
- Calculation of entry and exit points for turns, straight-line trajectories, and intercept points.
- Heading Change Angle (HCA) computation and optimization for parallel intercepts.
- Utility functions for generating intercept trajectories and optimizing maneuvers.

Dependencies:
-------------
- numpy
- pymap3d
- geographiclib.geodesic
- scipy.optimize
- .transform.Transform (local module)

Classes:
--------
- None (uses Transform class from .transform)

Functions:
----------
- normalize_angle(angle: float) -> float
    Normalize an angle to the range (-180, 180].
- normalize_heading(heading: float) -> float
    Normalize a heading to the range (0, 360].
- normalize_turn(angle: float) -> float
    Normalize a turn angle to the range [-360, 360].
- reciprocal_heading(heading: float) -> float
    Compute the reciprocal (opposite) heading.
- distance2D_fast(lat1, long1, lat2, long2) -> float
    Fast great-circle distance using the Haversine formula.
- distance2D(lat1, long1, lat2, long2) -> float
    Precise geodesic distance using Karney's algorithm (WGS84 ellipsoid).
- distance3D(lat1, long1, h1, lat2, long2, h2) -> float
    3D Euclidean distance between two geodetic points.
- turn_radius(TAS, load_factor) -> float
    Calculate turn radius for given speed and load factor.
- turn_rate(TAS, load_factor) -> float
    Calculate turn rate for given speed and load factor.
- ATA3D(lat_f, lon_f, alt_f, lat_tgt, lon_tgt, alt_tgt) -> float
    Compute 3D Antenna Train Angle (ATA) between aircraft and target.
- ATA2D(lat_f, lon_f, lat_tgt, lon_tgt) -> float
    Compute 2D Antenna Train Angle (ATA) between aircraft and target.
- collision_point(coord_tgt, TAS_tgt, coord_f, TAS_f) -> tuple[Transform, float, float]
    Compute collision point, collision ATA, and time to intercept.
- CATA(lat_tgt, lon_tgt, hdg_tgt, TAS_tgt, lat_f, lon_f, TAS_f) -> float
    Compute Collision Antenna Train Angle (CATA).
- rotation_matrix(roll, pitch, yaw) -> np.ndarray
    Compute rotation matrix for given roll, pitch, and yaw.
- local2geodetic(x_body, y_body, z_body, lat0, long0, alt0, roll, pitch, yaw) -> Transform
    Convert local aircraft coordinates to geodetic coordinates.
- aer2geodetic(azimuth, elevation, slant_range, ref, level=False) -> Transform
    Convert azimuth, elevation, and range to geodetic coordinates.
- turn_exit(coord_in, hdg_change, turn_radius, level=True) -> Transform
    Compute exit point of a turn maneuver.
- turn_entry(coord_out, hdg_change, turn_radius, level=True) -> Transform
    Compute entry point of a turn maneuver.
- straight_line(coord_in, TAS, time, level=False) -> Transform
    Compute new position after straight flight.
- get_TIP(coord_tgt, TAS_tgt, TAS_f, load_factor_f, HCA) -> Transform
    Compute intercept point for a given Heading Change Angle (HCA).
- get_TIPs(TAS_tgt, TAS_f, load_factor_f, nb_points, side='right') -> np.ndarray[Transform]
    Compute array of intercept points for a range of HCA values.
- get_HCA(coord_fighter, coord_target) -> float
    Compute Heading Change Angle (HCA) for parallel intercept.
- constrained_HCA(HCA, coord_tgt_1, coord_f_1, TAS_tgt, TAS_f, load_factor_f) -> float
    Compute error between desired HCA and actual collision angle.
- two_stage_HCA_optimizer(coord_tgt_1, coord_f_1, TAS_tgt, TAS_f, load_factor_f, debug=False) -> float
    Optimize HCA using coarse and fine search.

------
- All angles are in degrees unless otherwise specified.
- All distances are in meters unless otherwise specified.
- The Transform class is assumed to encapsulate latitude, longitude, altitude, roll, pitch, and yaw.
- Some functions assume level flight (altitude unchanged).
- For advanced intercept trajectory modeling, see archived functions at the end of the file.

"""
import numpy as np
import pandas as pd
import pymap3d as pm3d
from geographiclib.geodesic import Geodesic
from scipy.optimize import minimize_scalar
from .transform import Transform



geod = Geodesic.WGS84  # using WGS84 ellipsoid for inverse geodesic calculations


def normalize_angle(angle):
    """Return an angle (in degrees) in the range (-180, 180]. 
    Works with scalars, NumPy arrays, or pandas Series.
    """
    normalized = ((np.asarray(angle) - 180) % 360) - 180
    # Handle the -180 case → map to 180
    normalized = np.where(normalized == -180, 180, normalized)
    
    # Preserve input type
    if isinstance(angle, pd.Series):
        return pd.Series(normalized, index=angle.index)
    elif np.isscalar(angle):
        return float(normalized)
    else:
        return normalized

def normalize_heading(heading: float) -> float:
    """Return a heading (in degrees) in the range (0, 360]."""
    return heading % 360


def normalize_turn(angle: float) -> float:
    """Return a turn angle in the range [-360, 360] degrees."""
    # Reduce angle within 0 to 720
    angle = angle % 720

    # If it's greater than 360, wrap it to the negative side
    if angle > 360:
        angle -= 720

    return angle


def reciprocal_heading(heading: float) -> float:
    """Calculate the reciprocal heading (opposite direction) of a given heading in degrees."""
    return normalize_heading(heading + 180)


def distance2D_fast(lat1: float, long1: float, lat2: float, long2: float) -> float:
    """Calculate the great-circle distance between two points on the Earth's surface using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        long1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        long2 (float): Longitude of the second point in degrees.

    Returns:
        float: Distance between the two points in meters.

    Notes:
        - This function assumes the Earth is a perfect sphere with radius 6,371,000 meters.

    """
    R = 6371000  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(long2 - long1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def distance2D(lat1: float, long1: float, lat2: float, long2: float) -> float:
    """Calculate the precise distance between two geographic coordinates using Karney's variant of Vincenty's formula.

    Earth is assumed to be a WGS84 ellipsoid.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        long1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        long2 (float): Longitude of the second point in degrees.

    Returns:
        float: Distance between the two points in meters. Convergence goes down to millimeters.

    """
    return geod.Inverse(lat1, long1, lat2, long2)['s12']


def distance3D(lat1: float, long1: float, h1: float, lat2: float, long2: float, h2: float) -> float:
    """Calculate the 3D slant range distance between two geodetic points specified by latitude, longitude, and height.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        long1 (float): Longitude of the first point in degrees.
        h1 (float): Height of the first point in meters.
        lat2 (float): Latitude of the second point in degrees.
        long2 (float): Longitude of the second point in degrees.
        h2 (float): Height of the second point in meters.

    Returns:
        float: The 3D distance between the two points in meters (euclidean norm).

    """
    x1, y1, z1 = pm3d.geodetic2ecef(lat1, long1, h1)
    x2, y2, z2 = pm3d.geodetic2ecef(lat2, long2, h2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5


def turn_radius(TAS: float, load_factor: float) -> float:
    """Calculate the turn radius (in meters) given the speed (in knots) and load factor (in g)."""
    g = 9.80665  # m/s^2, standard gravity

    if load_factor <= 1:
        raise ValueError("Load factor must be greater than 1 for a valid level turn.")

    turn_radius = (TAS) ** 2 / (g * np.sqrt(load_factor ** 2 - 1))

    return turn_radius


def turn_rate(TAS: float, load_factor: float) -> float:
    """Calculate the turn rate (in degrees per second) given the speed (in knots) and load factor (in g)."""
    g = 9.80665  # m/s^2, standard gravity

    if load_factor <= 1:
        raise ValueError("Load factor must be greater than 1 for a valid level turn.")

    turn_rate_rad = g * np.sqrt(load_factor ** 2 - 1) / TAS

    return np.rad2deg(turn_rate_rad)


def ATA3D(lat_f: float, lon_f: float, alt_f: float, lat_tgt: float, lon_tgt: float, alt_tgt: float) -> float:
    """Calculate the 3D Antenna Train Angle (ATA) between an aircraft and a target.

    Args:
        lat_f (float): Latitude of the fighter (degrees).
        lon_f (float): Longitude of the fighter (degrees).
        alt_f (float): Altitude of the fighter (meters).
        lat_tgt (float): Latitude of the target (degrees).
        lon_tgt (float): Longitude of the target (degrees).
        alt_tgt (float): Altitude of the target (meters).

    Returns:
        float: The 3D ATA (Antenna Train Angle) in degrees (always ≥ 0).

    Notes:
        - The ATA is computed as the Euclidean norm of azimuth and elevation angles.
        - The result is normalized to a heading angle.
        - The ATA is always ≥ 0 (no distinction between left or right)

    """
    azimuth, elevation, _ = pm3d.geodetic2aer(lat_f, lon_f, alt_f, lat_tgt, lon_tgt, alt_tgt)
    ATA = np.sqrt(azimuth ** 2 + elevation ** 2)  # Calculate the 3D ATA
    return normalize_angle(ATA)  # Normalize the azimuth to a heading


def ATA2D(lat_f: float, lon_f: float, lat_tgt: float, lon_tgt: float) -> float:
    """Calculate the 2D Antenna Train Angle (ATA) between an aircraft (fighter) and a target.

    Args:
        lat_f (float): Latitude of the fighter in degrees.
        lon_f (float): Longitude of the fighter in degrees.
        lat_tgt (float): Latitude of the target in degrees.
        lon_tgt (float): Longitude of the target in degrees.

    Returns:
        float: The 2D ATA in degrees between -180° and 180°. Positive values indicate the target is to the right
        of the fighter.

    """
    ATA, _, _ = pm3d.geodetic2aer(lat_f, lon_f, 0, lat_tgt, lon_tgt, 0)  # Azimuth from fighter to target
    return normalize_angle(ATA)  # Normalize the ATA between -180 and 180°


def collision_point(coord_tgt: Transform, TAS_tgt: float, coord_f: Transform, TAS_f: float) -> tuple[Transform,float,float]:
    """Calculate the collision point between two aircraft given their positions, headings, and speeds.

    Args:
        coord_tgt (Transform): The target aircraft's position and orientation (latitude, longitude, altitude, yaw).
        TAS_tgt (float): True Airspeed of the target aircraft in meters per second.
        coord_f (Transform): The fighter aircraft's position and orientation (latitude, longitude, altitude, yaw).
        TAS_f (float): True Airspeed of the fighter aircraft in meters per second.

    Returns:
        coord_coll (Transform): The collision point as a Transform object (latitude, longitude, altitude, yaw).
        CATA (float): Collision Antenna Train Angle (azimuth from fighter to collision point) in degrees.
        time_to_intercept (float): Time to intercept in seconds.

    Raises:
        ValueError: If the altitudes of the target and fighter differ by more than 100 meters, or if there is no intercept solution.

    Notes:
        - Assumes a 2D collision calculation (elevation set to zero).

    """
    if abs(coord_tgt.alt - coord_f.alt) > 100:
        print(f"Target altitude: {coord_tgt.alt} m, Fighter altitude: {coord_f.alt} m")
        raise ValueError("The target and fighter coordinates must be the same for a collision point calculation.")

    # Azimuth, elevation, and slant range to the intercept point
    az_inter, elev_inter, range_init = pm3d.geodetic2aer(coord_f.lat, coord_f.lon, coord_f.alt, coord_tgt.lat,
                                                         coord_tgt.lon, coord_tgt.alt)

    elev_inter = 0  # Set elevation to 0 for a 2D collision point calculation
    ATA = (coord_tgt.yaw - az_inter) % 360  # Angle to target aircraft

    # Polynomial for the time to intercept
    coeffs = [TAS_f ** 2 - TAS_tgt ** 2, 2 * range_init * TAS_tgt * np.cos(np.deg2rad(ATA)), -range_init ** 2]
    roots = np.roots(coeffs)  # Solve the polynomial for time to intercept

    if np.iscomplex(roots).any():
        print("No solution collision")
        raise ValueError("There is no intercept solution")
    else:
        time_to_intercept = max(roots)  # Take the real root for time to intercept

    # Calculate the collision point
    lat_coll, lon_coll, alt_coll = pm3d.aer2geodetic(coord_tgt.yaw, elev_inter, TAS_tgt * time_to_intercept,
                                                     coord_tgt.lat, coord_tgt.lon, coord_tgt.alt)
    # Get the azimuth of the intercept point
    CATA, _, _ = pm3d.geodetic2aer(lat_coll, lon_coll, alt_coll, coord_f.lat, coord_f.lon, coord_f.alt)

    # Create a Transform object for the collision point
    coord_coll = Transform(lat=lat_coll, lon=lon_coll, alt=alt_coll, roll=0, pitch=0, yaw=CATA)
    return coord_coll, CATA, time_to_intercept


def CATA(lat_tgt: float, lon_tgt: float, hdg_tgt: float, TAS_tgt: float, lat_f: float, lon_f: float, TAS_f: float) -> float:
    """Calculate the CATA (Collision Antenna Train Angle) in degrees between a fighter and a target."""
    return collision_point(lat_tgt, lon_tgt, hdg_tgt, TAS_tgt, lat_f, lon_f, TAS_f)[3]


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Calculate the rotation matrix for given roll, pitch, and yaw angles.

    Args:
        roll (float): Roll angle in degrees
        pitch (float): Pitch angle in degrees
        yaw (float): Yaw angle in degrees

    Returns:
        np.ndarray: 3x3 rotation matrix

    """
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])

    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])

    return R_z @ R_y @ R_x


def local2geodetic(x_body: float, y_body: float, z_body: float, lat0: float, long0: float, alt0: float,
                   roll: float, pitch: float, yaw: float) -> Transform:
    """Convert local aircraft coordinates to geodetic coordinates.

    Args:
        x_body (float): coordinate along aircraft's x-axis in meters (x>0 to the front)
        y_body (float): coordinate along aircraft's y-axis in meters (y>0 to the right)
        z_body (float): coordinate along aircraft's z-axis in meters (z>0 downwards)
        lat0 (float): aircraft latitude in degrees
        long0 (float): aircraft longitude in degrees
        alt0 (float): aircraft altitude in meters
        pitch (float): Pitch angle in degrees
        roll (float): Roll angle in degrees
        yaw (float): Yaw angle in degrees

    Returns:
        Transform: Geodetic coordinates (latitude, longitude, altitude) in degrees and meters

    """
    R = rotation_matrix(roll, pitch, yaw)
    body_coords = np.array([x_body, y_body, z_body])
    local_coords = R @ body_coords
    az = np.arctan2(local_coords[1], local_coords[0])  # Azimuth in radians
    elev = np.arctan2(local_coords[2], np.sqrt(local_coords[0]**2 + local_coords[1]**2))  # Elevation in radians
    range = np.linalg.norm(local_coords)  # Slant range in meters
    lat, lon, alt = pm3d.aer2geodetic(np.rad2deg(az), np.rad2deg(elev), range, lat0, long0, alt0)
    # Return a Transform object with the geodetic coordinates
    return Transform(lat=lat, lon=lon, alt=alt, roll=roll, pitch=pitch, yaw=yaw)


def aer2geodetic(azimuth: float, elevation: float, slant_range: float, ref: Transform, level: bool = False) -> Transform:
    """Convert az, el, range coordinates to geodetic coordinates.

    Args:
        azimuth (float): Azimuth angle in degrees
        elevation (float): Elevation angle in degrees
        slant_range (float): Slant range in meters
        ref (Transform): Reference Transform object for the aircraft's position
        level (bool): If True, the altitude will be set to the reference altitude.

    Returns:
        Transform: Geodetic coordinates (latitude, longitude, altitude) in degrees and meters

    """
    lat, lon, alt = pm3d.aer2geodetic(azimuth, elevation, slant_range, ref.lat, ref.lon, ref.alt)
    if level:
        alt = ref.alt  # If level is True, set altitude to the reference altitude

    return Transform(lat=lat, lon=lon, alt=alt, roll=ref.roll, pitch=ref.pitch, yaw=ref.yaw)


def turn_exit(coord_in: Transform, hdg_change: float, turn_radius: float, level: bool = True) -> Transform:
    """Calculate the exit point of a turn maneuver given initial position, heading, heading change, and turn radius.

    Args:
        coord_in (Transform): Initial position and heading as a Transform object.
        hdg_change (float): Change in heading during the turn, in degrees.
        turn_radius (float): Radius of the turn, in feet.
        level (bool, optional): If True, maintains the initial altitude during the turn. Defaults to True.

    Returns:
        Transform: Transform object representing the exit point's latitude, longitude, altitude (in meters),
        and heading (in degrees).

    """
    # Convert headings to radians
    hdg_out = normalize_heading(coord_in.yaw + hdg_change)
    # Compute the trajectory in the body frame (with roll=0, pitch=0 to get a level turn)
    x = turn_radius * np.sin(np.deg2rad(abs(hdg_change)))
    y = np.sign(hdg_change) * turn_radius * (1 - np.cos(np.deg2rad(hdg_change)))

    # Convert local coordinates to geodetic coordinates
    coord_out = local2geodetic(x, y, 0, coord_in.lat, coord_in.lon, coord_in.alt, 0, 0, coord_in.yaw)
    coord_out.yaw = hdg_out  # Set the yaw to the new heading after the turn
    if level:
        coord_out.alt = coord_in.alt
    return coord_out


def turn_entry(coord_out: Transform, hdg_change: float, turn_radius: float, level: bool = True) -> Transform:
    """Calculate the entry point of a turn maneuver given the exit point, heading change, and turn radius.

    The function uses turn_exit with a opposite heading change to find the entry point.

    Args:
        coord_out (Transform): Exit position and heading as a Transform object.
        hdg_change (float): Change in heading during the turn, in degrees.
        turn_radius (float): Radius of the turn, in feet.
        level (bool, optional): If True, maintains the initial altitude during the turn. Defaults to True.

    Returns:
        Transform: Transform object representing the exit point's latitude, longitude, altitude (in meters),
        and heading (in degrees).

    """
    # going back to the entry point is equivalent to getting to the exit point of a turn with a reversed heading change
    # Calculate the entry point by reversing the heading change
    coord_out.yaw = reciprocal_heading(coord_out.yaw)
    coord_in = turn_exit(coord_out, -hdg_change, turn_radius, level)
    # Set the yaw to the reciprocal heading
    coord_in.yaw = reciprocal_heading(coord_in.yaw)
    return coord_in


def straight_line(coord_in: Transform, TAS: float, time: float, level: bool = False) -> Transform:
    """Calculate the new position of an aircraft after flying straight at a given airspeed for a specified time.

    Args:
        coord_in (Transform): Initial position and orientation of the aircraft.
        TAS (float): True Airspeed in m/s.
        time (float): Time interval in seconds.
        level (bool, optional): If True, maintains constant altitude for a level trajectory. Defaults to False.

    Returns:
        Transform: The updated position and orientation of the aircraft after the specified time.

    """
    # Convert local coordinates to geodetic coordinates
    coord_out = local2geodetic(TAS*time, 0, 0, coord_in.lat, coord_in.lon, coord_in.alt,
                               coord_in.roll, coord_in.pitch, coord_in.yaw)
    if level:
        coord_out.alt = coord_in.alt  # correct for round earth if we want a level trajectory
    return coord_out


def get_TIP(coord_tgt: Transform, TAS_tgt: float, TAS_f: float, load_factor_f: float, HCA: float) -> Transform:
    """Calculate the intercept point for a given HCA (Heading Change Angle).

    Args:
        coord_tgt (Transform): Target coordinates
        TAS_tgt (float): Speed of the target in meters per second
        TAS_f (float): Speed of the fighter in meters per second
        load_factor_f (float): Load factor of the fighter
        HCA (float): Heading Change Angle in degrees

    Returns:
        Transform: Intercept point in geodetic coordinates

    """
    turn_radius_f = turn_radius(TAS_f, load_factor_f)  # Calculate the turn radius of the fighter
    turn_rate_f = turn_rate(TAS_f, load_factor_f)  # Calculate the turn rate of the fighter
    body_TIP_x = TAS_tgt * abs(HCA) / turn_rate_f - np.sign(HCA)*turn_radius_f * np.sin(np.deg2rad(HCA))
    body_TIP_y = np.sign(HCA)*turn_radius_f * (1 - np.cos(np.deg2rad(HCA)))
    body_TIP_z = 0  # Assume the intercept point is at the same altitude as the target
    # Convert local TIP to geodetic coordinates
    coord_TIP = local2geodetic(body_TIP_x, body_TIP_y, body_TIP_z,
                               coord_tgt.lat, coord_tgt.lon, coord_tgt.alt, 0, 0, coord_tgt.yaw)
    coord_TIP.yaw = normalize_turn(HCA)  # Normalize the HCA to a heading
    return coord_TIP


def get_TIPs(TAS_tgt: float, TAS_f: float, load_factor_f: float, nb_points: int, side: str = 'right')-> np.ndarray[Transform]:
    """Calculate the intercept points for a given target speed, fighter speed, load factor, and number of points.

    Args:
        TAS_tgt (float): Speed of the target in meters per second
        TAS_f (float): Speed of the fighter in meters per second
        load_factor_f (float): Load factor of the fighter
        nb_points (int): Number of points to calculate for the intercept trajectory
        side (str): Side to calculate the TIPs for ('left' or 'right')

    Returns:
        np.ndarray: Array of intercept points in the fighter's reference frame (x in front, y to the right)

    """
    turn_radius_f = turn_radius(TAS_f, load_factor_f)  # Calculate the turn radius of the fighter
    turn_rate_f = turn_rate(TAS_f, load_factor_f)  # Calculate the turn rate of the fighter
    if side not in ['right', 'left']:
        raise ValueError("side must be 'right' or 'left'")
    if side == 'right':
        HCA = np.linspace(0, 360, nb_points)
    else:
        HCA = np.linspace(0, -360, nb_points)
    HCA_rad = np.deg2rad(HCA)  # Convert HCA to radians
    TIP_x = TAS_tgt * abs(HCA) / turn_rate_f - np.sign(HCA)*turn_radius_f * np.sin((HCA_rad))
    TIP_y = np.sign(HCA)*turn_radius_f * (1 - np.cos(HCA_rad))
    TIPs = np.stack((TIP_x, TIP_y), axis=1)

    return TIPs


def get_HCA(coord_fighter: Transform, coord_target: Transform) -> float:
    """Calculate the Heading Change Angle (HCA) required for a fighter to turn parallel to a target.

    The function computes the difference in yaw (heading) between the fighter and the target,
    determines the direction of turn (left or right), and adjusts the angle if the turn is away
    from the target to ensure the shortest path is taken.

    Args:
        coord_fighter (Transform): The current position and orientation of the fighter.
        coord_target (Transform): The position and orientation of the target.

    Returns:
        float: The heading change angle (in degrees) required for the fighter to turn parallel to the target.

    """
    heading = normalize_heading(coord_fighter.yaw)
    vector_t_ned = pm3d.aer2ned(heading, 0, 1000)
    vector_f_ned = pm3d.geodetic2ned(
        coord_target.lat, coord_target.lon, coord_target.alt,
        coord_fighter.lat, coord_fighter.lon, coord_fighter.alt
    )
    crossProd = np.cross(vector_f_ned[:2], vector_t_ned[:2])
    HCA = normalize_turn(coord_target.yaw - coord_fighter.yaw)
    if HCA * crossProd > 0:  # we are turning away from the target
        # get the angle complementary to 360 (ex : instead of turning left 90, we need to turn right 270)
        HCA = HCA - np.sign(HCA) * 360
    # print(f"Final HCA: {HCA:.2f}°")
    return HCA


def constrained_HCA(HCA: float, coord_tgt_1: Transform, coord_f_1: Transform, TAS_tgt: float, TAS_f: float,
                    load_factor_f: float) -> float:
    """Calculate the error between the desired Heading Change Angle (HCA) and the actual angle at the collision point.

    This function is used as cost function for HCA optimization to make sure than the heading of the fighter when it
    reaches the collision point is as close as possible to the HCA associated to the targeted TIP.

    Args:
        HCA (float): Heading Change Angle in degrees.
        coord_tgt_1 (Transform): Target's initial coordinate and orientation.
        coord_f_1 (Transform): Follower's initial coordinate and orientation.
        TAS_tgt (float): True Airspeed of the target.
        TAS_f (float): True Airspeed of the fighter.
        load_factor_f (float): Load factor for the fighter.

    Returns:
        float: The absolute error between the collision angle and the HCA.

    """
    coord_TIP_1 = get_TIP(coord_tgt_1, TAS_tgt, TAS_f, load_factor_f, HCA)
    coord_TIP_1.yaw = coord_tgt_1.yaw
    _, CATA, _ = collision_point(coord_TIP_1, TAS_tgt, coord_f_1, TAS_f)
    error = abs(normalize_angle(CATA - coord_f_1.yaw) - HCA)
    return error


def two_stage_HCA_optimizer(coord_tgt_1: Transform, coord_f_1: Transform, TAS_tgt: float, TAS_f: float,
                            load_factor_f: float, debug: bool = False) -> float:
    """Optimize the Heading Change Angle (HCA) using a two-stage search process.

    This function performs a coarse search over the whole range of HCA values (-360 to 360), followed by a fine search
    around the best coarse result to minimize heading error. It uses the `constrained_HCA` function
    to evaluate the error for each candidate angle.

    Args:
        coord_tgt_1 (Transform): Target coordinates for stage 1.
        coord_f_1 (Transform): Reference coordinates for stage 1.
        TAS_tgt (float): True Airspeed of the target.
        TAS_f (float): True Airspeed of the reference.
        load_factor_f (float): Load factor for the reference.
        debug (bool, optional): If True, prints debug information during optimization. Defaults to False.

    Returns:
        float: The optimized Heading Change Angle (HCA) in degrees.

    """
    # --- Stage 1: Coarse search every 5 degrees
    coarse_HCA = np.arange(-355, 355, 5)
    best_error = 360
    for angle in coarse_HCA:
        error = constrained_HCA(angle, coord_tgt_1, coord_f_1, TAS_tgt, TAS_f, load_factor_f)
        if error < best_error:
            best_error = error
            best_HCA = angle
            if debug:
                print(f"🔍 Coarse search improved: HCA={best_HCA:.2f}°, heading error={error:.2f}°")
    if debug:
        print(f"🔍 Best coarse result: HCA={best_HCA:.2f}°, heading error={error:.2f}°")

    # --- Stage 2: Fine search ±5° around best_angle
    bounds = (best_HCA - 5, best_HCA + 5)
    result = minimize_scalar(
        constrained_HCA,
        args=(coord_tgt_1, coord_f_1, TAS_tgt, TAS_f, load_factor_f),
        bounds=bounds,
        method='bounded',
        options={'xatol': 0.1}
    )
    if debug:
        print(f"Result:{result}")
        print(f"✅ Refined result: Best HCA={result.x:.2f}°, error={error:.4f}s")

    return result.x


#####################    ARCHIVES    ############################
# def get_TIPs(coord_tgt: Transform, TAS_tgt: float, TAS_f: float, load_factor_f: float) -> list[Transform]:
#     """Calculate a list of Target Intercept Points (TIPs) for a fighter aircraft to intercept a target.

#     For a range of Heading Change Angles (HCA), this function computes the intercept points in the target's reference
#     frame, converts them to geodetic coordinates, and returns them as Transform objects.

#     Args:
#         coord_tgt (Transform): The geodetic position and orientation of the target (latitude, longitude, altitude, yaw).
#         TAS_tgt (float): True Airspeed of the target (meters per second).
#         TAS_f (float): True Airspeed of the fighter (meters per second).
#         load_factor_f (float): Load factor (g) for the fighter, used to calculate turn radius and rate.

#     Returns:
#         list[Transform]: A sorted list of Transform objects representing the intercept points, ordered by yaw angle.

#     """
#     turn_radius_f = turn_radius(TAS_f, load_factor_f)  # Calculate the turn radius of the fighter
#     turn_rate_f = turn_rate(TAS_f, load_factor_f)  # Calculate the turn rate of the fighter
#     TIPs = []  # Initialize an empty array to store the intercept points

#     for i in range(1, 36):
#         HCA = i * 10
#         # Calculate the intercept point for each HCA in the tgt reference frame (x in front, y to the right,z down)
#         body_TIP_x = TAS_tgt * abs(HCA) / turn_rate_f - turn_radius_f * np.sin(np.deg2rad(abs(HCA)))
#         body_TIP_y = np.sign(HCA) * turn_radius_f * (1 - np.cos(np.deg2rad(HCA)))
#         body_TIP_z = 0  # Assuming the intercept point is at the same altitude as the target

#         # Convert local TIP to geodetic coordinates
#         lat_TIP, lon_TIP, alt_TIP = local2geodetic(body_TIP_x, body_TIP_y, body_TIP_z, coord_tgt.lat,
#                                                    coord_tgt.lon, coord_tgt.alt, 0, 0, coord_tgt.yaw)
#         # Create a Transform object for the intercept point
#         TIPs.append(Transform(lat=lat_TIP, lon=lon_TIP, alt=alt_TIP, roll=0, pitch=0, yaw=normalize_turn(HCA)))

#         # Convert local TIP to geodetic coordinates
#         lat_TIP, lon_TIP, alt_TIP = local2geodetic(body_TIP_x, -body_TIP_y, body_TIP_z,
#                                                    coord_tgt.lat, coord_tgt.lon, coord_tgt.alt, 0, 0, coord_tgt.yaw)
#         # Create a Transform object for the intercept point
#         TIPs.append(Transform(lat=lat_TIP, lon=lon_TIP, alt=alt_TIP, roll=0, pitch=0, yaw=normalize_turn(-HCA)))

#     return sorted(TIPs, key=lambda tip: tip.yaw)

# def intercept_traj(coord_tgt: Transform, coord_f: Transform, TAS_tgt: float, TAS_f: float, load_factor_f: float,
#                    init_turn: float, debug: bool = False) -> tuple[float, float, float, float, float, Transform]:
#     """Calculate the intercept trajectory for a fighter aircraft to intercept a target aircraft.

#     The function models a two-turn intercept maneuver:
#     1. The fighter performs an initial turn.
#     2. The fighter flies straight to a collision point with the Target Interception Point (TIP).
#     3. The fighter performs a second turn to complete the intercept.

#     Args:
#         coord_tgt (Transform): Initial position and orientation of the target aircraft.
#         coord_f (Transform): Initial position and orientation of the fighter aircraft.
#         TAS_tgt (float): True airspeed of the target aircraft (meters/second).
#         TAS_f (float): True airspeed of the fighter aircraft (meters/second).
#         load_factor_f (float): Load factor (g) for the fighter's turn.
#         init_turn (float): Initial turn angle for the fighter (degrees).
#         debug (bool, optional): If True, prints debug information. Defaults to False.

#     Returns:
#         tuple: A tuple containing:
#             - total_time (float): Total time required for the intercept trajectory (seconds).
#             - CATA (float): Collision angle to attack (degrees).
#             - time_1 (float): Time to complete the initial turn (seconds).
#             - time_2 (float): Time to reach the collision point after the initial turn (seconds).
#             - time_3 (float): Time to complete the second turn (seconds).
#             - coord_coll (Transform): Position and orientation at the collision/intercept point.

#     """
#     turn_radius_f = turn_radius(TAS_f, load_factor_f)  # Calculate the turn radius of the fighter
#     turn_rate_f = turn_rate(TAS_f, load_factor_f)  # Calculate the turn rate of the fighter

#     # time 1 is the time when the fighter exits the first turn.
#     # time 2 is the time when the fighter starts the second turn.
#     coord_f_1 = turn_exit(coord_f, init_turn, turn_radius_f)  # Calculate the exit point of the first turn
#     time_1 = abs(init_turn) / turn_rate_f  # Calculate the time to complete the first turn
#     coord_tgt_1 = straight_line(coord_tgt, TAS_tgt, time_1)  # Calculate the target position at t1

#     # Calculate the collision point at t1
#     coord_coll, CATA, time_2 = collision_point(coord_tgt_1, TAS_tgt, coord_f_1, TAS_f)
#     # print(f"CATA init: {CATA:.1f}°, Time 2: {time_2:.2f} seconds")  # Print the initial CATA and time to intercept
#     coord_f_2 = straight_line(coord_f_1, TAS_f, time_2)
#     HCA = get_HCA(coord_f_1, coord_tgt_1)
#     # HCA= normalize_turn(coord_tgt.yaw - CATA) ## TO BE UPDATED must be fighter 2 with target
#     if debug:
#         print(f'CATA: {CATA:.1f}°, TGT HDG:{coord_tgt.yaw:.1f}°, HCA: {HCA:.1f}°')
#     i = 0
#     error = 10
#     while (error > 0.1 and i < 3):
#         # Calculate the intercept point for the second turn
#         coord_TIP_1 = get_TIP(coord_tgt_1, TAS_tgt, TAS_f, load_factor_f, HCA)
#         coord_TIP_1.yaw = coord_tgt.yaw  # Set the yaw of the intercept point to the target yaw
#         coord_TIP_2 = straight_line(coord_tgt, TAS_tgt, time_2)  # Calculate the target position at t1
#         HCA = get_HCA(coord_f_2, coord_TIP_2)  # TO BE UPDATED must be fighter 2 with target
#         coord_coll, CATA, time_2 = collision_point(coord_TIP_1, TAS_tgt, coord_f_1, TAS_f)
#         error = abs(normalize_angle(HCA - coord_tgt.yaw + CATA))
#         # print(f'CATA: {CATA:.1f}°, HCA: {HCA:.1f}°, error:{error:.2f}°, Time 2: {time_2:.2f} seconds')
#         i += 1
#     if i == 3 and debug:
#         print("Warning: Maximum iterations reached, the intercept trajectory may not be accurate.")

#     time_3 = abs(HCA) / turn_rate_f  # Calculate the time to complete the second turn
#     total_time = time_1 + time_2 + time_3  # Total time for the intercept trajectory
#     return total_time, CATA, time_1, time_2, time_3, coord_coll


# def intercept_traj(coord_tgt, coord_f, TAS_tgt, TAS_f, load_factor_f, init_turn, debug=False):
#     turn_radius_f = turn_radius(TAS_f, load_factor_f)  # Calculate the turn radius of the fighter
#     turn_rate_f = turn_rate(TAS_f, load_factor_f)  # Calculate the turn rate of the fighter

#     coord_f_1 = turn_exit(coord_f, init_turn, turn_radius_f)  # Calculate the exit point of the first turn
#     time_1 = abs(init_turn) / turn_rate_f  # Calculate the time to complete the first turn
#     coord_tgt_1 = straight_line(coord_tgt, TAS_tgt, time_1)  # Calculate the target position at t1
#     # print(f"Time to first turn exit: {time_1:.2f} seconds")
#     # Calculate the collision point at t1
#     coord_coll, CATA, time_2 = collision_point(coord_tgt_1, TAS_tgt, coord_f_1, TAS_f)
#     HCA = two_stage_HCA_optimizer(coord_tgt_1, coord_f_1, TAS_tgt, TAS_f, load_factor_f, debug)
#     print(f"HCA:{HCA}")
#     # Calculate the intercept point for the second turn
#     coord_TIP_1 = get_TIP(coord_tgt_1, TAS_tgt, TAS_f, load_factor_f, HCA)
#     coord_TIP_1.yaw = coord_tgt.yaw  # Set the yaw of the intercept point to the target yaw
#     coord_coll, CATA, time_2 = collision_point(coord_TIP_1, TAS_tgt, coord_f_1, TAS_f)

#     time_3 = abs(HCA)/turn_rate_f  # Calculate the time to complete the second turn
#     total_time = time_1 + time_2 + time_3  # Total time for the intercept trajectory
#     return total_time, CATA, time_1, time_2, time_3, coord_coll
