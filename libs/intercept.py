"""Calculate the optimal intercept trajectory for a fighter aircraft to intercept a moving target.

This function determines the tangent headings and intersection points with the Target Intercept Polygon (TIP),

Workflow:
    - Computes the relative position of the fighter to the target in the target's body frame.
    - For both left and right 360-degree turns, casts rays in the direction of the fighter's relative motion.
    - Checks ray intersections with the TIP polygon to find possible intercepts.
    - Optimizes turn angles to be tangent to the TIP polygon and selects the fastest intercept path.
    - Generates trajectories for both the fighter and the target based on optimized turn angles and predicted paths.
    - Calculates the horizontal error between fighter and target at intercept (mainly due to flat earth assumption).
    - Optionally plots the solution and exports to Tacview format.

"""
import numpy as np
import plotly.graph_objects as go
import pymap3d as pm3d  # Importing pymap3d for geodetic to ECEF conversions
from numpy.linalg import norm
from scipy.optimize import minimize_scalar

from . import (
    geo,
    tacview,
)
from . import trajectory as traj
from .trajectory import Trajectory  # Importing the Trajectory class for trajectory handling
from .transform import Transform  # Importing the Transform class for coordinate transformations


def intercept(coord_f: Transform, coord_tgt: Transform, TAS_f: float, TAS_tgt: float, load_factor_f: float,  # noqa: C901
              plots: bool = True, debug: bool = False) -> tuple[Trajectory,Trajectory,float]:
    """Calculate the optimal intercept trajectory for a fighter aircraft to intercept a moving target.

    The function determines the tangent headings and intersection points with the Target Intercept Polygon (TIP),
    refines the solution, and generates the corresponding fighter and target trajectories. It can optionally plot the
    solution and export to Tacview format.

    The function takes the following steps :
        - In the body frame of the target, the relative position of the fighter is calculated using rel_pos.
        - For both a left and right 360 turn, rays are cast in the direction of the relative motion of the fighter.
        ray_polygon_intersection_np checks whether the ray intercepts with the polygon formed by the intercept points.
        - Then turn angles are optimized to be tangent to the intercept points and the fastest path to the target is
        selected.
        - Trajectory for both the fighter and the target is generated based on the optimized turn angles and the
        predicted paths.
        - Finally, the horizontal error between the fighter and target at intercept is calculated. Horizontal error is
        mostly due to the flat earth assumption.
        - Optionally, the solution can be plotted and exported to Tacview format.

    Args:
        coord_f (Transform): Initial position and orientation of the fighter (latitude, longitude, altitude, yaw).
        coord_tgt (Transform): Initial position and orientation of the target (latitude, longitude, altitude, yaw).
        TAS_f (float): True Airspeed of the fighter (meters/second).
        TAS_tgt (float): True Airspeed of the target (meters/second).
        load_factor_f (float): Load factor (g) for the fighter's turn performance.
        plots (bool, optional): If True, plots the solution and trajectories. Defaults to True.
        debug (bool, optional): If True, prints debug information during computation. Defaults to False.

    Returns:
        tuple[Trajectory, Trajectory, float]:
            - Trajectory: Fighter's trajectory as a sequence of coordinates and timestamps.
            - Trajectory: Target's trajectory as a sequence of coordinates and timestamps.
            - float: Final horizontal error (distance in meters) between fighter and target at intercept.

    Notes:
        - If no valid trajectory is found, returns np.nan.
        - When fighter and target are too far away, one degree of heading change may not be sufficient to achieve
        intercept. Try increasing the number of steps for the coarse search (nb_steps).

    """
    def rel_pos(coord_f: Transform, coord_tgt: Transform) -> np.ndarray:
        """Calculate the relative position of a fighter aircraft to a target in the target's body frame.

        Args:
            coord_f (Transform): Transform object representing the fighter's position and orientation.
            coord_tgt (Transform): Transform object representing the target's position and orientation.

        Returns:
            numpy.ndarray: Relative position vector [x, y, z] in meters, expressed in the target's body frame.
            x: Forward/backward position (positive is forward)
            y: Left/right position (positive is right)
            z: Up/down position (positive is down)

        """
        az, el, dist = pm3d.geodetic2aer(coord_f.lat, coord_f.lon, coord_f.alt, coord_tgt.lat, coord_tgt.lon, coord_tgt.alt)
        el=0  # Assuming iso altitude for simplicity, can be adjusted if needed

        # Adjust azimuth into target's body frame (relative bearing)
        rel_bearing = geo.normalize_angle(az - coord_tgt.yaw)
        # print(f"Relative Bearing: {rel_bearing} degrees, Elevation: {el} degrees, Distance: {dist} m")

        # Convert to radians
        rb_rad = np.deg2rad(rel_bearing)
        el_rad = np.deg2rad(el)

        # Compute relative position in target's body frame
        x = dist * np.cos(el_rad) * np.cos(rb_rad)
        y = dist * np.cos(el_rad) * np.sin(rb_rad)
        z = -dist * np.sin(el_rad)  # Down is positive in NED
        return np.array([x, y, z])

    def turn_exit(rel_pos: np.ndarray, hdg_f: float, TAS_f: float, load_factor_f: float, hdg_tgt: float, TAS_tgt: float,
                  turn_angle: float) -> tuple[np.ndarray, float]:
        """Calculate the exit position and heading of a fighter aircraft after performing a turn in the target body frame.

        Args:
            rel_pos (np.ndarray): Relative position of the fighter in the target's body frame.
            hdg_f (float): Heading of the fighter in degrees.
            TAS_f (float): True airspeed of the fighter in m/s.
            load_factor_f (float): Load factor of the fighter (G-forces).
            hdg_tgt (float): Heading of the target in degrees.
            TAS_tgt (float): True airspeed of the target in m/s.
            turn_angle (float): Turn angle of the fighter in degrees.

        Returns:
            tuple[np.ndarray, float]: Exit position and exit heading of the fighter.
            exit_pos is in meters in the target body frame.

        """
        # Calculate turn radius and rate
        turn_radius_f = geo.turn_radius(TAS_f, load_factor_f)
        turn_rate_f = geo.turn_rate(TAS_f, load_factor_f)

        # Calculate the fighter heading relative to the target
        rel_bearing=geo.normalize_angle(hdg_f - hdg_tgt)

        # Calculate delta_x and delta_y in the fighter's turn frame
        delta_x = turn_radius_f * np.sin(np.deg2rad(abs(turn_angle)))
        delta_y = np.sign(turn_angle) * turn_radius_f * (1 - np.cos(np.deg2rad(turn_angle)))
        delta_pos= np.array([delta_x, delta_y, 0])  # Delta position in the fighter's turn frame

        Rot_matrix=geo.rotation_matrix(0,0,rel_bearing)  # Rotation matrix to rotate delta position into the target's body frame
        delta_pos_tgt = Rot_matrix @ delta_pos  # Rotate delta position into the target's body frame
        offset_tgt_speed = TAS_tgt*abs(turn_angle)/turn_rate_f * np.array([-1,0,0])
        exit_pos = rel_pos + delta_pos_tgt+offset_tgt_speed  # Exit position in the target's body frame
        exit_rel_bearing = geo.normalize_heading(rel_bearing + turn_angle)  # Exit heading in degrees
        return exit_pos,exit_rel_bearing

    def ray_polygon_intersection_np(rel_pos:np.ndarray, direction: np.ndarray, polygon: np.ndarray)-> np.ndarray | None:
        """Calculate the intersection point of a ray with a polygon.

        This function determines where a ray, defined by an origin and direction, intersects the edges of a polygon in
        2D space.
        It returns the closest intersection point in the direction of the ray, or None if there is no valid intersection.

        Args:
            rel_pos (np.ndarray): Relative position of the ray origin in the target's body frame (x, y, z).
            direction (np.ndarray): Direction of the ray as a unit vector (dx, dy).
            polygon (np.ndarray): Polygon vertices as a numpy array of shape (N, 2), where N is the number of vertices.

        Returns:
            np.ndarray | None: Intersection point as a numpy array (x, y), or None if no intersection is found or if the
            ray origin is inside the polygon.

        Notes:
            - Only 2D intersection is considered (altitude is ignored).
            - If the ray origin is inside the polygon (odd number of intersections), None is returned.

        """
        xa, ya,_= rel_pos  # Ray origin, don't take into account altitude for 2D intersection
        dx, dy = direction

        # Segment start and end points
        p1 = polygon
        p2 = np.roll(polygon, -1, axis=0)

        # Segment vectors
        ex = p2[:, 0] - p1[:, 0]
        ey = p2[:, 1] - p1[:, 1]

        # Determinant (2D cross product)
        det = dx * ey - dy * ex
        parallel = np.abs(det) < 1e-10  # Treat near-zero determinant as parallel

        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((p1[:, 0] - xa) * ey - (p1[:, 1] - ya) * ex) / det
            u = ((p1[:, 0] - xa) * dy - (p1[:, 1] - ya) * dx) / det

        # Filter valid intersections: forward along ray (t ≥ 0) and within segment (0 ≤ u ≤ 1)
        mask = (~parallel) & (t >= 0) & (u >= 0) & (u <= 1)
        # hdg = np.rad2deg(np.arctan2(dy, dx))  # Calculate the angle of the ray direction


        if not np.any(mask):
            # print(f"Direction:{hdg:.2f}, No intersections found.")
            return None

        t_valid = t[mask]
        # u_valid = u[mask]
        # print(f"Direction:{hdg:.2f}, Valid intersections found: {np.sum(mask)}")
        # print("t_valid",t_valid)
        # print("u_valid", u_valid)
        t_min_index = np.argmin(t_valid)
        t_closest = t_valid[t_min_index]
        if np.sum(mask) % 2 == 1:  # If there is only one valid intersection
            # fighter is inside TIP
            return None

        # Compute intersection point using parametric equation of the ray
        intersection_point = np.array([xa, ya]) + t_closest * direction
        return intersection_point

    def find_tangent_headings_refined(coord_f: Transform, coord_tgt: Transform, TIPs_L: np.ndarray, TIPs_R: np.ndarray,  # noqa: C901
                                      TAS_f: float, TAS_tgt: float, load_factor_f: float, turn: str = 'right',
                                      coarse_steps: int = 360, refine_window: float = 1, tol: float = 1e-8) -> \
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find two tangent headings for a fighter aircraft to intercept a target polygon (TIPs).

        This function performs a coarse scan of possible turn angles, identifies the most separated pair of tangent
        headings that intersect the target polygon, and then refines each heading to achieve tangency with the polygon
        edge. It calculates the positions, intersection points, times, and Head-on Crossing Angles (HCA)
        for each tangent heading.

        Parameters
        ----------
        coord_f : Transform
            The fighter's position and orientation.
        coord_tgt : Transform
            The target's position and orientation.
        TIPs_L : np.ndarray
            Nx3 array of polygon vertices representing the left TIPs in the target's body frame.
        TIPs_R : np.ndarray
            Nx3 array of polygon vertices representing the right TIPs in the target's body frame.
        TAS_f : float
            True airspeed of the fighter (m/s).
        TAS_tgt : float
            True airspeed of the target (m/s).
        load_factor_f : float
            Load factor (g) for the fighter's turn.
        turn : str, optional
            Direction of turn, either 'right' or 'left'. Default is 'right'.
        coarse_steps : int, optional
            Number of coarse scan steps for turn angles. Default is 360.
        refine_window : float, optional
            Window (degrees) around coarse heading for refinement. Default is 1.
        tol : float, optional
            Tolerance for heading refinement optimization. Default is 1e-8.

        Returns
        -------
        turns : np.ndarray
            Array of two refined tangent turn angles (degrees).
        points_1 : np.ndarray
            Array of two positions (x, y) after the initial turn.
        points_2 : np.ndarray
            Array of two intersection points (x, y) with the polygon.
        times : np.ndarray
            Array of shape (2, 4) with [total_time, time1, time2, time3] for each heading.
        all_rel_pos_out : np.ndarray
            Array of all relative positions after initial turns (for coarse scan).
        intersections : np.ndarray
            Array of intersection points found during coarse scan.
        pos_out_inter : np.ndarray
            Array of positions after initial turn for each intersection found.
        HCAs : np.ndarray
            Array of two Head-on Crossing Angles (degrees) for each tangent heading.

        Notes
        -----
        - If no valid tangent headings are found, returns arrays filled with NaN or empty arrays.
        - The function uses geometric calculations and optimization to ensure tangency and timing accuracy.
        - Requires supporting functions: rel_pos, geo.turn_rate, turn_exit, geo.normalize_heading, geo.normalize_angle, ray_polygon_intersection_np, norm.

        """
        rel_pos_init = rel_pos(coord_f, coord_tgt)
        turn_rate_f = geo.turn_rate(TAS_f, load_factor_f)  # Turn rate in degrees per second

        def intersection_error(turn_angle: float) -> float:
            """Cost function for the optimization. Cost is the angle difference between the ray and the TIP segment."""
            rel_pos_out, rel_bearing_out = turn_exit(rel_pos_init, coord_f.yaw, TAS_f, load_factor_f, coord_tgt.yaw, TAS_tgt, turn_angle)

            # Relative exit speed in target frame
            exit_rel_speed = np.array([
                TAS_f * np.cos(np.deg2rad(rel_bearing_out)) - TAS_tgt,
                TAS_f * np.sin(np.deg2rad(rel_bearing_out))
            ])
            ray_dir = exit_rel_speed / norm(exit_rel_speed)  # Unit vector

            # Choose left or right TIPs based on relative position
            TIPs = TIPs_L if rel_pos_out[1] < 0 else TIPs_R

            xa, ya, _ = rel_pos_out
            dx, dy = ray_dir

            # Segment start and end
            p1 = TIPs
            p2 = np.roll(TIPs, -1, axis=0)
            ex = p2[:, 0] - p1[:, 0]
            ey = p2[:, 1] - p1[:, 1]

            # Compute determinant to check parallelism
            det = dx * ey - dy * ex
            parallel = np.abs(det) < 1e-10

            with np.errstate(divide='ignore', invalid='ignore'):
                t = ((p1[:, 0] - xa) * ey - (p1[:, 1] - ya) * ex) / det
                u = ((p1[:, 0] - xa) * dy - (p1[:, 1] - ya) * dx) / det

            # Find intersected segment
            mask = (~parallel) & (t >= 0) & (u >= 0) & (u <= 1)

            if not np.any(mask):
                return 1e6  # Penalize rays that miss the polygon

            # Get the closest intersected segment
            t_valid = t[mask]
            idx_valid = np.where(mask)[0]
            closest_idx = idx_valid[np.argmin(t_valid)]

            # Vector of the intersected segment
            seg_vec = np.array([ex[closest_idx], ey[closest_idx]])
            seg_vec /= norm(seg_vec)

            # Tangency error: angle between ray and segment
            cos_theta = np.clip(np.dot(ray_dir, seg_vec), -1.0, 1.0)
            theta= 180-np.rad2deg(np.arccos(cos_theta))  # Angle in degrees

            if debug:
                print(f"Turn angle: {turn_angle:.2f}°, cos theta: {cos_theta:.2f}, theta:{theta}°")
            return theta   # Optional: np.degrees(angle_error) if you prefer degrees

        # Step 1: Coarse scan
        coarse_headings = []
        if turn == 'right':
            turn_angles = np.linspace(0, 360, coarse_steps, endpoint=False)
            # turn_angles = np.linspace(160, 190, 60, endpoint=False)
        else:
            turn_angles = np.linspace(0, -360, coarse_steps, endpoint=False)
        all_rel_pos_out=[]
        intersections = []
        pos_out_inter=[]


        for turn_angle in turn_angles:

            rel_pos_out, rel_bearing_out = turn_exit(rel_pos_init, coord_f.yaw, TAS_f, load_factor_f,coord_tgt.yaw, TAS_tgt, turn_angle)
            all_rel_pos_out.append(rel_pos_out)
            if rel_pos_out[1] < 0:
                TIPs=TIPs_L  # Use left TIPs if the fighter is left of the target
            else:
                TIPs=TIPs_R
            exit_rel_speed=np.array([TAS_f * np.cos(np.deg2rad(rel_bearing_out))-TAS_tgt, TAS_f * np.sin(np.deg2rad(rel_bearing_out))])  # Exit speed in the target's body frame
            ray_dir = exit_rel_speed/norm(exit_rel_speed)  # Normalize the ray direction
            ray_dir_angle= geo.normalize_heading(np.rad2deg(np.arctan2(ray_dir[1], ray_dir[0])))  # Calculate the angle of the ray direction
            intersection = ray_polygon_intersection_np(rel_pos_out, ray_dir, TIPs)
            if intersection is None:
                # if debug:
                    # print(f"Turn:{turn_angle:.2f}°, Heading:{rel_bearing_out:.2f}°\tray direction: {ray_dir_angle:.2f}\tNo intersection found")
                continue
            else:
                intersections.append(intersection)  #save for plotting later
                pos_out_inter.append(rel_pos_out[:2])  # Save the position after the initial turn
                coarse_headings.append(turn_angle)  # Store the ray direction angle
                if debug:
                    print(f"Turn:{turn_angle:.2f}°, Heading:{rel_bearing_out:.2f}°\tray direction: {ray_dir_angle:.2f}\tIntersection found at {intersection}, Distance: {np.min(norm(TIPs - intersection, axis=1)):.2f} m")

        if len(coarse_headings) == 0:
            # print(f"Fighter heading:{coord_f.yaw:.2f}°\tTarget heading:{coord_tgt.yaw:.2f}°\tNo valid headings found in coarse scan.")
            return np.array([]), np.empty((0, 2)),np.empty((0, 2)),np.empty((0,4)),np.array(all_rel_pos_out), np.empty((0,2)),np.empty((0,2)),np.array([])
        elif len(coarse_headings) == 1:
            best_pair = (coarse_headings[0], coarse_headings[0])  # If only one heading, use it for both
            # print(f"Fighter heading:{coord_f.yaw:.2f}°\tTarget heading:{coord_tgt.yaw:.2f}°\tOnly one valid heading found in coarse scan.")
            return np.array([]), np.empty((0, 2)),np.empty((0, 2)),np.empty((0,4)),np.array(all_rel_pos_out), np.array(intersections),np.array(pos_out_inter),np.array([])  # No valid pair found
        else:
            # Step 2: Pick the most separated pair
            coarse_headings = np.unique(np.round(coarse_headings, 4))
            max_sep = 0
            best_pair = (None, None)

            for i in range(len(coarse_headings)):
                for j in range(i + 1, len(coarse_headings)):
                    h1, h2 = coarse_headings[i], coarse_headings[j]
                    sep = np.abs((h1 - h2 + 180) % 360 - 180)
                    if sep > max_sep:
                        max_sep = sep
                        best_pair = (h1, h2)

            if debug:
                print(f"🔎Best pair of headings: {best_pair}, Max separation: {max_sep:.2f} °")
            if best_pair[0] is None:
                print("Not possible to pair.")
                return np.array([]), np.empty((0, 2)),np.empty((0, 2)),np.empty((0,4)),np.array(all_rel_pos_out), np.array(intersections),np.array(pos_out_inter),np.array([])  # No valid pair found

        # Step 3: Refine both tangent headings
        turns = np.zeros(2)
        points_2 = np.zeros((2, 2)) #point at the TIP
        points_1 = np.zeros((2, 2)) #point after the initial turn
        times= np.zeros((2,4))  # Initialize times array
        HCAs = np.zeros(2)  # Initialize HCA array

        for idx, h in enumerate(best_pair):
            result = minimize_scalar(
                intersection_error,
                bracket=(h - refine_window, h + refine_window),
                method='brent',
                options={'xtol': tol}
            )
            if not result.success or (result.success and np.abs(result.fun) > 5): #if after optimization ray is not tangent (>2° angle) invalidate the result
                if debug:
                    print(f"Refinement failed for heading {h} with result: {result.message} angle was {result.fun}")
                turns[idx] = np.nan
                points_1[idx] = np.nan
                points_2[idx] = np.nan
                times[idx] = np.nan
                HCAs[idx] = np.nan
                continue
            if debug:
                print("Result:",result)
            turns[idx] = result.x

            # Compute intersection point again using refined heading
            rel_pos_init = rel_pos(coord_f, coord_tgt)
            rel_pos_out, rel_bearing_out = turn_exit(rel_pos_init, coord_f.yaw, TAS_f, load_factor_f, coord_tgt.yaw, TAS_tgt, result.x)
            if rel_pos_out[1] < 0:
                TIPs=TIPs_L  # Use left TIPs if the fighter is left of the target
            else:
                TIPs=TIPs_R
            exit_rel_speed=np.array([TAS_f * np.cos(np.deg2rad(rel_bearing_out))-TAS_tgt, TAS_f * np.sin(np.deg2rad(rel_bearing_out))])  # Exit speed in the target's body frame
            ray_dir = exit_rel_speed/norm(exit_rel_speed)
            intersection = ray_polygon_intersection_np(rel_pos_out, ray_dir, TIPs)
            time1=abs(result.x)/turn_rate_f #time to perform the initial turn
            time2 = norm(intersection - rel_pos_out[:2])/norm(exit_rel_speed) if intersection is not None else np.nan #time to reach the intersection point

            #get HCA
            HCA=geo.normalize_angle(0 - rel_bearing_out)
            if debug:
                print(f"Rel hdg: {rel_bearing_out:.2f}°, HCA: {HCA:.2f}°, Rel_pos_out: {rel_pos_out[1]:.2f}, sin f bearing: {np.sin(np.deg2rad(rel_bearing_out)):.2f}°, Ray dir: {ray_dir[1]:.2f}")

            # if rel_pos_out[1]*ray_dir[1] > 0:  # If the relative position and ray direction are in the same direction
            if rel_pos_out[1]*ray_dir[1] > 0:  # If the relative position and ray direction are in the same direction
                HCA = HCA - np.sign(HCA)*360
            if debug:
                print(f"HCA step 3: {HCA:.2f}°")

            time3 =  abs(HCA)/turn_rate_f# time to turn from TIP to target
            # print(f"Refined heading: {result.x:.2f} degrees, HCA: {HCA:.2f}°, Time1: {time1:.2f} s, Time2: {time2:.2f} s, Time3: {time3:.2f} s")
            total_time = time1 + time2 + time3

            if intersection is not None:
                points_1[idx] = rel_pos_out[:2]  # Store the position after the initial turn
                points_2[idx] = intersection
                times[idx] = [total_time, time1, time2, time3]
                HCAs[idx] = HCA
            else:
                points_1[idx] = np.nan
                points_2[idx] = np.nan
                times[idx] = [np.nan, np.nan, np.nan, np.nan]
                HCAs[idx] = np.nan

        # print(f"size intersections: {len(intersections)}")
        return turns, points_1,points_2, times,np.array(all_rel_pos_out), np.array(intersections),np.array(pos_out_inter),HCAs  # Return headings, intersection points, and all relative positions

    def get_trajectory(coord_f: Transform, TAS_f: float, coord_tgt: Transform, TAS_tgt:float, init_turn: float,
                       times: np.ndarray, HCA: float, level: bool = True) -> tuple[Trajectory, Trajectory]:
        """Generate the trajectories for the interception.

        The fighter's trajectory consists of three segments:
            1. Initial turn from starting position.
            2. Straight-line flight after the turn.
            3. Final turn to intercept the target.

        The target's trajectory is a straight-line path from its starting position.

        Args:
            coord_f (Transform): Initial coordinate and orientation of the fighter.
            TAS_f (float): True airspeed of the fighter (meters/second).
            coord_tgt (Transform): Initial coordinate and orientation of the target.
            TAS_tgt (float): True airspeed of the target (meters/second).
            init_turn (float): Initial turn angle for the fighter (radians).
            times (np.ndarray): Array containing [total_time, time_1, time_2, time_3] for maneuver segments (seconds).
            HCA (float): Heading change angle for the final turn (radians).
            level (bool, optional): If True, assumes level flight for all segments. Defaults to True.

        Returns:
            tuple[Trajectory, Trajectory]:
                - Trajectory of the fighter (merged from all maneuver segments).
                - Trajectory of the target (straight-line path).

        """
        turn_radius_f = geo.turn_radius(TAS_f, load_factor_f)  # Calculate turn radius in meters
        #initialize coordinates arrays
        coords_f1=[]
        coords_f2=[]
        coords_f3=[]
        coords_tgt=[]
        total_time,time_1, time_2, time_3 = times  # Initialize times
        #Generate time vector
        times = np.arange(0, total_time, 1)  # Time vector in seconds
        times_tgt=np.append(times, [total_time])  # Add the final point
        #Generate target trajectory
        for t in times_tgt:
            coords_tgt.append(geo.straight_line(coord_tgt, TAS_tgt, t, level=level))  # Target trajectory
        traj_tgt = Trajectory(np.array(coords_tgt),times_tgt)
        traj_tgt.params= {'Name': 'Kh-101', 'Color': 'Red', 'Type': 'Air+Missile', 'Pilot': 'Target'},  # Orange for target
        traj_tgt.ID=101

        #Generate fighter trajectory
        #1st segment: Initial turn
        times_f1=np.append(np.arange(0, time_1, 1),[time_1])  # Time vector for the first turn
        for i in range(0, int(time_1) + 1):
            coords_f1.append(geo.turn_exit(coord_f, (i)/time_1*init_turn, turn_radius_f, level=level))  # Fighter trajectory
        coords_f1.append(geo.turn_exit(coord_f, init_turn, turn_radius_f, level=level))
        traj_f1 = Trajectory(np.array(coords_f1), times_f1)  # Convert coordinates and times to trajectory

        coord_f_1 = traj_f1.coords[-1]  # Store the last coordinate after the first turn

        # 2nd segment: Straight flight
        times_f2=np.append(np.arange(1, time_2, 1),[time_2])  # Time vector for the second turn
        for i in range(1, int(time_2) + 1):
            coords_f2.append(geo.straight_line(coord_f_1, TAS_f, i, level=level))
        coords_f2.append(geo.straight_line(coord_f_1, TAS_f, time_2, level=level))

        traj_f2 = Trajectory(np.array(coords_f2), times_f2)  # Convert coordinates and times to trajectory
        traj_f2.time_offset(time_1)  # Offset the time vector for the second turn

        coord_f_2= traj_f2.coords[-1]  # Store the exact coordinate after the second turn

        # 3rd segment: Final turn
        times_f3 = np.append(np.arange(1, time_3, 1),[time_3])  # Combine time vectors for the fighter trajectory
        for i in range(1,int(time_3)+1):
            coords_f3.append(geo.turn_exit(coord_f_2, HCA*(i)/time_3, turn_radius_f, level=level))
        coords_f3.append(geo.turn_exit(coord_f_2, HCA, turn_radius_f, level=level))  # Final turn to intercept the target

        traj_f3 = Trajectory(np.array(coords_f3), times_f3)  # Convert coordinates and times to trajectory
        traj_f3.time_offset(time_1 + time_2)  # Offset the time vector for the second turn

        traj_f = traj.merge_trajectories([traj_f1, traj_f2, traj_f3])
        traj_f.params={'Name': 'T-38', 'Color': 'Blue', 'Type':'Air+FixedWing', 'Pilot': 'Fighter'},  # Blue for fighter
        traj_f.ID=1

        return traj_f, traj_tgt

    def plot_geodesic(traj_f: Trajectory, traj_tgt: Trajectory) -> None:
        """Plot the fighter and target trajectories on a map using Plotly."""

        def extract_coords(coord_list: list[Transform]) -> tuple[list[float], list[float]]:
            lats = [c.lat for c in coord_list]
            lons = [c.lon for c in coord_list]
            return lats, lons

        tgt_lats, tgt_lons = extract_coords(traj_tgt.coords)
        f_lats, f_lons = extract_coords(traj_f.coords)

        fig2 = go.Figure()

        # Target trajectory
        fig2.add_trace(go.Scattermap(lat=tgt_lats,lon=tgt_lons,mode='lines+markers',name='Target',line=dict(width=1, color='orange'),marker=dict(size=5, color='orange')))
        # Fighter trajectory
        fig2.add_trace(go.Scattermap(lat=f_lats,lon=f_lons,mode='lines+markers',name='Fighter',line=dict(width=1, color='blue'),marker=dict(size=5, color='blue')))

        # Start markers
        fig2.add_trace(go.Scattermap(lat=[tgt_lats[0]],lon=[tgt_lons[0]],mode='markers+text',name='Target Start',marker=dict(size=8, color='orange'),text=['Target Start'],))
        fig2.add_trace(go.Scattermap(lat=[f_lats[0]],lon=[f_lons[0]],mode='markers+text',name='Fighter Start',marker=dict(size=8, color='blue'),text=['Fighter Start'],))
        # fig2.add_trace(go.Scattermap(lat=[coord_f_1.lat],lon=[coord_f_1.lon],mode='markers+text',name='Fighter Start',marker=dict(size=8, color='black'),text=['Fighter F1'],))
        # fig2.add_trace(go.Scattermap(lat=[coord_f_2.lat],lon=[coord_f_2.lon],mode='markers+text',name='Fighter Start',marker=dict(size=8, color='black'),text=['Fighter F2'],))
        # fig2.add_trace(go.Scattermap(lat=[coord_f_1_rel.lat],lon=[coord_f_1_rel.lon],mode='markers+text',name='Fighter Start',marker=dict(size=8, color='grey'),text=['Fighter F1 rel'],))
        # fig2.add_trace(go.Scattermap(lat=[coord_f_2_rel.lat],lon=[coord_f_2_rel.lon],mode='markers+text',name='Fighter Start',marker=dict(size=8, color='grey'),text=['Fighter F2 rel'],))
        # fig2.add_trace(go.Scattermap(lat=[coord_TIP.lat], lon=[coord_TIP.lon], mode='markers+text', marker=dict(size=8, color='purple'), text=['TIP Point'],))

        # Center map around midpoint of both tracks
        center_lat = (tgt_lats[0] + f_lats[0]) / 2
        center_lon = (tgt_lons[0] + f_lons[0]) / 2


        fig2.update_layout(
            title='Fighter and Target Trajectories on Map',
            map=dict(
                style='satellite',  # or 'carto-positron', 'stamen-terrain', etc.
                center=dict(lat=center_lat, lon=center_lon),
                zoom=10
            ),
            height=700,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        fig2.show()

    def plot_tangent_solution_plotly(TIPs: np.ndarray, all_rel_pos_out_R: np.ndarray, all_rel_pos_out_L: np.ndarray,
                                     point_1: np.ndarray, point_2: np.ndarray, rel_pos_init: np.ndarray, idx: int,
                                     intersections: np.ndarray, pos_out_inter: np.ndarray) -> None:
        """Visualizes tangent intersection solutions in the target body frame.

        All coordinates are in [x,y] format (x>0 forward of the target, y>0 right of the target)
        Plots the following elements:
            1. TIP polygon (closed loop).
            2. Fighter's turn paths (right and left).
            3. Lines between each pair of tangent points (point_1 and point_2).
            4. Lines between intersection points and corresponding output positions.
            5. Tangent intersection points.
            6. Initial relative position of the fighter.

        Args:
            TIPs (np.ndarray): Array of TIP polygon vertices, shape (N, 2).
            all_rel_pos_out_R (np.ndarray): Array of fighter's right turn path positions, shape (M, 2).
            all_rel_pos_out_L (np.ndarray): Array of fighter's left turn path positions, shape (K, 2).
            point_1 (np.ndarray): Array of end of initial turn position with a tangent solution, shape (2,).
            point_2 (np.ndarray): Array of tangent intersection points, shape (L, 2).
            rel_pos_init (np.ndarray): Initial relative position of the fighter, shape (2,).
            idx (int): Index of the currently selected tangent solution.
            intersections (np.ndarray): Array of intersection points, shape (P, 2).
            pos_out_inter (np.ndarray): Array of output positions corresponding to intersections, shape (P, 2).

        Returns:
            None

        """
        fig = go.Figure()

        # 1. Plot the TIP polygon (closed loop)
        TIPs_closed = np.vstack([TIPs, TIPs[0]])
        fig.add_trace(go.Scatter(
            x=TIPs_closed[:, 0],
            y=TIPs_closed[:, 1],
            mode='markers+lines',
            name='TIP Polygon',
            line=dict(color='blue', width=2),
            marker=dict(color='blue', size=6, symbol='circle'),
        ))

        # 2. Plot the fighter’s turn path
        if all_rel_pos_out_R.size > 0:
            fig.add_trace(go.Scatter(
                x=all_rel_pos_out_R[:, 0],
                y=all_rel_pos_out_R[:, 1],
                mode='markers',
                name='Fighter Turn Path (R)',
                marker=dict(color='orange', size=6, symbol='circle')
            ))

        if all_rel_pos_out_L.size > 0:
            fig.add_trace(go.Scatter(
                x=all_rel_pos_out_L[:, 0],
                y=all_rel_pos_out_L[:, 1],
                mode='markers',
                name='Fighter Turn Path (L)',
                marker=dict(color='yellow', size=6, symbol='circle')
            ))

        # 2.5. Plot lines between each point_1 and point_2 (if available)
        if point_1.size > 0 and point_2.size > 0 and len(point_1) == len(point_2):
            for i in range(len(point_1)):
                color= 'green' if i == idx else 'red'
                dash_style = 'solid' if i == idx else 'dash'
                fig.add_trace(go.Scatter(
                    x=[point_1[i, 0], point_2[i, 0]],
                    y=[point_1[i, 1], point_2[i, 1]],
                    mode='lines',
                    line=dict(color=color,width=2, dash=dash_style),
                    name=f'Path {i+1}',
                    showlegend=False
                ))

        if intersections.size > 0 and pos_out_inter.size > 0 and len(intersections) == len(pos_out_inter):
            for i in range(len(intersections)):
                color= 'blue'
                dash_style = 'dot'
                fig.add_trace(go.Scatter(
                    x=[intersections[i, 0], pos_out_inter[i, 0]],
                    y=[intersections[i, 1], pos_out_inter[i, 1]],
                    mode='lines',
                    line=dict(color=color,width=2, dash=dash_style),
                    name=f'Path {i+1}',
                    showlegend=False
                ))

        # 3. Plot the tangent intersection points
        if point_2.size > 0:
            fig.add_trace(go.Scatter(
                x=point_2[:, 0],
                y=point_2[:, 1],
                mode='markers+text',
                name='Tangent Points',
                text=[f'Tangent {i+1}' for i in range(len(point_2))],
                textposition='top center',
                marker=dict(color='red', size=10, symbol='x')
            ))

        # 4. Optional: plot initial relative position of fighter
        if rel_pos_init is not None:
            fig.add_trace(go.Scatter(
                x=[rel_pos_init[0]],
                y=[rel_pos_init[1]],
                mode='markers+text',
                text=["Fighter Init"],
                name='Fighter Start',
                textposition='bottom right',
                marker=dict(color='black', size=10, symbol='star')
            ))

        # Final layout
        fig.update_layout(
            title='Tangent Intersections on TIP Polygon',
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            width=1300,
            height=800,
            showlegend=True,
            template='plotly_white'
        )

        # Invert Y-axis (positive down) and preserve aspect ratio
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            autorange="reversed"
        )

        fig.show()

    nb_TIPs = 360
    nb_steps = 360*2  # Number of steps for coarse search
    refine_window=1  # Refinement window in degrees
    TIPs_L=geo.get_TIPs(TAS_tgt,TAS_f,load_factor_f,nb_TIPs,side='left')
    TIPs_R=geo.get_TIPs(TAS_tgt,TAS_f,load_factor_f,nb_TIPs,side='right')
    if debug:
        print("➡️ Right turn")
    turn_R, points_1_R,points_2_R,times_R,all_rel_pos_out_R,intersections_R,pos_out_inter_R,HCA_R = find_tangent_headings_refined(coord_f,coord_tgt,TIPs_L,TIPs_R,TAS_f,TAS_tgt,load_factor_f,turn='right',coarse_steps=nb_steps, refine_window=refine_window)
    if debug:
        print("⬅️ Left turn")
    turn_L, points_1_L,points_2_L,times_L,all_rel_pos_out_L,intersections_L,pos_out_inter_L,HCA_L = find_tangent_headings_refined(coord_f,coord_tgt,TIPs_L,TIPs_R,TAS_f,TAS_tgt,load_factor_f,turn='left',coarse_steps=nb_steps, refine_window=refine_window)

    turns = np.concatenate([turn_R, turn_L])
    points_1 = np.concatenate([points_1_R, points_1_L])
    points_2=np.concatenate([points_2_R, points_2_L])  # points_2 is the intersection points
    times = np.concatenate([times_R, times_L])
    HCA = np.concatenate([HCA_R, HCA_L])  # Combine HCA from both sides
    TIPs=np.concatenate([TIPs_L, TIPs_R])  # Combine TIPs from both sides
    intersections = np.concatenate([intersections_R, intersections_L])  # Combine intersections from both sides
    pos_out_inter = np.concatenate([pos_out_inter_R, pos_out_inter_L])

    if times.size > 0 and not np.isnan(times).all():
        min_idx = np.nanargmin(times[:, 0])  # Find the index of the minimum time
        if debug:
            print("Minimum time", times[min_idx, :])
        traj_f, traj_tgt = get_trajectory(coord_f, TAS_f, coord_tgt, TAS_tgt, turns[min_idx], times[min_idx],HCA[min_idx])
        last_coord_f = traj_f.coords[-1]  # Last coordinate of the fighter trajectory
        last_coord_tgt = traj_tgt.coords[-1]  # Last coordinate of the target trajectory
        error=geo.distance2D(last_coord_f.lat,last_coord_f.lon,last_coord_tgt.lat,last_coord_tgt.lon)
        # print(f"Fighter heading:{coord_f.yaw:.2f}°\tTarget heading:{coord_tgt.yaw:.2f}°\tHorizontal error : {error:.2f} m")
        # print(f"Minimum time index: {min_idx}, Time: {times[min_idx, 0]:.2f} s")
        if plots:
            rel_pos_init = rel_pos(coord_f, coord_tgt)  # Initial relative position of the fighter to the target
            plot_tangent_solution_plotly(TIPs,all_rel_pos_out_R,all_rel_pos_out_L,points_1,points_2,rel_pos_init[:2],min_idx,intersections,pos_out_inter)
            plot_geodesic(traj_f, traj_tgt)  # Plot the geodesic path of the fighter and the target
            tacview.traj2tacview(traj_f,traj_tgt)
        return traj_f,traj_tgt,error
    else:
        print("No valid trajectory found.")
        return np.nan



