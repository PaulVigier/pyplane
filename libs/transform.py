"""Transform class for representing coordinates in both ECEF and geodetic reference frame as well as euler angles."""

import pymap3d as pm3d

class Transform:
    """Transform class for representing coordinates in both ECEF and geodetic reference frame as well as euler angles.

    This class allows initialization from either ECEF coordinates (x, y, z) or geodetic coordinates (latitude,
    longitude, altitude), and provides methods to set and retrieve these coordinates and orientation angles.
    It also supports rotation by updating the angles.

    Attributes:
        x (float): ECEF x-coordinate in meters.
        y (float): ECEF y-coordinate in meters.
        z (float): ECEF z-coordinate in meters.
        lat (float): Geodetic latitude in degrees.
        lon (float): Geodetic longitude in degrees.
        alt (float): Geodetic altitude in meters.
        roll (float): Roll angle in degrees.
        pitch (float): Pitch angle in degrees.
        yaw (float): Yaw angle in degrees.

    Methods:
        __init__(...): Initialize the Transform object using either ECEF or geodetic coordinates.
        __str__(): Return a string representation of the Transform object.
        init_geodetic(...): Initialize the object with geodetic coordinates and orientation angles.
        set_geodetic(...): Set the geodetic coordinates.
        set_ecef(...): Set the ECEF coordinates.
        set_angles(...): Set the orientation angles.
        rotate(...): Rotate the transform by given roll, pitch, and yaw angles.
        get_ecef(): Get the ECEF coordinates.
        get_geodetic(): Get the geodetic coordinates.
        get_angles(): Get the orientation angles.

    """

    def __init__(self, x: float=None, y: float=None, z: float=None, lat: float=None, lon: float=None, alt: float=None,
                 roll: float=0, pitch: float=0, yaw: float=0)-> None:
        """Initialize the Transform object using either ECEF or geodetic coordinates.

        Provide either (x, y, z) or (lat, lon, alt).

        Args:
            x (float): ECEF x-coordinate in meters
            y (float): ECEF y-coordinate in meters
            z (float): ECEF z-coordinate in meters
            lat (float): Geodetic latitude in degrees
            lon (float): Geodetic longitude in degrees
            alt (float): Geodetic altitude in meters
            roll (float): Roll angle in degrees
            pitch (float): Pitch angle in degrees
            yaw (float): Yaw angle in degrees

        """
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

        if None not in (lat, lon, alt):
            # Initialize with geodetic → compute ECEF
            self.lat = lat
            self.lon = lon
            self.alt = alt
            self.x, self.y, self.z = pm3d.geodetic2ecef(lat, lon, alt)
        elif None not in (x, y, z):
            # Initialize with ECEF → compute geodetic
            self.x = x
            self.y = y
            self.z = z
            self.lat, self.lon, self.alt = pm3d.ecef2geodetic(x, y, z)
        else:
            raise ValueError("Must provide either (lat, lon, alt) or (x, y, z)")

    def __str__(self)-> str:
        """Return a string representation of the Transform object."""
        return f"Transform(x={self.x}, y={self.y}, z={self.z}, lat={self.lat}, long={self.lon}, alt={self.alt}, " \
               f"roll={self.roll}, pitch={self.pitch}, yaw={self.yaw})"

    def set_geodetic(self, lat: float, lon: float, alt: float)-> None:
        """Set the geodetic coordinates.

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
            alt (float): Altitude in meters.

        """
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.x, self.y, self.z = pm3d.geodetic2ecef(lat, lon, alt)

    def set_ecef(self, x: float, y: float, z: float)-> None:
        """Set the ECF coordinates.

        Args:
            x (float): ECF x-coordinate in meters
            y (float): ECF y-coordinate in meters
            z (float): ECF z-coordinate in meters

        """
        self.x = x
        self.y = y
        self.z = z
        self.lat, self.lon, self.alt = pm3d.ecef2geodetic(x, y, z)

    def set_angles(self, roll: float, pitch: float, yaw: float)-> None:
        """Set the angles.

        Args:
            roll (float): Roll angle in degrees
            pitch (float): Pitch angle in degrees
            yaw (float): Yaw angle in degrees

        """
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def rotate(self, roll: float, pitch: float, yaw: float)-> None:
        """Rotate the transform by given roll, pitch, and yaw angles.

        Args:
            roll (float): Roll angle in degrees
            pitch (float): Pitch angle in degrees
            yaw (float): Yaw angle in degrees

        """
        self.roll =(self.roll+ roll) % 360
        self.pitch = (self.pitch + pitch) % 360
        self.yaw = (self.yaw + yaw) % 360

    def get_ecef(self) -> list[float]:
        """Get the ECF coordinates.

        Returns:
            list: ECF coordinates [x, y, z] in meters

        """
        return [self.x, self.y, self.z]

    def get_geodetic(self) -> list[float]:
        """Get the geodetic coordinates.

        Returns:
            list: Geodetic coordinates [latitude, longitude, altitude] in degrees and meters

        """
        return [self.lat, self.lon, self.alt]

    def get_angles(self) -> list[float]:
        """Get the angles.

        Returns:
            list: Angles [roll, pitch, yaw] in degrees

        """
        return [self.roll, self.pitch, self.yaw]

