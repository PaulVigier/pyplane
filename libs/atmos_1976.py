"""atmos_1976.py.

This module provides functions to compute properties of the 1976 Standard Atmosphere model, including pressure,
temperature, density, speed of sound, and dynamic viscosity at a given altitude. It also includes utilities
for converting between calibrated airspeed (CAS) and true airspeed (TAS), and for finding altitudes corresponding
to specific pressure or density ratios.

Functions:
----------
std_atmosphere(ha, units='ft', silent=True)
    Calculates atmospheric properties at a given altitude using the 1976 Standard Atmosphere model.
    Returns delta, theta, sigma, Pressure, Temp, Density, Speed of sound, dynamic viscosity.
density_alt(sigma, units='ft')
    Returns the altitude corresponding to a given density ratio (sigma).
pressure_alt(delta, units='ft')
    Returns the altitude corresponding to a given pressure ratio (delta).
pressure_ratio(altitude, units='ft')
    Returns the pressure ratio (delta) at a given altitude.
temperature_ratio(altitude, units='ft')
    Returns the temperature ratio (theta) at a given altitude.
density_ratio(altitude, units='ft')
    Returns the density ratio (sigma) at a given altitude.
pressure(altitude, units='ft')
    Returns the pressure at a given altitude.
temperature(altitude, units='ft')
    Returns the temperature at a given altitude.
density(altitude, units='ft')
    Returns the density at a given altitude.
speed_of_sound(altitude, units='ft')
    Returns the speed of sound at a given altitude.
viscosity(altitude, units='ft')
    Returns the dynamic viscosity at a given altitude.
CAStoTAS(CAS, altitude, temp=0, Offset=True, units='ft')
    Converts Calibrated Airspeed (CAS) to True Airspeed (TAS) using the standard atmosphere model.
TAStoCAS(TAS, altitude, temp=0, Offset=True, units='ft')
    Converts True Airspeed (TAS) to Calibrated Airspeed (CAS) using the standard atmosphere model.

Notes
-----
- All calculations are based on the 1976 Standard Atmosphere model.
- Functions support both imperial ('ft') and metric ('m') units.
- Some functions allow for temperature offsets from ISA conditions.

"""

import numpy as np
from . import constants as cst
from scipy.optimize import fsolve
import pandas as pd

def std_atmosphere(ha: float | int | np.ndarray | pd.Series, units: str = 'ft', silent: bool = True)->tuple:
    """Calculate atmosphere parameters.

    Return delta, theta, sigma, Pressure, Temp, Density, Speed of sound, dynamic viscosity at a given altitude based
    on the 1976 Standard Atmosphere model.

    Parameters
    ----------
    ha : float, int, np.ndarray, pd.Series
        Altitude(s) above sea level.
    units : 'ft' or 'm'
        Input and output units.
    silent : bool
        If False, print results for each input (only when scalar or small array).

    """
    # Constants
    Ra = 1716.49
    g = 32.174049
    gamma = 1.4
    Su = 198.72
    bu = 2.2697E-8

    # Convert to numpy array
    is_scalar = np.isscalar(ha)
    ha_array = np.atleast_1d(ha).astype(float)

    if units == 'm':
        ha_array = ha_array / 0.3048
    elif units != 'ft':
        raise TypeError("Inconsistent unit. Should be 'm' or 'ft'")

    # Altitude, lapse rates, temps, pressures
    H1 = np.array([
        0, 36089, 65617, 104987, 154199, 167323,
        232940, 278385, 298556, 360892, 393701
    ])

    A1 = np.array([
        -3.566E-3, 0, 5.486E-4, 1.540E-3, 0,
        -1.540E-3, -1.1004E-3, 0, -137.382, 6.584E-3, 0
    ])

    T1 = [
        518.67, 389.97, 389.97, 411.57, 487.17,
        487.17, 386.37, 336.361, 473.743, 432
    ]
    T1.append(T1[9] + A1[9] * (H1[10] - H1[9]))  # Temp at 120 km

    P1 = [
        2116.21662, 472.688, 114.345, 18.129, 2.31634,
        1.39805, 0.082632, 0.0077986
    ]
    for i in range(7, 10):
        T_base = T1[i]
        P_prev = P1[-1]
        delta_H = H1[i+1] - H1[i]
        P1.append(P_prev * np.exp(-g / Ra / T_base * delta_H))

    Rho1 = np.array(P1[0:10]) / np.array(T1[0:10]) / Ra

    def compute_single(h_ft: float | int | np.ndarray | pd.Series) -> tuple:
        idx_0 = list(map(lambda i: i > h_ft, H1)).index(True) - 1
        a0 = A1[idx_0]
        T0 = T1[idx_0]
        H0 = H1[idx_0]
        P0 = P1[idx_0]
        Rho0 = Rho1[idx_0]

        if idx_0 != 8:
            Ta = T0 + a0 * (h_ft - H0)
            if a0 != 0:
                Pa = P0 * (Ta / T0) ** (-g / a0 / Ra)
                Rhoa = Rho0 * (Ta / T0) ** (-1 * (g / a0 / Ra + 1))
            else:
                Pa = P0 * np.exp(-g / Ra / Ta * (h_ft - H0))
                Rhoa = Rho0 * np.exp(-g / Ra / Ta * (h_ft - H0))
        else:
            Ta = T0 + a0 * np.sqrt(1 - ((h_ft - H0) / (-19.9429 * 3281)) ** 2)
            Pa = P0 * (Ta / T0) ** (-g / a0 / Ra)
            Rhoa = Rho0 * (Ta / T0) ** (-1 * (g / a0 / Ra + 1))

        a = np.sqrt(gamma * Ra * Ta)
        mu = bu * Ta ** 1.5 / (Ta + Su)

        delta = Pa / P1[0]
        theta = Ta / T1[0]
        sigma = Rhoa / Rho1[0]

        if units == 'm':
            Pa *= 47.8803
            Ta /= 1.8
            Rhoa *= 515.379
            a *= 0.3048
            mu *= 47.8803

        return delta, theta, sigma, Pa, Ta, Rhoa, a, mu

    # Vectorized computation
    results = np.vectorize(compute_single, otypes=[float]*8)(ha_array)

    # If input was scalar, return scalars
    if is_scalar:
        return tuple(res[0] for res in results)

    # Return array of results
    return tuple(np.array(res) for res in results)


def density_alt(sigma :float, units:str='ft')-> float:
    """Return the altitude at a given density ratio.

    The function is based on the 1976 Standard Atmosphere model.
    units : 'ft' input and outputs in imperial units
            'm' input and outputs in metric units
    """
    if units not in ['ft', 'm']:
        raise TypeError("Inconsistent unit. Should be 'm' or 'ft'")
    min_alt = 0
    max_alt = 393701
    while max_alt - min_alt > 1:
        current_alt = (min_alt + max_alt) / 2

        _, _, temp_sigma, _, _, _, _, _ = std_atmosphere(current_alt)
        if temp_sigma > sigma:
            min_alt = current_alt
        else:
            max_alt = current_alt

    if units == 'm':
        current_alt = current_alt * 0.3048

    return round(current_alt)


def pressure_alt(delta: float, units: str = 'ft') -> float:
    """Calculate the altitude corresponding to a given pressure ratio using the 1976 Standard Atmosphere model.

    Parameters
    ----------
    delta : float
        Pressure ratio for which to find the corresponding altitude.
    units : str, optional
        Unit system for input and output altitude. Accepts 'ft' for imperial units or 'm' for metric units.
        Default is 'ft'.

    Returns
    -------
    float
        Altitude at which the given pressure ratio occurs, rounded to the nearest integer.
        The unit of the returned value matches the `units` parameter.

    Raises
    ------
    TypeError
        If `units` is not 'ft' or 'm'.

    Notes
    -----
    - The function uses a binary search algorithm to find the altitude where the pressure ratio matches `delta`.
    - The altitude range is limited between 0 and 393,701 feet.
    - If metric units are requested, the result is converted from feet to meters.

    """
    if units not in ['ft', 'm']:
        raise TypeError("Inconsistent unit. Should be 'm' or 'ft'")

    min_alt = 0
    max_alt = 393701
    while max_alt - min_alt > 1:
        current_alt = (min_alt + max_alt) / 2

        temp_delta, _, _, _, _, _, _, _ = std_atmosphere(current_alt)
        if temp_delta > delta:
            min_alt = current_alt
        else:
            max_alt = current_alt

    if units == 'm':
        current_alt = current_alt * 0.3048

    return round(current_alt)


def pressure_ratio(altitude: float, units: str = 'ft') -> float:
    """Calculate the pressure ratio at a given altitude based on the 1976 Standard Atmosphere model.

    Args:
        altitude (float): The altitude at which to calculate the pressure ratio.
        units (str, optional): The units of altitude ('ft' for feet or 'm' for meters). Defaults to 'ft'.

    Returns:
        float: The pressure ratio (static pressure divided by sea-level standard pressure) at the specified altitude.

    Raises:
        ValueError: If the units are not supported or altitude is out of valid range.

    Note:
        This function relies on the `std_atmosphere` function to perform the actual calculation.

    """
    result, _, _, _, _, _, _, _ = std_atmosphere(altitude, units)
    return result


def temperature_ratio(altitude: float, units: str = 'ft') -> float:
    """Calculate the temperature ratio at a given altitude based on the 1976 Standard Atmosphere model.

    Args:
        altitude (float): The altitude at which to calculate the temperature ratio.
        units (str, optional): The units of altitude ('ft' for feet or 'm' for meters). Defaults to 'ft'.

    Returns:
        float: The temperature ratio (static temperature divided by sea-level standard temperature) at the specified altitude.

    Raises:
        ValueError: If the units are not supported or altitude is out of valid range.

    Note:
        This function relies on the `std_atmosphere` function to perform the actual calculation.

    """
    _, result, _, _, _, _, _, _ = std_atmosphere(altitude, units)
    return result


def density_ratio(altitude: float, units: str = 'ft') -> float:
    """Calculate the density ratio at a given altitude based on the 1976 Standard Atmosphere model.

    Args:
        altitude (float): The altitude at which to calculate the density ratio.
        units (str, optional): The units of altitude ('ft' for feet or 'm' for meters). Defaults to 'ft'.

    Returns:
        float: The density ratio (static density divided by sea-level standard density) at the specified altitude.

    Raises:
        ValueError: If the units are not supported or altitude is out of valid range.

    Note:
        This function relies on the `std_atmosphere` function to perform the actual calculation.

    """
    _, _, result, _, _, _, _, _ = std_atmosphere(altitude, units)
    return result


def pressure(altitude: float, units: str = 'ft') -> float:
    """Calculate the pressure at a given altitude based on the 1976 Standard Atmosphere model.

    Args:
        altitude (float): The altitude at which to calculate the pressure.
        units (str, optional): The units of altitude ('ft' for feet or 'm' for meters). Defaults to 'ft'.

    Returns:
        float: The pressure at the specified altitude.

    Raises:
        ValueError: If the units are not supported or altitude is out of valid range.

    Note:
        This function relies on the `std_atmosphere` function to perform the actual calculation.

    """
    _, _, _, result, _, _, _, _ = std_atmosphere(altitude, units)
    return result


def temperature(altitude: float, units: str = 'ft') -> float:
    """Calculate the temperature at a given altitude based on the 1976 Standard Atmosphere model.

    Args:
        altitude (float): The altitude at which to calculate the temperature.
        units (str, optional): The units of altitude ('ft' for feet or 'm' for meters). Defaults to 'ft'.

    Returns:
        float: The temperature at the specified altitude.

    Raises:
        ValueError: If the units are not supported or altitude is out of valid range.

    Note:
        This function relies on the `std_atmosphere` function to perform the actual calculation.

    """
    _, _, _, _, result, _, _, _ = std_atmosphere(altitude, units)
    return result


def density(altitude: float, units: str = 'ft') -> float:
    """Calculate the density at a given altitude based on the 1976 Standard Atmosphere model.

    Args:
        altitude (float): The altitude at which to calculate the density.
        units (str, optional): The units of altitude ('ft' for feet or 'm' for meters). Defaults to 'ft'.

    Returns:
        float: The density at the specified altitude.

    Raises:
        ValueError: If the units are not supported or altitude is out of valid range.

    Note:
        This function relies on the `std_atmosphere` function to perform the actual calculation.

    """
    _, _, _, _, _, result, _, _ = std_atmosphere(altitude, units)
    return result


def speed_of_sound(altitude: float, units: str = 'ft') -> float:
    """Calculate the speed of sound at a given altitude based on the 1976 Standard Atmosphere model.

    Args:
        altitude (float): The altitude at which to calculate the speed of sound.
        units (str, optional): The units of altitude ('ft' for feet or 'm' for meters). Defaults to 'ft'.

    Returns:
        float: The speed of sound at the specified altitude.

    Raises:
        ValueError: If the units are not supported or altitude is out of valid range.

    Note:
        This function relies on the `std_atmosphere` function to perform the actual calculation.

    """
    _, _, _, _, _, _, result, _ = std_atmosphere(altitude, units)
    return result


def viscosity(altitude: float, units: str = 'ft') -> float:
    """Calculate the viscosity at a given altitude based on the 1976 Standard Atmosphere model.

    Args:
        altitude (float): The altitude at which to calculate the viscosity.
        units (str, optional): The units of altitude ('ft' for feet or 'm' for meters). Defaults to 'ft'.

    Returns:
        float: The viscosity at the specified altitude.

    Raises:
        ValueError: If the units are not supported or altitude is out of valid range.

    Note:
        This function relies on the `std_atmosphere` function to perform the actual calculation.

    """
    _, _, _, _, _, _, _, result = std_atmosphere(altitude, units)
    return result


def CAStoTAS(CAS: float, altitude: float, temp: float = 0, Offset: bool = True, units: str = 'ft') -> float:  # noqa: N802
    """Convert Calibrated Airspeed (CAS) to True Airspeed (TAS) using the standard atmosphere model.

    Parameters
    ----------
    CAS : float
        Calibrated Airspeed (CAS) in knots.
    altitude : float
        Altitude at which the conversion is performed. Units are specified by `units`.
    temp : float, optional
        If Offset is True, this is the temperature offset (in °C) from ISA conditions.
        If Offset is False, this is the actual air temperature (in °C).
    Offset : bool, optional
        If True, `temp` is interpreted as the offset from ISA temperature.
        If False, `temp` is interpreted as the actual temperature.
    units : str, optional
        Unit system for input and output. 'ft' for imperial units, 'm' for metric units.

    Returns
    -------
    float
        True Airspeed (TAS) in knots (if units='ft') or meters per second (if units='m').

    Notes
    -----
    - Uses the standard atmosphere model for calculations.
    - Requires external constants and functions: `cst.A_SL_KT`, `cst.P_SL_PSF`, `pressure`, `temperature`, `cst.GAMMA`, `cst.R_IMPERIAL`, `cst.K_TO_R`, `cst.FPS_TO_KT`.

    """
    # Get Qc from CAS
    Qc = ((1 + 0.2 * (CAS / cst.A_SL_KT) ** 2) ** (7 / 2) - 1) * cst.P_SL_PSF
    Pa = pressure(altitude, units)
    Mach = np.sqrt(5 * ((Qc / Pa + 1) ** (2 / 7) - 1))
    if Offset:
        Air_temp = temperature(altitude * 0.3048, units='m') + temp
    else:
        Air_temp = temp + 273.15
    a = np.sqrt(cst.GAMMA * cst.R_IMPERIAL * Air_temp * cst.K_TO_R)
    TAS = Mach * a * cst.FPS_TO_KT

    return TAS


def TAStoCAS(TAS: float, altitude: float, temp: float = 0, Offset: bool = True, units: str = 'ft') -> float:
    """Convert True Airspeed (TAS) to Calibrated Airspeed (CAS) using the standard atmosphere model for mach <1.

    Parameters
    ----------
    TAS : float
        True Airspeed (TAS) in knots.
    altitude : float
        Altitude at which the conversion is performed. Units are specified by `units`.
    altitude : float
        Altitude at which the conversion is performed. Units are specified by `units`.
    temp : float, optional
        If Offset is True, this is the temperature offset (in °C) from ISA conditions.
        If Offset is False, this is the actual air temperature (in °C).
    Offset : bool, optional
        If True, `temp` is interpreted as the offset from ISA temperature.
        If False, `temp` is interpreted as the actual temperature.
    units : str, optional
        Unit system for input and output. 'ft' for imperial units, 'm' for metric units.

    Returns
    -------
    float
        True Airspeed (TAS) in knots (if units='ft') or meters per second (if units='m').

    Notes
    -----
    - Uses the standard atmosphere model for calculations.

    """
    if units not in ['ft', 'm']:
        raise ValueError("Invalid units. Please use 'ft' or 'm'.")

    if Offset:
        Air_temp = temperature(altitude, units) + temp
    else:
        Air_temp = temp + 273.15
    a = np.sqrt(cst.GAMMA * cst.R_IMPERIAL * Air_temp * cst.K_TO_R)* cst.FPS_TO_KT
    Mach = TAS / a

    if Mach > 1:
        qc = cst.P_SL_PSF*pressure_ratio(altitude, units) * (166.921*Mach**7/(7*Mach**2-1)**(5/2)-1) #Erb's formula C102
        CAS_init = 800
        print(qc)

        def implicit_eq(CAS: float) -> float:
            left = CAS
            right = cst.A_SL_KT * 0.881284 * np.sqrt(
                (qc / cst.P_SL_PSF + 1) * (1 - 1 / (7 * (CAS / cst.A_SL_KT)**2))**(5/2)
            )
            return left - right  # we want this to be 0

        CAS = fsolve(implicit_eq, CAS_init)[0]
        return CAS

    CAS = cst.A_SL_KT * np.sqrt(
        5 * (
            (
                pressure_ratio(altitude, units) * ((1 + 0.2 * Mach ** 2) ** (7 / 2) - 1) + 1
            ) ** (2 / 7)
            - 1
        )
    )
    return CAS
