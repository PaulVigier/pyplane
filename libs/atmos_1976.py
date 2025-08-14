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


def std_atmosphere(ha: float, units: str = 'ft', silent: bool = True) -> tuple:
    """Return delta, theta, sigma, Pressure, Temp, Density, Speed of sound,
    dynamic viscosity at a given altitude based on Standard Atmosphere model from 1976.

    Parameters
    ----------
    ha : float
        Altitude above sea level.
    units : 'ft' input and outputs in imperial units
            'm' input and outputs in metric units
    silent : 'True' print the results in the console
             'False' do not print the results in the console

    """  # noqa: D205
    # Constants

    Ra = 1716.49   # specific gas constant for air (ft lb / slug R)
    g = 32.174049  # acceleration due to gravity (ft/s^2)
    gamma = 1.4
    Su = 198.72  # Sutherland's constant (R)
    bu = 2.2697E-8  # (slug/ft-s-sqrt(R))

    if units == 'm':
        ha = ha / 0.3048
    elif units != 'ft':
        raise TypeError("Inconsistent unit. Should be 'm' or 'ft'")

    # parameters that define the standard atmosphere...
    # altitudes (ft)
    H1 = [
        0,         # 0: beginning of troposphere
        36089,     # 1: tropopause, start of isothermal stratosphere
        65617,     # 2: start of lower stratosphere lapse layer
        104987,    # 3: start of upper stratosphere lapse layer
        154199,    # 4: stratopause, start of isothermal mesophere
        167323,    # 5: start of lower mesosphere lapse layer
        232940,    # 6: start of upper mesosphere lapse layer
        278385,    # 7: mesopause, start of isothermal thermosphere
        298556,    # 8: start of thermosphere ellipse lapse layer (91-120 km)
        360892,    # 9: thermosphere lapse layer (110-120 km)
        393701     # 10: top of current calculations (120 km)
    ]

    # lapse rates  (R/ft)
    A1 = [
        -3.566E-3,   # 0: beginning of troposphere
        0,           # 1: tropopause, start of isothermal stratosphere
        5.486E-4,    # 2: start of lower stratosphere lapse layer
        1.540E-3,    # 3: start of upper stratosphere lapse layer
        0,           # 4: stratopause, start of isothermal mesophere
        -1.540E-3,   # 5: start of lower mesosphere lapse layer
        -1.1004E-3,  # 6: start of upper mesosphere lapse layer
        0,           # 7: mesopause, start of isothermal thermosphere
        -137.382,    # 8: thermosphere ellipse lapse layer (different formula)
        6.584E-3,    # 9: thermosphere linear lapse layer (110-120 km)
        0            # 10: top of calcs (120 km)
    ]

    # base layer Temp definitions
    T1 = [
        518.67,     # 0: SSL temp
        389.97,     # 1: tropopause, start of isothermal stratosphere
        389.97,     # 2: start of lower stratosphere lapse layer
        411.57,     # 3: start of upper stratosphere lapse layer
        487.17,     # 4: stratopause, start of isothermal mesophere
        487.17,     # 5: start of lower mesosphere lapse layer
        386.37,     # 6: start of upper mesosphere lapse layer
        336.361,    # 7: mesopause, start of isothermal thermosphere
        473.743,    # 8: thermosphere elliptic lapse layer (diff formula)
        432         # 9: thermosphere linear lapse layer (110-120 km)
    ]
    T1.append(T1[9] + A1[9] * (H1[10] - H1[9]))  # 10: temp at 120 km

    P1 = [
        2116.21662,     # 0: SSL temp
        472.688,        # 1: tropopause, start of isothermal stratosphere
        114.345,        # 2: start of lower stratosphere lapse layer
        18.129,         # 3: start of upper stratosphere lapse layer
        2.31634,        # 4: stratopause, start of isothermal mesophere
        1.39805,        # 5: start of lower mesosphere lapse layer
        0.082632,       # 6: start of upper mesosphere lapse layer
        0.0077986       # 7: mesopause, start of isothermal thermosphere
    ]

    P1.append(P1[7] * np.exp(-1 * (g / Ra / T1[7]) * (H1[8] - H1[7])))
    P1.append(P1[8] * np.exp(-1 * (g / Ra / T1[8]) * (H1[9] - H1[8])))
    P1.append(P1[9] * np.exp(-1 * (g / Ra / T1[9]) * (H1[10] - H1[9])))

    Rho1 = np.divide(np.array(P1[0:10]), np.array(T1[0:10])) / Ra

    idx_0 = list(map(lambda i: i > ha, H1)).index(True) - 1
    # idx_1 = list(map(lambda i: i >= ha, H1)).index(True)
    a0 = A1[idx_0]  # lapse rate for the layer
    T0 = T1[idx_0]  # temperature at the base of the layer
    H0 = H1[idx_0]  # altitude at the base of the layer
    P0 = P1[idx_0]  # pressure at the base of the layer
    Rho0 = Rho1[idx_0]  # density at the base of the layer

    if idx_0 != 8:  # the linear and zero lapse layers (all layers 8)
        Ta = T0 + a0 * (ha - H0)
        if a0 != 0:
            Pa = P0 * (Ta / T0) ** (-g / a0 / Ra)
            Rhoa = Rho0 * (Ta / T0) ** (-1 * (g / a0 / Ra + 1))
        else:
            Pa = P0 * np.exp(-1 * (g / Ra / Ta) * (ha - H0))
            Rhoa = Rho0 * np.exp(-1 * (g / Ra / Ta) * (ha - H0))

    else:  # 91-120 km elliptic temperature profile layer
        Ta = T0 + a0 * np.sqrt(1 - ((ha - H0) / (-19.9429 * 3281)) ** 2)
        Pa = P0 * (Ta / T0) ** (-g / a0 / Ra)
        Rhoa = Rho0 * (Ta / T0) ** (-1 * (g / a0 / Ra + 1))

    a = np.sqrt(gamma * Ra * Ta)
    mu = bu * Ta ** 1.5 / (Ta + Su)
    delta = Pa / P1[0]
    theta = Ta / T1[0]
    sigma = Rhoa / Rho1[0]

    if units == 'ft' and not silent:
        print(
            f"{'  delta:'.ljust(21)}{delta:0.4f}\n"
            f"{'  theta:'.ljust(20)}{theta:0.4f}\n"
            f"{'  sigma:'.ljust(20)}{sigma:0.4f}\n"
            f"{'  P (psf):'.ljust(20)}{Pa:0.2f}\n"
            f"{'  T (R):'.ljust(20)}{Ta:0.2f}\n"
            f"{'  rho (slug/ft^3):'.ljust(20)}{Rhoa:0.3e}\n"
            f"{'  a (fps):'.ljust(20)}{a:0.1f}\n"
            f"{'  mu (slug/ft-s):'.ljust(20)}{mu:0.3e}\n"
        )

    elif units == 'm':
        Pa = Pa * 47.8803
        Ta = Ta / 1.8
        Rhoa = Rhoa * 515.379
        a = a * 0.3048
        mu = mu * 47.8803

        if not silent:
            print(
                f"{'  delta:'.ljust(21)} {delta:0.4f}\n"
                f"{'  theta:'.ljust(20)} {theta:0.4f}\n"
                f"{'  sigma:'.ljust(20)} {sigma:0.4f}\n"
                f"{'  P (Pa):'.ljust(20)} {Pa:0.2f}\n"
                f"{'  T (K):'.ljust(20)} {Ta:0.2f}\n"
                f"{'  rho (kg/m^3):'.ljust(20)} {Rhoa:0.3e}\n"
                f"{'  a (m/s):'.ljust(20)} {a:0.1f}\n"
                f"{'  mu (kg/m-s):'.ljust(20)} {mu:0.3e}\n"
            )

    return delta, theta, sigma, Pa, Ta, Rhoa, a, mu


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
    - Requires external constants and functions: `cst.A_SL_KT`, `cst.P_SL_PSF`, `pressure`, `temperature`, `cst.GAMMA`,
                                                `cst.R_IMPERIAL`, `cst.K_TO_R`, `cst.FPS_TO_KT`.

    """
    if Offset:
        Air_temp = temperature(altitude * 0.3048, units='m') + temp
    else:
        Air_temp = temp + 273.15
    a = np.sqrt(cst.GAMMA * cst.R_IMPERIAL * Air_temp * cst.K_TO_R)
    Mach = TAS / (a * cst.FPS_TO_KT)
    if Mach > 1:
        raise ValueError("Mach number > 1, cannot convert TAS to CAS")
    CAS = cst.A_SL_KT * np.sqrt(
        5 * (
            (
                pressure_ratio(altitude, units) * ((1 + 0.2 * Mach ** 2) ** (7 / 2) - 1) + 1
            ) ** (2 / 7)
            - 1
        )
    )
    return CAS
