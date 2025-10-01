"""Constants.py.

This module defines physical constants and conversion factors commonly used in aeromodeling and engineering calculations.

It includes values for temperature, pressure, density, and speed of sound at sea level, as well as gravity in both
imperial and metric units.
Distance and speed conversion factors between SI and Imperial units are provided.
Atmospheric constants such as the specific gas constant and ratio of specific heats are included.
Temperature conversion factors between Kelvin and Rankine are defined.
Additional constants include epoch time offsets, geographical declination, and specific parameters for the C-12 aircraft model.

Attributes:
    T_SL_K (float): Temperature at sea level in Kelvin.
    P_SL_PSF (float): Pressure at sea level in pounds per square foot.
    RHO_SL_SCF (float): Density at sea level in slugs per cubic foot.
    A_SL_KT (float): Speed of sound at sea level in knots.
    C_TO_K_OFFSET (float): Celsius to Kelvin offset.
    G_IMPERIAL (float): Gravity in feet per second squared.
    G_METRIC (float): Gravity in meters per second squared.
    FT_TO_M (float): Feet to meters conversion factor.
    M_TO_FT (float): Meters to feet conversion factor.
    FT_TO_NM (float): Feet to nautical miles conversion factor.
    NM_TO_FT (float): Nautical miles to feet conversion factor.
    NM_TO_M (float): Nautical miles to meters conversion factor.
    M_TO_NM (float): Meters to nautical miles conversion factor.
    KT_TO_FPS (float): Knots to feet per second conversion factor.
    FPS_TO_KT (float): Feet per second to knots conversion factor.
    MS_TO_KT (float): Meters per second to knots conversion factor.
    KT_TO_MS (float): Knots to meters per second conversion factor.
    R_IMPERIAL (float): Specific gas constant in imperial units.
    R_SI (float): Specific gas constant in SI units.
    GAMMA (float): Ratio of specific heats.
    R_TO_K (float): Rankine to Kelvin conversion factor.
    K_TO_R (float): Kelvin to Rankine conversion factor.
    EPOCH_OFFSET (int): Epoch time offset for 1/1/2025 00:00:00.
    DECLINATION_KEDW (float): Magnetic declination at Edwards AFB as of 04/01/2025.
    C12_PROP_DIAMETER (float): C-12 propeller diameter in feet.
    C12_WING_AREA (float): C-12 wing area in square feet.
    C12_L_AOA (float): Distance from AOA vane to root of MAC for C-12 in feet.
    C12_MAC (float): Mean aerodynamic chord for C-12 in feet.

"""

# Physical constants and conversion factors

# Temperature, pressure, density, speed of sound at sea level
T_SL_K = 288.15         # K, temperature at sea level
P_SL_PSF = 2116.22      # psf, pressure at sea level
RHO_SL_SCF = 0.002378   # slug/ft^3, density at sea level
A_SL_KT = 661.4786      # kt, speed of sound at sea level
A_SL_MS = 340.294       # m/s, speed of sound at sea level
C_TO_K_OFFSET = 273.15  # K, Celsius to Kelvin offset
G_IMPERIAL = 32.2       # ft/s^2, gravity
G_METRIC = 9.80665      # m/s^2, gravity

# Distance conversions
FT_TO_M = 0.3048
M_TO_FT = 1 / FT_TO_M
FT_TO_NM = 1 / 6076.12
NM_TO_FT = 6076.12
NM_TO_M = 1852
M_TO_NM = 1 / NM_TO_M

# Speed conversions
KT_TO_FPS = 1.68781
FPS_TO_KT = 1 / KT_TO_FPS
MS_TO_KT = 3600 / 1852
KT_TO_MS = 1852 / 3600

# Atmospheric constants
R_IMPERIAL = 1716.49   # ft-lbf/slug-R
R_SI = 287.05287       # J/kg-K
GAMMA = 1.4            # Ratio of specific heats

# Temperature conversions
R_TO_K = 5 / 9           # Rankine to Kelvin
K_TO_R = 9 / 5           # Kelvin to Rankine
C_TO_K_OFFSET = 273.15  # Celsius to Kelvin offset
K_TO_C_OFFSET = -273.15 # Kelvin to Celsius offset

#Weight conversion
LB_TO_KG=0.45359237     #pound to kg
KG_TO_LB=1/0.45359237   #kg to pound

# Time constants
EPOCH_OFFSET = 1735689600  # Epoch time for 1/1/2025 00:00:00

# Geographical constants
DECLINATION_KEDW = 11.34   # E at Edwards 04/01/2025

# C-12 specific constants
C12_PROP_DIAMETER = 8.208      # ft
C12_WING_AREA = 303            # sqft
C12_L_AOA = (33.5 - 14.20 + 171.23) / 12  # ft, distance from AOA van to root of MAC
C12_MAC = 70.41 / 12           # ft
