import xarray as xr
import numpy as np
import pandas as pd

# 📌 Atlas & Ulbrich (1977) fall speed
def atlas_ulbrich_velocity(diameter):
    """
    This function calculates the fall speed of raindrops based on the Atlas and Ulbrich (1977) model.
    The input diameter is in millimeters.
    """
    vd = 17.67 * ((diameter / 10) ** 0.67)  # diámetro en mm → cm

    return vd

# 📌 Define variable attributes
def attr():
    atributes = {
    'Client': {'short_name': 'Client', 'long_name': 'Quality indicator', 'units': '', 'dtype': 'str'},
    'Synop_Code': {'short_name': 'Synop_Code', 'long_name': 'SYNOP code', 'units': '', 'dtype': 'str'},
    'r_int': {'short_name': 'r_int', 'long_name': 'Rain intensity', 'units': 'mm/h', 'dtype': 'float'},
    'rl_int': {'short_name': 'rl_int', 'long_name': 'Liquid rain intensity', 'units': 'mm/h', 'dtype': 'float'},
    'rs_int': {'short_name': 'rs_int', 'long_name': 'Solid rain intensity', 'units': 'mm/h', 'dtype': 'float'},
    'r_acc': {'short_name': 'r_acc', 'long_name': 'Rain amount accumulated', 'units': 'mm', 'dtype': 'float'},
    'MOR': {'short_name': 'MOR', 'long_name': 'Visibility in precipitation', 'units': 'm', 'dtype': 'float'},
    'ref': {'short_name': 'ref', 'long_name': 'Radar Reflectivity', 'units': 'dBZ', 'dtype': 'float'},
    'n_t': {'short_name': 'n_t', 'long_name': 'Total drop measured', 'units': 'count', 'dtype': 'float'},
    'raw': {'short_name': 'raw', 'long_name': 'raw data', 'units': 'count', 'dtype': 'float'},
    'vd' : {'short_name': 'vd', 'long_name': 'Fall speed (Atlas&Ulbrich)', 'units': 'm/s', 'dtype': 'float'}
    }
    return atributes

# 📌 Define diameter and velocity classes
def diam_vel_classes():
    """
    This function returns the diameter and velocity classes used in the calculations.
    """
    diameter_classes = np.array([
    0.125, 0.250, 0.375, 0.500, 0.750, 1.000, 1.250, 1.500, 1.750,
    2.000, 2.500, 3.000, 3.500, 4.000, 4.500, 5.000, 5.500, 6.000,
    6.500, 7.000, 7.500, 8.000
    ])
    velocity_classes = np.array([
        0.100, 0.200, 0.400, 0.600, 0.800, 1.000, 1.400, 1.800, 2.200,
        2.600, 3.000, 3.400, 4.200, 5.000, 5.800, 6.600, 7.400, 8.200,
        9.000, 10.000
    ])
    return diameter_classes, velocity_classes


# 📌 Define function to calculate nd
def calculate_nd(raw, diameter, velocity, delta_t=60):
    """
    A. Tokay, D. B. Wolff, and W. A. Petersen, “Evaluation of the New Version
    of the Laser-Optical Disdrometer, OTT Parsivel2,” Journal of Atmospheric 
    and Oceanic Technology, vol. 31, no. 6, pp. 1276–1288, Jun. 2014, 
    doi: 10.1175/jtech-d-13-00174.1.
    """

    # Differences between bins
    delta_D = diameter.diff("diameter")  # length 21
    # To leave it with length 22 like the original
    delta_D = xr.concat([diameter[0], delta_D], dim="diameter")

    # Effective area A_j (en m²): A_j = 228 mm * (20 mm - D_i / 2)
    A = 228 * (20 - diameter / 2)  # mm²
    A = A / 1e6  # convert to m²

    # Calculate denominator (22, 20)
    denominator = delta_t * delta_D * velocity * A

    # Calculate N(D) = sum_j (raw / denominador)
    nd = (raw / denominator).sum(dim="velocity")

    # Add attributes
    nd.attrs.update({
        'short_name': 'nd',
        'long_name': 'Field N(d)',
        'units': 'mm⁻¹ m⁻³',
        'dtype':'float'
    })

    return nd

def calculate_parameters_dsd(nd, diameter):
    """
    A. Tokay, D. B. Wolff, and W. A. Petersen, “Evaluation of the New Version
    of the Laser-Optical Disdrometer, OTT Parsivel2,” Journal of Atmospheric 
    and Oceanic Technology, vol. 31, no. 6, pp. 1276–1288, Jun. 2014, 
    doi: 10.1175/jtech-d-13-00174.1.
    """

    # Step 1: Calculate rainfall velocity by Ulbrich (1984)
    vd = 17.67 * ((diameter / 10) ** 0.67)  # diámetro en mm → cm

    # Differences between bins
    delta_D = diameter.diff("diameter")  # length 21
    # To leave it with length 22 like the original
    delta_D = xr.concat([diameter[0], delta_D], dim="diameter")

    # Effective area A_j (en m²): A_j = 228 mm * (20 mm - D_i / 2)
    A = 228 * (20 - diameter / 2)  # mm²
    A = A / 1e6  # convert to m²
    # Constants
    pi = np.pi
    rho_w = 1.0  # g/cm^3

    # 1. Rain rate R (mm/h)
    R = (6 * pi / 1e4) * ((diameter ** 3) * vd * nd * delta_D).sum(dim="diameter")

    # 2. Liquid water content W (g/m³)
    W = (pi * rho_w / 6000) * ((diameter ** 3) * nd * delta_D).sum(dim="diameter")

    # 3. Total drop concentration N_T (m⁻³)
    N_T = (nd * delta_D).sum(dim="diameter")

    # 4. Radar reflectivity Z (mm⁶/m³)
    Z = ((diameter ** 6) * nd * delta_D).sum(dim="diameter")

    # 5. Mean diameter D_m (mm)
    D_m = ((diameter ** 4) * nd * delta_D).sum(dim='diameter') / ((diameter ** 3) * nd * delta_D).sum(dim='diameter')

    # 6. Normalized intercept parameter N_w (m⁻³ mm⁻¹)
    N_w = ((4 ** 4) / pi * rho_w) * ((1e3 * W) / D_m ** 4)

    # Asignar atributos
    R.attrs.update({'short_name': 'r_int', 'long_name': 'Rain intensity', 'units': 'mm/h','dtype': 'float'})
    W.attrs.update({'short_name': 'lwc', 'long_name': 'Liquid water content', 'units': 'g m⁻³','dtype': 'float'})
    N_T.attrs.update({'short_name': 'n_t', 'long_name': 'Total drop concentration', 'units': 'm⁻³','dtype': 'float'})
    Z.attrs.update({'short_name': 'ref', 'long_name': 'Radar reflectivity factor', 'units': 'mm⁶/m³','dtype': 'float'})
    D_m.attrs.update({'short_name': 'd_m', 'long_name': 'Mean diameter', 'units': 'mm','dtype': 'float'})
    N_w.attrs.update({'short_name': 'n_w', 'long_name': 'Normalized intercept parameter', 'units': 'm⁻³ mm⁻¹','dtype': 'float'})

    return R, W, N_T, Z, D_m, N_w

def get_events(ds, time_lenght=10, time_break=5, tot_counts=5):
    """
    param ds: Xarray dataset with count bin (cnt_bin) matrix
    param time_lenght: Minimum duration of the event
    param time_break: Maximum duration of a time break between events
    param tot_counts: Minimum total number of particles within the count bin spectra
    """
    ev = ds.nd.sum(dim='diameter').where(ds.nd.sum(dim='diameter') > tot_counts).compute()
    ev = ev[ev.notnull()]
    a = ev.time.diff('time').to_dataframe('date') 
    sec = pd.Timedelta('10min') 
    breaks = a >= sec
    groups = breaks.cumsum()
    start = [i[1].index.min() for i in groups.groupby('date') if i[1].shape[0] > time_lenght]
    end = [i[1].index.max() for i in groups.groupby('date') if i[1].shape[0] > time_lenght]
    return start, end

def Annual_Precipitation_Cycle(R, t):
    """
    This function calculates the annual precipitation cycle for a given 
    R (rain rate [mm/h]) and t (time sample of measurement in minutes [min]).
    """
    r_int_minute = (R / t).to_series() # mm/min

    r_int_month = r_int_minute.resample('M').sum(min_count=1) # mm/month

    annual_cycle = r_int_month.groupby(r_int_month.index.month).mean() # average monthly rainfall rate

    return r_int_month, annual_cycle

def Annual_Precipitation_rain_gauge(r_g):
    """
    This function calculates the annual precipitation cycle for a given 
    r_g (rain gauge [mm]).
    The input r_g is a time series of rain gauge measurements.
    """
    r_int_month = r_g.resample('ME').sum(min_count=1) # mm/month

    mean_monthly = r_int_month.groupby(r_int_month.index.month).mean() # average monthly rainfall rate

    return r_int_month, mean_monthly

def months_indices():
    """ This function creates an array of month indices for plotting purposes.
    It returns an array of month abbreviations and their corresponding indices.
    """
    months = [
        'J', 'F', 'M', 'A', 'M', 'J',
        'J', 'A', 'S', 'O', 'N', 'D'
    ]

    # Crear los índices de los meses
    x = np.arange(len(months))

    return x, months