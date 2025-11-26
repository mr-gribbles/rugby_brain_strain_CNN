"""
Module to compute the UBrIC score from angular acceleration and velocity time series data.
This is a recreation of the Development of a Metric for Predicting Brain Strain Responses Using Head Kinematics paper
Comments should describe the steps taken to compute UBrIC as per the paper.
Paper can be found here: https://doi.org/10.1007/s10439-018-2015-9"""

import numpy as np
import pandas as pd
import math
from scipy import integrate

# Critical values for UBrIC calculation (from Table 4 in the reference paper)
w_cr = np.array([211.0, 171.0, 115.0])          
a_cr = np.array([20e3, 10.3e3, 7.76e3])         

def ubric_term(wp, ap):
    """
    Compute UBrIC term for a single axis.
    This is the part of equation 2 in the paper that is being summed within the square brackets.
    
    Args:
        wp (float): Normalized peak relative velocity.
        ap (float): Normalized peak acceleration.
    Returns:
        float: UBrIC term for the axis.
    """
    ratio = ap / wp
    return wp + (ap - wp) * math.exp(-(ratio))

def acceleration_to_velocity(acc, time):
    """
    Paper requires angular velocity time series, but input data provides angular acceleration.
    This function integrates angular acceleration to obtain angular velocity.
    Args:
        acc (np.ndarray): Angular acceleration time series.
        time (np.ndarray): Time vector corresponding to the acceleration data.

    Returns:
        vel (np.ndarray): Angular velocity time series.
    """
    vel = np.concatenate(([0.0], integrate.cumulative_trapezoid(acc, time)))
    return vel

def compute_ubric(acc_values, vel_values):
    """
    Compute UBrIC score from angular acceleration and velocity time series.

    Args:
        acc_values (np.ndarray): 3xN array of angular acceleration time series for X, Y, Z axes.
        vel_values (np.ndarray): 3xN array of angular velocity time series for X
    Returns:
        ubric (float): Computed UBrIC score.
    """

    # Calculate peak values for each axis
    a_vals = np.max(np.abs(acc_values), axis=1)
    # Calculate peak-to-peak velocity for each axis. Equation 8 in the paper
    w_vals = np.max(vel_values, axis=1) - np.min(vel_values, axis=1)

    # Normalize by critical values
    w_prime = w_vals / w_cr
    a_prime = a_vals / a_cr

    # Compute UBrIC terms for each axis
    t_x = ubric_term(w_prime[0], a_prime[0])
    t_y = ubric_term(w_prime[1], a_prime[1])
    t_z = ubric_term(w_prime[2], a_prime[2])


    r = 2.0 # reccommended value from the paper

    # Compute overall UBrIC score (Equation 2)
    ubric = (t_x**r + t_y**r + t_z**r)**(1/r)
    return ubric

def read_impact(path):
    """
    Reads a CSV file containing time series data for angular acceleration,
    computes angular velocity, and calculates the UBrIC score.
    Args:
        path (str): Path to the CSV file.
    Returns:
        ubric_score (float): Computed UBrIC score.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Unnamed: 0": "time"})
    time_col = "time"
    
    time = df[time_col].astype(float).to_numpy()
    acc_x = df["ang_x"].astype(float).to_numpy()
    acc_y = df["ang_y"].astype(float).to_numpy()
    acc_z = df["ang_z"].astype(float).to_numpy()
    acc_values = np.array([acc_x, acc_y, acc_z])
    vel_x = acceleration_to_velocity(acc_x, time)
    vel_y = acceleration_to_velocity(acc_y, time)
    vel_z = acceleration_to_velocity(acc_z, time)
    vel_values = np.array([vel_x, vel_y, vel_z])
    ubric_score = compute_ubric(acc_values, vel_values)
    return ubric_score






