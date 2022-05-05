#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    new_p = np.zeros((3, 1))
    new_v = np.zeros((3, 1))
    new_q = Rotation.identity()
    R = q.as_matrix()
    # update position
    new_p = p + v*dt + 1/2 * (R @ (a_m - a_b) + g) * dt**2

    # update velocity
    new_v = v + (R @ (a_m - a_b) + g) * dt

    # update rotation 
    delta_q = Rotation.from_rotvec(((w_m - w_b)*dt).flatten())
    new_q = q * delta_q

    # update bias and gravitational vector, but they don't change
    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    R = q.as_matrix()

    calibrated_acce = (a_m - a_b).flatten()
    acceleration_ssm = np.array([
        [0, -calibrated_acce[2], calibrated_acce[1]],
        [calibrated_acce[2], 0, -calibrated_acce[0]],
        [-calibrated_acce[1], calibrated_acce[0], 0]
    ]) # skew symmetrical form of acceleration vector

    # construct Fx
    Fx = np.identity(18)
    Fx[0:3,3:6] = dt * np.identity(3)
    Fx[3:6, 6:9] = - R @ acceleration_ssm * dt
    Fx[3:6, 9:12] = - R * dt
    Fx[3:6, 15:18] = dt * np.identity(3)
    Fx[6:9, 6:9] = Rotation.from_rotvec(
            ((w_m - w_b)*dt).flatten()
        ).as_matrix().T
    Fx[6:9, 12:15] = - dt * np.identity(3)
    
    # construct Fi (18,12)
    Fi = np.vstack((
        np.zeros((3,12)),
        np.identity(12),
        np.zeros((3,12))
    ))

    # construct Qi (12,12)
    Qi = np.zeros((12,12))
    Qi[0:3,0:3] = accelerometer_noise_density**2 * dt**2 * np.identity(3)
    Qi[3:6,3:6] = gyroscope_noise_density**2 * dt**2 * np.identity(3)
    Qi[6:9,6:9] = accelerometer_random_walk**2 * dt * np.identity(3)
    Qi[9:12,9:12] = gyroscope_random_walk**2 * dt * np.identity(3)

    P = error_state_covariance

    new_P = Fx @ P @ Fx.T + Fi @ Qi @ Fi.T
    # return an 18x18 covariance matrix
    return new_P


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state
    R = q.as_matrix()
    # compute the innovation next state
    Pc = R.T @ (Pw - p)
    Pc_xy = Pc[0:2]/Pc[2]
    innovation = uv - Pc_xy
    innovation_norm = np.linalg.norm(innovation)
    if innovation_norm < error_threshold: # perform observation update, otherwise do nothing
        # construct Ht
        Ht = np.zeros((2,18))
        Pc = Pc.flatten()
        uv = uv.flatten()
        partial_zt_Pc = 1/Pc[2]*np.array([
            [1, 0, -Pc_xy[0].item()],
            [0, 1, -Pc_xy[1].item()]
        ]) #(2,3)
        partial_Pc_delta_theta = np.array((
            [0, -Pc[2], Pc[1]],
            [Pc[2], 0, -Pc[0]],
            [-Pc[1], Pc[0], 0]
        ))
        partial_Pc_delta_p = - R.T #(3,3)
        partial_zt_delta_theta = partial_zt_Pc @ partial_Pc_delta_theta
        partial_zt_delta_p = partial_zt_Pc @ partial_Pc_delta_p
        Ht[:,0:3] = partial_zt_delta_p
        Ht[:,6:9] = partial_zt_delta_theta

        # construct Kt and new error state covariance
        SIGMA = error_state_covariance
        Kt = SIGMA @ Ht.T @ np.linalg.inv(Ht @ SIGMA @ Ht.T + Q)
        new_SIGMA = (np.identity(18) - Kt @ Ht) @ SIGMA @ (np.identity(18) - Kt @ Ht).T + Kt @ Q @ Kt.T

        # compute new nominal state
        delta_x = Kt @ innovation # (18,2) @ (2,1) 
        delta_p = delta_x[0:3]
        delta_v = delta_x[3:6]
        delta_q = delta_x[6:9]
        delta_a_b = delta_x[9:12]
        delta_w_b = delta_x[12:15]
        delta_g = delta_x[15:18]
        p = p + delta_p
        v = v + delta_v
        q = q * Rotation.from_rotvec(delta_q.flatten())
        a_b = a_b + delta_a_b
        w_b = w_b + delta_w_b
        g = g + delta_g
        error_state_covariance = new_SIGMA

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
