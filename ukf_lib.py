import BasicQuaternionFunctions as bqf
from transforms3d.euler import quat2euler, euler2quat, mat2euler, euler2mat
import numpy as np


# convert an array of euler to rotation
def arr_rot2euler(rots, n):
    euler = np.zeros((3, n))
    for i in range(n):
        euler[:, i] = mat2euler(rots[:, :, i])
    return euler


# convert an array of euler to quaternion
def arr_quat2euler(quats, n):
    euler = np.zeros((3, n))
    for i in range(n):
        euler[:, i] = quat2euler(quats[:, i])
    return euler


# simple integration with just rotation w
def integration_w(imu_vals, imu_ts, imu_n):
    qt = np.matrix([1, 0, 0, 0]).T
    qt_arr = np.zeros((4, imu_n))
    qt_arr[:, 0] = np.asarray(qt.T)
    for i in range(1, imu_n):
        dt = imu_ts[0, i] - imu_ts[0, i - 1]
        wt = np.matrix(imu_vals[3:6, i]).T
        qt = bqf.multiply(qt, bqf.exp(wt * dt))
        qt_arr[:, i] = np.asarray(qt.T)
    return qt_arr


# calculate bias and scale using first n samples
def bias_scale(imu, n):
    scale_accel = 3300.0 / 1023 / 300
    scale_gyros = 3300.0 / 1023 * np.pi / 180 / 3.33
    bias = np.array([0, 0, 0, 0, 0, 0])
    for i in range(n):
        for j in range(6):
            bias[j] += imu[j, i]
    bias /= n
    scale = [scale_accel, scale_gyros]
    return bias, scale


# unbias the imu data, flip sign of ax,ay, and reorder wz,wx,wy to wx,wy,wz
def unbias_reorder(imu_vals, n, bias, scale):
    data = np.zeros((6, n), dtype=float)
    for i in range(n):
        data[0, i] = -1 * (imu_vals[0, i] - bias[0]) * scale[0]
        data[1, i] = -1 * (imu_vals[1, i] - bias[1]) * scale[0]
        data[2, i] = (imu_vals[2, i] - bias[2]) * scale[0] + 1
        data[5, i] = (imu_vals[3, i] - bias[3]) * scale[1]
        data[3, i] = (imu_vals[4, i] - bias[4]) * scale[1]
        data[4, i] = (imu_vals[5, i] - bias[5]) * scale[1]
    return data


# q0 is 4D vector in quaternion, w0 is 3D vector, P0 and Q are 3x3 covariance
def prediction(q0, w0, P0, Q, dt, weight_m, weight_c):
    n = 3  # dimension of state vector
    S = np.matrix(np.linalg.cholesky(n * (P0 + Q)))

    W = np.matrix(np.zeros((4, 2 * n + 1)))  # matrix for storing sigma points in quaternion

    W[:, 0] = np.matrix([1, 0, 0, 0]).T  # sigma point W0
    for i in range(n):
        W[:, i + 1] = bqf.exp(S[:, i])  # find sigma points in quaternion
    for i in range(n):
        W[:, i + n + 1] = bqf.exp(-1 * S[:, i])  # find sigma points in quaternion

    Y = np.matrix(np.zeros((4, 2 * n + 1)))  # matrix for storing sigma points after process in quaternion

    for i in range(2 * n + 1):
        Y[:, i] = bqf.multiply(bqf.multiply(q0, W[:, i]),
                               bqf.exp(w0 * dt))  # find sigma points Y after process in quaternion

    qk, e_i = bqf.quaternion_averaging(Y, weight_m, Y[:, 0], 2 * n + 1, 0.0001)  # quaternion average

    P = np.matrix(np.zeros((3, 3)))  # matrix for storing P covariance
    for i in range(2 * n + 1):
        P += weight_c[0, i] * (e_i[:, i] * e_i[:, i].T)  # calculate covariance

    return qk, P, Y, e_i


# q0 is 4D vector in quaternion, w0 is 3D vector, P0 and R are 3x3 covariance
def update(q0, z0, Y, e_i, P0, R, dt, weight_m, weight_c):
    n = 3  # dimension of state vector
    Z = np.matrix(np.zeros((3, 2 * n + 1)))  # matrix for storing sigma points in quaternion

    g = np.matrix([0, 0, 0, 1]).T  # gravity g in quaternion

    for i in range(2 * n + 1):
        g_ = bqf.multiply(bqf.multiply(bqf.conjugate(Y[:, i]), g), Y[:, i])  # calculate g'
        Z[:, i] = g_[1:4, 0]

    Z_m = np.sum(np.multiply(weight_m, Z), axis=1)  # mean of sigma points Z
    Pzz = np.matrix(np.zeros((3, 3)))  # covariance of Z, Pzz
    for i in range(2 * n + 1):
        Pzz += weight_c[0, i] * ((Z[:, i] - Z_m) * (Z[:, i] - Z_m).T)

    Pxz = np.matrix(np.zeros((3, 3)))  # cross-covariance of xz, Pxz
    for i in range(2 * n + 1):
        Pxz += weight_c[0, i] * (e_i[:, i] * (Z[:, i] - Z_m).T)

    Pvv = Pzz + R  # add noise R to covariance Pzz
    v = z0 - Z_m
    K = Pxz * np.linalg.inv(Pvv)  # kalman gain
    qk = bqf.multiply(q0, bqf.exp(K * v))  # find qk
    P = P0 - K * Pvv * K.T  # find P covariance
    return qk, P
