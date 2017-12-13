import numpy as np


# quaternion averaging
def quaternion_averaging(q, weight, q0, n, epsilon):
    q = np.matrix(q)
    weight = np.matrix(weight)
    q_average = np.matrix(q0)
    for t in range(100):
        e_v_arr = np.matrix(np.zeros((3, n)))
        for i in range(n):
            q_e_i = multiply(inverse(q_average), q[:, i])
            e_v_i = log(q_e_i)
            e_v_i_norm = euler_norm(e_v_i)
            if e_v_i_norm != 0:
                constant = -np.pi + np.mod(e_v_i_norm + np.pi, 2 * np.pi)
                e_v_arr[:, i] = constant * e_v_i / e_v_i_norm

        e_v = np.sum(np.multiply(weight, e_v_arr), axis=1)
        e_v_norm = euler_norm(e_v)

        if e_v_norm != 0:
            q_average = multiply(q_average, exp(e_v))
        if e_v_norm < epsilon:
            return q_average, e_v_arr
    return q_average, e_v_arr


def multiply(q, p):
    qs = np.asscalar(q[0, 0])
    ps = np.asscalar(p[0, 0])
    qv = np.matrix(q[1:, 0])
    pv = np.matrix(p[1:, 0])
    m = np.matrix(np.zeros((4, 1)))
    m[0, 0] = qs * ps - qv.T * pv
    m[1:, 0] = qs * pv + ps * qv + np.cross(qv, pv, axis=0)
    return m


def norm(q):
    qs = np.asscalar(q[0, 0])
    qv = np.matrix(q[1:, 0])
    return np.asscalar(np.sqrt(qs ** 2 + qv.T * qv))


def euler_norm(qv):
    qv = np.matrix(qv)
    return np.asscalar(np.sqrt(qv.T * qv))


def log(q):
    qs = np.asscalar(q[0, 0])
    qv = np.matrix(q[1:, 0])
    if euler_norm(qv) == 0:
        return np.matrix([0, 0, 0]).T
    return 2 * qv / euler_norm(qv) * np.arccos(qs / norm(q))


def exp(w):
    qv = np.matrix(w) / 2.0
    qv_norm = euler_norm(qv)

    if qv_norm == 0:
        return np.matrix([1, 0, 0, 0]).T

    m = np.matrix(np.zeros((4, 1)))
    m[0, 0] = np.cos(qv_norm)
    m[1:, 0] = qv / qv_norm * np.sin(qv_norm)
    return m


def conjugate(q):
    q = np.matrix(q)
    q[1:, 0] = - q[1:, 0]
    return q


def inverse(q):
    q_conj = conjugate(q)
    return q_conj / (norm(q) ** 2)


def rotation(q, s):
    x = np.matrix(np.zeros((4, 1)))
    x[1:, 0] = s
    q_inv = inverse(q)
    return multiply(multiply(q, x), q_inv)
