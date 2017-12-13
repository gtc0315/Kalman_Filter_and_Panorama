import numpy as np


def closest_ts(imu_ts, ts):
    ref_ts = np.abs(np.matrix(imu_ts) - ts)
    return np.argmin(ref_ts)


def rc2i(r, c, M, N):
    return r * N + c


def spherical2cartesian(longitude, latitude, r):
    x = r * np.multiply(np.sin(latitude), np.cos(longitude))
    y = r * np.multiply(np.sin(latitude), np.sin(longitude))
    z = r * np.cos(latitude)
    return x, y, z


def cartesian2spherical(x, y, z):
    r = np.power(np.power(x, 2) + np.power(y, 2) + np.power(z, 2), 0.5)
    latitude = np.arccos(np.divide(z, r))  # 0 ~ pi
    longitude = np.arctan2(y, x)  # -pi ~ pi
    return longitude, latitude, r


def pixel2spherical(r, c, M, N):
    longitude = -30.0 / 180 * np.pi + 1.0 * c / N * 60 / 180 * np.pi
    latitude = np.pi / 2 - 22.5 / 180 * np.pi + 1.0 * r / M * 45 / 180 * np.pi
    return longitude, latitude


def rowcol_panorama(longitude, latitude, M_panorama, N_panorama):
    col = np.around((longitude + np.pi) / (2 * np.pi) * N_panorama)
    row = np.around(latitude / np.pi * M_panorama)
    return row, col


def camera2world(x, R, p):
    y = R * (x - p)
    return y


def estimated_time_to_finish(iter, n, dt, skip):
    t = (n - iter) / (skip + 1) * dt
    perc = iter * 100 / n
    print "image " + str(iter) + " | " + str(perc) + "% | time remaining:" + str(round(t / 60, 3)) + " min"
    return 0


def unpackimg(img, M, N):
    rgb = np.zeros((3, M * N), dtype=np.uint8)
    rc = np.zeros((2, M * N), dtype=int)
    for r in range(M):
        for c in range(N):
            i = rc2i(r, c, M, N)
            rgb[:, i] = img[r, c, :]
            rc[:, i] = np.array([r, c])
    return rc, rgb
