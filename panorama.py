import numpy as np
from transforms3d.euler import quat2euler, euler2quat, mat2euler, euler2mat
import load_data
import timeit
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import panorama_lib as pan

#####User input: choose a dataset here#####
dataset = "13"
training = False  # change to False if there is no vicon data
use_vic = False  # change use_vic to True to generate panorama with vicon data
skip = 0  # panorama will use 1/(skip+1) of total images, choose a bigger value to speed up
###########################################

if __name__ == "__main__":
    start = timeit.default_timer()
    t = start

    with open('ukf_data_' + str(dataset), 'rb') as rf:
        # load ukf data and format it based on training status
        if training:
            [vic_euler, vic_ts, imu_euler_integration, imu_euler_prediction, imu_euler_ukf, imu_ts] = pickle.load(rf)
        else:
            [imu_euler_integration, imu_euler_prediction, imu_euler_ukf, imu_ts] = pickle.load(rf)

    # load cam data
    cfile = "cam/cam" + dataset + ".p"
    ts = load_data.tic()
    camd = load_data.read_data(cfile)
    load_data.toc(ts, "Data import")

    # format cam data
    cam_vals = camd["cam"]
    cam_ts = camd["ts"]
    cam_n = len(cam_ts[0])

    # M=num of row, N=num of column, pixels=total pixel number
    M = len(cam_vals[:, 0, 0, 0])
    N = len(cam_vals[0, :, 0, 0])
    pixels = M * N

    # size of panorama
    M_panorama = 900
    N_panorama = 1920

    # empty array to store panorama
    img_panorama = np.zeros((M_panorama + 1, N_panorama + 1, 3), dtype=np.uint8)

    for i in range(cam_n):
        # use 1/(skip+1) of total images for panorama, to speed up in case of large number of images
        if i % (skip + 1) != 0:
            continue

        # unpack an image from data
        img = cam_vals[:, :, :, i]

        # unpack a rotation euler from either vicon or ukf
        if training and use_vic:
            ts = pan.closest_ts(vic_ts, cam_ts[0, i])
            euler = vic_euler[:, ts]
            name = "vicon"
        else:
            ts = pan.closest_ts(imu_ts, cam_ts[0, i])
            euler = imu_euler_ukf[:, ts]
            name = "ukf"

        # euler to rotation matrix
        R = euler2mat(euler[0], euler[1], euler[2])
        # unpack an image to 2 matrix for storing index and value
        rc, rgb = pan.unpackimg(img, M, N)

        r_matrix = np.matrix(rc[0, :])  # a vector of row index for all pixels
        c_matrix = np.matrix(rc[1, :])  # a vector of col index for all pixels

        # find longitude and latitude for all pixels
        longitude, latitude = pan.pixel2spherical(r_matrix, c_matrix, M, N)

        # find coordinate in camera frame for all pixels
        XYZ_c = np.matrix(np.zeros((3, pixels)))
        XYZ_c[0, :], XYZ_c[1, :], XYZ_c[2, :] = pan.spherical2cartesian(longitude, latitude, 1)

        # find coordinate in world frame for all pixels
        XYZ_w = pan.camera2world(XYZ_c, R, np.matrix([0, 0, 0.1]).T)

        # find new longitude and latitude for all pixels
        longitude, latitude, dist = pan.cartesian2spherical(XYZ_w[0, :], XYZ_w[1, :], XYZ_w[2, :])

        # find corresponding row and col in panorama image
        row, col = pan.rowcol_panorama(longitude, latitude, M_panorama, N_panorama)

        # copy rgb value from image to corresponding panorama image
        for j in range(pixels):
            r = int(row[0, j])
            c = int(col[0, j])

            if img_panorama[r, c, 0] == 0 and img_panorama[r, c, 1] == 0 and img_panorama[r, c, 2] == 0:
                img_panorama[r, c, :] = rgb[:, j]

        # calculate time left to finish
        pan.estimated_time_to_finish(i, cam_n, round(timeit.default_timer() - t), skip)
        t = timeit.default_timer()

    # store panorama array as image
    im = Image.fromarray(img_panorama)
    im.save("img_panorama_" + str(dataset) + "_" + name + ".jpeg")

    # plot image
    plt.imshow(im)
    plt.show()

    print "panorama done (" + str(round(timeit.default_timer() - start, 2)) + ' seconds)'
