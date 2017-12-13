import numpy as np
import load_data
import timeit
from draw import draw_euler
from test import testing
import pickle
import ukf_lib

#####User input: choose a dataset here#####
dataset = "13"
training = False  # change to False if there is no vicon data
###########################################

if __name__ == '__main__':
    start = timeit.default_timer()  # start timer

    ifile = "imu/imuRaw" + dataset + ".p"
    if training:
        vfile = "vicon/viconRot" + dataset + ".p"

    # load data
    ts = load_data.tic()
    imud = load_data.read_data(ifile)
    if training:
        vicd = load_data.read_data(vfile)
    load_data.toc(ts, "Data import")

    # format data
    imu_vals = imud['vals']
    imu_ts = imud['ts']
    imu_n = len(imu_ts[0])
    if training:
        vic_vals = vicd['rots']
        vic_ts = vicd['ts']
        vic_n = len(vic_ts[0])
    bias, scale = ukf_lib.bias_scale(imu_vals, 100)
    imu_vals = ukf_lib.unbias_reorder(imu_vals, imu_n, bias, scale)

    if training:
        vic_euler = ukf_lib.arr_rot2euler(vic_vals, vic_n)

    # simple integration
    print "starting simple integration..."
    qt_arr_integration = ukf_lib.integration_w(imu_vals, imu_ts, imu_n)

    qt = np.matrix([1.0, 0.0, 0.0, 0.0]).T  # qt at t = 0
    P = np.matrix(0.0001 * np.eye(3))
    Q = np.matrix(0.0001 * np.eye(3))
    R = np.matrix(0.01 * np.eye(3))

    # empty array to store values
    qt_arr_prediction = np.zeros((4, imu_n))
    qt_arr_prediction[:, 0] = np.asarray(qt.T)
    qt_arr_ukf = np.zeros((4, imu_n))
    qt_arr_ukf[:, 0] = np.asarray(qt.T)

    # weights in ukf
    weight_m = np.matrix(np.ones((1, 2 * 3 + 1))) / (2.0 * 3)
    weight_m[0, 0] = 0
    weight_c = np.matrix(np.ones((1, 2 * 3 + 1))) / (2.0 * 3)
    weight_c[0, 0] = 2

    # UKF with just prediction
    print "starting prediction only..."
    for i in range(1, imu_n):
        dt = imu_ts[0, i] - imu_ts[0, i - 1]
        wt = np.matrix(imu_vals[3:6, i]).T  # rotation
        zt = np.matrix(imu_vals[0:3, i]).T  # measurements (acceleration)

        qt, P, Y, e_i = ukf_lib.prediction(qt, wt, P, Q, dt, weight_m, weight_c)  # prediction step
        qt_arr_prediction[:, i] = np.asarray(qt.T)
        # print str(i * 100 / imu_n) + "%"

    # UKF with prediction and update
    print "starting UKF..."
    qt = np.matrix([1.0, 0.0, 0.0, 0.0]).T  # reset qt at t=0
    P = np.matrix(0.0001 * np.eye(3))  # reset P covariance
    for i in range(1, imu_n):
        dt = imu_ts[0, i] - imu_ts[0, i - 1]
        wt = np.matrix(imu_vals[3:6, i]).T  # rotation
        zt = np.matrix(imu_vals[0:3, i]).T  # measurements (acceleration)

        qt, P, Y, e_i = ukf_lib.prediction(qt, wt, P, Q, dt, weight_m, weight_c)  # prediction step
        qt, P = ukf_lib.update(qt, zt, Y, e_i, P, R, dt, weight_m, weight_c)  # update step
        qt_arr_ukf[:, i] = np.asarray(qt.T)
        # print str(i * 100 / imu_n) + "%"

    print "done in " + str(round(timeit.default_timer() - start, 2)) + ' seconds'

    # convert results from quaternion to euler
    imu_euler_integration = ukf_lib.arr_quat2euler(qt_arr_integration, imu_n)
    imu_euler_prediction = ukf_lib.arr_quat2euler(qt_arr_prediction, imu_n)
    imu_euler_ukf = ukf_lib.arr_quat2euler(qt_arr_ukf, imu_n)

    # store data in pickle for image panorama
    if training:
        store_data = [vic_euler, vic_ts, imu_euler_integration, imu_euler_prediction, imu_euler_ukf, imu_ts]
    else:
        store_data = [imu_euler_integration, imu_euler_prediction, imu_euler_ukf, imu_ts]

    with open('ukf_data_' + str(dataset), 'wb') as wf:
        pickle.dump(store_data, wf)

    if training:
        draw_euler(training, imu_euler_integration, imu_euler_prediction, imu_euler_ukf, imu_ts, dataset, vic_euler,
                   vic_ts)
    else:
        draw_euler(training, imu_euler_integration, imu_euler_prediction, imu_euler_ukf, imu_ts, dataset)
