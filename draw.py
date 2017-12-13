import matplotlib.pyplot as plt


def draw_euler(training, imu_euler_integration, imu_euler_prediction, imu_euler_ukf, imu_ts, dataset, vic_euler=0,
               vic_ts=0):
    plt.figure(1)
    plt.subplot(311)
    if training:
        plt.plot(vic_ts[0], vic_euler[0, :], 'r--', label='vicon')
    plt.plot(imu_ts[0], imu_euler_integration[0, :], 'g', label='integrat')
    plt.plot(imu_ts[0], imu_euler_prediction[0, :], 'b--', label='predict')
    plt.plot(imu_ts[0], imu_euler_ukf[0, :], 'k', label='UKF')
    plt.legend(prop={'size': 14}, bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
    plt.ylabel('roll (rad)', fontsize=14)

    plt.subplot(312)
    if training:
        plt.plot(vic_ts[0], vic_euler[1, :], 'r--', label='vicon')
    plt.plot(imu_ts[0], imu_euler_integration[1, :], 'g', label='integrat')
    plt.plot(imu_ts[0], imu_euler_prediction[1, :], 'b--', label='predict')
    plt.plot(imu_ts[0], imu_euler_ukf[1, :], 'k', label='UKF')
    plt.ylabel('pitch (rad)', fontsize=14)

    plt.subplot(313)
    if training:
        plt.plot(vic_ts[0], vic_euler[2, :], 'r--', label='vicon')
    plt.plot(imu_ts[0], imu_euler_integration[2, :], 'g', label='integrat')
    plt.plot(imu_ts[0], imu_euler_prediction[2, :], 'b--', label='predict')
    plt.plot(imu_ts[0], imu_euler_ukf[2, :], 'k', label='UKF')
    plt.xlabel('ts', fontsize=14)
    plt.ylabel('yaw (rad)', fontsize=14)

    plt.suptitle("DATASET(" + str(dataset) + ")", fontsize=14)
    plt.show()
