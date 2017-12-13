import BasicQuaternionFunctions as bqf
import numpy as np
from transforms3d.euler import quat2euler, euler2quat, mat2euler, euler2mat


# for testing purpose
def testing():
    q1 = np.matrix([-np.sin(np.pi), 3, 4, 3]).T
    q2 = np.matrix([4, 3.9, -1, -3]).T
    x = bqf.multiply(q1, q2)
    print "\nmultiply"
    print x
    print "true result: [1.3,3,36.7,-6.6]"
    print "\nnorm"
    print bqf.norm(x)
    print "true result: 37.4318"
    print "\nconjugate"
    print bqf.conjugate(x)
    print "\ninverse"
    print bqf.inverse(x)
    print "\nexp(w)"
    print bqf.exp(np.matrix([0, 2, 0]).T)
    print "true result: [0.54,0,0.84,0]"
    print "\nexp(w)"
    print bqf.exp(np.matrix([0, 0, 0]).T)
    print "true result: [1,0,0,0]"
    print "\nlog(q)"
    print bqf.log(np.matrix([1, 0, 0, 0]).T)
    print "true result: [0,0,0]"
    print "\nlog(q)"
    print bqf.log(np.matrix([np.cos(1), 0, np.sin(1), 0]).T)
    print "true result: [0,2,0]"
    print "\nquaternion average"
    q_ = np.matrix(np.zeros((4, 3)))
    q_[:, 0] = np.matrix(euler2quat(0, np.deg2rad(170), 0)).T
    q_[:, 1] = np.matrix(euler2quat(0, np.deg2rad(-101), 0)).T
    q_[:, 2] = np.matrix(euler2quat(0, np.deg2rad(270), 0)).T
    q_4 = np.matrix(euler2quat(0, np.deg2rad(-127), 0)).T
    a = np.matrix(np.ones((1, 3))) / 3.0
    q_, e_ = bqf.quaternion_averaging(q_, a, q_[:, 2], 0.1)
    print q_
    print "true result" + str(q_4)
