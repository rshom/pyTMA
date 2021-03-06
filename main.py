'''Extended Kalman Filter (EKF) localization'''

__author__ = 'Russell Shomberg'

import numpy as np
import math
import matplotlib.pyplot as plt

# EKF state covariance
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)])**2

#  Simulation parameter
Qsim = np.diag([0.2, np.deg2rad(1.0)])**2
Rsim = np.diag([1.0, np.deg2rad(10.0)])**2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

show_animation = False

def calc_input():
    v = 1.0 # [m/s]
    p = np.deg2rad(5.) # [rad/s]
    return np.array([[v,p]]).T

 
def motion_model(x,u):
    F = np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0]
    ])

    B = np.array([
        [DT * math.cos(x[2, 0]), 0 ],
        [DT * math.sin(x[2, 0]), 0 ],
        [0.0,                    DT]
    ])

    return (F@x)+(B@u)


def true_motion(x, u):

    u = np.array([[
        u[0,0] + np.random.randn() * Rsim[0,0],
        u[1,0] + np.random.randn() * Rsim[1,1]
        ]]).T
    return motion_model(x, u), u


def observation(x, RFID):
    z = np.zeros((0,3))

    for idx in range(len(RFID[:,0])):

        dx = RFID[idx,0] - x[0,0]
        dy = RFID[idx,1] - x[1,0]
        d = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy,dx)
    
        if d <= MAX_RANGE: # landmark is detected
            # record with some added noise
            dn = d + np.random.randn() * Qsim[0,0]
            anglen = angle + np.random.randn() * Qsim[1,1]
            # zidx is the identity
            zidx = np.array([dn,anglen, idx])
            z = np.vstack((z,zidx))
    return z

def ekf_slam( xEst, PEst, u, z):
    S = STATE_SIZE
    xEst[0:S] = motion_model(xEst[0:S],u)
    G, Fx = jacob_motion(xEst[0:S] , u)

    PEst[0:S,0:S] = G.T * PEst[0:S,0:S] * G + Fx.T * Cx *Fx

    initP = np.eye(2)

    # update
    #for idx in range(len(z

    return xEst, PEst

def jacob_motion(x,u):

    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
        (STATE_SIZE, LM_SIZE * calc_n_LM(x)))))

    jF = np.array([[0.0, 0.0, -DT * u[0] * math.sin(x[2,0])],
                   [0.0, 0.0, DT * u[0] * math.cos(x[2,0])],
                   [0.0, 0.0, 0.0]
                   ])

    G = np.eye(STATE_SIZE) + Fx.T * jF * Fx

    return G, Fx

def main():
    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([
        [10.0, -2.0],
        [15.0, 10.0],
        [3.0,  15.0],
        [-5.0, 20.0],
        [0.0,  11.0],
        [-1.0, 0.0]
    ])
    
    xTrue = np.zeros((STATE_SIZE,1))
    xEst = xTrue
    xDR = np.zeros((STATE_SIZE,1))

    PEst = np.eye(STATE_SIZE)

    # history
    hxTrue = xTrue
    hxDR = xDR

    while SIM_TIME >= time:
        time += DT
        u = calc_input()
        
        xTrue,ud = true_motion(xTrue, u)
        xDR = motion_model(xDR, u)
        
        # History
        hxTrue = np.hstack((hxTrue, xTrue))
        hxDR = np.hstack((hxDR, xDR))

        observations = observation(xTrue,RFID)

        xEst, PEst = ekf_slam(xEst, PEst, ud, observations)
        
        if show_animation:
            plt.cla()
            
            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(hxTrue[0,:],
                     hxTrue[1,:], "-b")
            plt.plot(hxDR[0,:],
                     hxDR[1,:], "-k")
            
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)
            

if __name__ == '__main__':
    main()


        
        
