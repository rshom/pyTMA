'''Simple Target Motion Analysis Demo'''

__author__ = 'Russell Shomberg'

import numpy as np
import math
import matplotlib.pyplot as plt

target = np.empty((100,4))
target[:] = [-200,100,5,2]

ownShip = np.empty((100,4))
ownShip[:50] = [0,0,0,5]

samplingTime = 1
runTime = 100

xRel = target-ownShip

time = np.arange(0,runTime,samplingTime)

def calcCourse(ship, T):
    for ii in range(len(ship[:-1])):
        F = np.array([[1, 0, T, 0],
                      [0, 1, 0, T],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        ship[ii+1] = F@(ship[ii])
    return ship


target = calcCourse(target,1.0)

turnTime = 20
ownShip[:turnTime+1] = calcCourse(ownShip[:turnTime+1],1.0)
ownShip[turnTime,2:] = [2,2]
ownShip[turnTime:] = calcCourse(ownShip[turnTime:],1.0)

Xrel = target-ownShip

bearings = np.arctan2(Xrel[:,0],Xrel[:,1])
noise = np.random.normal(0, 1, bearings.shape)*np.deg2rad(.1)
observations = bearings + noise

Xest = np.array([-200,100,2,1])
Pest = np.eye(4)
zHist = np.array([np.arctan2(Xest[0],Xest[1])])
Xhist = Xest

Q = np.diag([1.0, 1.0, 1.0, 1.0])**2  # predict state covariance
R = np.diag([1.0, 1.0, 1.0, 1.0])**2  # Observation covariance

T = samplingTime
fig,ax = plt.subplots()
for k in range(runTime-1):
    # Predict motion based on current estimate
    F = np.array([[1, 0, T, 0],
                  [0, 1, 0, T],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    U = np.array([ownShip[k+1,0] - ownShip[k,0] - T*ownShip[k,2],
                 ownShip[k+1,1] - ownShip[k,1] - T*ownShip[k,3],
                 ownShip[k+1,2]- ownShip[k,2],
                  ownShip[k+1,3]- ownShip[k,3]])


    Xpred = F@Xest - U


    Ppred = F@Pest@F.T

    zPred = np.arctan2(Xpred[0],Xpred[1])
    zHist = np.hstack((zHist,zPred))

    # Compare prediction to observation
    y = observations[k]-zPred

    # Update estimates
    H = np.array([Xest[1]/(Xest[0]**2+Xest[1]**2),
                  -Xest[1]/ (Xest[0]**2+Xest[1]**2),
                  0,
                  0])

    S = H@Ppred@H.T+R
    G = Ppred@H.T@np.linalg.inv(S)
    
    Xest = Xpred+G*y

    Pest = (np.eye(len(Xest)) - G@H)@Ppred

    Xhist = np.vstack((Xhist,Xest))

    course = Xhist + ownShip[:k+2]

    plt.cla()
    ax.plot(course[:,0],course[:,1],'.', label="Target Est.")
    ax.plot(target[:k,0],target[:k,1], label="Target True")
    ax.plot(ownShip[:k,0],ownShip[:k,1], label="OwnShip")
    plt.axis("equal")
    ax.legend()
    plt.pause(.1)



fig,ax = plt.subplots()
ax.plot(np.rad2deg(zHist),'.', label="Estimate")
ax.plot(np.rad2deg(bearings), label="True")
ax.plot(np.rad2deg(observations),'.', label="Observed")
ax.set_title("Bearing")
ax.legend()


fig,ax = plt.subplots()
ax.plot(np.sqrt(Xhist[:,0]**2+Xhist[:,1]**2),'.', label="Estimate")
ax.plot(np.sqrt(Xrel[:,0]**2+Xrel[:,1]**2), label="True")
ax.set_title("Range")
ax.legend()

'''
fig,ax = plt.subplots()
ax.plot(Xhist[:,1], label="Estimate")
ax.plot(Xrel[:,1], label="True")
ax.set_title("Y Relative")
ax.legend()

                
fig,ax = plt.subplots()
ax.plot(Xhist[:,0], label="Estimate")
ax.plot(Xrel[:,0], label="True")
ax.set_title("X Relative")
ax.legend()
'''

plt.show()
