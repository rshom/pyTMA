'''Simple Target Motion Analysis Demo'''

__author__ = 'Russell Shomberg'

import numpy as np
import math
import matplotlib.pyplot as plt

target = np.empty((100,4))
target[:] = [-200,100,5,2]

ownShip = np.empty((100,4))
ownShip[:50] = [0,0,0,5]
ownShip[50:] = [0,0,5,0]

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

ownShip[:51] = calcCourse(ownShip[:51],1.0)
ownShip[50,2:] = [5,0]
ownShip[50:] = calcCourse(ownShip[50:],1.0)

Xrel = target-ownShip

bearings = np.arctan2(Xrel[:,0],Xrel[:,1])
noise = np.random.normal(0, 1, bearings.shape)*np.deg2rad(1)
observations = bearings + noise

Xest = np.array([[-100,50,10,-15]]).T
Pest = np.eye(4)
zHist = np.array([0])
Xhist = Xest

Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0])**2  # predict state covariance
R = np.diag([1.0, 1.0, 1.0, 1.0])**2 # Observation x,y position covariance

T = samplingTime
for k in range(runTime-1):
    # Predict motion based on current estimate
    F = np.array([[1, 0, T, 0],
                  [0, 1, 0, T],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    U = np.array([[ownShip[k+1,0] - ownShip[k,0] - T*ownShip[k,2]],
                  [ownShip[k+1,1] - ownShip[k,1] - T*ownShip[k,3]],
                  [ownShip[k+1,2]- ownShip[k,2]],
                  [ownShip[k+1,3]- ownShip[k,3]]])


    Xpred = F@Xest - U
    Ppred = F@Pest@F.T

    zPred = np.arctan2(Xpred[0],Xpred[1])
    zHist = np.hstack((zHist,zPred))

    # Compare prediction to observation
    y = observations[k]-zPred

    # Update estimates
    H = np.array([
        Xest[1,0]/ (Xest[0,0]**2+Xest[1,0]**2),
         -Xest[1,0]/ Xest[0,0]**2+Xest[1,0]**2,
         0,
         0])

    S = H@Ppred@H.T+R
    G = Ppred@H.T@np.linalg.inv(S)

    Xest = Xpred+G*y
    Pest = (np.eye(len(Xest)) - G@H)@Ppred

    print(y)
    print(Xpred)
    print(Xest)
    print(Xhist)
    print()
    Xhist = np.vstack((Xhist,Xest))

    

fig,ax = plt.subplots()
ax.plot(zHist)
ax.plot(bearings)
plt.show()
                

    
