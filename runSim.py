'''Simple Target Motion Analysis Demo'''

__author__ = 'Russell Shomberg'

import numpy as np
import math
import matplotlib.pyplot as plt

from subcon import *

runTime = 50.
dT = 1

def main():
    # Generate ownship and target ship
    ownship = Ship( np.array([ 0,0,2,2 ] ))
    target = Ship( np.array([ 0,50,0,2 ] ))

    # Take initial bearing to target and create contact
    contact = Contact( sonar_bearing( ownship, target ) )

    # Record history
    ownshipHist = ownship.X
    targetHist = target.X
    contactHist = contact.xEst

    time = 0
    while time < runTime:
        time += dT
        U = np.array([0,0,0,0])
        
        # move ships
        if time == 20:
            Xt, U = ownship(dT, newCourse=[-2,2])
        else:
            Xo, _ = ownship(dT)

        Xt, U = target(dT)

        # update contact
        bearing = sonar_bearing( ownship, target )
        xEst = contact( bearing, U, dT )

        # Record history
        ownshipHist = np.vstack((ownshipHist, ownship.X))
        targetHist = np.vstack((targetHist,target.X))
        contactHist = np.vstack((contactHist,contact.xEst+ownship.X))

        # Plot progress
        plt.cla()
        plt.plot(ownshipHist[:,0],ownshipHist[:,1], label="OwnShip")
        plt.plot(targetHist[:,0],targetHist[:,1], label="Target")
        plt.plot(contactHist[:,0],contactHist[:,1], label="Contact")
        plt.axis("equal")
        plt.legend()
        plt.pause(.1)

    return


if __name__=='__main__':
    plt.style.use('ggplot')
    main()
    plt.show()
