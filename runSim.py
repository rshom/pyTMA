'''Simple Target Motion Analysis Demo'''

__author__ = 'Russell Shomberg'

import numpy as np
import math
import matplotlib.pyplot as plt
import time

from subcon import *

runTime = 60.
dT = 10.0

def main():
    # Generate ownship and target ship
    ownship = Ship( np.array([ 5.,15.,3, 1] ))
    target = Ship( np.array([ 50.,50.,4, 2 ] ) )
    
    # Take initial bearing to target and create contact
    contact = Contact( sonar_bearing( ownship, target ) )
    #contact.xEst = target.X-ownship.X

    # Record history
    ownshipHist = ownship.X
    targetHist = target.X
    contactHist = contact.xEst+ownship.X
    relHist = mpc2polar(xy2mpc(contact.xEst))

    time = 0
    while time < runTime*dT:
        #print()
        #print(time)
        # move ships
        '''
        if time == 10*dT:
            Xo, U = ownship.update(dT, newCourse=[0,6])
        elif time == 30*dT:
            Xo, U = ownship.update(dT, newCourse=[6,0])
        elif time == 50*dT:
            Xo, U = ownship.update(dT, newCourse=[0,6])
        elif time == 70*dT:
            Xo, U = ownship.update(dT, newCourse=[6,0])
        elif time == 90*dT:
            Xo, U = ownship.update(dT, newCourse=[0,6])
        else:
            Xo, U = ownship.update(dT)
        '''
        if time == 10*dT:
            Xo, U = ownship.update(dT, newCourse=[3,3])
        elif time == 20*dT:
            Xo, U = ownship.update(dT, newCourse=[3,0])
        elif time == 30*dT:
            Xo, U = ownship.update(dT, newCourse=[3,3])
        else:
            Xo, U = ownship.update(dT)

        Xt, _ = target.update(dT)

        time += dT

        # update contact
        bearing = sonar_bearing( ownship, target )
        xEst, yEst = contact.MPCEKF( bearing, U, dT )

        # Record history
        ownshipHist = np.vstack((ownshipHist, ownship.X))
    
        targetHist = np.vstack((targetHist,target.X))
        contactHist = np.vstack((contactHist,contact.xEst+ownship.X))
        #relcontactHist = np.vstack((relHist, MPC2polar(yEst)))

        # Plot progress
        plt.cla()
        plt.plot(ownshipHist[:,0],ownshipHist[:,1],'b-', label="OwnShip")
        plt.plot(targetHist[:,0],targetHist[:,1],'r-', label="Target")
        plt.plot(contactHist[:,0],contactHist[:,1],'r.', label="Contact")
        plt.axis("equal")
        #plt.ylim((-10,500))
        #plt.xlim((-10,500))
        #plt.axis((-10,5000,-10,5000))
        plt.title("COP")
        plt.legend()
        plt.pause(.01*dT)

    
    build_plots( contactHist, targetHist, ownshipHist )
    return


if __name__=='__main__':
    for _ in range(1):
        print("starting SIM")
        main()
        print()
        print("Complete")
        print()
        plt.show()

