'''Simple Target Motion Analysis Demo'''

__author__ = 'Russell Shomberg'

import numpy as np
import math
import matplotlib.pyplot as plt

from subcon import *

runTime = 200.
dT = 1.0

def main():
    # Generate ownship and target ship
    ownship = Ship( np.array([ 0.,0.,-3, 5] ))
    target = Ship( np.array([ -100.,50.,3.5,3.5 ] ) )

    # Take initial bearing to target and create contact
    contact = Contact( sonar_bearing( ownship, target ) )
    #contact.xEst = target.X-ownship.X

    # Record history
    ownshipHist = ownship.X
    targetHist = target.X
    contactHist = contact.xEst+ownship.X
    relHist = mpc2polar(xy2mpc(contact.xEst))

    time = 0
    while time < runTime:
        #print()
        #print(time)
        # move ships
        if time == 30:
            Xo, U = ownship.update(dT, newCourse=[5,3])
        elif time == 70:
            Xo, U = ownship.update(dT, newCourse=[3,5])
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
        plt.plot(ownshipHist[:,0],ownshipHist[:,1],'b', label="OwnShip")
        plt.plot(targetHist[:,0],targetHist[:,1],'r', label="Target")
        plt.plot(contactHist[:,0],contactHist[:,1],'.', label="Contact")
        plt.axis("equal")
        plt.title("COP")
        plt.legend()
        plt.pause(.1*dT)

    
    build_plots( contactHist, targetHist, ownshipHist )
    return


if __name__=='__main__':
    plt.style.use('ggplot')
    main()
    print()
    print("Complete")
    print()
    plt.show()

