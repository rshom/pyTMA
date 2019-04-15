'''Simple Target Motion Analysis Demo'''

__author__ = 'Russell Shomberg'

import numpy as np
import math
import matplotlib.pyplot as plt

from subcon import *

runTime = 100.
dT = 1.0

def main():
    # Generate ownship and target ship
    ownship = Ship( np.array([ 0.,0.,0, 5] ))
    target = Ship( np.array([ -50.,50.,3.5,3.5 ] ) )

    # Take initial bearing to target and create contact
    contact = Contact( sonar_bearing( ownship, target ) )
    contact.xEst = target.X-ownship.X

    # Record history
    ownshipHist = ownship.X
    targetHist = target.X
    contactHist = contact.xEst+ownship.X

    time = 0
    while time < runTime:
        time += dT
        # move ships
        if time == 50:
            Xo, U = ownship.update(dT, newCourse=[2,2])
            print(U)
        else:
            Xo, U = ownship.update(dT)

        Xt, _ = target.update(dT)

        # update contact
        bearing = sonar_bearing( ownship, target )


        xEst = contact.MPCEKF( bearing, U, dT )

        # Record history
        ownshipHist = np.vstack((ownshipHist, ownship.X))
        targetHist = np.vstack((targetHist,target.X))
        contactHist = np.vstack((contactHist,contact.xEst+ownship.X))

        # Plot progress
        plt.cla()
        plt.plot(ownshipHist[:,0],ownshipHist[:,1],'b', label="OwnShip")
        plt.plot(targetHist[:,0],targetHist[:,1],'r', label="Target")
        plt.plot(contactHist[:,0],contactHist[:,1],'.', label="Contact")
        plt.axis("equal")
        plt.title("COP")
        plt.legend()
        plt.pause(.1)
        print()
        print(time)

    return


if __name__=='__main__':
    plt.style.use('ggplot')
    main()
    print()
    print("Complete")
    print()
    plt.show()

