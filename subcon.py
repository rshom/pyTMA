'''Submarine Control Library'''

import numpy as np

_SONAR_BEARING_ERROR = .1
_MIN_DETECTION_RANGE = 30.0
_MAX_TARGET_SPEED = 5.0

# predict state covariance
Q = np.diag(
    [1.0
    ,1.0
    ,1.0
    ,1.0
])**2  

# Observation covariance
R = np.diag(
    [1.0
    ,1.0
    ,1.0
    ,1.0
])**2  

class Contact:
    def __init__( self, bearing ):
        '''Create a new contact assuming the most dangerous solution to start'''
        
        print("New contact bears {}".format(np.rad2deg(bearing)))
        
        # assume target is close
        distance = _MIN_DETECTION_RANGE
        xRel = distance*np.sin(bearing)
        yRel = distance*np.cos(bearing)
        
        # assume target is fast
        speed = _MAX_TARGET_SPEED
        
        # assume target is on collision course
        uRel = -speed*np.sin(bearing)
        vRel = -speed*np.cos(bearing)
        
        self.xEst = np.array([ xRel, yRel, uRel, vRel])
        self.xEst = np.array([ -50,-50,0,1 ] )
        self.pEst = np.eye(len(self.xEst))
        

    def __call__( self, bearing, ownshipAcceleration, dT ):
        ''' Generate new estimate using Extended Kalman Filter'''

        ## Jacobian
        motionModel = np.array([[1, 0, dT, 0],
                                [0, 1, 0, dT],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        # Predict based on motion model and input from ship motion U
        xPred = motionModel@self.xEst - ownshipAcceleration
        pPred = motionModel@self.pEst@motionModel.T#+Q*.001
        bearingPred = np.arctan2(xPred[0],xPred[1])

        # Compare bearing prediction and observation
        bearingError = bearing-bearingPred

        # Update estimates
        ## Jacobian for observation
        jH = np.array([
            self.xEst[1]/(self.xEst[0]**2+self.xEst[1]**2), 
            -self.xEst[1]/ (self.xEst[0]**2+self.xEst[1]**2),
            0,
            0])

        ## Correction for observation
        S = jH@pPred@jH.T+R
        print(S)
        G = pPred@jH.T@np.linalg.inv(S)

        ## final estimate
        self.xEst = xPred+G*bearingError
        self.pEst = (np.eye(len(self.xEst)) - G@jH)@pPred

        return self.xEst
    

class Ship:

    def __init__( self, state ):
        self.X = state

    def __call__( self, dT, newCourse=None ):
        '''step ship forward and record path'''

        xPrev = self.X
        if newCourse:
            self.X[2] = newCourse[0]
            self.X[3] = newCourse[1]

        #heading = np.deg2rad(heading)
        #self.X[2] = speed*np.sin(heading)
        #self.X[3] = speed*np.cos(heading)

        motionModel = np.array([[1, 0, dT, 0],
                                [0, 1, 0, dT],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        self.X = motionModel@self.X
        

        # Ship motion as input to relative motion of other ships
        U = np.array([self.X[0] - xPrev[0] - dT*xPrev[2],
                      self.X[1] - xPrev[1] - dT*xPrev[3],
                      self.X[2] - xPrev[2],
                      self.X[3] - xPrev[3]])
        
        return self.X, U


    def change_course( self, heading, speed, dT ):
        '''Update the course while maintaining current position'''




def sonar_bearing( ownship, ship ):
    '''Return a bearing affected by noise to a target ship'''
    
    xRel = ship.X-ownship.X
    bearing = np.arctan2( xRel[0], xRel[1] )
    noise = np.random.normal(0,1) * _SONAR_BEARING_ERROR
    return bearing+np.deg2rad(noise)
