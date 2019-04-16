'''Submarine Control Library'''

import numpy as np

_SONAR_BEARING_ERROR = .01
_MEAN_RANGE = 100.0
_RANGE_VARIANCE = 1.0
_SPEED_VARIANCE = 1.0

_MIN_DETECTION_RANGE = 50.0
_MAX_TARGET_SPEED = 5.0


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
        self.pEst = np.eye(len(self.xEst))

        yEst = np.array([ 0.0, 0.0, bearing, 1.0/_MEAN_RANGE ])
        self.xEst = polar2xy( yEst )

        self.pEst = np.diag([ _SPEED_VARIANCE/_MEAN_RANGE,
                              _SPEED_VARIANCE/_MEAN_RANGE,
                              _SONAR_BEARING_ERROR,
                              _RANGE_VARIANCE/_MEAN_RANGE**2
        ])**2

        #self.pEst = np.diag([ 10., 10., 1., 10.])**2
        

    def EKF( self, bearing, ownshipAcceleration, dT ):
        ''' Generate new estimate using Extended Kalman Filter'''

        # predict state covariance
        Q = 1. * np.diag([1.0,1.0,.1,.1])**2
        ### this is wrong because it is the noise caused by the process
        ### This should be relatively low or on the same order as the
        ### process noise I add in the model

        # Observation covariance
        R = 1 * np.diag([1,1,1.0,1.0])**2  

        ## Jacobian
        motionModel = np.array([[1, 0, dT, 0],
                                [0, 1, 0, dT],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        # Predict based on motion model and input from ship motion U
        xPred = motionModel@self.xEst - ownshipAcceleration
        pPred = motionModel@self.pEst@motionModel.T+Q

        bearingPred = np.arctan2(xPred[0],xPred[1])

        # Compare bearing prediction and observation
        bearingError = bearing-bearingPred

        # Update estimates
        ## Jacobian for observation
        jH = np.array(
            [self.xEst[1]/(self.xEst[0]**2+self.xEst[1]**2)
            ,-self.xEst[1]/ (self.xEst[0]**2+self.xEst[1]**2)
            ,0
            ,0
        ])

        ## Correction for observation
        S = jH@pPred@jH.T+R

        G = pPred@jH.T@np.linalg.inv(S)

        ## final estimate
        self.xEst = xPred+G*bearingError
        self.pEst = (np.eye(len(self.xEst)) - G@jH)@pPred

        return self.xEst

    def MPCEKF( self, B, U, dT ):

        # Convert to modified polar coords
        Y = xy2polar( self.xEst )

        a = np.array([ dT*Y[0] - Y[3]*( U[0]*np.cos(B) - U[1]*np.sin(B) ),
                       1 + dT*Y[1] - Y[3]*( U[0]*np.sin(B) + U[1]*np.cos(B) ),
                       Y[0] - Y[3]*( U[2]*np.cos(B) - U[3]*np.sin(B) ),
                       Y[1] - Y[3]*( U[2]*np.sin(B) - U[3]*np.cos(B) )
        ])

        yPred = np.array([ ( a[1]*a[2] - a[0]*a[3] )/( a[0]**2+a[1]**2 ),
                           ( a[0]*a[2] + a[1]*a[3] )/( a[0]**2+a[1]**2 ),
                           Y[2] + np.arctan2( a[0],a[1] ),
                           Y[3]/np.sqrt( a[0]**2+a[1]**2 )
        ])

        # state prediction jacobian
        F = polarJacobian( Y, U, a, dT )

        pPred = F@self.pEst@F.T # is this right

        # observation jacobian
        jH = np.array([0,0,1,0])

        S = jH@pPred@jH.T+(_SONAR_BEARING_ERROR)**2
        #G = pPred@jH.T@np.linalg.inv(S)
        G = (pPred@jH.T)*(S)**-1

        yEst = yPred+G*( B - jH@yPred )
        self.pEst = ( np.eye(len(Y)) - G@jH )@pPred
        
        self.xEst = polar2xy( yEst )

        print(yEst)
        print(self.xEst)
        print()
        
        return self.xEst


class Ship:

    def __init__( self, state ):
        self.X = state

    def update( self, dT, newCourse=None ):
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
    bearing = np.arctan2( xRel[0],xRel[1] )
    noise = np.random.normal(0,1) * _SONAR_BEARING_ERROR
    return bearing+np.deg2rad(noise)


def xy2polar(X):
    '''Converts to modified polar coords'''

    Y = np.array( [ ( X[2]*X[1]-X[3]*X[0] )/(X[0]**2+X[1]**2), # bearing_rate
                    ( X[2]*X[0]+X[3]*X[1] )/(X[0]**2+X[1]**2), # range_rate /range
                    np.arctan2(X[0],X[1]),                     # bearing
                    1.0/np.sqrt(X[0]**2+X[1]**2)               # 1/range
    ])
    return Y

def polar2xy(Y):
    '''convert from modified polar coords'''

    X = (1/Y[3])*np.array([np.sin(Y[2]),
                           np.cos(Y[2]),
                           Y[1]*np.sin(Y[2]) + Y[0]*np.cos(Y[2]),
                           Y[1]*np.cos(Y[2]) - Y[0]*np.sin(Y[2])
    ])
    return X

def polarJacobian(Y, U, a, dT):
    '''Compute F matrix in modified polar coords'''

    d11 = (-a[0]*( a[1]*a[2] - a[0]*a[3] ) - a[1]*( a[0]*a[2] + a[1]*a[3] ) )/( a[0]**2 + a[1]**2 )**2
    d21 = (-a[0]*( a[0]*a[2] + a[1]*a[3] ) + a[1]*( a[1]*a[2] - a[0]*a[3] ) )/( a[0]**2 + a[1]**2 )**2
    d31 = a[1]/( a[0]**2 + a[1]**2 )
    d41 = -a[2]*Y[3]/( a[0]**2 + a[1]**2 )**(3.0/2.0)
    d32 = -a[0]/( a[0]**2 + a[1]**2 )
    d42 = -a[1]*Y[3]/( a[0]**2 + a[1]**2 )**(3.0/2.0)
    d13 = a[1]/( a[0]**2 + a[1]**2 )
    
    e14 = -( U[0]*np.cos(Y[2]) - U[1]*np.sin(Y[2]) )
    e24 = -( U[0]*np.sin(Y[2]) + U[1]*np.cos(Y[2]) )
    e34 = -( U[2]*np.cos(Y[2]) - U[3]*np.sin(Y[2]) )
    e44 = -( U[2]*np.sin(Y[2]) + U[3]*np.cos(Y[2]) )
    e13 = -Y[3]*e24
    e23 = Y[3]*e14
    e33 = -Y[3]*e44
    e43 = Y[3]*e34

    C = np.array([[ 0,0,0,0 ],
                  [ 0,0,0,0 ],
                  [ 0,0,1,0 ],
                  [ 0,0,0, 1.0/np.sqrt (a[0]**2 + a[1]**2 ) ]
    ])
    
    D = np.array([[ d11, -d21, d13, d32],
                  [ d21, d11, -d32, d13],
                  [ d31, d32,  0,   0 ],
                  [ d41, d42,  0,   0 ]])
    
    E = np.array([[ dT, 0, e13, e14 ],
                  [ 0, dT, e23, e24 ],
                  [ 1, 0,  e33, e34 ],
                  [ 0, 1,  e43, e44 ]])
        
    return C+D@E
