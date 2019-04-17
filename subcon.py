'''Submarine Control Library'''

import numpy as np
import matplotlib.pyplot as plt


_SONAR_BEARING_ERROR = 0.1
_MEAN_RANGE = 200.0
_RANGE_VARIANCE = .2
_SPEED_VARIANCE = 0.01

_MIN_DETECTION_RANGE = 10.0
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

        yEst = np.array([ 0.0, 0.0, bearing, 1.0/_MIN_DETECTION_RANGE ])
        self.yEst = np.array([ 0.0, 0.0, bearing, 1.0/_MEAN_RANGE ])
        self.xEst = mpc2xy( yEst )

        self.pEst = np.diag([ _SPEED_VARIANCE/_MEAN_RANGE,
                              _SPEED_VARIANCE/_MEAN_RANGE,
                              np.deg2rad(_SONAR_BEARING_ERROR),
                              _RANGE_VARIANCE/_MEAN_RANGE**2
        ])

        print(self.pEst)
        ''''
        self.pEst = np.diag([ _MAX_TARGET_SPEED/_MIN_DETECTION_RANGE,
                              _MAX_TARGET_SPEED/_MIN_DETECTION_RANGE,
                              np.deg2rad(_SONAR_BEARING_ERROR),
                              _RANGE_VARIANCE/_MEAN_RANGE**2
        ])
        '''
        #self.pEst = np.diag([ 10., 10., 1., 1.])**2
        

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

        return self.xEst, xy2mpc(self.xEst)

    def MPCEKF( self, B, U, dT ):

        # Convert to modified polar coords
        #Y = xy2mpc( self.xEst )
        Y = self.yEst
        a = np.array([ dT*Y[0] - Y[3]*( U[0]*np.cos(B) - U[1]*np.sin(B) ),
                       1 + dT*Y[1] - Y[3]*( U[0]*np.sin(B) + U[1]*np.cos(B) ),
                       Y[0] - Y[3]*( U[2]*np.cos(B) - U[3]*np.sin(B) ),
                       Y[1] - Y[3]*( U[2]*np.sin(B) - U[3]*np.cos(B) )
        ])

        yPred = np.array([ ( a[1]*a[2] - a[0]*a[3] )/( a[0]**2 + a [1]**2 ),
                           ( a[0]*a[2] + a[1]*a[3] )/( a[0]**2 + a[1]**2 ),
                           Y[2] + np.arctan2( a[0], a[1] ),
                           Y[3]/np.sqrt( a[0]**2 + a[1]**2 )
        ])

        # state prediction jacobian
        F = polarJacobian( Y, U, a, dT )

        pPred = F@self.pEst@F.T # is this right

        # observation jacobian
        jH = np.array([0,0,1,0])

        S = jH@pPred@jH.T+np.deg2rad(_SONAR_BEARING_ERROR)**2
        #G = pPred@jH.T@np.linalg.inv(S)
        G = (pPred@jH.T)*(S)**-1

        self.yEst = yPred+G*( B - jH@yPred )
        self.pEst = ( np.eye(len(Y)) - G@jH )@pPred
        
        self.xEst = mpc2xy( self.yEst )

        return self.xEst, self.yEst


class Ship:

    def __init__( self, state ):
        self.X = state

    def update( self, dT, newCourse=None ):
        '''step ship forward and record path'''

        Qsim = np.random.randn(4)*.1
        Qsim[2:3] = 0
        xPrev = self.X.copy()

        if newCourse:
            self.X[2] = newCourse[0]
            self.X[3] = newCourse[1]

        motionModel = np.array([[1, 0, dT, 0],
                                [0, 1, 0, dT],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        self.X = motionModel@self.X+Qsim

        

        # Ship motion as input to relative motion of other ships
        U = np.array([self.X[0] - (xPrev[0] + dT*xPrev[2]),
                      self.X[1] - (xPrev[1] + dT*xPrev[3]),
                      self.X[2] - xPrev[2],
                      self.X[3] - xPrev[3]
        ])

        return self.X, U


def sonar_bearing( ownship, ship ):
    '''Return a bearing affected by noise to a target ship'''
    
    xRel = ship.X-ownship.X
    bearing = np.arctan2( xRel[0],xRel[1] )
    noise = np.random.normal(0,1) * np.deg2rad(_SONAR_BEARING_ERROR)
    return bearing+noise


def xy2mpc(X):
    '''Converts to modified polar coords'''

    Y = np.array( [ ( X[2]*X[1]-X[3]*X[0] )/(X[0]**2+X[1]**2), # bearing_rate
                    ( X[2]*X[0]+X[3]*X[1] )/(X[0]**2+X[1]**2), # range_rate /range
                    np.arctan2(X[0],X[1]),                     # bearing
                    1.0/np.sqrt(X[0]**2+X[1]**2)               # 1/range
    ])
    return Y

def mpc2xy(Y):
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

def mpc2polar( Y ):
    '''Returns array of [Bearing, Bearing Rate, Range, Range Rate] to target'''
    return np.array([ np.rad2deg(Y[2]),
                      np.rad2deg(Y[0]),
                      #Y[3],
                      1/Y[3],
                      #Y[2]
                      Y[2]/Y[3]
    ])

def convert_hist_polar( hist, ownshipHist ):
    '''Returns array of vectors [Bearing, Bearing Rate, Range, Range Rate] to target'''
    if len(hist)!=len(ownshipHist):
        print("Lengths must match")
        return 0
    xRel = hist-ownshipHist
    Y = np.zeros_like(xRel)
    ii = 0
    for val in xRel:
        Y[ii] = mpc2polar(xy2mpc(val))
        ii+=1
    
    return Y

def build_plots( contactHist, targetHist, ownshipHist ):
    '''build and display some error checking plots at the end'''
    contactRelHist = convert_hist_polar( contactHist, ownshipHist )
    fig,ax = plt.subplots(2,2)
    plt.tight_layout()
    ax[0,0].plot(contactRelHist[:,0], '.')
    ax[0,1].plot(contactRelHist[:,1], '.')
    ax[1,0].plot(contactRelHist[:,2], '.')
    ax[1,1].plot(contactRelHist[:,3], '.')

    targetRelHist = convert_hist_polar( targetHist, ownshipHist )
    ax[0,0].plot(targetRelHist[:,0], '-')
    ax[0,1].plot(targetRelHist[:,1], '-')
    ax[1,0].plot(targetRelHist[:,2], '-')
    ax[1,1].plot(targetRelHist[:,3], '-')
    
    #[Bearing, Bearing Rate, Range, Range Rate] to target
    ax[0,0].set_title("Bearing")

    ax[0,1].set_title("Bearing Rate")
    
    ax[1,0].set_title("Range")
    
    ax[1,1].set_title("Range Rate")
    '''
    ax[0,0].set_ylim((-180,180))
    ax[0,1].set_ylim((-180,180))
    ax[1,0].set_ylim(0, max(targetRelHist[:,2]))
    ax[1,1].set_ylim(0, max(targetRelHist[:,3]))
    '''
