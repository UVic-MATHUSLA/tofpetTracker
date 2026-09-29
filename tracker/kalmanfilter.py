import numpy as np
from numpy.linalg import inv
import functools
print = functools.partial(print, flush=True) # make python actually flush the output!

class KalmanFilterFind():
    """A simplified Kalman filter that only do foward filtering without recording internal status"""
    def init_filter(self, m0, V0, H0, Xf0, Cf0, Rf0):
        self.Xf = Xf0
        self.Cf = Cf0
        self.chift_total = 0
        self.Ndim_meas = len(m0)
        self.Ndim_stat = len(Xf0)
        self.identity_measure = np.identity(self.Ndim_meas)
        self.identity_state = np.identity(self.Ndim_stat)

    # A): Forward iterations
    def update_matrix(self, Vi, Hi, Fi, Qi):
        """Update the predicted next state (Xp_i), state covariance (Cp_i) and residual covariance (Rp_i)"""
        Xp_i = Fi@self.Xf
        Cp_i = Fi@self.Cf@Fi.T + Qi
        Rp_i = Vi + Hi@Cp_i@Hi.T

        self.Xp_i = Xp_i
        self.Cp_i = Cp_i
        self.Rp_i = Rp_i
        self.Vi = Vi
        self.Hi = Hi
        self.Qi = Qi

    def forward_predict_chi2(self, mi):
        """Calculate the prediction chi2 with the next measurement"""
        rp_i = mi - self.Hi@self.Xp_i
        chi2_predict_i = rp_i.T @ inv(self.Rp_i) @ rp_i

        return chi2_predict_i

    def forward_filter(self, mi):
        """Run filter on the next measurement"""
        rp_i = mi - self.Hi@self.Xp_i                           # Get residual
        K = self.Cp_i.dot(self.Hi.T).dot(inv(self.Rp_i))        # Kalman gain (K)
        Xf = self.Xp_i + K@rp_i # Filtered state: combo of predicted state, measured values, covariance matrix, Kalman Gain
        Cf = (self.identity_state - K@self.Hi).dot(self.Cp_i)   # Filtered covariance matrix
        rf = mi - self.Hi@Xf                                    # Filtered residual
        Rf = (self.identity_measure - self.Hi@K).dot(self.Vi)   # Residual Covariance matrix
        try:    chi2_filtered = rf.T @ inv(Rf) @ rf             # Chi2 contribution
        except: chi2_filtered = 0

        # Update the internal state
        self.Xf = Xf
        self.Cf = Cf
        self.chift = chi2_filtered
        self.chift_total+=chi2_filtered

        return chi2_filtered


class KalmanFilter():
    """This class define the Core algorithm of Kalman filter (RTS smoothing)"""
    def __init__(self):
        """Suppose there are N+1 measurements starting from index 0"""
        # Measurement
        self.m=[]           # measurements                      [0...N]
        self.V=[]           # measurement uncertainty matrix    [0...N]
        self.H=[]           # measurement matrices              [0...N]
        # States
        self.Xp=[]          # state predicted (Extrapolated)    [_,1...N] *Initial value needs a placeholder
        self.Xf=[]          # state filtered                    [0...N]
        self.Xsm=[]         # state smoothed                    [0...N] *------INIT when smoothing
        # Extrapolation Matrices
        self.F=[]           # Extrapolate function              [0...N-1] 
        self.Q=[]           # Extrapolate uncertainty           [0...N-1]
        # Variation Matrices
        self.Cp=[]          # state Cov predicted               [_,1...N] *Initial value needs a placeholder
        self.Cf=[]          # state Cov filtered                [0...N]
        self.Csm=[]         # state Cov smoothed                [0...N] *------INIT when smoothing
        # Residuals
        self.rp=[]          # residual predicted                [_,1...N] *Initial value needs a placeholder
        self.Rp=[]          # residual Cov predicted            [_,1...N] *Initial value needs a placeholder
        self.Rf=[]          # residual Cov filtered             [0...N]
        self.Rsm=[]         # residual Cov smoothed             [0...N] *------INIT when smoothing
        # Gains
        self.K=[]           # Forward Kalman gain               [0...N] *Initial value derived from Cf[0] and H[0]
        self.A=[]           # Backward Kalman gain              [0...N-1]
        # Chi-squares
        self.chift=[]       # Forward (filtering) chi2          [0...N] *Initial value is constant, chift[0] = 0
        self.chism=[]       # Backward (smoothing) chi2         [0...N] *------INIT when smoothing

        self.SMOOTH_STEP_TRIED = False # Flag to indicate if the current smoothing step is already executed by "smooth_step_try"

    def init_filter(self, m0, V0, H0, Xf0, Cf0, Rf0):
        self.m.append(m0)
        self.V.append(V0)
        self.H.append(H0)
        self.Xf.append(Xf0)
        self.Cf.append(Cf0)
        self.Rf.append(Rf0)
        K0 = Cf0@H0.T@inv(V0)
        self.K.append(K0)
        self.chift.append(0)

        # For those "predicted" variables, fill them with initial state to align the index
        self.Xp.append(Xf0)
        self.Cp.append(Cf0)
        self.rp.append(np.zeros_like(m0))
        self.Rp.append(np.identity(len(m0)))

        self.Ndim_meas = len(m0)
        self.Ndim_stat = len(Xf0)
        self.identity_measure = np.identity(self.Ndim_meas)
        self.identity_state = np.identity(self.Ndim_stat)

    # A): Forward iterations
    def forward_predict(self, mi, Vi, Hi, Fi, Qi):
        Xp_i = Fi@self.Xf[-1]           # Predicted next state (Xp_i)
        Cp_i = Fi@self.Cf[-1]@Fi.T + Qi # Predicted next state covariance (Cp_i)
        rp_i = mi - Hi@Xp_i             # Predicted residual (rp_i)
        Rp_i = Vi + Hi@Cp_i@Hi.T        # Predicted residual covariance (Rp_i)
        
        # Append input parameters
        self.m.append(mi)
        self.V.append(Vi)
        self.H.append(Hi)
        self.F.append(Fi)
        self.Q.append(Qi)
        # Append predicted values
        self.Xp.append(Xp_i)
        self.Cp.append(Cp_i)
        self.rp.append(rp_i)
        self.Rp.append(Rp_i)

    def forward_filter(self):
        H = self.H[-1]
        Cp = self.Cp[-1]

        K = Cp.dot(H.T).dot(inv(self.Rp[-1]))               # Kalman Gain K
        Xf = self.Xp[-1] + K@self.rp[-1]# Filtered state: combo of predicted state, measured values, covariance matrix, Kalman Gain
        Cf = (self.identity_state - K@H).dot(Cp)            # Filtered Covariance Matrix
        rf = self.m[-1] - H@Xf                              # Filtered residual
        Rf = (self.identity_measure - H@K).dot(self.V[-1])  # Residual Covariance matrix
        chi2 = rf.T @ inv(Rf) @ rf                          # Chi2 contribution

        # Append filtered values
        self.K.append(K)
        self.Xf.append(Xf)
        self.Cf.append(Cf)
        self.Rf.append(Rf)
        self.chift.append(chi2)
        self.chift_total = sum(self.chift)

    # B): Backward Recursion
    def init_smooth(self):
        # Initialized with the last filtered state (X) and Covariance (C)
        self.Xsm.append(self.Xf[-1])
        self.Csm.append(self.Cf[-1])
        self.Rsm.append(0)
        self.chism.append(0)
        self.STEPS_LIST  = range(len(self.Xf)-2,-1,-1) # make a list from N-1 to 0 (inclusive)
        self.CURRENT_STEP=self.STEPS_LIST[0]

    def smooth_step_try(self):
        i = self.CURRENT_STEP
        if i == -1:
            print("Smoothing done, the current state is already the first step")
            return -1
        
        # Kalman Gain A
        try:
            A = self.Cf[i].dot(self.F[i].T).dot(inv(self.Cp[i+1]))
        except Exception as e:
            # print(self.Cp, i)
            # print(inv(np.diag(np.diag(self.Cp[i+1]))))
            A = self.Cf[i].dot(self.F[i].T).dot(inv(np.diag(np.diag(self.Cp[i+1]))))
            print("   Error during smoothing:", e)
        
        Xsm = self.Xf[i] + A.dot(self.Xsm[0] - self.Xp[i+1])            # State
        Csm = self.Cf[i] + A.dot(self.Csm[0] - self.Cp[i+1]).dot(A.T)   # Covariance Matrix
        rsm = self.m[i] - self.H[i]@Xsm                                 # Residual
        Rsm = self.V[i] - self.H[i]@Csm@self.H[i].T                     # Covariance of Residual
        chism = rsm.T @ inv(Rsm) @ rsm                                  # Chi2 contribution

        # Record the result to be used again in "smooth_step()"
        self.SMOOTH_STEP_TRIED = True
        self.Xsm_temp = Xsm
        self.Csm_temp = Csm
        self.Rsm_temp = Rsm
        self.A_temp = A
        self.chism_temp = chism
        return chism

    def smooth_step(self, drop=False):
        """
        Run smoothing with the option to ignore outlider
        drop (bool): If True, revert the information of the current step
        """
        i = self.CURRENT_STEP
        if i == -1:
            print("Smoothing done, the current state is already the first step")
            return -1

        # Use the other function to run smoothing step
        if not self.SMOOTH_STEP_TRIED:
            self.smooth_step_try()
        self.SMOOTH_STEP_TRIED = False

        # If user choose to drop this step, revert the contribution to the smoothed state
        if drop:
            # Modified Kalman Gain K (equation (12b) in Fruhwirth )
            Kmod = self.Csm_temp @ (self.H[i].T) @ inv(-self.V[i] + self.H[i]@self.Csm_temp@self.H[i].T)
            # Change the smoothed state
            self.Xsm_temp = self.Xsm_temp + Kmod @ (self.m[i] - self.H[i]@self.Xsm_temp)
            # Change the smoothed Covariance
            self.Csm_temp = (self.identity_state - Kmod @ self.H[i]) @ self.Csm_temp
            self.chism_temp = 0
        
        # Insert smoothed values
        self.Xsm.insert(0,self.Xsm_temp)        # Insert the smoothed state
        self.Csm.insert(0,self.Csm_temp)        # Insert the smoothed covariance matrix
        self.Rsm.insert(0,self.Rsm_temp)        # Insert the residual covariance matrix
        self.A.insert(0,self.A_temp)            # Insert the Kalman Gain A
        self.chism.insert(0,self.chism_temp)    # Insert the chi2 contribution
        self.chism_total = sum(self.chism)      # Total chi-square
        # Move backward by one step
        self.CURRENT_STEP-=1
        
    def backward_smooth(self):
        self.init_smooth()
        while self.CURRENT_STEP>=0:
            self.smooth_step()