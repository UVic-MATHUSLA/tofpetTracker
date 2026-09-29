import copy
import numpy as np
import scipy as sp
import iminuit

from tracker import kalmanfilter as KF
from tracker import datatypes


class HitPair:
    def __init__(self, hits, dr_range=[-150,150]):
        self.dr_range = dr_range
        self.hit_pair_info = HitPair.get_hit_pair_info(hits, dr_range=dr_range)


    def exists_pair(self, hit1, hit2):
        ind_pair = (min(hit1.ind, hit2.ind), max(hit1.ind, hit2.ind))
        return ind_pair in self.hit_pair_info


    def exists_hit(self, hit1):
        ind = hit1.ind
        keys = self.hit_pair_info.keys()
        for key in keys:
            if ind in key:
                return True
        return False


    def pop_hit(self, hit):
        keys = list(self.hit_pair_info.keys())
        for key in keys:
            if hit.ind in key:
                self.hit_pair_info.pop(key)


    def pop_ind(self, ind):
        keys = list(self.hit_pair_info.keys())
        for key in keys:
            if ind in key:
                self.hit_pair_info.pop(key)


    def len(self):
        return len(self.hit_pair_info.keys())


    @staticmethod
    def get_hit_pair_info(hits, dr_range=[-100, 100]):
        hit_pair_info = dict()

        for i in range(len(hits)):
            for j in range(i+1, len(hits)):
                if hits[i].layer == hits[j].layer: continue

                ind_pair = tuple(sorted([hits[i].ind, hits[j].ind]))
                dr = np.linalg.norm([hits[i].x-hits[j].x, hits[i].y-hits[j].y, hits[i].z-hits[j].z]) - 29.979* abs(hits[i].t-hits[j].t)
                if dr < dr_range[0] or dr > dr_range[1]: continue   # Speed cut
                hit_pair_info[ind_pair] = dr
        return hit_pair_info

class hit:
    @staticmethod
    def make_hits(x, y, z, t, ylayers):
        Y_LAYERS = ylayers
        det_width  = 4      # 4cm per bar
        det_height = 1      #[cm]
        time_resolution = 1 #[ns], single channel
        refraction_index = 1.58

        unc_trans = det_width/np.sqrt(12)
        unc_long = time_resolution*sp.constants.c/1e7/np.sqrt(2)/refraction_index
        UNC_T = time_resolution/np.sqrt(2)  # ns
        UNC_Y = det_height/np.sqrt(12)      # uncertainty in thickness, cm

        hits=[]
        for i, layer in enumerate(Y_LAYERS):
            if layer%2==1: hits.append(datatypes.Hit(x[i], y[i], z[i], t[i], unc_trans, UNC_Y, unc_long, UNC_T, layer, i))
            else: hits.append(datatypes.Hit(x[i], y[i], z[i], t[i], unc_long, UNC_Y, unc_trans, UNC_T, layer, i))

        return hits

    @staticmethod
    def gen_hits(x0=0, y0=0, z0=0, t0=0, Ax=0.3, Az=0.2, At=1/28, N_LAYERS = 8):
        Y_LAYERS = 12_00 + np.arange(N_LAYERS) * 80

        det_width  = 4      # 4cm per bar
        det_height = 1      #[cm]
        time_resolution = 1 #[ns], single channel
        refraction_index = 1.58

        unc_trans = det_width/np.sqrt(12)
        unc_long = time_resolution*sp.constants.c/1e7/np.sqrt(2)/refraction_index
        unc_vert = det_height/np.sqrt(12)
        unc_time = time_resolution

        hits=[]
        hits_truth=[]
        for i in range(N_LAYERS):
            dy = Y_LAYERS[i] - Y_LAYERS[0]
            hits_truth.append(datatypes.Hit(x0+Ax*dy, Y_LAYERS[i], z0+Az*dy, t0+At*dy , 0, 0, 0, 0, i, i, 0, 0))

            if i%2==1:
                hits.append(datatypes.Hit(hits_truth[-1].x//det_width*det_width,
                                          hits_truth[-1].y,
                                          hits_truth[-1].z+np.random.normal(0,unc_long),
                                          hits_truth[-1].t+np.random.normal(0,unc_time),
                                          unc_trans, unc_vert, unc_long, unc_time, i, i, 0, 0))
            else:
                hits.append(datatypes.Hit(hits_truth[-1].x+np.random.normal(0,unc_long),
                                          hits_truth[-1].y,
                                          hits_truth[-1].z//det_width*det_width,
                                          hits_truth[-1].t+np.random.normal(0,unc_time),
                                          unc_long, unc_vert, unc_trans, unc_time, i, i, 0, 0))

        par_truth = [x0, z0, t0, Ax, Az, At]
        return hits, hits_truth, np.array(par_truth)


class track:
    @staticmethod
    def init_state(hits):
        """m0, V0, H0, Xf0, Cf0, Rf0"""
        dt = hits[1].t - hits[0].t
        dx = hits[1].x - hits[0].x
        dy = hits[1].y - hits[0].y
        dz = hits[1].z - hits[0].z

        # Initial State Vector X0
        Xf0 = np.array([hits[1].x, hits[1].z, hits[1].t, dx/dy, dz/dy, dt/dy])
        
        # Initial Covariance P0
        J=np.array([[ 0       , 0           , 0       , 0       , 1       , 0             , 0     , 0     ],
                    [ 0       , 0           , 0       , 0       , 0       , 0             , 1     , 0     ],
                    [ 0       , 0           , 0       , 0       , 0       , 0             , 0     , 1     ],
                    [- 1 / dy, dx / (dy*dy) , 0       , 0       , 1 / dy  , - dx / (dy*dy), 0     , 0     ],
                    [0       , dz / (dy*dy) , - 1 / dy, 0       , 0       , - dz / (dy*dy), 1 / dy, 0     ],
                    [0       , dt / (dy*dy) , 0       , - 1 / dy, 0       , - dt / (dy*dy), 0     , 1 / dy]])

        err0 = np.diag([hits[0].x_err, hits[0].y_err, hits[0].z_err, hits[0].t_err,
                        hits[1].x_err, hits[1].y_err, hits[1].z_err, hits[1].t_err])**2
        Cf0 = J.dot(err0).dot(J.T)

        # the rest
        m0 = np.array([hits[1].x, hits[1].z, hits[1].t])
        V0 = np.diag([hits[1].x_err, hits[1].z_err, hits[1].t_err])**2
        Rf0 = np.diag([hits[1].x_err, hits[1].z_err, hits[1].t_err])**2
        H0 = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]])

        return m0, V0, H0, Xf0, Cf0, Rf0


    @staticmethod
    def add_measurement(hit, dy, velocity=None, multiple_scattering_p=500, multiple_scattering_length=0.06823501107481977):
        """Calculate matrix when adding a new hit"""
        mi = np.array([hit.x, hit.z, hit.t])                            # measurement
        Vi = np.diag([hit.x_err, hit.z_err, hit.t_err])**2              # measurement uncertainty
        Hi = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]])    # measurement matrix
        Fi = np.array([[1, 0, 0,dy, 0, 0],
                       [0, 1, 0, 0,dy, 0],
                       [0, 0, 1, 0, 0,dy],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
        Qi = track.update_Q(dy, *velocity, multiple_scattering_p, multiple_scattering_length) if velocity is not None else 0
        return mi, Vi, Hi, Fi, Qi


    @staticmethod
    def update_Q(dy, Ax, Az, At, multiple_scattering_p=500, multiple_scattering_length=0.06823501107481977):
        # precalculate some numbers
        Ax2 = Ax**2
        Az2 = Az**2
        At2 = At**2
        dy2 = dy**2
        P4P5 = (1+Ax2+Az2)
        P4P52 = P4P5*P4P5
        sin_theta = np.power(Ax**2+Az**2+1, -1/2)

        # Q_block1 = np.array([[(1+Ax2)*P4P5,      Ax*Az*P4P5 , (Ax-1)*P4P5*At],
        #                      [  Ax*Az*P4P5,    (1+Az2)*P4P5,  (Az-1)*P4P5*At],
        #                      [ (Ax-1)*P4P5*At,  (Az-1)*P4P5*At, (Ax2+Az2)*At2]])

        Q_block1 = np.array([[(1+Ax2)*P4P5,      Ax*Az*P4P5 , Ax*P4P52*At],
                             [  Ax*Az*P4P5,    (1+Az2)*P4P5,  Az*P4P52*At],
                             [ Ax*P4P52*At,    Az*P4P52*At,   (Ax2+Az2)*P4P52*At2]])

        # Q_block1 = np.array([[(1+Ax2)*P4P5,      Ax*Az*P4P5 , 0],
        #                      [  Ax*Az*P4P5,    (1+Az2)*P4P5,  0],
        #                      [ 0,0,0]])

        Q = np.block([[Q_block1*dy2, Q_block1*dy],
                      [Q_block1*dy , Q_block1]])
        # Q = np.block([[Q_block1*0, Q_block1*0],
        #               [Q_block1*0 , Q_block1]])

        sigma_ms2 = track.scattering_angle(multiple_scattering_length/sin_theta, momentum_MeV=multiple_scattering_p)**2
        Q = Q * sigma_ms2
        return Q


    @staticmethod
    def update_Q_partial(dy, Ax, Az, At, multiple_scattering_p=500, multiple_scattering_length=0.06823501107481977):
        # precalculate some numbers
        Ax2 = Ax**2
        Az2 = Az**2
        At2 = At**2
        dy2 = dy**2
        P4P5 = (1+Ax2+Az2)
        P4P52 = P4P5*P4P5
        sin_theta = np.power(Ax**2+Az**2+1, -1/2)

        Q_block1 = dy2 * np.array([[ (1+Ax2)*P4P5,   Ax*Az*P4P5, Ax*P4P52*At],
                                    [  Ax*Az*P4P5, (1+Az2)*P4P5, Az*P4P52*At],
                                    [ Ax*P4P52*At,  Az*P4P52*At, (Ax2+Az2)*P4P52*At2]])

        sigma_ms2 = track.scattering_angle(multiple_scattering_length/sin_theta, momentum_MeV=multiple_scattering_p)**2
        Q_block1 = Q_block1 * sigma_ms2
        return Q_block1


    @staticmethod
    def scattering_angle(l_rad_relative, momentum_MeV):
        sigma_ms = 13.6 * np.sqrt(l_rad_relative) * (1 + 0.038 * np.log(l_rad_relative)) / momentum_MeV
        return sigma_ms 


    @staticmethod
    def run_kf(hits, initial_state=None, initial_cov=None, multiple_scattering=False, propagate_state_0="False"):
        kf = KF.KalmanFilter()

        # Set initial state using first two hits to initiate
        m0, V0, H0, Xf0, Cf0, Rf0 = track.init_state(hits)
        if initial_state is not None: Xf0 = initial_state
        if initial_cov is not None: Cf0 = initial_cov
        kf.init_filter( m0, V0, H0, Xf0, Cf0, Rf0)

        # Feed all measurements to KF
        start_ind = 2 if initial_state is None else 1
        for i in range(start_ind,len(hits)):
            # get updated matrix
            hit = hits[i]
            dy = hits[i].y-hits[i-1].y

            # If you don't need multiple scattering:
            if not multiple_scattering: mi, Vi, Hi, Fi, Qi = track.add_measurement(hit, dy)
            else:
                Ax, Az, At = kf.Xf[-1][3:]
                velocity = [Ax, Az, At] #[Ax/At, 1/At, Az/At]
                mi, Vi, Hi, Fi, Qi = track.add_measurement(hit, dy, velocity=velocity)

            kf.forward_predict(mi, Vi, Hi, Fi, Qi)
            kf.forward_filter()

        kf.backward_smooth()
        return kf


    @staticmethod
    def run_kf_fast(hits, initial_state=None, initial_cov=None, multiple_scattering=False):
        kf = KF.KalmanFilterFind()

        # Set initial state using first two hits to initiate
        m0, V0, H0, Xf0, Cf0, Rf0 = track.init_state(hits)
        if initial_state is not None: Xf0 = initial_state
        if initial_cov is not None: Cf0 = initial_cov
        kf.init_filter(m0, V0, H0, Xf0, Cf0, Rf0)

        # Feed all measurements to KF
        start_ind = 2 if initial_state is None else 1
        for i in range(start_ind,len(hits)):
            # get updated matrix
            hit = hits[i]
            dy = hits[i].y-hits[i-1].y

            # If you don't need multiple scattering:
            if not multiple_scattering: mi, Vi, Hi, Fi, Qi = track.add_measurement(hit, dy)
            else:
                Ax, Az, At = kf.Xf[-1][3:]
                velocity = [Ax, Az, At] #[Ax/At, 1/At, Az/At]
                mi, Vi, Hi, Fi, Qi = track.add_measurement(hit, dy, velocity=velocity)

            kf.update_matrix(Vi, Hi, Fi, Qi)
            kf.forward_filter(np.array([mi.x, mi.z, mi.t]))
        return kf


    @staticmethod
    def group_hits_by_layer(hits, used_index=[]):
        # Get layers and group hits by layer
        layers = np.unique([hit.layer for hit in hits])
        HitsLayerGrouped = {layer:[] for layer in layers}
        for ihit in range(len(hits)):
            if hits[ihit].ind not in used_index:
                HitsLayerGrouped[hits[ihit].layer].append(hits[ihit])
        # Remove empty layer
        for layer in list(HitsLayerGrouped.keys()):
            if HitsLayerGrouped[layer] == []:
                HitsLayerGrouped.pop(layer)
        return HitsLayerGrouped


    @staticmethod
    def cov_point_track(point, track, point_unc=None):
        x, y, z, t = point
        dy = y - track.y0
        # Covariance
        jac = np.array([[1, 0, 0, dy,  0,  0],
                        [0, 1, 0,  0, dy,  0],
                        [0, 0, 1,  0,  0, dy]])
        covariance = jac @ track.cov @ jac.T

        # Add the uncertainty of the point
        if point_unc is not None:
            if np.array(point_unc).ndim == 1:
                x_err, y_err, z_err, t_err = point_unc
                covariance += np.diag(np.array([x_err, z_err, t_err])**2)
            elif np.array(point_unc).ndim == 2:
                covariance += point_unc

        return covariance


    @staticmethod
    def chi2_point_track(point, track_this, point_unc=None, multiple_scattering=True, speed_constraint=False,\
            multiple_scattering_p=500, multiple_scattering_length=0.06823501107481977):
        """ 
        Calculate the chi-squre distance between a 4-D point [x,y,z,t] and a track parameterized 
        by [x0, y0, z0, t0, Ax, Az, At] where Ax = dx/dy, Az = dz/dy, At = dt/dy
        point (list): [x,y,z,t]
        track (namedtuple): namedtuple("Track", ["x0", "y0", "z0", "t0", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        point_unc (1d-list or 2d-list or None): [x_err,y_err,z_err,t_err]
        chi2 (float):chi-square distance between the point and the track
        """
        if track_this.Ay == 1:
            track_this = track_this
            point = point
            point_unc = point_unc
        else:
            ind_flip = 2 if (track_this.Az == 1) else 0
            track_this = general.flip_track(track_this, [1, ind_flip])
            point = general.flip_list(point, [1, ind_flip])
            if point_unc is not None:
                if np.array(point_unc).ndim == 1:
                    point_unc = general.flip_list(point_unc, [1, ind_flip])
                else:
                    point_unc = general.flip_matrix(point_unc, [1, ind_flip])

        x, y, z, t = point
        dy = y - track_this.y0
        Ax = track_this.Ax
        Az = track_this.Az
        At = track_this.At
        if speed_constraint: At = np.sqrt(Ax**2+Az**2+1)/(sp.constants.c*1e-7)

        # Residual
        track_x = track_this.x0 + Ax*dy
        track_z = track_this.z0 + Az*dy
        track_t = track_this.t0 + At*dy
        residual = np.array([track_x-x, track_z-z, track_t-t])

        # Covariance
        # jac=np.array([[	1,  0,	0,  dy,   0,   0],
        #               [	0,  1,  0,   0,  dy,   0],
        #               [	0,  0, 	1,   0,   0,  dy]])
        # Equivalent to uisng Jacobian, covariance = jac @ track_this.cov @ jac.T faster
        covariance = track_this.cov[:3,:3] + dy*track_this.cov[3:,:3] + dy*track_this.cov[:3,3:] + dy*dy*track_this.cov[3:,3:]

        # Add the uncertainty of the point
        if point_unc is not None:
            if np.array(point_unc).ndim == 1:
                x_err, y_err, z_err, t_err = point_unc
                covariance += np.diag(np.array([x_err, z_err, t_err])**2)
            elif np.array(point_unc).ndim == 2:
                covariance += point_unc

        # Add the uncertainty from multiple scattering in the last layer
        if multiple_scattering:
            Q_partial = track.update_Q_partial(dy, Ax, Az, At, multiple_scattering_p, multiple_scattering_length)
            covariance += Q_partial*4

        # Finally, calculate Chi-square with total covariance
        chi2 = residual.T @ np.linalg.inv(covariance) @ residual
        return chi2


    @staticmethod
    def chi2_point_track_time(point, track_this, point_unc=None, multiple_scattering=True,\
            multiple_scattering_p=500, multiple_scattering_length=0.06823501107481977):
        """ 
        Calculate the chi-squre distance between a 4-D point [x,y,z,t] and a track parameterized 
        by [x0, y0, z0, t0, Ax, Az, At] where Ax = dx/dy, Az = dz/dy, At = dt/dy, use time as the parameter
        point (list): [x,y,z,t]
        track (namedtuple): namedtuple("Track", ["x0", "y0", "z0", "t0", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        point_unc (1d-list or 2d-list or None): [x_err,y_err,z_err,t_err]
        chi2 (float): chi-square distance between the point and the track
        """
        x, y, z, t = point
        dt = t - track_this.t0

        # Residual
        track_x = track_this.x0 + track_this.Ax/track_this.At*dt
        track_y = track_this.y0 + 1/track_this.At*dt
        track_z = track_this.z0 + track_this.Az/track_this.At*dt
        residual = np.array([track_x-x, track_y-y, track_z-z])

        # Covariance
        Ax = track_this.Ax
        Az = track_this.Az
        At = track_this.At
        jac = np.array([[1, 0, -Ax/At, dt/At,     0, -Ax*dt/At**2],
                        [0, 0,  -1/At,     0,     0,  -1*dt/At**2],
                        [0, 1, -Az/At,     0, dt/At, -Az*dt/At**2]])
        covariance = jac @ track_this.cov @ jac.T

        # Add the uncertainty of the point
        if point_unc is not None:
            if np.array(point_unc).ndim == 1:
                x_err, y_err, z_err, t_err = point_unc
                covariance += np.diag(np.array([x_err, z_err, t_err])**2)
            elif np.array(point_unc).ndim == 2:
                covariance += point_unc

        # Add the uncertainty from multiple scattering in the last layer
        if multiple_scattering:
            Q_partial = track.update_Q_partial(dt, Ax, Az, At, multiple_scattering_p, multiple_scattering_length)
            covariance += Q_partial*4

        # Finally, calculate Chi-square with total covariance
        chi2 = residual.T @ np.linalg.inv(covariance) @ residual
        return chi2


    @staticmethod
    def closest_approach_midpoint(tr1, tr2):
        """
        Calculate the closest approach between two lines
        tr1, tr2 (list): ["x0", "y0", "z0", "vx", "vy", "vz", "t0"]
        return: midpoint([x,y,z,t]), distance
        """
        rel_v = tr2[3:6] - tr1[3:6]
        rel_v2 = np.dot(rel_v, rel_v)

        # Find the time at midpoint
        displacement = tr1[:3] - tr2[:3] # position difference
        t_ca = (np.dot(displacement, rel_v) + np.dot((tr2[3:6]*tr2[6] - tr1[3:6]*tr1[6]), rel_v))/rel_v2

        pos1 = tr1[:3] + tr1[3:6]*(t_ca - tr1[6])
        pos2 = tr2[:3] + tr2[3:6]*(t_ca - tr2[6])
        midpoint = (pos1 + pos2)*(0.5)
        midpoint = np.append(midpoint, t_ca)
        distance = np.linalg.norm((pos1- pos2))
        return midpoint, distance


    @staticmethod
    def closest_approach_midpoint_Track(track1, track2):
        """
        Calculate the closest approach between two tracks
        track1, track2 (namedtuple): namedtuple("Track", ["x0", "y0", "z0", "t0", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        return: midpoint([x,y,z,t]), distance
        """
        tr1 = np.array([track1.x0, track1.y0, track1.z0, track1.Ax/track1.At, track1.Ay/track1.At, track1.Az/track1.At, track1.t0])
        tr2 = np.array([track2.x0, track2.y0, track2.z0, track2.Ax/track2.At, track2.Ay/track2.At, track2.Az/track2.At, track2.t0])
        return track.closest_approach_midpoint(tr1,tr2)


    @staticmethod
    def line_distance(tr1,tr2,time_center):
        """ 
        Calculate the distance of two lines at a certain time
        tr1, tr2 (list): ["x0", "y0", "z0", "vx", "vy", "vz", "t0"])
        time (float): the time to calculate distance
        """
        pos1 = tr1[:3] + tr1[3:6]*(time_center - tr1[6])
        pos2 = tr2[:3] + tr2[3:6]*(time_center - tr2[6])
        displacement = pos1-pos2
        return np.dot(displacement,displacement) 


    @staticmethod
    def position(track, t=None, y=None, x=None, z=None):
        """
        Calculate the position of a track at a given time or y-coordinate
        track (namedtuple): namedtuple("Track", ["x0", "y0", "z0", "t0", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        t (float): the time to calculate position
        y (float): the y-coordinate to calculate position
        x (float): the x-coordinate to calculate position
        z (float): the z-coordinate to calculate position
        return: [x, y, z, t]
        """
        if y is not None:
            dy = y-track.y0
            x = track.x0 + track.Ax*dy
            z = track.z0 + track.Az*dy
            t = track.t0 + track.At*dy 
        elif t is not None:
            dt = t-track.t0
            x = track.x0 + track.Ax/track.At*dt
            y = track.y0 + 1/track.At*dt
            z = track.z0 + track.Az/track.At*dt
        return np.array([x,y,z,t])


    @staticmethod
    def distance_to_point(line, point):
        """
        Calculate the distance from a line to a point
        line (namedtuple): namedtuple("Track", ["x0", "y0", "z0", "t0", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        point (list): [x, y, z, t]
        return: distance
        """
        track_position = track.position(line, t=point[3])
        return np.linalg.norm((track_position-point)[:3])


    @staticmethod
    def guess_track(hits):
        # Guess initial value
        x0_init = hits[0].x
        z0_init = hits[0].z
        t0_init = hits[0].t
        
        dy=hits[-1].y-hits[0].y
        Ax_init = (hits[-1].x-hits[0].x)/dy
        Az_init = (hits[-1].z-hits[0].z)/dy
        At_init = (hits[-1].t-hits[0].t)/dy
        return  (x0_init, z0_init,t0_init,Ax_init,Az_init,At_init)


    @staticmethod
    def fit_track(hits, guess):
        x0_init, z0_init,t0_init,Ax_init,Az_init,At_init = guess

        m = iminuit.Minuit(chi2_track(hits),x0=x0_init, z0=z0_init, t0=t0_init, Ax=Ax_init,Az=Az_init, At=At_init)
        # m.fixed["y0"]=True
        m.limits["x0"]=(-100000,100000)
        m.limits["z0"]=(-100000,100000)
        m.limits["t0"]=(-100,1e5)
        m.limits["Ax"]=(-10,10) # Other
        m.limits["Az"]=(-10,10)
        m.limits["At"]=(0.001,0.2) # Beam direction; From MKS unit to cm/ns = 1e2/1e9=1e-7
        m.errors["x0"]=0.1
        m.errors["z0"]=0.1
        m.errors["t0"]=0.1
        m.errors["Ax"] = 0.001
        m.errors["At"] = 0.0001
        m.errors["Az"] = 0.001

        m.migrad()  # run optimiser
        m.hesse()   # run covariance estimator
        return m


    @staticmethod
    def fit_track_scattering(hits, guess):
        x0_init, z0_init,t0_init,Ax_init,Az_init,At_init = guess

        m = iminuit.Minuit(chi2_track_scattering(hits),x0=x0_init, z0=z0_init, t0=t0_init, Ax=Ax_init,Az=Az_init, At=At_init)
        m.limits["x0"]=(-100000,100000)
        m.limits["z0"]=(-100000,100000)
        m.limits["t0"]=(-100,1e5)
        m.limits["Ax"]=(-10,10) # Other
        m.limits["Az"]=(-10,10)
        m.limits["At"]=(0.001,0.2) # Beam direction; From MKS unit to cm/ns = 1e2/1e9=1e-7
        m.errors["x0"]=0.1
        m.errors["z0"]=0.1
        m.errors["t0"]=0.1
        m.errors["Ax"] = 0.0001
        m.errors["At"] = 0.0001
        m.errors["Az"] = 0.0001

        m.migrad()  # run optimiser
        m.hesse()   # run covariance estimator
        return m


    @staticmethod
    def fit_track_ana(hits, scattering = False, iters = 2):
        y0 = hits[0].y

        X = np.array([hit.x for hit in hits])
        Z = np.array([hit.z for hit in hits])
        T = np.array([hit.t for hit in hits])
        H = np.array([[1,hit.y-y0] for hit in hits])

        if scattering is False:
            Vx_inv = np.diag([1/hit.x_err for hit in hits ])
            Vz_inv = np.diag([1/hit.z_err for hit in hits ])
            Vt_inv = np.diag([1/hit.t_err for hit in hits ])
        elif scattering is True:
            if iters == 0:
                Param_all, Error_all, chi2_all = track.fit_track_ana(hits, scattering=False)
            else:
                iters -= 1
                Param_all, Error_all, chi2_all = track.fit_track_ana(hits, scattering=True, iters=iters)

            # chi2_object = chi2_track_scattering(hits)
            Vx, Vz = chi2_track_scattering(hits).get_cov(Param_all[3], Param_all[4])
            Vx_inv = np.linalg.inv(Vx)
            Vz_inv = np.linalg.inv(Vz)
            Vt_inv = np.diag([1/hit.t_err for hit in hits ])

        Error_x = np.linalg.inv(H.T @ Vx_inv @ H)
        Param_x = Error_x @ (H.T @ Vx_inv @ X)
        chi2_x = (X - H@Param_x).T @ Vx_inv @ (X - H@Param_x)

        Error_z = np.linalg.inv(H.T @ Vz_inv @ H)
        Param_z = Error_z @ (H.T @ Vz_inv @ Z)
        chi2_z = (Z - H@Param_z).T @ Vz_inv @ (Z - H@Param_z)

        Error_t = np.linalg.inv(H.T @ Vt_inv @ H)
        Param_t = Error_t @ (H.T @ Vt_inv @ T)  
        chi2_t = (T - H@Param_t).T @ Vt_inv @ (T - H@Param_t)

        Error_all = np.array([  [Error_x[0,0],            0,            0, Error_x[0,1],            0,            0],
                                [           0, Error_z[0,0],            0,            0, Error_z[0,1],            0],
                                [           0,            0, Error_t[0,0],            0,            0, Error_t[0,1]],
                                [Error_x[1,0],            0,            0, Error_x[1,1],            0,            0],
                                [           0, Error_z[1,0],            0,            0, Error_z[1,1],            0],
                                [           0,            0, Error_t[1,0],            0,            0, Error_t[1,1]]])
        Param_all = [Param_x[0],Param_z[0],Param_t[0], Param_x[1],Param_z[1],Param_t[1]]
        chi2_all = chi2_x + chi2_z + chi2_t

        return  Param_all, Error_all, chi2_all


# -------------------------------------
# LS fit
# ------------------------------------
class chi2_track:
    def __init__(self, hits):
        self.hits=hits
        self.func_code = iminuit.util.make_func_code(['x0', 'z0', 't0', 'Ax', 'Az', 'At'])
    def __call__(self, x0, z0, t0, Ax, Az, At):
        error=0
        y0 = 10
        for hit in self.hits:
            dy = (hit.y - self.hits[1].y)
            model_x = x0 + Ax*dy
            model_z = z0 + Az*dy
            model_t = t0 + At*dy
            error+= np.sum(np.power([(model_t-hit.t)/hit.t_err,(model_x-hit.x)/hit.x_err,(model_z-hit.z)/hit.z_err],2))
        return error


# -------------------------------------
# LS fit with error on time
# ------------------------------------
# class chi2_track:
#     def __init__(self, hits):
#         self.hits=hits
#         self.func_code = iminuit.util.make_func_code(['x0', 'y0', 'z0', 't0', 'vx', 'vy', 'vz'])
#     def __call__(self, x0, y0, z0, t0, vx, vy, vz):
#         error_squared=0
#         for hit in self.hits:
#             dt = (hit.t - t0)
#             model_x = x0 + vx*dt
#             model_y = y0 + vy*dt
#             model_z = z0 + vz*dt
#             err_x2 = hit.x_err**2 + vx * hit.t_err
#             err_y2 = hit.y_err**2 + vy * hit.t_err
#             err_z2 = hit.z_err**2 + vz * hit.t_err
#             error_squared += np.sum([(model_x-hit.x)**2/err_x2,(model_x-hit.x)**2/err_y2,(model_z-hit.z)**2/err_z2])
#         return error_squared


# -------------------------------------
# LS fit with multiple scattering
# ------------------------------------
class chi2_track_scattering:
    def __init__(self, hits):
        self.hits=hits
        self.func_code = iminuit.util.make_func_code(['x0', 'z0', 't0', 'Ax', 'Az', 'At'])


    def __call__(self, x0, z0, t0, Ax, Az, At):
        chi2_distance = 0

        cov_x, cov_z = self.get_cov(Ax, Az)
        residuals = []

        for hit in self.hits:
            dy = (hit.y - self.hits[0].y)
            model_x = x0 + Ax*dy
            model_z = z0 + Az*dy
            model_t = t0 + At*dy
            residuals.append([model_x-hit.x, model_z-hit.z, model_t-hit.t])
        t_err = [hit.t_err for hit in self.hits]

        residuals=np.array(residuals)

        chi2_distance = residuals[:,0].T @ np.linalg.inv(cov_x) @residuals[:,0] +\
                        residuals[:,1].T @ np.linalg.inv(cov_z) @residuals[:,1] +\
                        np.sum([(residuals[:,2]/t_err)**2])
        return chi2_distance


    def get_theta0(self, sin_theta, p=500):
        L_Al =  0.4
        L_Sc = 1.0              # [cm] Scintillator
        L_r_Al = 24.0111/2.7    # [cm] Radiation length Aluminum/density of Aluminum
        L_r_Sc = 43             # [cm] Radiation length Scintillator (Saint-Gobain paper)

        L_rad = L_Al / L_r_Al + L_Sc / L_r_Sc   # [rad lengths] orthogonal to Layer
        L_rad /= sin_theta                      # [rad lengths] in direction of track

        sigma_ms = 13.6 * np.sqrt(L_rad) * (1 + 0.038 * np.log(L_rad))
        sigma_ms /= p # [MeV] Divided by 1000 MeV
        return sigma_ms


    def get_cov(self, Ax, Az):
        sin_theta = np.power(Ax**2+Az**2+1, -1/2)
        theta_0 = self.get_theta0(sin_theta)
        hits = self.hits

        cov_xz = np.zeros((len(self.hits), len(self.hits)))
        for i in range(1,len(self.hits)):
            for j in range(1,len(self.hits)):
                cov_xz[i,j] = np.sum([(hits[i].y-hits[k].y)*(hits[j].y-hits[k].y)*theta_0**2 for k in range(0, min(i,j))])

        cov_x = cov_xz* Ax**2
        cov_z = cov_xz* Az**2
        cov_x+=np.diag([hit.x_err**2 for hit in hits])
        cov_z+=np.diag([hit.z_err**2 for hit in hits])
        return np.array(cov_x), np.array(cov_z)


class vertex:
    @staticmethod
    def score_seed(seed_par):
        """ 
        Calculate a score for the vertex seed, this is purely empirical.
        Lower score means better seed quality and should be used first.
        """
        x0,y0,z0,t0, midpoint_chi2, dist_seed, N_compatible_tracks, N_compatible_track_distance, seed_track_unc, seed_track_chi2,seed_track_dist, seed_opening_angle = seed_par

        # Score based on the following items:
        # - Seed chi2
        # - Seed distance
        # - Seed uncertainty
        # - Seed starting point (Higher priority to ones closer to the IP)
        # - Seed track uncertainty
        # - Number of compatible tracks

        # score = 10*midpoint_chi2  + dist_seed + 0.1*y0 + 0.2*seed_track_unc -50*N_compatible_tracks + 0.3*N_compatible_track_distance
        # float score_layers = -100.0* (tracks.first->chi_s.size()+tracks.second->chi_s.size());
        # return closest_dist + score_layers + chi2*10.0 - compatible_tracks*20.0;

        # score = 3*midpoint_chi2  + dist_seed + 0.1*y0 + 0.1*z0 + 0.2*seed_track_unc -50*N_compatible_tracks + 0.3*N_compatible_track_distance
        # score = 3*midpoint_chi2  + dist_seed + 0.2*seed_track_unc
        # score = dist_seed*0.5 + midpoint_chi2*10 - N_compatible_tracks*50 - seed_opening_angle*200 - seed_track_dist
        score = dist_seed + midpoint_chi2*10 - N_compatible_tracks*100 - seed_opening_angle*40 #+ seed_track_dist
        return score


class general:
    @staticmethod
    def flip_list(l, flip_index=[0,1]):
        if flip_index[0] == flip_index[1]: return l
        l = copy.copy(l)
        l[flip_index[0]], l[flip_index[1]] = l[flip_index[1]], l[flip_index[0]]
        return l


    @staticmethod
    def flip_matrix(m, flip_index=[0,1]):
        if flip_index[0] == flip_index[1]: return m
        m = np.array(m)
        m[flip_index, :] = m[flip_index[::-1], :]
        m[:, flip_index] = m[:, flip_index[::-1]]
        return m


    @staticmethod
    def flip_hit(hit_t, flip_index=[0,1]):
        """
        Flip the two coordinates of a hit_t
        hit_t: datatypes.Hit
        flip_index: list, [ind1, ind2], both index should be from {0,1,2}
        """
        if flip_index[0] == flip_index[1]: return hit_t
        hit_t = list(hit_t)
        hit_t[flip_index[0]], hit_t[flip_index[1]] = hit_t[flip_index[1]], hit_t[flip_index[0]]
        hit_t[flip_index[0]+4], hit_t[flip_index[1]+4] = hit_t[flip_index[1]+4], hit_t[flip_index[0]+4]
        hit_new = datatypes.Hit(*hit_t)
        return hit_new


    @staticmethod
    def flip_track(track_t, flip_index=[0,1]):
        """
        Flip the two coordinates of a track
        track_t: datatypes.Track
        flip_index: list, [ind1, ind2], both index should be from {0,1,2}
        """
        # Track = namedtuple("Track", ["x0", "y0", "z0", "t0", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        if flip_index[0] == flip_index[1]: return track_t
        cov = track_t.cov
        chi2 = track_t.chi2
        ind = track_t.ind
        hits = track_t.hits
        hits_filtered = track_t.hits_filtered
        track_t = list(track_t)

        # xyzt and error
        track_t[flip_index[0]], track_t[flip_index[1]] = track_t[flip_index[1]], track_t[flip_index[0]]
        track_t[flip_index[0]+4], track_t[flip_index[1]+4] = track_t[flip_index[1]+4], track_t[flip_index[0]+4]
        # covariance matrix
        cov_new = general.flip_matrix(cov, flip_index=flip_index)
        # filtered hits
        hits_filtered_new = [general.flip_list(hit_i, flip_index=flip_index) for hit_i in hits_filtered]
        hit_new = datatypes.Track(*track_t[:8], cov_new, chi2, ind, hits, hits_filtered_new)
        return hit_new


    @staticmethod
    def flip_vertex(vertex_t, flip_index=[0,1]):
        """
        Flip the two coordinates of a track
        track_t: datatypes.Track
        flip_index: list, [ind1, ind2], both index should be from {0,1,2}
        """
        # Vertex = namedtuple("Vertex", ["x0", "y0", "z0", "t0", "cov", "chi2", "tracks"])
        if flip_index[0] == flip_index[1]: return vertex_t
        cov = vertex_t.cov
        chi2 = vertex_t.chi2
        tracks = vertex_t.tracks
        vertex_t = list(vertex_t)

        # xyzt
        vertex_t[flip_index[0]],vertex_t[flip_index[1]] = vertex_t[flip_index[1]],vertex_t[flip_index[0]]
        # covariance matrix
        cov_new = general.flip_matrix(cov, flip_index=flip_index)
        vertex_new = datatypes.Vertex(*vertex_t[:4], cov_new, chi2, tracks)
        return vertex_new


class processing:
    @staticmethod
    def drop_hits(hits, efficiency, seed):
        rng = np.random.default_rng(seed)
        hits_keep_mask = rng.binomial(1, efficiency, len(hits))
        for i in range(len(hits))[::-1]:
            if hits_keep_mask[i]==0:
                hits.pop(i)
        return hits

