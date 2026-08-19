import copy
import numpy as np
from numpy.linalg import inv
import scipy as sp
import scipy.constants
import scipy.stats

# Internal modules
from . import utilities as Util
from . import kalmanfilter as KF
from . import datatypes
import functools; print = functools.partial(print, flush=True) #make python actually flush the output!

# ----------------------------------------------------------------------
class TrackFinder:
    def __init__(self, parameters=None, method="recursive", debug=False):
        """
        Initialize the TrackFinder
        ---
        parameters: dict of parameters for the track finding
        method: string, the method to use for track finding
        debug: boolean, whether to print debug information
        """
        self.method = method                        # {"recursive", "greedy"}
        self.debug = debug
        self.parameters={
            "cut_track_SeedSpeed": 1,               # In the unit of c. Limit the maximum speed formed by the seed.
            "cut_track_HitAddChi2": 15,             # Only used when method is "greedy"
            "cut_track_HitDropChi2": -1,            # Set to -1 to turn off (default = 15)
            "cut_track_HitProjectionSigma": 10,     # Number of sigmas
            "cut_track_TrackChi2Reduced": 3,        # Only use this for track with 3 hits
            "cut_track_TrackChi2Prob": 0.9,         # Chi-square probablity (calculated from chi2_cdf(x, DOF))
            "cut_track_TrackNHitsMin": 3,           # Minimum number of hits per track
            "cut_track_TrackSpeed": [25,35],        # [cm/ns], [speed_low, speed_high]. 30 is the speed of light
            "fit_track_MultipleScattering": False,
            "cut_track_MultipleScatteringFind": False,
            "fit_track_Method": "backward",         # Choose one of {"backward", "forward", "forward-seed", "least-square", "least-square-ana"}
            "fit_track_LeastSquareIters":2,         # No need to change
            "multiple_scattering_p": 500,           # [MeV/c] momentum of multiple scattering, 
            "multiple_scattering_length": 0.06823501107481977 # [1] material thickness in the unit of scattering length
        }


    def run(self, hits):
        """
        hits: list of hits to be used for track finding
        ---
        Run all three rounds of kalman filter: find, drop, fit
        This function is a mixer of self.find() and self.filter_smooth()
        ---
        returns a list of tracks, each track is a namedtuple:
        ("Track", ["x0", "y0", "z0", "t", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        """
        self.hits = copy.copy(hits)
        self.hits_grouped = Util.track.group_hits_by_layer(self.hits)
        self.total_layers = len(list(self.hits_grouped.keys()))
        self.tracks = []
        self.seeds = self.seeding(self.hits)
        self.hit_pair = Util.HitPair(self.hits)

        # Check if the minimum number of hits is larger than the total number of layers. If so, return empty tracks
        if self.parameters["cut_track_TrackNHitsMin"] > self.total_layers:
            return self.tracks

        hit_found_inds = []
        for track_TrackNHitsMin in range(self.parameters["cut_track_TrackNHitsMin"], self.total_layers+1)[::-1]:
            if self.debug: print(f"\n\n======Looking for track with {track_TrackNHitsMin} hits======")
            hits_found_all = []
            self.seeds_unused = []
            while len(self.seeds) > 0:
                if len(self.hits_grouped.keys()) < self.parameters["cut_track_TrackNHitsMin"]: # If not enough hits left:
                    break
                
                # ------------------------------------
                # Round 1: Find hits that belongs to one track
                seed = self.seeds[-1]; 
                print(f"Number of seeds: {len(self.seeds)}, Number of unused seeds: {len(self.seeds_unused)}")
                if self.debug: print(f"--- New seed --- \n [Seed]: {seed}")
                hits_found, track_chi2 = self.find_once(self.hits, self.hits_grouped, seed, self.hit_pair)
                # Remove the current seed no matter the track is good or not:
                self.seeds.pop(-1)
                # Apply cuts: if not enough hits, drop this track
                print(len(hits_found))
                if len(hits_found) < track_TrackNHitsMin:
                    if self.debug: print(f"   finding failed (adding), not enough hits. Hits found: {len(hits_found)}")
                    # Keep the seeds that potentially matches to a track
                    if len(hits_found) >= self.parameters["cut_track_TrackNHitsMin"]:
                        self.seeds_unused.append(seed)  
                        # if len(hits_found)==track_TrackNHitsMin-1:
                        #     self.remove_related_seeds(self.seeds, hits_found)
                    continue

                # Sort the hits by time before running the filter
                hits_found.sort(key = lambda hit: hit.t)

                # ------------------------------------
                # Round 2: Run filter and smooth with the option to drop outlier during smoothing
                kalman_result, inds_dropped = self.filter_smooth(hits_found, drop_chi2=self.parameters["cut_track_HitDropChi2"])
                inds_dropped.sort(reverse=True)
                for ind in inds_dropped:
                    hits_found.pop(ind)
                # If not enough hits, drop this track
                if len(hits_found)<track_TrackNHitsMin:
                    if self.debug: print(f"   finding failed (dropping), not enough hits. Hits found: {len(hits_found)}")
                    # Keep the seeds that potentially matches to a track
                    if len(hits_found) >= self.parameters["cut_track_TrackNHitsMin"]:
                        self.seeds_unused.append(seed)
                    continue 

                # ------------------------------------
                # Round 3: Run filter again on found hits
                if self.parameters["fit_track_Method"]=="backward": 
                    # Run filter backwards, no smoothing
                    kalman_result, inds_dropped = self.filter_smooth(hits_found[::-1], drop_chi2=-1)   # This time we run the filter backwards
                    track_output = self.prepare_output_back(kalman_result, hits_found, track_ind = len(self.tracks))

                elif self.parameters["fit_track_Method"]=="forward":
                    # Run filter forward with smoothing, use first two hits to initialize
                    kalman_result, inds_dropped = self.filter_smooth(hits_found, drop_chi2=-1)
                    track_output = self.prepare_output(kalman_result, hits_found, track_ind = len(self.tracks))

                elif self.parameters["fit_track_Method"]=="forward-seed": 
                    # Run filter forward with smoothing, use first and last hit to initialize and set the initial state
                    m0, V0, H0, Xf0, Cf0, Rf0 = Util.track.init_state([hits_found[-1],hits_found[0]]) # Use the first two hits to initiate
                    kalman_result = Util.track.run_kf(hits_found, initial_state=Xf0, initial_cov=Cf0, multiple_scattering=True)
                    track_output = self.prepare_output_v2(kalman_result, hits_found, track_ind = len(self.tracks))

                elif self.parameters["fit_track_Method"]=="least-square": 
                    # Run least square fit
                    guess = Util.track.guess_track(hits_found)
                    fit_ls = Util.track.fit_track_scattering(hits_found,guess) if self.parameters["fit_track_MultipleScattering"] else Util.track.fit_track(hits_found,guess) 
                    popt = fit_ls.values
                    pcov = fit_ls.covariance
                    chi2 = fit_ls.fval
                    track_output = self.prepare_output_ls(popt, pcov, chi2, hits_found, track_ind = len(self.tracks))

                elif self.parameters["fit_track_Method"]=="least-square-ana": 
                    # Run analytical least square fit (less iteration during minimization)
                    popt,pcov,chi2 = Util.track.fit_track_ana(hits_found, scattering = self.parameters["fit_track_MultipleScattering"], iters = self.parameters["fit_track_LeastSquareIters"]) 
                    track_output = self.prepare_output_ls(popt, pcov, chi2, hits_found, track_ind = len(self.tracks))

                # Cut on chi2 probablity
                ndof = 3 * len(hits_found) - 6
                track_chi2 = track_output.chi2
                track_chi2_prob = sp.stats.chi2.cdf(track_chi2, ndof)
                track_chi2_reduced = track_chi2 / ndof
                if (track_chi2_prob > self.parameters["cut_track_TrackChi2Prob"] and ndof > 3) or \
                   (track_chi2_reduced > self.parameters["cut_track_TrackChi2Reduced"] and ndof <= 3):
                    if self.debug: print(f" Track vetoed, chi2 too large. Chi2/nodf: {track_chi2}/{ndof}, prob = {track_chi2_prob}")
                    continue

                # Cut on speed
                state = track_output # Track is a namedtuple("Track", ["x0", "y0", "z0", "t", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
                speed = np.linalg.norm([state.Ax / state.At, state.Az / state.At, 1 / state.At])  
                if not (self.parameters["cut_track_TrackSpeed"][0] < speed < self.parameters["cut_track_TrackSpeed"][1]):
                    if self.debug: print(f" Track vetoed. Speed of the track: {speed}[cm/ns]")
                    continue
                elif self.debug:
                    print(f" [Track found]", track_output)
                    print(" Added hits:")
                    for t in hits_found:
                        print("  ", t)

                self.tracks.append(track_output)            # Add the found track to the list of tracks
                self.remove_related_hits_seeds(hits_found)  # Remove seeds that shares hits of the found track
                hits_found_all.extend(hits_found)           # Add the found hits to the list of all found hits

            hit_found_inds.extend([hit.ind for hit in hits_found_all])  # Add the indices of the found hits to the list of found hit indices
            hit_found_inds.sort(reverse=True)                           # Sort the indices in reverse order to avoid index shifting when removing hits
            self.remove_related_seeds(self.seeds_unused, hits_found_all)# Remove seeds that shares hits of the found track
            self.seeds = copy.copy(self.seeds_unused)                   # Reset the seeds to the unused seeds for the next iteration
            # print("------------------decreasinglayer")
            # print(hit_found_inds)
            # for ind in hit_found_inds:
            #     self.hits.pop(ind)
            # Group the remaining hits
            # self.hits_grouped = Util.track.group_hits_by_layer(self.hits, used_index = hit_found_inds)

        if self.debug:
            print("=========Track finding finished=========")
            print("Tracks found:")
            for t in self.tracks:
                print(t)

        return self.tracks


    def seeding(self, hits, used_index=[]):
        """
        hits: list of hits to be used for track finding
        used_index: list of indices of hits that have already been used
        ---
        Find seed for tracks
        Returns a pair of index of the hit (not the hit itself!) and a score
        ---
        The score is a tuple of (abs(dy), dr), where dy is the difference in y between the two hits, 
        and dr is the distance between the two hits. The seeds are sorted by score, 
        with the best seed having the smallest abs(dy)
        """
        c = sp.constants.c / 1e7        # [cm/ns]
        seeds=[]
        for i in range(len(hits)):
            for j in range(i+1, len(hits)):
                if (hits[i].y == hits[j].y) or (hits[i].layer == hits[j].layer):
                    continue
                if hits[i].ind in used_index   or  hits[j].ind in used_index:
                    continue
                dx = hits[i].x- hits[j].x       # Difference in x
                dy = hits[i].y- hits[j].y       # Difference in y
                dz = hits[i].z- hits[j].z       # Difference in z
                dt = hits[i].t- hits[j].t       # Difference in time
                # ds = np.abs((dx**2+dy**2+dz**2)/c**2-dt**2)
                # ds = ds/dt**2
                # if ds>self.parameters["cut_track_SeedSpeed"]:
                #     continue
                # seeds.append([i,j,ds,-abs(dy)])
                
                dr = np.linalg.norm([dx,dy,dz]) # Distance between the two hits
                ds = abs(dr/c - abs(dt))        # Difference between the distance and the time difference (in units of c)
                if ds > 1:
                    continue
                seeds.append([i,j, dr, abs(dy)])

        # Sorting seeds from small to large score, placing the best one at the end
        seeds.sort(key=lambda s: (s[3], s[2]))
        return seeds


    def find_once(self, hits, hits_layer_grouped, seed, hit_pair):
        """
        hits: list of hits to be used for track finding
        hits_layer_grouped: dict of hits grouped by layer
        seed: list of two indices of hits to be used as the seed for track finding
        hit_pair: HitPair object to check if two hits are compatible
        ---
        Find hits that belongs to one track using the seed
        ---
        Returns a list of hits that belongs to the track and the chi2 of the track
        """
        #### General info ####
        LAYERS = np.sort(list(hits_layer_grouped.keys()))

        ##### Seed ####
        # Check the direction of seed by comparing the time of two hits
        seed_hits = [hits[seed[0]], hits[seed[1]]]
        # Always have the first hit to be first in time
        if (seed_hits[0].t > seed_hits[1].t):
            seed_hits = seed_hits[::-1]
        seed_start_layer = seed_hits[0].layer
        seed_stop_layer  = seed_hits[1].layer
        # Check if needed to find backward or forward
        if (seed_hits[0].layer > seed_hits[1].layer):
            TRACK_DIRECTION = 0 # Downward track
            FIND_BACKWARD_LAYERS = LAYERS[np.argmax(LAYERS>seed_stop_layer):]
        else:
            TRACK_DIRECTION = 1 # Upward track
            FIND_BACKWARD_LAYERS = LAYERS[:np.argmax(LAYERS>seed_stop_layer)-1][::-1]

        FIND_FORWARD = True # Always find forward
        FIND_BACKWARD= True if len(FIND_BACKWARD_LAYERS)>1 else False  # Find backwards only when there are more than one layer before the second hit in of the seed

        ##### Find ####
        hits_found = [seed_hits[1]]
        chi2_found = 0
        if FIND_BACKWARD:
            step_pre = seed_hits[1].y # Keep track of the y of the previous step

            kf_find = KF.KalmanFilterFind()
            kf_find.init_filter(*Util.track.init_state(seed_hits))
            
            if self.debug: print(" Finding backward in layers", FIND_BACKWARD_LAYERS)
            if self.method=="recursive":
                hits_found_backward, chi2 = self.find_in_layers_recursive(hits, hits_layer_grouped, FIND_BACKWARD_LAYERS, kf_find, step_pre)
            else:
                hits_found_backward, chi2 = self.find_in_layers_greedy(hits, hits_layer_grouped, FIND_BACKWARD_LAYERS, kf_find, step_pre, hit_pair)
            # Order of found hits also needs to be reversed for backward finding
            hits_found_backward = hits_found_backward[::-1]
            hits_found_backward.extend(hits_found)
            hits_found = hits_found_backward
        else:
            hits_found = seed_hits

        if FIND_FORWARD:
            if len(hits_found) < 2:
                return [], []

            seed_hits = hits_found[:2]# Reset the seed to be the first and the last hit
            step_pre = seed_hits[1].y # Keep track of the y of the previous step
            if max(LAYERS) == seed_hits[1].layer:
                return hits_found, chi2_found

            if (seed_hits[0].layer > seed_hits[1].layer):
                FIND_FORWARD_LAYERS = LAYERS[:np.argmax(LAYERS>seed_hits[1].layer)-1][::-1]
            else:
                FIND_FORWARD_LAYERS  = LAYERS[np.argmax(LAYERS>seed_hits[1].layer):] 

            kf_find = KF.KalmanFilterFind()
            kf_find.init_filter(*Util.track.init_state(seed_hits)) # Set initial state using two hits specified by the seed
            if self.debug: print(" Finding forward in layers", FIND_FORWARD_LAYERS )
            if self.method=="recursive":
                hits_found_forward, chi2 = self.find_in_layers_recursive(hits, hits_layer_grouped, FIND_FORWARD_LAYERS, kf_find, step_pre)
            else:
                hits_found_forward, chi2 = self.find_in_layers_greedy(hits, hits_layer_grouped, FIND_FORWARD_LAYERS, kf_find, step_pre, hit_pair = hit_pair)
            chi2_found = chi2
            hits_found = hits_found[:2] + hits_found_forward
   
        return hits_found,chi2_found


    def find_in_layers_greedy(self, hits, hits_layer_grouped, layers_to_scan, kf_find, step_pre, cut_chi2=True, hit_pair=None):
        """
        hits: list of hits to be used for track finding
        hits_layer_grouped: dict of hits grouped by layer
        layers_to_scan: list of layers to scan for hits
        kf_find: KalmanFilterFind object to keep track of the state of the filter
        step_pre: float, the y of the previous step
        cut_chi2: boolean, whether to cut on chi2 when adding hits
        hit_pair: HitPair object to check if two hits are compatible
        ---
        Find the hit that has minimum chi2 in each layer
        ---
        Returns a list of hits that belongs to the track and the chi2 of the track
        """
        hits_found = []
        for layer in layers_to_scan:
            hits_thislayer = hits_layer_grouped[layer]
            if len(hits_thislayer)==0:
                continue
            
            # Use the pre-calculated hit pair info to see if there are any compatible hits
            if hit_pair is not None:
                # If there is no hit to match the previous hit, break the loop immediately
                if len(hits_found)>0 and (not hit_pair.exists_hit(hits_found[-1])):
                    print("--no matched hit")
                    break

            hit = hits_thislayer[0]         # Get one hit
            step_this = hits_thislayer[0].y # Get the prediction matrix
            dy = step_this - step_pre       # Step size

            Ax, Az, At = kf_find.Xf[3:]     # Get the velocity from the Kalman filter state
            velocity = [Ax, Az, At] if self.parameters["cut_track_MultipleScatteringFind"] else None # Velocity is needed for multiple scattering

            # Calculate matrices. Only need to do once for all this in the same layer
            _, Vi, Hi, Fi, Qi = Util.track.add_measurement(hits_thislayer[0], dy, velocity, self.parameters["multiple_scattering_p"],self.parameters["multiple_scattering_length"]) 
            kf_find.update_matrix(Vi, Hi, Fi, Qi) # pass matrices to KF

            Xp = kf_find.Xp_i                       # Get the predicted state from the Kalman filter
            Xp_unc = np.sqrt(np.diag(kf_find.Rp_i)) # Get the uncertainty of the predicted state from the Kalman filter
            N_sigma = self.parameters["cut_track_HitProjectionSigma"]   # Number of sigmas to use for the cut
            # Use the total uncertainty of the prediction plus the measurement
            unc_total = [np.linalg.norm([hit.x_err,Xp_unc[0]]), np.linalg.norm([hit.z_err,Xp_unc[1]]), np.linalg.norm([hit.t_err,Xp_unc[2]])]
            # Function to test if new measurement is within N_sigma times the uncertainty ellipsoid
            test_measurement_incompatible = lambda x,z,t: abs(x-Xp[0])>unc_total[0]*N_sigma or \
                                                          abs(z-Xp[1])>unc_total[1]*N_sigma or \
                                                          abs(t-Xp[2])>unc_total[2]*N_sigma or \
                                                          ((x-Xp[0])/unc_total[0])**2 + ((z-Xp[1])/unc_total[1])**2 + ((t-Xp[2])/unc_total[2])**2 > N_sigma**2

            # Calculate chi2 for all hits in the next layer
            # chi2_predict = [kf_find.forward_predict_chi2(np.array([mi.x, mi.z, mi.t])) for mi in hits_thislayer]
            chi2_predict=[]
            chi2_predict_inds =[]
            for imeasurement, m in enumerate(hits_thislayer):
                # Use the pre-calculated hit pair info to narrow down the search
                if hit_pair is not None:
                    if len(hits_found)>0 and (not hit_pair.exists_pair(hits_found[-1], m)):
                        continue
                if test_measurement_incompatible(m.x, m.z, m.t):
                    continue
                else:
                    chi2 = kf_find.forward_predict_chi2(np.array([m.x, m.z, m.t]))
                    chi2_predict.append(chi2)
                    chi2_predict_inds.append(imeasurement)
            if len(chi2_predict) == 0:
                continue

            # Find the hit with minimum chi2
            chi2_min_idx = np.argmin(chi2_predict)
            if chi2_predict[chi2_min_idx]<self.parameters["cut_track_HitAddChi2"] or not cut_chi2:
                # Save the hit either if the chi2 is lower than the threshold, or the cut is disabled
                hits_found.append(hits_thislayer[chi2_predict_inds[chi2_min_idx]])
                # Update the step and the Kalman filter
                step_pre = step_this
                mi = hits_found[-1]
                kf_find.forward_filter(np.array([mi.x, mi.z, mi.t]))
                if self.debug: print("  Hit found:", mi, "; chi2", chi2_predict[chi2_min_idx],chi2_predict[chi2_min_idx]<self.parameters["cut_track_HitAddChi2"], cut_chi2)
            else:
                if self.debug: print(f"  No hits added from layer {layer}. Chi2 of hits {chi2_predict}. Hits", np.array(hits_thislayer)[chi2_predict_inds])

        return hits_found, kf_find.chift_total


    def remove_related_hits_seeds(self, hits_found):
        """
        hits_found: list of hits that belongs to the found track
        ---
        Remove seeds that shares hits of the found track
        Remove hits that are already used in the found track
        """
        hits_found_inds = [hit.ind for hit in hits_found]
        hits_found_inds.sort(reverse=True)
        # Remove seeds backwards to not change the index
        for i in reversed(range(len(self.seeds))):
            seed = self.seeds[i]
            if (self.hits[seed[0]].ind in hits_found_inds) or (self.hits[seed[1]].ind in hits_found_inds):
                self.seeds.pop(i)
        # Redo the grouping
        for layer in list(self.hits_grouped.keys()):
            hits = self.hits_grouped[layer]
            for ihit in reversed(range(len(hits))):
                if hits[ihit].ind in hits_found_inds:
                    hits.pop(ihit)
            if len(hits)==0:
                self.hits_grouped.pop(layer)
        # Clean up the hit pair:
        for ind in hits_found_inds:
            self.hit_pair.pop_ind(ind)


    def remove_related_seeds(self, seeds, hits_found):
        """
        seeds: list of seeds to be checked
        hits_found: list of hits that belongs to the found track
        ---
        Remove seeds that shares hits of the found track
        """
        hits_found_inds = [hit.ind for hit in hits_found]
        hits_found_inds.sort(reverse=True)
        # Remove seeds backwards to not change the index
        for i in reversed(range(len(seeds))):
            seed = seeds[i]
            if (self.hits[seed[0]].ind in hits_found_inds) or (self.hits[seed[1]].ind in hits_found_inds):
                seeds.pop(i)


    def filter_smooth(self, hits, drop_chi2=-1):
        """
        hits: list of hits to be used for track fitting
        drop_chi2: float, the chi2 threshold for dropping hits during smoothing.
        ---
        Run the forward filter and backward smooth at once
        ---
        Returns the Kalman filter result and the indices of dropped hits
        """
        kf = KF.KalmanFilter()
        m0, V0, H0, Xf0, Cf0, Rf0 = Util.track.init_state(hits) # Use the first two hits to set initial state
        kf.init_filter(m0, V0, H0, Xf0, Cf0, Rf0)               # Initialize the Kalman filter with the initial state and covariance

        # Feed all measurements to KF
        for i in range(2,len(hits)):
            hit = hits[i]               # Get the current hit
            dy  = hits[i].y-hits[i-1].y # Calculate the step size in y
            Ax, Az, At = kf.Xf[-1][3:]  # Get the velocity from the last filtered state
            velocity = [Ax, Az, At] if self.parameters["fit_track_MultipleScattering"] else None  # Velocity is needed for multiple scattering
            mi, Vi, Hi, Fi, Qi = Util.track.add_measurement(hit, dy, velocity, self.parameters["multiple_scattering_p"],self.parameters["multiple_scattering_length"])
            
            # pass to KF
            kf.forward_predict(mi, Vi, Hi, Fi, Qi)  # Predict the next state based on the current state and the measurement
            kf.forward_filter()                     # Update the state based on the measurement

        # Filter backward
        dropped_inds = []
        if drop_chi2<0:
            kf.backward_smooth()
        else:
            # Manually go through all steps to check if it exceed drop_chi2
            kf.init_smooth()
            while kf.CURRENT_STEP >= 0:
                chi2_temp = kf.smooth_step_try()
                dropped = chi2_temp > drop_chi2
                if dropped:
                    dropped_inds.append(kf.CURRENT_STEP)
                    if self.debug: print(f"   hit dropped with chi2 {chi2_temp}. Hit {hits[kf.CURRENT_STEP][:6]}")
                # Finishing the current step
                kf.smooth_step(drop = dropped)

        return kf, dropped_inds


    def prepare_output_back(self, kalman_result, hits_found_temp, track_ind=0):
        """
        kalman_result: Kalman filter result from the backward filter
        hits_found_temp: list of hits that belongs to the found track
        track_ind: int, the index of the track in the list of tracks
        ---
        Turn the Kalman filter result into a Track object
        ---
        Returns a Track object
        """
        # propagate the KF result from the second hit to the first hit
        hits_found = hits_found_temp[::-1]              # Reverse the order of hits to have the first hit at the beginning
        Ax, Az, At = kalman_result.Xsm[0][3:]           # Get the velocity from the first smoothed state
        velocity = [Ax, Az, At] if self.parameters["fit_track_MultipleScattering"] else None       # Velocity is needed for multiple scattering
        mi, Vi, Hi, Fi, Qi = Util.track.add_measurement(hits_found[0], hits_found[0].y - hits_found[1].y, velocity, self.parameters["multiple_scattering_p"],self.parameters["multiple_scattering_length"])
        state_predicted_step_0 = Fi@kalman_result.Xsm[0]# Propagate the first smoothed state to the first hit
        x0 = state_predicted_step_0[0]                  # Get the x position from the predicted state
        z0 = state_predicted_step_0[1]                  # Get the z position from the predicted state
        t0 = state_predicted_step_0[2]                  # Get the time from the predicted state
        Ax = state_predicted_step_0[3]                  # Get the x velocity from the predicted state
        Az = state_predicted_step_0[4]                  # Get the z velocity from the predicted state
        At = state_predicted_step_0[5]                  # Get the time velocity from the predicted state
        y0 = hits_found[0].y                            # Get the y position from the first hit
        Ay = 1                                          # Slope of Y vs Y, which is always 1
        # Propagate the KF result from the second hit to the first hit
        hits_filtered = [[xsm[0], hit.y, xsm[1], xsm[2]] for hit,xsm in zip(hits_found[1:], kalman_result.Xsm)]
        hits_filtered.insert(0, [x0,y0,z0,t0])
        # Add the covariance of one additional layer:
        cov = kalman_result.Cf[-1]
        chi2 = kalman_result.chift_total
        ind = track_ind
        # Get the indices of the hits in the found track
        hits = [hit.ind for hit in hits_found[::-1]]

        track_result = kalman_result.Xf[-1]
        x0 = track_result[0]
        z0 = track_result[1]
        t0 = track_result[2]
        Ax = track_result[3]
        Az = track_result[4]
        At = track_result[5]
        y0 = hits_found[-1].y
        Ay = 1
        # Track is a namedtuple("Track", ["x0", "y0", "z0", "t", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        track = datatypes.Track(x0, y0, z0, t0, Ax, Ay, Az, At, cov, chi2, ind, hits, hits_filtered)
        return track

