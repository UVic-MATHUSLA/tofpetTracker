import copy
import numpy as np
from numpy.linalg import inv
import scipy as sp
import functools
print = functools.partial(print, flush=True) # make python actually flush the output!

# Internal modules
from tracker import utilities as Util
from tracker import kalmanfilter as KF
from tracker import datatypes


class TrackFinder:
    def __init__(self, parameters=None, method="recursive", debug=False):
        self.method = method # {"recursive", "greedy"}
        self.debug = debug
        self.parameters={
            "cut_track_SeedSpeed": 1,               # in the unit of c. Limit the maximum speed formed by the seed.
            "cut_track_HitAddChi2": 15,             # Only used when method is "greedy"
            "cut_track_HitDropChi2": 15,            # Set to -1 to turn off
            "cut_track_HitProjectionSigma": 10,     # Number of sigmas
            "cut_track_TrackChi2Reduced": 3,        # Only use this for track with 3 hits
            "cut_track_TrackChi2Prob": 0.9,         # Chi-square probablity (calculated from chi2_cdf(x, DOF))
            "cut_track_TrackNHitsMin": 3,           # Minimum number of hits per track
            "cut_track_TrackSpeed": [25,35],        # [cm/ns], [speed_low, speed_high]. 30 is the speed of light
            "fit_track_MultipleScattering": False,
            "cut_track_MultipleScatteringFind": False,
            "fit_track_Method": "backward", # One of: backward, forward, forward-seed, least-square, least-square-ana
            "fit_track_LeastSquareIters":2, # No need to change
            "multiple_scattering_p": 500, # [MeV/c] momentum of multiple scattering, 
            "multiple_scattering_length": 0.06823501107481977 # [1] material thickness in the unit of scattering length
        }

    def run(self, hits):
        """
        Run all three rounds of kalman filter: find, drop, fit
        This function is a mixer of self.find() and self.filter_smooth()
        """
        self.hits = copy.copy(hits)
        self.hits_grouped = Util.track.group_hits_by_layer(self.hits)
        self.total_layers = len(list(self.hits_grouped.keys()))
        self.tracks = []
        self.seeds = self.seeding(self.hits)
        self.hit_pair = Util.HitPair(self.hits)

        if self.debug:
            print(f"TrackFinder input: {len(self.hits)} hits, {self.total_layers} layers, {len(self.seeds)} seeds")

        if self.parameters["cut_track_TrackNHitsMin"]>self.total_layers:
            if self.debug:
                print("Track finding stopped: fewer layers than cut_track_TrackNHitsMin")
            return self.tracks

        hit_found_inds = []
        for track_TrackNHitsMin in range(self.parameters["cut_track_TrackNHitsMin"], self.total_layers+1)[::-1]:
            if self.debug: print(f"\n====Looking for track with {track_TrackNHitsMin} hits====\n")
            hits_found_all = []
            self.seeds_unused = []
            while len(self.seeds) > 0:
                if len(self.hits_grouped.keys())<self.parameters["cut_track_TrackNHitsMin"]: # If not enough hits left:
                    break

                # Round 1: Find hits that belongs to one track
                seed = self.seeds[-1]
                if self.debug: print(f"--- New seed --- \n [Seed]: {seed}")
                hits_found, track_chi2 = self.find_once(self.hits, self.hits_grouped, seed, self.hit_pair)
                # Remove the current seed no matter the track is good or not:
                self.seeds.pop(-1)

                # Apply cuts: if not enough hits, drop this track
                if len(hits_found) < track_TrackNHitsMin:
                    if self.debug: print(f"Track finding failed (adding), not enough hits only found: {len(hits_found)}")
                    # Keep the seeds that potentially matches to a track
                    if len(hits_found) >= self.parameters["cut_track_TrackNHitsMin"]:
                        self.seeds_unused.append(seed)
                        # print("seed added back")
                        # if len(hits_found)==track_TrackNHitsMin-1:
                        #     self.remove_related_seeds(self.seeds, hits_found)
                    continue

                # Sort the hits by time before running the filter
                hits_found.sort(key = lambda hit: hit.t)

                # Round 2: Run filter and smooth with the option to drop outlier during smoothing
                kalman_result, inds_dropped = self.filter_smooth(hits_found, drop_chi2=self.parameters["cut_track_HitDropChi2"])
                inds_dropped.sort(reverse = True)
                for ind in inds_dropped:
                    hits_found.pop(ind)
                # If not enough hits, drop this track
                if len(hits_found) < track_TrackNHitsMin:
                    if self.debug: print(f"Track finding failed (dropping), not enough hits only found: {len(hits_found)}")
                    # Keep the seeds that potentially matches to a track
                    if len(hits_found) >= self.parameters["cut_track_TrackNHitsMin"]:
                        self.seeds_unused.append(seed)
                    continue 

                # Round 3: Run filter again on found hits using method specified by user
                if self.parameters["fit_track_Method"] == "backward":
                    # Run filter backwards, no smoothing
                    kalman_result, inds_dropped = self.filter_smooth(hits_found[::-1], drop_chi2=-1)
                    track_output = self.prepare_output_back(kalman_result, hits_found, track_ind = len(self.tracks))

                elif self.parameters["fit_track_Method"] == "forward":
                    # Run filter forward with smoothing, use first two hits to initialize
                    kalman_result, inds_dropped = self.filter_smooth(hits_found, drop_chi2=-1)
                    track_output = self.prepare_output(kalman_result, hits_found, track_ind = len(self.tracks))

                elif self.parameters["fit_track_Method"] == "forward-seed": 
                    # Run filter forward with smoothing, set initial state using first and last hit
                    m0, V0, H0, Xf0, Cf0, Rf0 = Util.track.init_state([hits_found[-1], hits_found[0]])
                    kalman_result = Util.track.run_kf(hits_found, initial_state=Xf0, initial_cov=Cf0, multiple_scattering=True)
                    track_output = self.prepare_output_v2(kalman_result, hits_found, track_ind=len(self.tracks))

                elif self.parameters["fit_track_Method"] == "least-square":
                    # Run least square fit
                    guess = Util.track.guess_track(hits_found)
                    fit_ls = Util.track.fit_track_scattering(hits_found,guess) if self.parameters["fit_track_MultipleScattering"] else Util.track.fit_track(hits_found,guess) 
                    popt = fit_ls.values
                    pcov = fit_ls.covariance
                    chi2 = fit_ls.fval
                    track_output = self.prepare_output_ls(popt, pcov, chi2, hits_found, track_ind=len(self.tracks))

                elif self.parameters["fit_track_Method"] == "least-square-ana":
                    # Run analytical least square fit (less iteration during minimization)
                    popt, pcov, chi2 = Util.track.fit_track_ana(hits_found, scattering=self.parameters["fit_track_MultipleScattering"], iters=self.parameters["fit_track_LeastSquareIters"]) 
                    track_output = self.prepare_output_ls(popt, pcov, chi2, hits_found, track_ind=len(self.tracks))

                # Cut on chi2 probablity
                ndof = 3 * len(hits_found) - 6
                track_chi2 = track_output.chi2
                track_chi2_prob = sp.stats.chi2.cdf(track_chi2, ndof)
                track_chi2_reduced = track_chi2/ndof
                if (track_chi2_prob > self.parameters["cut_track_TrackChi2Prob"] and ndof > 3) or \
                   (track_chi2_reduced > self.parameters["cut_track_TrackChi2Reduced"] and ndof <= 3):
                    if self.debug:
                        print(f"Track vetoed, chi2 too large. Chi2/nodf: {track_chi2}/{ndof}, prob = {track_chi2_prob}")
                    continue

                # Cut on speed
                state = track_output # Track is a namedtuple("Track", ["x0", "y0", "z0", "t", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
                speed = np.linalg.norm([state.Ax/state.At, state.Az/state.At, 1/state.At])
                if not (self.parameters["cut_track_TrackSpeed"][0] < speed < self.parameters["cut_track_TrackSpeed"][1]):
                    if self.debug: print(f"Track vetoed. Speed of the track: {speed}[cm/ns]")
                    continue
                elif self.debug:
                    print(f" [Track found]", track_output) 
                    print("Added hits:")
                    for t in hits_found:
                        print("   ", t)

                # Add track and remove other seeds that shares hits of the found track
                self.tracks.append(track_output)
                self.remove_related_hits_seeds(hits_found)
                hits_found_all.extend(hits_found)

            # Remove hits that are already added to track
            hit_found_inds.extend([hit.ind for hit in hits_found_all])
            hit_found_inds.sort(reverse=True)
            self.remove_related_seeds(self.seeds_unused, hits_found_all)
            self.seeds = copy.copy(self.seeds_unused)
            # print("------------------decreasinglayer")
            # print(hit_found_inds)
            # for ind in hit_found_inds:
            #     self.hits.pop(ind)
            # Group the remaining hits
            # self.hits_grouped = Util.track.group_hits_by_layer(self.hits, used_index = hit_found_inds)

        if self.debug:
            print("\n========Track finding finished=======\n")
            print("Tracks found:")
            for t in self.tracks:
                print("   ", t)
        return self.tracks


    # def find(self, hits):
    #     self.seeds = self.seeding(hits)
    #     self.hits = copy.copy(hits)
    #     self.hits_grouped = Util.track.group_hits_by_layer(self.hits)
    #     self.tracks_found = []
    #     while len(self.seeds)>0:
    #         seed = self.seeds[-1]
    #         hits_found, track_chi2 = self.find_once(self.hits, self.hits_grouped, seed, 1)


    #         # Remove the current seed no matter the track is good or not:
    #         self.seeds.pop(-1)         
    #         # Apply cuts: if not enough hits, drop this track
    #         if len(hits_found)<self.parameters["cut_track_TrackNHitsMin"]: continue
    #         ndof = 3*len(hits_found) - 6
    #         if ndof<=0: continue
    #         track_chi2_reduced = track_chi2/ndof
    #         # If chi2 is too large, drop this track
    #         if track_chi2_reduced>self.parameters["cut_track_TrackChi2Reduced"]:
    #             continue
    #         # Attach the track if it pass the cuts
    #         self.tracks_found.append(hits_found)
    #         # Remove other seeds that are in the track
    #         self.remove_related_hits_seeds(hits_found)
    #     return self.tracks_found


    def seeding(self, hits, used_index=[]):
        """
        Find seed for tracks
        Returns a pair of index of the hit (not the hit itself!) and a score
        """
        c = sp.constants.c/1e7 # [cm/ns]
        seeds = []
        for i in range(len(hits)):
            for j in range(i+1, len(hits)):
                if(hits[i].y == hits[j].y) or (hits[i].layer == hits[j].layer): continue
                if hits[i].ind in used_index or hits[j].ind in used_index: continue
                dx = hits[i].x- hits[j].x
                dy = hits[i].y- hits[j].y
                dz = hits[i].z- hits[j].z
                dt = hits[i].t- hits[j].t
                dr = np.linalg.norm([dx,dy,dz])
                ds = abs(dr/c - abs(dt))
                if ds > 1:
                    continue
                seeds.append([i, j, dr, abs(dy)])

        # Sort by score, larger gap = better, small to large is placing best one at end
        seeds.sort(key=lambda s: (s[3], s[2]))
        return seeds


    def find_once(self, hits, hits_layer_grouped, seed, hit_pair):
        LAYERS = np.sort(list(hits_layer_grouped.keys()))
        # Check the direction of seed by comparing the time of two hits
        seed_hits = [hits[seed[0]], hits[seed[1]]]
        # Always have the first hit to be first in time
        if (seed_hits[0].t > seed_hits[1].t): seed_hits = seed_hits[::-1]
        seed_start_layer = seed_hits[0].layer
        seed_stop_layer  = seed_hits[1].layer
        # Check if needed to find backward or forward
        if (seed_hits[0].layer > seed_hits[1].layer):
            TRACK_DIRECTION = 0 # Downward track
            FIND_BACKWARD_LAYERS = LAYERS[np.argmax(LAYERS > seed_stop_layer):]
        else:
            TRACK_DIRECTION = 1 # Upward track
            FIND_BACKWARD_LAYERS = LAYERS[:np.argmax(LAYERS > seed_stop_layer)-1][::-1]

        # Always find forward, only find backward when there is more than one layer before second hit in of seed
        FIND_FORWARD = True
        FIND_BACKWARD = True if len(FIND_BACKWARD_LAYERS)>1 else False

        # Find: Start with the second hit of the seed, and find backward and forward
        hits_found = [seed_hits[1]]
        chi2_found = 0
        # Find backward first, then find forward
        if FIND_BACKWARD:
            step_pre = seed_hits[1].y # Keep track of the y of the previous step

            kf_find = KF.KalmanFilterFind()
            kf_find.init_filter(*Util.track.init_state(seed_hits))

            if self.debug: print("   Finding backward in layers", FIND_BACKWARD_LAYERS)
            if self.method == "recursive":
                hits_found_backward, chi2 = self.find_in_layers_recursive(hits, hits_layer_grouped, FIND_BACKWARD_LAYERS, kf_find, step_pre)
            else:
                hits_found_backward, chi2 = self.find_in_layers_greedy(hits, hits_layer_grouped, FIND_BACKWARD_LAYERS, kf_find, step_pre, hit_pair)
            # Order of found hits also needs to be reversed for backward finding
            hits_found_backward = hits_found_backward[::-1] 
            hits_found_backward.extend(hits_found)
            hits_found = hits_found_backward
        else:
            hits_found = seed_hits
        # Find forward
        if FIND_FORWARD:
            if len(hits_found) < 2:
                return [], []
            # Reset seed to be first and last hit: seed_hits = [hits_found[0], hits_found[-1]]
            seed_hits = hits_found[:2]
            step_pre = seed_hits[1].y # Keep track of the y of the previous step

            if max(LAYERS) == seed_hits[1].layer:           # If last hit in last layer, no need to find forward
                return hits_found, chi2_found
            if (seed_hits[0].layer > seed_hits[1].layer):   # If first hit in a layer after second hit, find forward in layers before second hit
                FIND_FORWARD_LAYERS = LAYERS[:np.argmax(LAYERS > seed_hits[1].layer)-1][::-1]
            else:
                FIND_FORWARD_LAYERS  = LAYERS[np.argmax(LAYERS > seed_hits[1].layer):] 

            kf_find = KF.KalmanFilterFind()
            kf_find.init_filter(*Util.track.init_state(seed_hits)) # Set initial state using two hits specified by the seed

            if self.debug: print(".  Finding forward in layers", FIND_FORWARD_LAYERS )
            if self.method == "recursive":
                hits_found_forward,chi2 = self.find_in_layers_recursive(hits, hits_layer_grouped, FIND_FORWARD_LAYERS, kf_find, step_pre)
            else:
                hits_found_forward,chi2 = self.find_in_layers_greedy(hits, hits_layer_grouped, FIND_FORWARD_LAYERS, kf_find, step_pre, hit_pair)
            chi2_found = chi2
            hits_found = hits_found[:2] + hits_found_forward

        return hits_found,chi2_found

    def find_in_layers_greedy(self, hits, hits_layer_grouped, layers_to_scan, kf_find, step_pre, cut_chi2=True, hit_pair=None):
        """Find the hit that has minimum chi2 in each layer"""
        hits_found = []
        for layer in layers_to_scan:
            hits_thislayer = hits_layer_grouped[layer]
            if len(hits_thislayer) == 0:
                continue

            # Use the pre-calculated hit pair info to see if there are any compatible hits
            if hit_pair is not None:
                # If there is no hit to match the previous hit, break the loop immediately
                if len(hits_found) > 0 and (not hit_pair.exists_hit(hits_found[-1])):
                    print("--no matched hit")
                    break

            hit = hits_thislayer[0]             # Get one hit
            step_this = hits_thislayer[0].y     # Get the prediction matrix 
            dy = step_this - step_pre           # Step size
            Ax, Az, At = kf_find.Xf[3:]         # Get the state vector

            # Velocity is needed for multiple scattering
            velocity = [Ax, Az, At] if self.parameters["cut_track_MultipleScatteringFind"] else None
             # Calculate matrices. Only need to do once for all this in the same layer
            _, Vi, Hi, Fi, Qi = Util.track.add_measurement(hits_thislayer[0],dy,velocity,self.parameters["multiple_scattering_p"],self.parameters["multiple_scattering_length"])
            
            kf_find.update_matrix(Vi, Hi, Fi, Qi)       # pass matrices to KF
            Xp = kf_find.Xp_i                           # Use the Predicted location to limit the search range
            Xp_unc = np.sqrt(np.diag(kf_find.Rp_i))     # Use the uncertainty of the prediction to limit the search range
            N_sigma = self.parameters["cut_track_HitProjectionSigma"]   # Get the number of sigma to use for the cut

            # Use the total uncertainty of the prediction plus the measurement
            unc_total = [np.linalg.norm([hit.x_err,Xp_unc[0]]), np.linalg.norm([hit.z_err,Xp_unc[1]]), np.linalg.norm([hit.t_err,Xp_unc[2]])]
            test_measurement_incompatible = lambda x,z,t: abs(x-Xp[0])>unc_total[0]*N_sigma or \
                                                          abs(z-Xp[1])>unc_total[1]*N_sigma or \
                                                          abs(t-Xp[2])>unc_total[2]*N_sigma or \
                                                        ((x-Xp[0])/unc_total[0])**2 + ((z-Xp[1])/unc_total[1])**2 + ((t-Xp[2])/unc_total[2])**2 > N_sigma**2

            # Calculate chi2 for all hits in the next layer
            chi2_predict=[]
            chi2_predict_inds =[]
            for imeasurement, m in enumerate(hits_thislayer):
                # Use the pre-calculated hit pair info to narrow down the search
                if hit_pair is not None:
                    if len(hits_found) > 0 and (not hit_pair.exists_pair(hits_found[-1], m)): continue

                # A more strict test of compatibility
                if test_measurement_incompatible(m.x, m.z, m.t): continue
                else:
                    chi2 = kf_find.forward_predict_chi2(np.array([m.x, m.z, m.t]))
                    chi2_predict.append(chi2)
                    chi2_predict_inds.append(imeasurement)
            if len(chi2_predict)==0:
                continue

            # Find the hit with minimum chi2
            chi2_min_idx = np.argmin(chi2_predict)
            if chi2_predict[chi2_min_idx] < self.parameters["cut_track_HitAddChi2"] or not cut_chi2:
                # Save the hit either if the chi2 is lower than the threshold, or the cut is disabled
                hits_found.append(hits_thislayer[chi2_predict_inds[chi2_min_idx]])
                # Update the step and the Kalman filter
                step_pre = step_this
                mi = hits_found[-1]
                kf_find.forward_filter(np.array([mi.x, mi.z, mi.t]))
                if self.debug: print("   Hit found:", mi, "; chi2", chi2_predict[chi2_min_idx],chi2_predict[chi2_min_idx]<self.parameters["cut_track_HitAddChi2"], cut_chi2)
            else:
                if self.debug: print(f"   No hits added from layer {layer}. Chi2 of hits {chi2_predict}. Hits", np.array(hits_thislayer)[chi2_predict_inds])

        return hits_found, kf_find.chift_total


    def find_in_layers_recursive(self, hits, hits_layer_grouped, layers_to_scan, kf_find, step_pre):
        """Find the hits that has minimum chi2 in total"""
        self.found_hit_groups = []
        self.found_chi2_groups = []

        # Run the recrusive finding
        current_layer_ind = -1
        found_hits_inds = []
        found_chi2s = []
        self._find_in_layers_recursive(hits_layer_grouped, layers_to_scan, kf_find, step_pre, current_layer_ind, found_hits_inds, found_chi2s)

        if len(self.found_hit_groups)==0:
            return [], 0

        # Find the group with minimum chi2 per hit
        n_hits = [max(len(i),1) for i in self.found_hit_groups] # limit to be at least 1 to not mess up the divide in the following line
        chi2_reduced = np.sum(self.found_chi2_groups, axis=1) / n_hits
        ind_minchi2 = np.argmin(chi2_reduced)
        hits_found = [hits[i] for i in self.found_hit_groups[ind_minchi2]]
        return hits_found, chi2_reduced[ind_minchi2]


    def _find_in_layers_recursive(self, hits_layer_grouped, layers_to_scan, kf_find, step_pre, current_layer_ind, found_hits_inds, found_chi2s):
        current_layer_ind+=1 

        if current_layer_ind>=len(layers_to_scan):
            self.found_hit_groups.append(found_hits_inds)
            self.found_chi2_groups.append(found_chi2s)
            return

        layer = layers_to_scan[current_layer_ind]
        hits_thislayer = hits_layer_grouped[layer]

        step_this = hits_thislayer[0].y     # Get the prediction matrix 
        dy = step_this - step_pre           # Step size
        step_pre = step_this                # Update the step for the next recursion
        Ax, Az, At = kf_find.Xf[3:]         # Get the state vector

        # Velocity is needed for multiple scattering
        velocity = [Ax, Az, At]     if self.parameters["cut_track_MultipleScatteringFind"] else None
        # Calculate matrices. Only need to do once for all this in the same layer
        _, Vi, Hi, Fi, Qi = Util.track.add_measurement(hits_thislayer[0],dy,velocity,self.parameters["multiple_scattering_p"],self.parameters["multiple_scattering_length"])
        kf_find.update_matrix(Vi, Hi, Fi, Qi) # pass matrices to KF

        # Predicted location 
        Xp = kf_find.Xp_i
        Xp_unc = np.sqrt(np.diag(kf_find.Rp_i))
        N_sigma = self.parameters["cut_track_HitProjectionSigma"]
        # Use the total uncertainty of the prediction plus the measurement
        hit = hits_thislayer[0]
        unc_total = [np.linalg.norm([hit.x_err,Xp_unc[0]]), np.linalg.norm([hit.z_err,Xp_unc[1]]), np.linalg.norm([hit.t_err,Xp_unc[2]])]
        test_measurement_incompatible = lambda x,z,t: abs(x-Xp[0])>unc_total[0]*N_sigma or \
                                                      abs(z-Xp[1])>unc_total[1]*N_sigma or \
                                                      abs(t-Xp[2])>unc_total[2]*N_sigma or \
                                                      ((x-Xp[0])/unc_total[0])**2 + ((z-Xp[1])/unc_total[1])**2 + ((t-Xp[2])/unc_total[2])**2 > N_sigma**2    

        # calculate chi2 for all hits in the next layer
        for imeasurement, m in enumerate(hits_thislayer):
            # Limit our search to the hits close to prediction:
            if test_measurement_incompatible(m.x, m.z, m.t): continue

            # Make copys for hits and chi2s for each recursion
            found_hits_inds_i = copy.deepcopy(found_hits_inds)
            found_chi2s_i = copy.deepcopy(found_chi2s)
            kf_find_i = copy.deepcopy(kf_find)
            # Run Kalman filter
            chi2 = kf_find_i.forward_filter(np.array([m.x, m.z, m.t]))
            found_hits_inds_i.append(m.ind)
            found_chi2s_i.append(chi2)
            self._find_in_layers_recursive(hits_layer_grouped, layers_to_scan, kf_find_i, step_pre, current_layer_ind, found_hits_inds_i, found_chi2s_i)
        return


    def remove_related_hits_seeds(self, hits_found):
        hits_found_inds = [hit.ind for hit in hits_found]
        hits_found_inds.sort(reverse=True)
        # Remove seeds: need to do backwards to not change the index
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
        hits_found_inds = [hit.ind for hit in hits_found]
        hits_found_inds.sort(reverse=True)
        # Remove seeds: need to do backwards to not change the index
        for i in reversed(range(len(seeds))):
            seed = seeds[i]
            if (self.hits[seed[0]].ind in hits_found_inds) or (self.hits[seed[1]].ind in hits_found_inds):
                seeds.pop(i)


    def filter_smooth(self, hits, drop_chi2=-1):
        """
        Run the forward filter and backward smooth at once
        hits (list): A list of all hits in a track
        drop_chi2 (float): for values less than zero, disable dropping
                           for values zero, the steps with chi2 larger than this number will be dropped
        """
        kf = KF.KalmanFilter()

        # Set initial state using first two hits to initiate it
        m0, V0, H0, Xf0, Cf0, Rf0 = Util.track.init_state(hits)
        kf.init_filter( m0, V0, H0, Xf0, Cf0, Rf0)

        # Feed all measurements to KF
        for i in range(2,len(hits)):
            # Get updated matrix
            hit = hits[i]
            dy  = hits[i].y-hits[i-1].y
            Ax, Az, At = kf.Xf[-1][3:]
            # Velocity is needed for multiple scattering
            velocity = [Ax, Az, At] if self.parameters["fit_track_MultipleScattering"] else None
            mi, Vi, Hi, Fi, Qi = Util.track.add_measurement(hit,dy,velocity,self.parameters["multiple_scattering_p"],self.parameters["multiple_scattering_length"])

            # pass to KF
            kf.forward_predict(mi, Vi, Hi, Fi, Qi)
            kf.forward_filter()

        # Filter backward
        dropped_inds = []
        if drop_chi2<0:
            kf.backward_smooth()
        else:
            # Manually go through all steps to check if it exceed drop_chi2
            kf.init_smooth()
            while kf.CURRENT_STEP>=0:
                chi2_temp = kf.smooth_step_try()
                dropped =  chi2_temp > drop_chi2
                if dropped:
                    dropped_inds.append(kf.CURRENT_STEP)
                    if self.debug: print(f"   hit dropped with chi2 {chi2_temp}. Hit {hits[kf.CURRENT_STEP][:6]}")
                # Finishing the current step
                kf.smooth_step(drop = dropped)

        return kf, dropped_inds


    def prepare_output(self, kalman_result, hits_found, track_ind=0):
        """ Turn the Kalman filter result into a Track object"""
        # Propagate the KF result from the second hit to the first hit
        Ax, Az, At = kalman_result.Xsm[0][3:]
        # Velocity is needed for multiple scattering
        velocity = [Ax, Az, At] if self.parameters["fit_track_MultipleScattering"] else None

        mi, Vi, Hi, Fi, Qi = Util.track.add_measurement(hits_found[0], hits_found[0].y - hits_found[1].y, velocity,
                                            self.parameters["multiple_scattering_p"],self.parameters["multiple_scattering_length"])
        Xp_i = Fi@kalman_result.Xsm[0]
        Cp_i = Fi@kalman_result.Csm[0]@Fi.T + Qi
        rp_i = mi - Hi@Xp_i
        Rp_i = Vi + Hi@Cp_i@Hi.T
        K = Cp_i.dot(Hi.T).dot(inv(Rp_i))       # Kalman Gain (K)
        Xf = Xp_i + K@rp_i# Filtered state: combo of predicted state, measured values, covariance matrix, Kalman gain
        Cf = (np.identity(len(Xf)) - K@Hi).dot(Cp_i)
        state_predicted_step_0 = Xf
        statecov_predicted_step_0 = Cf 

        # Add the covariance of one additional layer:
        mi, Vi, Hi, Fi, Qi = Util.track.add_measurement(hits_found[0], -1.5, velocity,
                                            self.parameters["multiple_scattering_p"],self.parameters["multiple_scattering_length"])
        cov = statecov_predicted_step_0 + Qi
        chi2 = kalman_result.chift_total
        ind = track_ind
        hits = [hit.ind for hit in hits_found]

        x0 = state_predicted_step_0[0]
        y0 = hits_found[0].y
        z0 = state_predicted_step_0[1]
        t0 = state_predicted_step_0[2]
        Ax = state_predicted_step_0[3]
        Ay = 1 # Slope of Y vs Y, which is always 1
        Az = state_predicted_step_0[4]
        At = state_predicted_step_0[5]

        hits_filtered = [[xsm[0], hit.y, xsm[1], xsm[2]] for hit,xsm in zip(hits_found[1:], kalman_result.Xsm)]
        hits_filtered.insert(0, [x0,y0,z0,t0])

        # Track is a namedtuple("Track", ["x0", "y0", "z0", "t", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        track = datatypes.Track(x0, y0, z0, t0, Ax, Ay, Az, At, cov, chi2, ind, hits, hits_filtered)
        return track

    def prepare_output_v2(self, kalman_result, hits_found, track_ind=0):
        """ Turn the Kalman filter result into a Track object"""
        # propagate the KF result from the second hit to the first hit
        Ax, Az, At = kalman_result.Xsm[0][3:]

        # Velocity is needed for multiple scattering
        velocity = [Ax, Az, At] if self.parameters["fit_track_MultipleScattering"] else None
        state_predicted_step_0 = kalman_result.Xsm[0]
        statecov_predicted_step_0 = kalman_result.Csm[0]

        # Add the covariance of one additional layer:
        mi, Vi, Hi, Fi, Qi = Util.track.add_measurement(hits_found[0], -1.5, velocity,
                                            self.parameters["multiple_scattering_p"],self.parameters["multiple_scattering_length"])
        cov = statecov_predicted_step_0 + Qi
        chi2 = kalman_result.chift_total
        ind = track_ind
        hits = [hit.ind for hit in hits_found]

        x0 = state_predicted_step_0[0]
        y0 = hits_found[0].y
        z0 = state_predicted_step_0[1]
        t0 = state_predicted_step_0[2]
        Ax = state_predicted_step_0[3]
        Ay = 1 # Slope of Y vs Y, which is always 1
        Az = state_predicted_step_0[4]
        At = state_predicted_step_0[5]

        hits_filtered = [[xsm[0], hit.y, xsm[1], xsm[2]] for hit,xsm in zip(hits_found[0:], kalman_result.Xsm)]

        # Track is a namedtuple("Track", ["x0", "y0", "z0", "t", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        track = datatypes.Track(x0, y0, z0, t0, Ax, Ay, Az, At, cov, chi2, ind, hits, hits_filtered)
        return track


    def prepare_output_back(self, kalman_result, hits_found_temp, track_ind=0):
        """Turn the Kalman filter result into a Track object"""
        # propagate the KF result from the second hit to the first hit
        hits_found = hits_found_temp[::-1]
        Ax, Az, At = kalman_result.Xsm[0][3:]

        # Velocity is needed for multiple scattering
        velocity = [Ax, Az, At] if self.parameters["fit_track_MultipleScattering"] else None

        mi, Vi, Hi, Fi, Qi = Util.track.add_measurement(hits_found[0], hits_found[0].y - hits_found[1].y, velocity,
                                            self.parameters["multiple_scattering_p"],self.parameters["multiple_scattering_length"])
        state_predicted_step_0 = Fi@kalman_result.Xsm[0]
        x0 = state_predicted_step_0[0]
        y0 = hits_found[0].y
        z0 = state_predicted_step_0[1]
        t0 = state_predicted_step_0[2]
        Ax = state_predicted_step_0[3]
        Ay = 1 # Slope of Y vs Y, which is always 1
        Az = state_predicted_step_0[4]
        At = state_predicted_step_0[5]

        hits_filtered = [[xsm[0], hit.y, xsm[1], xsm[2]] for hit,xsm in zip(hits_found[1:], kalman_result.Xsm)]
        hits_filtered.insert(0, [x0,y0,z0,t0])

        # Add the covariance of one additional layer:
        cov = kalman_result.Cf[-1]
        chi2 = kalman_result.chift_total
        ind = track_ind
        hits = [hit.ind for hit in hits_found[::-1]]

        track_result = kalman_result.Xf[-1]
        x0 = track_result[0]
        y0 = hits_found[-1].y
        z0 = track_result[1]
        t0 = track_result[2]
        Ax = track_result[3]
        Ay = 1 # Slope of Y vs Y, which is always 1
        Az = track_result[4]
        At = track_result[5]

        # Track is a namedtuple("Track", ["x0", "y0", "z0", "t", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        track = datatypes.Track(x0, y0, z0, t0, Ax, Ay, Az, At, cov, chi2, ind, hits, hits_filtered)
        return track


    def prepare_output_ls(self, popt, pcov, chi2, hits_found, track_ind):
        x0, z0, t0, Ax, Az, At  = popt
        y0 = hits_found[0].y
        Ay = 1
        hits = [hit.ind for hit in hits_found[::-1]]
        hits_filtered=[[x0 + Ax*(hit.y-y0), hit.y, z0 + Az*(hit.y-y0), t0 + At*(hit.y-y0)] for hit in hits_found]
        track = datatypes.Track(x0, y0, z0, t0, Ax, Ay, Az, At, pcov, chi2, track_ind, hits, hits_filtered)
        return track

