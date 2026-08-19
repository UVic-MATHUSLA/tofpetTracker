import copy
import os, sys
from collections import namedtuple
import numpy as np
from numpy.linalg import inv
import scipy as sp
import scipy.constants
import iminuit

# Internal modules
from . import utilities as Util
from . import kalmanfilter as KF
from . import datatypes
import functools; print = functools.partial(print, flush=True) #make python actually flush the output!


class chi2_vertex:
    def __init__(self, tracks):
        self.tracks = tracks
        self._parameters = {'x0': None, 'y0': None, 'z0': None, 't0': None}
    def __call__(self, x0, y0, z0, t0):
        error = 0
        point = [x0, y0, z0, t0]
        for track in self.tracks:
            error += Util.track.chi2_point_track(point, track, multiple_scattering=True, speed_constraint=False)
            # error += Util.track.chi2_point_track(point_temp, track_temp, multiple_scattering=True, speed_constraint=False)
        return error


class VertexFitter:
    def __init__(self, parameters=None, method="ls", debug=False):
        self.debug = debug
        self.method = method # {"ls", "kalman"}
        self.trackinfo =  namedtuple("trackinfo", ["track_ind", "track_chi2", "track_vertex_chi2", "track_vertex_dist"])
        self.parameters={
            "cut_vertex_SeedDist": 300,                 # [cm], close approach distance between two tracks of the seed
            "cut_vertex_SeedChi2": 50,                  # chi-square cut of the seed
            "cut_vertex_TrackChi2Reduced": -1,          # NOT USED
            "cut_vertex_TrackAddDist": 300,             # Distance cut to add a track
            "cut_vertex_TrackAddChi2": 30,              # chi2 cut to add a track
            "cut_vertex_TrackDropChi2": 15,             # chi2 cut to drop a track
            "cut_vertex_VertexChi2Reduced": 5,          # Vertex chi2-square cut
            "multiple_scattering_p": 500,               # [MeV/c] momentum of multiple scattering, 
            "multiple_scattering_length": 0.06823501107481977 # [1] material thickness in the unit of scattering length
        }

    def run(self, tracks):
        """
        tracks: list of Track objects
        ---
        Run the vertex fitter on a list of tracks
        ---
        return: list of Vertex objects
        """
        self.tracks = tracks
        self.tracks_remaining = copy.copy(tracks)
        self.tracks_remaining_info = [] # a list of track infomation. Each element is [track_ind, track_chi2, track_vertex_chi2, track_vertex_dist]
        self.seeding(tracks)
        if self.debug: 
            print("\n---------Looking for vertex  --------")
            print(" Seeds:")
            for seed in self.seeds:
                print(seed)

        self.vertices = []
        while len(self.seeds)>0:
            # ------------------------------------
            # Round 1: Find tracks that belongs to this vertex
            seed = self.seeds[-1]
            tracks_found, vertex_fit = self.find_once(self.tracks, seed)
            self.seeds.pop(-1)
            if len(tracks_found)==0:
                continue

            vertex_location = np.array(vertex_fit.values) 
            vertex_cov = np.array(vertex_fit.covariance) 
            vertex_chi2 = vertex_fit.fval 
            vertex_tracks = tracks_found
            vertex_ndof = 3*len(vertex_tracks)-4
            # ------------------------------------
            # Round 2: Drop outliers
            vertex_cov_xzt = copy.copy(vertex_cov)
            vertex_cov_xzt = np.delete(vertex_cov_xzt, 1,0)
            vertex_cov_xzt = np.delete(vertex_cov_xzt, 1,1)
            tracks_chi2 = [Util.track.chi2_point_track(vertex_location, track, vertex_cov_xzt) for track in tracks_found]
            while (max(tracks_chi2)>self.parameters["cut_vertex_TrackDropChi2"]):
                # Drop the track
                chi2_max = max(tracks_chi2)
                ind_drop = np.argmax(tracks_chi2)
                if self.debug: print(f"  track {tracks_found[ind_drop].ind} dropped with chi2 {chi2_max}")
                tracks_found.pop(ind_drop)
                if len(tracks_found)<2:
                    if self.debug: print("Not enough tracks for this seed")
                    break 
                # Update the fit and recalculate the track chi2
                vertex_fit = self.fit(tracks_found, vertex_location, hesse=False, strategy=0, tolerance=1)
                vertex_cov =  np.array(vertex_fit.covariance)
                vertex_cov_xzt = copy.copy(vertex_cov)
                vertex_cov_xzt = np.delete(vertex_cov_xzt, 1,0)
                vertex_cov_xzt = np.delete(vertex_cov_xzt, 1,1)
                tracks_chi2 = [Util.track.chi2_point_track(vertex_location, track, vertex_cov_xzt) for track in tracks_found]

            if len(tracks_found)<2:
                continue
            if vertex_chi2/vertex_ndof>self.parameters["cut_vertex_VertexChi2Reduced"]:
                if self.debug: print(f"Vertex vetoed. Chi2 too large. Chi2/ndof:{vertex_chi2}/{vertex_ndof}")
                continue

            # ------------------------------------
            # Round 3: Final fit, use smaller tolerance (0.1) and more accurate strategy (1)
            vertex_fit = self.fit(tracks_found, vertex_location, hesse=True, strategy=1, tolerance=0.1)
            if not vertex_fit.valid:
                if self.debug: print("Vertex vetoed, final fit failed.")
                continue
            vertex_location = np.array(vertex_fit.values) 
            vertex_cov = np.array(vertex_fit.covariance) 
            vertex_chi2 = vertex_fit.fval 
            vertex_ndof = 3*len(vertex_tracks)-4
            vertex_tracks = tracks_found
            tracks_found_inds = [t.ind for t in vertex_tracks]


            # Vertex = namedtuple("Vertex", ["x0", "y0", "z0", "t0", "cov", "chi2", "tracks"])
            self.vertices.append(datatypes.Vertex(*vertex_location, vertex_cov, vertex_chi2, tracks_found_inds))
            if self.debug:
                print(f"Vertex found! track indices: {tracks_found_inds}")
                print(f"  Tracks to vertex chi2:", tracks_chi2)
                print(f"  Vertex Chi2/DOF: {vertex_chi2:.1f}/{vertex_ndof}; N tracks: {len(vertex_tracks)}; x0,y0,z0,t0: {vertex_location}; Uncertainty: {np.sqrt(np.diag(vertex_cov))}")
            # Finally, remove used tracks
            self.remove_related_seeds(tracks_found)
        return self.vertices


    def find_once(self, tracks, seed):
        """
        tracks: list of Track objects
        seed: VertexSeed object
        ---
        Run the vertex finder on a list of tracks and a seed
        ---
        return: list of Track objects that belongs to the vertex, and the vertex fit result
        """
        # tracks may not be ordered.. Make a dictionary to map track index to track
        tracks = {track.ind:track for track in tracks}
        # Add hits from the seed first
        if self.debug: print("\n--- New seed for vertex --- \n  Seed", seed)
        seed_inds = [seed.trackind1, seed.trackind2]
        seed_midpoint = np.array([seed.x0, seed.y0, seed.z0, seed.t0])
        tracks_found = [tracks[seed_inds[0]], tracks[seed_inds[1]]]

        # Fit the seed
        m = self.fit(tracks_found, seed_midpoint, hesse=False, strategy=0, tolerance = 1) # Use strategy of 0 and large tolerance to do fast fit
        if (not m.valid) or (m.fval>self.parameters["cut_vertex_SeedChi2"]):
            if self.debug: print(f"  * Seed failed. Seed fit result valid: {m.valid}, seed chi2 {m.fval}")
            return [], []
        ndof = 3*len(tracks_found)-4
        vertex_location = np.array(m.values)
        vertex_err = np.array(m.errors)
        vertex_chi2 = m.fval
        m_final = m

        # Update the track info
        self.tracks_remaining_info = []
        for track in self.tracks_remaining:
            ind = track.ind
            chi2_track = track.chi2
            dist = Util.track.distance_to_point(track,vertex_location)
            chi2 = Util.track.chi2_point_track(vertex_location, track, point_unc=vertex_err)
            self.tracks_remaining_info.append(self.trackinfo(ind, chi2_track, chi2, dist))
        # Sort
        self.tracks_remaining_info.sort(key=lambda m: m.track_vertex_chi2) # sort by distance

        # Add hits until no longer passes cut, iteratively add tracks
        for i in range(len(self.tracks_remaining_info)):
            info = self.tracks_remaining_info[i]
            # Continue if track is from seed
            if info.track_ind in seed_inds:
                continue
            # Continue if track is too far from the seed
            if (info.track_vertex_chi2 > self.parameters["cut_vertex_TrackAddChi2"]) and\
               (info.track_vertex_dist > self.parameters["cut_vertex_TrackAddDist"]):
                if self.debug: print(f"  * Track [{info.track_ind:<2}] too far from vertex. Track dist to vertex: {info.track_vertex_chi2:.2f}, track chi2 to vertex: {info.track_vertex_dist:.2f}")
                continue

            tracks_found.append(tracks[info.track_ind])
            m = self.fit(tracks_found, vertex_location, hesse=False, strategy=0, tolerance = 1)
            ndof = 3*len(tracks_found)-4
            track_print = ','.join([f"{x:.1f}" for x in tracks[info.track_ind][:8]])
            vertex_print = ','.join([f"{x:.1f}" for x in list(m.values)])
            if (not m.valid) or ((m.fval-vertex_chi2)>self.parameters["cut_vertex_TrackAddChi2"]):
                if self.debug:
                    print(f" -- Track [{info.track_ind:<2}] removed.{' Valid ' if m.valid else 'Invalid'}. Vertex chi2_r {m.fval/ndof:<3.2f}, + {m.fval-vertex_chi2 :<7.2f}; Track: {track_print}; Vertex: {vertex_print}")                                   
                tracks_found.pop(-1)
                ndof = 3*len(tracks_found)-4
                continue

            m_final = m
            vertex_location = list(m.values)
            vertex_err = list(m.values)
            if self.debug: print(f" ++ Track [{info.track_ind:<2}] added to vertex. Vertex chi2_r {m.fval/ndof:<3.2f}, + {m.fval-vertex_chi2 :.2f}. Track: {track_print}; Vertex: {vertex_print}") 
            vertex_chi2 = m_final.fval

            # Update the track info
            for j in range(i+1, len(self.tracks_remaining_info)):
                ind = self.tracks_remaining_info[j].track_ind
                track = tracks[ind]
                chi2_track = track.chi2
                dist = Util.track.distance_to_point(track,vertex_location)
                chi2 = Util.track.chi2_point_track(vertex_location, track, point_unc=vertex_err)
                self.tracks_remaining_info[j] = self.trackinfo(ind, chi2_track, chi2, dist)
            # self.tracks_remaining_info.sort(key=lambda m: m.track_vertex_chi2) # sort by distance

        tracks_found_inds = [track.ind for track in tracks_found]
        # print(m_final.fval, m_final.values, tracks_found_inds)
        return tracks_found, m_final


    def remove_related_seeds(self, tracks_found):
        """
        tracks_found: list of Track objects
        ---
        Remove seeds that are related to the found tracks, and remove the found tracks from the remaining tracks
        """
        tracks_found_inds = [track.ind for track in tracks_found]
        tracks_found_inds.sort(reverse=True)
        # Remove seeds, do backwards to not change the index
        for i in reversed(range(len(self.seeds))):
            seed = self.seeds[i]
            if (seed.trackind1 in tracks_found_inds) or (seed.trackind2 in tracks_found_inds):
                self.seeds.pop(i)

        # Remove tracks
        for i in reversed(range(len(self.tracks_remaining))):
            if self.tracks_remaining[i].ind in tracks_found_inds:
                self.tracks_remaining.pop(i)


    def seeding(self, tracks):
        """
        tracks: list of Track objects
        ---
        Find seeds for vertex finding. A seed is a pair of tracks that are close to each other.
        The seed is defined by the midpoint of the closest approach of the two tracks,
        and the distance between the two tracks at that point.
        ---
        return: list of VertexSeed objects
        """
        seeds = []
        tracks = {track.ind:track for track in tracks}
        tracks_inds = list(tracks.keys())
        for ii in range(len(tracks_inds)):
            for jj in range(ii+1, len(tracks_inds)):
                i = tracks_inds[ii]
                j = tracks_inds[jj]

                # Cut on seed distance
                midpoint,dist_seed = Util.track.closest_approach_midpoint_Track(tracks[i], tracks[j])
                if dist_seed>self.parameters["cut_vertex_SeedDist"]:
                    if self.debug: print(f"  Seed ({i,j}) failed, distance is {dist_seed}")
                    continue

                # Check if tracks are duplicated
                if tracks[i].x0 == tracks[j].x0:
                    print("----------------Warning, track duplicated-------------")
                    print(i,j)

                # Cut on seed chi2 (estimated with closest approach midpoint)
                midpoint_chi2 = Util.track.chi2_point_track(midpoint, tracks[i])+ Util.track.chi2_point_track(midpoint, tracks[j])
                midpoint_err_sum = np.sqrt(np.sum(np.diag(Util.track.cov_point_track(midpoint, tracks[i]))+ np.diag(Util.track.cov_point_track(midpoint, tracks[j]))))
                if (midpoint_chi2>self.parameters["cut_vertex_SeedChi2"]):
                    if self.debug: print(f"  * Seed ({i,j}) failed, chi2 too large, chi2 is {midpoint_chi2}")
                    continue

                # print(i, j , midpoint_err_sum)
                v1 = [tracks[i].Ax/tracks[i].At, tracks[i].Az/tracks[i].At, tracks[i].Ay/tracks[i].At]
                v2 = [tracks[j].Ax/tracks[j].At, tracks[j].Az/tracks[j].At, tracks[i].Ay/tracks[j].At]
                seed_opening_angle = np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2))
                # print(i,j,seed_opening_angle)
                seed_track_unc = np.sum(np.diag(tracks[i].cov)) + np.sum(np.diag(tracks[j].cov))
                seed_track_chi2 =tracks[i].chi2+tracks[j].chi2
                seed_track_dist = np.linalg.norm([tracks[i].x0-tracks[j].x0, tracks[i].y0-tracks[j].y0, tracks[i].z0-tracks[j].z0])
                # print(i,j,seed_track_dist)

                # Check number of compatible tracks
                N_compatible_tracks = 0
                N_compatible_track_distance = 0
                N_compatible_hits = 0
                for k in range(len(tracks)):
                    if k in [i,j]:
                        continue
                    dist = Util.track.distance_to_point(tracks[k],midpoint)
                    chi2_this = Util.track.chi2_point_track(midpoint, tracks[k])
                    if dist<self.parameters["cut_vertex_TrackAddDist"] and chi2_this<self.parameters["cut_vertex_TrackAddChi2"]:
                        N_compatible_tracks = N_compatible_tracks + 1
                        N_compatible_hits += len(tracks[k].hits)
                        N_compatible_track_distance += dist
                N_compatible_track_distance = N_compatible_track_distance/N_compatible_tracks if N_compatible_tracks>0 else N_compatible_track_distance

                # Calculate the seed score
                seed_score = Util.vertex.score_seed([*midpoint, midpoint_chi2, dist_seed, N_compatible_tracks, N_compatible_track_distance, seed_track_unc, seed_track_chi2, seed_track_dist, seed_opening_angle])
                seed_found = datatypes.VertexSeed(*midpoint, 0, midpoint_chi2, dist_seed, N_compatible_tracks, i, j, seed_score)
                seeds.append(seed_found)

        # Sort the seeds, rank them reversely
        seeds.sort(key=lambda seed: seed.score, reverse=True)
        self.seeds = seeds
        return seeds


    def fit(self, tracks, guess, hesse=False, strategy=1, tolerance = 0.1, \
            limit = {"x0":(-20000,20000), "y0":(-1000,12000), "z0":(-20000,20000), "t0":(-100,1000)}):
        """
        tracks: list of Track objects
        guess: list of 4 floats, initial guess for the vertex position [x0, y0, z0, t0]
        hesse: bool, if True, run the HESSE algorithm to calculate the covariance matrix of the fit parameters
        strategy: int, choose one of {0,1,2}, 0 is the fastest. Default is 1
        tolerance: float, convergence is detected when edm < edm_max, where edm_max is calculated as 
                   edm_max = 0.002 * tolerance * errordef default tolerance is 0.1, 
                   set to larger value to speed up convergence
        limit: dict, limits for the fit parameters, keys are "x0", "y0", "z0", "t0", values are tuples of (min, max)
        ---
        Run the vertex fitter on a list of tracks, with an initial guess for the vertex position
        ---
        return: iminuit.Minuit object, which contains the fit result
        """
        x0_init, y0_init, z0_init, t0_init = guess
        m = iminuit.Minuit(chi2_vertex(tracks), x0=x0_init, y0=y0_init, z0=z0_init, t0=t0_init)
        m.strategy = strategy
        m.tol = tolerance
        m.limits["x0"] = limit["x0"]
        m.limits["y0"] = limit["y0"]
        m.limits["z0"] = limit["z0"]
        m.limits["t0"] = limit["t0"]
        m.errors["x0"] = 1
        m.errors["y0"] = 1
        m.errors["z0"] = 1
        m.errors["t0"] = 0.1
        m.migrad() # run optimiser
        if m.valid and hesse:
            m.hesse() # run covariance estimator
        return m

