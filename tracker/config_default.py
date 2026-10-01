parameters = {}

# Run parameters
parameters["debug"]=True                # Show debug info
parameters["debug_tracker"]=True        # Show debug info for tracker
parameters["debug_vertexer"]=True       # Show debug info for vertexer
parameters["print_n"]=10                # Print every n events
parameters["start_event"]=0             # 0-based index
parameters["end_event"]=-1            # -1 means all events
parameters["seed"]=1                    # Seed for random number generator (used for scintillator efficiency)
parameters["detector_efficiency"]=1     # Scintillator efficiency, any number between 0-1, 1 is 100%

# Global parameters:
parameters["multiple_scattering_p"]=500                         # [MeV/c] momentum of multiple scattering, 
parameters["multiple_scattering_length"]=0.06823501107481977    # Material thickness in the unit of attenuation length = thickness/attenuation_length

# Track parameters
parameters["cut_track_SeedSpeed"]=1                     # In the unit of c. Limit the maximum speed formed by the seed.
parameters["cut_track_HitAddChi2"]=100                   # Only used when method is "greedy"
parameters["cut_track_HitDropChi2"]=-1                  # Set to -1 to turn off
parameters["cut_track_HitProjectionSigma"]=20            # Number of sigmas
parameters["cut_track_TrackChi2Reduced"]=30             # Only use this for track with 3 hits
parameters["cut_track_TrackChi2Prob"]=0.5               # Chi-square probablity (calculated from chi2_cdf(x, DOF))
parameters["cut_track_TrackNHitsMin"]=3                 # Minimum number of hits per track
parameters["cut_track_TrackSpeed"] = [10,40]            # [cm/ns], [speed_low, speed_high]. 30 is the speed of light
parameters["fit_track_MultipleScattering"]=False        # Use multiple scattering in the track fitting
parameters["cut_track_MultipleScatteringFind"]=False    # Use multiple scattering in the track finding
parameters["fit_track_Method"]="backward"               # One of: backward, forward, forward-seed, least-square, least-square-ana

# Vertex parameters
parameters["cut_vertex_SeedDist"]=300               # Minimum distance between two tracks to form a vertex seed
parameters["cut_vertex_SeedChi2"]=25                # Maximum chi2 of the vertex seed
parameters["cut_vertex_TrackAddDist"]=300           # Minimum distance between a track and a vertex to add the track to the vertex
parameters["cut_vertex_TrackAddChi2"]=25            # Maximum chi2 of the track to add to the vertex
parameters["cut_vertex_TrackDropChi2"]=15           # Maximum chi2 of the track to drop from the vertex
parameters["cut_vertex_VertexChi2ReducedAdd"]=10    # Maximum chi2 of the vertex to add a track to the vertex
parameters["cut_vertex_VertexChi2Reduced"]=7        # Maximum chi2 of the vertex to keep the vertex