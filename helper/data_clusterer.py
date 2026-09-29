from __future__ import annotations
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from . import root
from .base import geo_loader as gl

_c = 29.9792458     # cm/ns
_n = 1.896          # fibre refractive index
_v = _c / _n        # speed of light in fibre (cm/ns)
_fibre_cache = {}
_bar_cache = {}
_height_to_layer = {0.0: 3, 79.1: 2, 161.7: 1, 243.9: 0}  # mapping of bar heights to layer numbers


def decode_detector_id(detector_id):
    detector_id = int(detector_id)
    return {"group": (detector_id // 100_000_000_000) % 1000,
            "layer": (detector_id // 100_000_000) % 1000,
            "bar": detector_id % 100_000,}


def load_raw_hits(path, tree_name="data"):
    path = Path(path)
    reader = root.tfile_reader(str(path))
    reader.get_tree(tree_name)
    rdf = root.ROOT.RDataFrame(reader.tree)
    return {"kind": "root", "path": str(path), "reader": reader, "rdf": rdf, "frame": None}


def load_geometry_assets(geometry_path):
    with open(geometry_path, "r", encoding="utf-8") as stream:
        geometry_data = json.load(stream)

    detector = gl.detector(geometry_data)
    channel_map = gl.build_ch_map(detector)
    channel_to_layer = {}
    for layer_id, layer_obj in detector.layers.items():
        for fibre in layer_obj.fibres.values():
            for channel in fibre.interfaces.keys():
                channel_to_layer[int(channel)] = int(layer_id)
    return detector, channel_map, channel_to_layer


def build_fibre_cache(channel_map):
    """
    Builds a cache of x and y coordinates for each channel in the channel map. 
    The cache is stored in a dictionary where the keys are the channel indices 
    and the values are dictionaries containing the x and y coordinates, lengths, 
    start and end positions, and other relevant information for each channel.
    ---
    channel_map : list, the channel map used for computing x and y coordinates
    ---
    returns: dict, a cache of x and y coordinates for each channel in the channel map
    """
    cache_key = id(channel_map)
    if cache_key in _fibre_cache:
        return _fibre_cache[cache_key]

    fibre = {}
    for channel_index, channel_entry in enumerate(channel_map):
        path, layer, direction = channel_entry
        name = np.asarray([getattr(item,"ID", np.nan) for item in path], dtype=str)
        x_cm = np.asarray([getattr(item, "x", np.nan) for item in path], dtype=float)
        y_cm = np.asarray([getattr(item, "y", np.nan) for item in path], dtype=float)
        item_lengths = np.asarray([item.length for item in path], dtype=float)
        fibre_length = np.sum(item_lengths)
        layer_id = _height_to_layer[layer]
        
        fibre[channel_index] = {"item": name,
                                "direction": direction,
                                "fibre_length": fibre_length,
                                "item_lengths": item_lengths,
                                "x_cm": x_cm,
                                "y_cm": y_cm,
                                "layer": layer_id}

    _fibre_cache[cache_key] = fibre
    return fibre

def build_bar_cache(detector):
    cache_key = id(detector)
    if cache_key in _bar_cache:
        return _bar_cache[cache_key]

    bar = {}
    for layer_id, layer in detector.layers.items():
        z = np.float32(layer.z * 10.0)  # geometry is in cm; data is in mm
        bar[z] = {}
        for fibre in layer.fibres.values():
            for b in fibre.bars.values():
                x_min, x_max, y_min, y_max = b.get_bounds(layer_id)
                bar[z][f'{b.ID}-{layer_id}'] = (([x_min * 10.0, x_max * 10.0], [y_min * 10.0, y_max * 10.0], z))

    _bar_cache[cache_key] = bar
    return bar


def compute_dist(channel_map, channel_pairs, time_pairs):
    '''
    vectorized version of get_dist that processes multiple hits at once.
    given lists of channel pairs and time pairs, calculates distances and times of arrival for all hits
    ...
    map : list, the channel map
    channel_pairs : list of [ch1, ch2] pairs
    time_pairs : list of [t1, t2] pairs
    ...
    returns: pandas DataFrame with columns [ref_channel, toa_ns, dist_cm, z_cm] for all hits
    '''
    channels_arr = np.asarray(channel_pairs)
    times_arr = np.asarray(time_pairs)

    channels = channels_arr[:, 0]
    z_values = np.fromiter((channel_map[ch][1] for ch in channels), dtype=float)
    fibre_paths = build_fibre_cache(channel_map)
    fibre_lengths = [channel["fibre_length"] for channel in (fibre_paths[ch] for ch in channels)]
    # Convert times to arrays in ns, replacing any -1 with 0
    t1s = np.where(times_arr[:, 0] != -1, times_arr[:, 0], 0) * 0.001
    t2s = np.where(times_arr[:, 1] != -1, times_arr[:, 1], 0) * 0.001

    # Vectorized calculations
    # Correct Time & Distance Calculation:
    tdiffs = t1s - t2s
    ttots = np.array(fibre_lengths) / _v
    ts = 0.5 * (ttots - np.abs(tdiffs))
    hitds = 0.5 * (np.array(fibre_lengths) - (tdiffs * _v))
    hitts = np.where(t1s < t2s, t1s - ts, t2s - ts)

    # Get bar_id and bar_bounds for hits based on the distance along the fibre using the premade fibre_cache
    hitassids = []
    for ch, dist in zip(channels, hitds):
        item_lengths = fibre_paths[ch]["item_lengths"]
        item_names = fibre_paths[ch]["item"]
        item_layer = fibre_paths[ch]["layer"]
        if dist > float(np.sum(item_lengths)) or dist < 0:
            hitassids.append(np.nan)
            continue
        remaining = dist
        for length, name in zip(item_lengths, item_names):
            remaining -= length
            if remaining < 0:
                hitassids.append(f"{str(name)}-{item_layer}")
                break

    return pd.DataFrame({'hitz': z_values, 'hitt': hitts, 'hitdt': tdiffs, 'hitd': hitds, 'hitchid': channels, 'hitassid': hitassids})


def compute_xy(channel_map, channels, dist_cm):
    """
    Calculates the x and y coordinates for a set of channels and distances using the provided channel map.
    ---
    channel_map : list, the channel map used for computing x and y coordinates
    channels : list or array, the channels for which to compute x and y coordinates
    dist_cm : list or array, the distances in centimeters corresponding to each channel
    ---
    returns: tuple of (x_values, y_values) arrays containing the computed x and y 
             coordinates for the given channels and distances
    """
    channels = np.asarray(channels, dtype=int)
    dist_cm = np.asarray(dist_cm, dtype=float)
    x_cm = np.zeros(len(channels), dtype=float)
    y_cm = np.zeros(len(channels), dtype=float)
    x_err = np.zeros(len(channels), dtype=float)
    y_err = np.zeros(len(channels), dtype=float)
    fibre_paths = build_fibre_cache(channel_map)

    for i, (ch, dist) in tqdm(enumerate(zip(channels, dist_cm)), total=len(channels), desc="Computing x and y coordinates"):
        direction = fibre_paths[ch]["direction"]
        item_lengths = fibre_paths[ch]["item_lengths"]
        item_xs_cm = fibre_paths[ch]["x_cm"]
        item_ys_cm = fibre_paths[ch]["y_cm"]
    
        if dist > float(np.sum(item_lengths)) or dist < 0:
            continue
        
        remaining = dist
        for cnt, (length, x, y) in enumerate(zip(item_lengths, item_xs_cm, item_ys_cm), start=1):
            remaining -= length
            if remaining < 0:
                if (cnt % 2) == 0:                  # even for bars
                    if (cnt % 4) == 0:              # if cnt % 4 == 0, we are moving towards start of bar so add i.e +(-f_len)
                        if direction == 'x':
                            x_cm[i] = x + remaining
                            y_cm[i] = y
                            x_err[i] = 50.0
                            y_err[i] = 20.0
                        else:
                            x_cm[i] = x
                            y_cm[i] = y - remaining
                            x_err[i] = 20.0
                            y_err[i] = 50.0
                        break
                    else:                           # if cnt % 4 != 0, we are moving towards end of bar so subtract i.e -(-f_len)
                        if direction == 'x':
                            x_cm[i] = x - remaining - length
                            y_cm[i] = y
                            x_err[i] = 50.0
                            y_err[i] = 20.0
                        else:
                            x_cm[i] = x
                            y_cm[i] = y + remaining + length
                            x_err[i] = 20.0
                            y_err[i] = 50.0
                        break
                else:
                    x_cm[i] = 0
                    y_cm[i] = 0
                    x_err[i] = 0.0
                    y_err[i] = 0.0
                    break
    return x_cm, y_cm, x_err, y_err


def build_hit_table(raw_input, channel_map, channel_to_layer):
    """
    Function to build a hit table from raw input data, channel map, and channel-to-layer mapping. 
    It processes each entry in the raw input, extracts hit information, computes distances and times 
    of arrival, and constructs a DataFrame for each entry. The resulting DataFrames are collected 
    into a list for further analysis or processing.
    ---
    raw_input : dict, contains the raw input data including the reader and frame
    channel_map : list, the channel map used for computing distances and times of arrival
    channel_to_layer : dict, mapping of channels to their corresponding layers
    max_entries : int or None, maximum number of entries to process (None for all entries)
    progress : int or None, frequency of progress updates (None for no updates)
    ---
    returns: list of pandas DataFrames, each containing hit information for an entry
    """
    # Get the reader, and number of entries from the raw input
    rdf = raw_input.get("rdf")
    source_columns = ["channelID1", "channelID2", "time1", "time2", "energy1", "energy2", "xi1", "xi2", "yi1", "yi2"]
    arrays = rdf.AsNumpy(source_columns)

    channel_pairs = np.column_stack((arrays["channelID1"], arrays["channelID2"]))
    time_pairs = np.column_stack((arrays["time1"], arrays["time2"]))

    hit_table = compute_dist(channel_map, channel_pairs, time_pairs)
    x_cm, y_cm, x_err, y_err = compute_xy(channel_map, hit_table["hitchid"].to_numpy(dtype=int), hit_table["hitd"].to_numpy(dtype=float))

    hit_table["hitx"] = x_cm * 10.0
    hit_table["hity"] = y_cm * 10.0
    hit_table["hitz"] = hit_table["hitz"] * 10.0
    hit_table["hitxerr"] = x_err
    hit_table["hityerr"] = y_err
    hit_table["hitzerr"] = 10.0
    hit_table["hitterr"] = 10.0
    hit_table["hitlay"] = hit_table["hitchid"].map(channel_to_layer)
    hit_table["hitdetid"] = np.arange(len(hit_table))
    hit_table["hitnrg1"] = arrays["energy1"]
    hit_table["hitnrg2"] = arrays["energy2"]
    hit_table["hitxi1"] = arrays["xi1"]
    hit_table["hitxi2"] = arrays["xi2"]
    hit_table["hity1"] = arrays["yi1"]
    hit_table["hity2"] = arrays["yi2"]

    hit_table = hit_table[[
            "hitx",
            "hity",
            "hitz",
            "hitt",
            "hitdt",
            "hitd",
            "hitxerr",
            "hityerr",
            "hitzerr",
            "hitterr",
            "hitlay",
            "hitdetid",
            "hitchid",
            "hitassid",
            "hitnrg1",
            "hitnrg2",
            "hitxi1",
            "hitxi2",
            "hity1",
            "hity2"]]

    return hit_table


def remove_time_outliers(cluster_df, sigma_cut=4.0):
    """
    Removes time outliers from a cluster of hits based on the median and median absolute deviation (MAD).
    Hits that are more than `sigma_cut` times the robust standard deviation away from the median are 
    considered outliers and removed.
    ---
    cluster_df : pandas DataFrame, the cluster of hits to process
    sigma_cut : float, the number of robust standard deviations to use as the cutoff for outliers
    ---
    returns: pandas DataFrame, the cluster of hits with outliers removed
    """
    # If the cluster has fewer than 4 hits, return it unchanged
    if len(cluster_df) < 4:
        return cluster_df.copy()

    # Extract the time of arrival values as a NumPy array of floats
    times = cluster_df["hitt"].astype(float).to_numpy()   # Convert to float for numerical stability
    median_time = np.nanmedian(times)                       # Compute the median time of arrival
    mad = np.nanmedian(np.abs(times - median_time))         # Compute the median absolute deviation (MAD) of the times
    # If the MAD is not finite or is less than or equal to zero, return the cluster unchanged
    if not np.isfinite(mad) or mad <= 0:
        return cluster_df.copy()

    # Compute the robust standard deviation using the MAD and the scaling factor for a normal distribution
    robust_sigma = 1.4826 * mad
    keep_mask = np.abs(times - median_time) <= sigma_cut * robust_sigma
    return cluster_df.loc[keep_mask].copy()


def finalize_cluster(cluster_df, cluster_id, min_hits=3, min_layers=3):
    """
    Finalizes a cluster of hits by removing outliers and checking validity criteria.
    ---
    cluster_df : pandas DataFrame, the cluster of hits to finalize
    cluster_id : int, the identifier for the cluster
    min_hits : int, the minimum number of hits required for a valid cluster
    min_layers : int, the minimum number of unique layers required for a valid cluster
    ---
    returns: tuple of (cleaned cluster DataFrame, manifest dictionary) or 
             (None, manifest dictionary) if the cluster is invalid
    """
    # Sort the cluster DataFrame by time of arrival to ensure consistent ordering
    cleaned = cluster_df.sort_values(["hitt"], kind="mergesort").copy()
    # Get the unique layers and groups present in the cleaned cluster
    layers = sorted({int(layer) for layer in cleaned["hitlay"].dropna().astype(int).tolist()}) if "hitlay" in cleaned.columns else []
    # Calculate the start time, end time, and span of the cluster based on the time of arrival values
    start_t = float(cleaned["hitt"].min()) if len(cleaned) > 0 else np.nan
    end_t = float(cleaned["hitt"].max()) if len(cleaned) > 0 else np.nan
    span_t = float(end_t - start_t) if len(cleaned) > 0 else np.nan

    # If not enough hits are present, return None and a manifest indicating the reason fror rejection
    if len(cleaned) < min_hits:
        return None, {"reason": "too_few_hits",
                      "hitevtid": cluster_id,
                      "t0": start_t,
                      "tf": end_t,
                      "deltat": span_t,
                      "nhits": len(cleaned),
                      "lays": layers}
    # If not enough layers are present, return None and a manifest indicating the reason for rejection
    if len(layers) > 0 and len(layers) < min_layers:
        return None, {"reason": "too_few_layers",
                      "hitevtid": cluster_id,
                      "t0": start_t,
                      "tf": end_t,
                      "deltat": span_t,
                      "nhits": len(cleaned),
                      "lays": layers}
    # If layers not increasing in time, return None and a manifest indicating the reason for rejection
    if not cleaned["hitlay"].is_monotonic_increasing:
        return None, {"reason": "not_increasing_time",
                      "hitevtid": cluster_id,
                      "t0": start_t,
                      "tf": end_t,
                      "deltat": span_t,
                      "nhits": len(cleaned),
                      "lays": layers}

    # Add cluster identifiers as int
    cleaned["hitevtid"] = cluster_id
    cleaned["hitdetid"] = cleaned["hitdetid"].astype(int)
    cleaned["hitevtid"] = cleaned["hitevtid"].astype(int)
    cleaned["hitchid"] = cleaned["hitchid"].astype(int)
    cleaned["hitlay"] = cleaned["hitlay"].astype(int)

    # Manifest dictionary summarizing the cluster's properties
    manifest = {"hitevtid": cluster_id,
                "t0": start_t,
                "tf": end_t,
                "deltat": span_t,
                "nhits": len(cleaned),
                "layers": ",".join(map(str, layers))}
    
    return cleaned, manifest


def cluster_hits_by_time(hit_table, time_gap_ns, max_cluster_span_ns, min_hits=3, min_layers=3):
    """
    Clusters hits in the hit table based on time proximity. Hits that are close in time are grouped 
    together into clusters, and each cluster is finalized by removing outliers and checking for validity 
    based on the specified parameters.
    ---
    hit_table : pandas DataFrame, the table of hits to cluster
    time_gap_ns : float, the maximum time gap between consecutive hits to be considered part of the same cluster
    max_cluster_span_ns : float, the maximum time span of a cluster
    min_hits : int, the minimum number of hits required for a valid cluster
    min_layers : int, the minimum number of unique layers required for a valid cluster
    sigma_cut : float, the number of robust standard deviations to use for outlier removal
    ---
    returns: tuple of (hits Dataframe with add cluster id, clustered hits DataFrame, cluster manifest DataFrame, rejected manifest DataFrame)
    """
    # Sort the hit table by time of arrival to ensure consistent ordering
    sorted_hits = hit_table.sort_values(["hitt"], kind="mergesort").reset_index(drop=True)
    accepted_hits = []
    accepted_hits_info = []
    rejected_hits_info = []
    current_hits = []
    cluster_id = 0
    cluster_ids = []

    # Define a nested function to flush the current cluster of hits and finalize it
    def flush(rows, current_cluster_id):
        """
        Finalizes the current cluster of hits and appends the results to the accepted or rejected lists.
        ---
        rows : list of dicts, the current cluster of hits to finalize
        current_cluster_id : int, the identifier for the current cluster
        """
        # Use nonlocal to access variables from the enclosing scope
        nonlocal accepted_hits, accepted_hits_info, rejected_hits_info
        cluster_these_hits = pd.DataFrame(rows)
        clustered_hits, manifest = finalize_cluster(cluster_these_hits, current_cluster_id, min_hits, min_layers)
        # If cluster is invalid, add the manifest to rejected_rows
        if clustered_hits is None:
            rejected_hits_info.append(manifest)
        # If cluster is valid, add the clustered hits and manifest
        else:
            accepted_hits.append(clustered_hits)
            accepted_hits_info.append(manifest)

    # Iterate over each hit to build clusters based on time proximity
    hit_iter = tqdm(sorted_hits.iterrows(), total=len(sorted_hits), desc="Clustering hits within " + str(time_gap_ns) + "ns") if tqdm is not None else sorted_hits.iterrows()
    for _, hit in hit_iter:
        hit_dict = hit.to_dict()
        cluster_ids.append(cluster_id)
        # If there are no current hits, start a new cluster with the current hit
        if len(current_hits) == 0:
            current_hits.append(hit_dict)
            continue
        
        prev_time = float(current_hits[-1]["hitt"])   # toa of the previous hit in the current cluster
        first_time = float(current_hits[0]["hitt"])   # toa of the first hit in the current cluster
        current_time = float(hit_dict["hitt"])        # toa of the current hit being processed
        time_gap = current_time - prev_time             # time between the current hit and the previous hit
        cluster_span = current_time - first_time        # time between the current hit and the first hit

        # If time gap or cluster span exceeds max, finalize and flush current cluster then start new one
        if time_gap > time_gap_ns or cluster_span > max_cluster_span_ns:
            flush(current_hits, cluster_id)
            cluster_id += 1
            current_hits = [hit_dict]
        # Else add the current hit to the current cluster
        else:
            current_hits.append(hit_dict)

    # Flush any remaining hits after the loop
    flush(current_hits, cluster_id)
    # Build dataframes for the accepted clusters and their info
    clustered_hits = pd.concat(accepted_hits, ignore_index=True)
    cluster_manifest = pd.DataFrame.from_records(accepted_hits_info)
    rejected_manifest = pd.DataFrame.from_records(rejected_hits_info)
    if len(clustered_hits) > 0:
        clustered_hits = clustered_hits.sort_values(["hitt"], kind="mergesort").reset_index(drop=True)

    return clustered_hits, cluster_manifest, rejected_manifest


def to_root(hit_table, output_file):
    """
    Converts a hit table to a ROOT file format.
    ---
    hit_table : pandas DataFrame, the table of hits to convert
    output_file : str, the path to the output ROOT file
    ---
    returns: None
    """
    tree_writer = root.tfile_writer("data", output_file)
    # Define branches for the hit data
    tree_writer.define_branch("hitx", "vector<float>")
    tree_writer.define_branch("hity", "vector<float>")
    tree_writer.define_branch("hitz", "vector<float>")
    tree_writer.define_branch("hitt", "vector<float>")
    tree_writer.define_branch("hitdt", "vector<float>")
    tree_writer.define_branch("hitd", "vector<float>")
    tree_writer.define_branch("hitxerr", "vector<float>")
    tree_writer.define_branch("hityerr", "vector<float>")
    tree_writer.define_branch("hitzerr", "vector<float>")
    tree_writer.define_branch("hitterr", "vector<float>")
    tree_writer.define_branch("hitlay", "vector<int>")
    tree_writer.define_branch("hitchid", "vector<int>")
    tree_writer.define_branch("hitdetid", "vector<int>")
    tree_writer.define_branch("hitevtid", "int")
    tree_writer.define_branch("hitassid", "vector<string>")
    tree_writer.define_branch("hitnrg1", "vector<float>")
    tree_writer.define_branch("hitnrg2", "vector<float>")

    _x = []
    _y = []
    _z = []
    _t = []
    _dt = []
    _d = []
    _xerr = []
    _yerr = []
    _zerr = []
    _terr = []
    _lay = []
    _chid = []
    _id = []
    _evtid = 0
    _assid = []
    _nrg1 = []
    _nrg2 = []

    current_cluster = None

    for hit in tqdm(hit_table.itertuples(index=False), total=len(hit_table), desc="Writing hits to ROOT file"):
        if current_cluster is None:
            current_cluster = hit.hitevtid
        elif hit.hitevtid != current_cluster:
            # Write the current batch to the ROOT file
            data = {"hitx": _x,
                    "hity": _y,
                    "hitz": _z,
                    "hitt": _t,
                    "hitdt": _dt,
                    "hitd": _d,
                    "hitxerr": _xerr,
                    "hityerr": _yerr,
                    "hitzerr": _zerr,
                    "hitterr": _terr,
                    "hitlay": _lay,
                    "hitchid": _chid,
                    "hitdetid": _id,
                    "hitevtid": _evtid,
                    "hitassid": _assid,
                    "hitnrg1": _nrg1,
                    "hitnrg2": _nrg2}
            tree_writer.fill(data)
            tree_writer.tree.Write()

            # Reset for the next event
            current_cluster = hit.hitevtid
            _x.clear()
            _y.clear()
            _z.clear()
            _t.clear()
            _dt.clear()
            _d.clear()
            _xerr.clear()
            _yerr.clear()
            _zerr.clear()
            _terr.clear()
            _lay.clear()
            _chid.clear()
            _id.clear()
            _evtid = 0
            _assid.clear()
            _nrg1.clear()
            _nrg2.clear()

        # Append the current hit's data to the batch lists
        _x.append(hit.hitx)
        _y.append(hit.hity)
        _z.append(hit.hitz)
        _t.append(hit.hitt)
        _dt.append(hit.hitdt)
        _d.append(hit.hitd)
        _xerr.append(hit.hitxerr)
        _yerr.append(hit.hityerr)
        _zerr.append(hit.hitzerr)
        _terr.append(hit.hitterr)
        _lay.append(hit.hitlay)
        _chid.append(hit.hitchid)
        _id.append(hit.hitdetid)
        _evtid = hit.hitevtid
        _assid.append(hit.hitassid)
        _nrg1.append(hit.hitnrg1)
        _nrg2.append(hit.hitnrg2)

    # Write any remaining hits after the loop
    if _x:
        data = {"hitx": _x,
                "hity": _y,
                "hitz": _z,
                "hitt": _t,
                "hitdt": _dt,
                "hitd": _d,
                "hitxerr": _xerr,
                "hityerr": _yerr,
                "hitzerr": _zerr,
                "hitterr": _terr,
                "hitlay": _lay,
                "hitchid": _chid,
                "hitdetid": _id,
                "hitevtid": _evtid,
                "hitassid": _assid,
                "hitnrg1": _nrg1,
                "hitnrg2": _nrg2}
        tree_writer.fill(data)

    tree_writer.file.cd()
    tree_writer.tree.Write()
    tree_writer.write_and_close()