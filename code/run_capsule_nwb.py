import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import os
import numpy as np
from pathlib import Path
import shutil
import json
import sys
import time
from datetime import datetime, timedelta
from packaging.version import parse

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc
import spikeinterface.widgets as sw

import sortingview.views as vv


# LOCAL
from version import version as __version__

matplotlib.use("agg")


### PARAMS ###
n_jobs = os.cpu_count()
job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=False)

preprocessing_params = dict(
    preprocessing_strategy="cmr",  # 'destripe' or 'cmr'
    min_preprocessing_duration=120,  # if less than this duration, processing is skipped (probably a test recording)
    highpass_filter=dict(freq_min=300.0, margin_ms=5.0),
    phase_shift=dict(margin_ms=100.0),
    detect_bad_channels=dict(
        method="coherence+psd",
        dead_channel_threshold=-0.5,
        noisy_channel_threshold=1.0,
        outside_channel_threshold=-0.3,
        n_neighbors=11,
        seed=0,
    ),
    remove_out_channels=False,
    remove_bad_channels=False,
    max_bad_channel_fraction_to_remove=1.1,
    common_reference=dict(reference="global", operator="median"),
    highpass_spatial_filter=dict(
        n_channel_pad=60,
        n_channel_taper=None,
        direction="y",
        apply_agc=True,
        agc_window_length_s=0.01,
        highpass_butter_order=3,
        highpass_butter_wn=0.01,
    ),
)

sorter_name = "kilosort2_5"
sorter_params = dict()

qm_params = {
    "presence_ratio": {"bin_duration_s": 60},
    "snr": {"peak_sign": "neg", "peak_mode": "extremum", "random_chunk_kwargs_dict": None},
    "isi_violation": {"isi_threshold_ms": 1.5, "min_isi_ms": 0},
    "rp_violation": {"refractory_period_ms": 1, "censored_period_ms": 0.0},
    "sliding_rp_violation": {
        "bin_size_ms": 0.25,
        "window_size_s": 1,
        "exclude_ref_period_below_ms": 0.5,
        "max_ref_period_ms": 10,
        "contamination_values": None,
    },
    "amplitude_cutoff": {
        "peak_sign": "neg",
        "num_histogram_bins": 100,
        "histogram_smoothing_value": 3,
        "amplitudes_bins_min_ratio": 5,
    },
    "amplitude_median": {"peak_sign": "neg"},
    "amplitude_cv": {
        "average_num_spikes_per_bin": 50,
        "percentiles": (5, 95),
        "min_num_bins": 10,
        "amplitude_extension": "spike_amplitudes",
    },
    "firing_range": {"bin_size_s": 5, "percentiles": (5, 95)},
    "synchrony": {"synchrony_sizes": (2, 4, 8)},
    "nearest_neighbor": {"max_spikes": 10000, "n_neighbors": 4},
    "nn_isolation": {"max_spikes": 10000, "min_spikes": 10, "n_neighbors": 4, "n_components": 10, "radius_um": 100},
    "nn_noise_overlap": {"max_spikes": 10000, "min_spikes": 10, "n_neighbors": 4, "n_components": 10, "radius_um": 100},
    "silhouette": {"method": ("simplified",)}
}
qm_metric_names = [
    "num_spikes",
    "firing_rate",
    "presence_ratio",
    "snr",
    "isi_violation",
    "rp_violation",
    "sliding_rp_violation",
    "amplitude_cutoff",
    "amplitude_median",
    "amplitude_cv",
    "synchrony",
    "firing_range",
    "drift",
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "nearest_neighbor",
    "silhouette"
]

sparsity_params = dict(method="radius", radius_um=100)

postprocessing_params = dict(
    sparsity=sparsity_params,
    waveforms_deduplicate=dict(
        ms_before=0.5,
        ms_after=1.5,
        max_spikes_per_unit=100,
        return_scaled=False,
        dtype=None,
        precompute_template=("average",),
        use_relative_path=True,
    ),
    waveforms=dict(
        ms_before=3.0,
        ms_after=4.0,
        max_spikes_per_unit=500,
        return_scaled=True,
        dtype=None,
        precompute_template=("average", "std"),
        use_relative_path=True,
    ),
    spike_amplitudes=dict(
        peak_sign="neg",
        return_scaled=True,
        outputs="concatenated",
    ),
    similarity=dict(method="cosine_similarity"),
    correlograms=dict(
        window_ms=50.0,
        bin_ms=1.0,
    ),
    isis=dict(
        window_ms=100.0,
        bin_ms=5.0,
    ),
    locations=dict(method="monopolar_triangulation"),
    template_metrics=dict(upsampling_factor=10, sparsity=None, include_multi_channel_metrics=True),
    principal_components=dict(n_components=5, mode="by_channel_local", whiten=True),
    quality_metrics=dict(qm_params=qm_params, metric_names=qm_metric_names, n_jobs=1),
)

curation_params = dict(
    duplicate_threshold=0.9,
    isi_violations_ratio_threshold=0.5,
    presence_ratio_threshold=0.8,
    amplitude_cutoff_threshold=0.1,
)

visualization_params = dict(
    timeseries=dict(n_snippets_per_segment=2, snippet_duration_s=0.5, skip=False),
    drift=dict(
        detection=dict(method="locally_exclusive", peak_sign="neg", detect_threshold=5, exclude_sweep_ms=0.1),
        localization=dict(ms_before=0.1, ms_after=0.3, radius_um=100.0),
        n_skip=30,
        alpha=0.15,
        vmin=-200,
        vmax=0,
        cmap="Greys_r",
        figsize=(10, 10),
    ),
)

data_folder = Path("../data")
scratch_folder = Path("../scratch")
results_folder = Path("../results")

if scratch_folder.is_dir():
    shutil.rmtree(scratch_folder)
scratch_folder.mkdir(exist_ok=True)
if results_folder.is_dir():
    shutil.rmtree(results_folder)
results_folder.mkdir(exist_ok=True)

tmp_folder = results_folder / "tmp"
if tmp_folder.is_dir():
    shutil.rmtree(tmp_folder)
tmp_folder.mkdir()

visualization_output = {}


if __name__ == "__main__":
    datetime_now = datetime.now()
    t_global_start = time.perf_counter()
    # SET DEFAULT JOB KWARGS
    si.set_global_job_kwargs(**job_kwargs)

    kachery_zone = os.getenv("KACHERY_ZONE", None)
    print(f"Kachery Zone: {kachery_zone}")

    if len(sys.argv) == 7:
        PREPROCESSING_STRATEGY = sys.argv[1]
        if sys.argv[2] == "true":
            REMOVE_OUT_CHANNELS = True
        else:
            REMOVE_OUT_CHANNELS = False
        if sys.argv[3] == "true":
            REMOVE_BAD_CHANNELS = True
        else:
            REMOVE_BAD_CHANNELS = False
        if sys.argv[4] == "true":
            DEBUG = True
        else:
            DEBUG = False
        DEBUG_DURATION = float(sys.argv[5]) if DEBUG else None
        if sys.argv[6] == "true":
            CONCAT = True
        else:
            CONCAT = False
    else:
        PREPROCESSING_STRATEGY = "cmr"
        REMOVE_OUT_CHANNELS = True
        REMOVE_BAD_CHANNELS = True
        DEBUG = False
        DEBUG_DURATION = False
        CONCAT = False

    assert PREPROCESSING_STRATEGY in [
        "cmr",
        "destripe",
    ], f"Preprocessing strategy can be 'cmr' or 'destripe'. {PREPROCESSING_STRATEGY} not supported."
    preprocessing_params["preprocessing_strategy"] = PREPROCESSING_STRATEGY
    preprocessing_params["remove_out_channels"] = REMOVE_OUT_CHANNELS
    preprocessing_params["remove_bad_channels"] = REMOVE_BAD_CHANNELS

    if DEBUG:
        print("DEBUG ENABLED")
        # when debug is enabled let's shorten some steps
        postprocessing_params["waveforms"]["max_spikes_per_unit"] = 200
        visualization_params["timeseries"]["n_snippets_per_segment"] = 1
        visualization_params["timeseries"]["snippet_duration_s"] = 0.1
        visualization_params["timeseries"]["skip"] = False
        # do not use presence ratio for short durations
        curation_params["presence_ratio_threshold"] = 0.1

    ### DATA LOADING SECTION ###

    ## NAB data loader ##
    input_format = "nwb"
    ecephys_nwb_files = [p for p in data_folder.iterdir() if ".nwb" in p.name]
    assert len(ecephys_nwb_files) == 1, "Provide one NWB file at a time"
    ecephys_nwb_file = ecephys_nwb_files[0]

    print(f"Global job kwargs: {si.get_global_job_kwargs()}")
    print(f"Preprocessing strategy: {PREPROCESSING_STRATEGY}")

    ####### PREPROCESSING #######
    print("\n\nPREPROCESSING")
    preprocessed_output_folder = tmp_folder / "preprocessed"

    datetime_start_preproc = datetime.now()
    t_preprocessing_start = time.perf_counter()

    recording_names = []
    preprocessing_notes = ""
    preprocessing_vizualization_data = {}

    recording_name_stem = ecephys_nwb_file.stem
    session_name = recording_name_stem
    recording = se.read_nwb_recording(ecephys_nwb_file)

    if DEBUG:
        recording_list = []
        for segment_index in range(recording.get_num_segments()):
            recording_one = si.split_recording(recording)[segment_index]
            recording_one = recording_one.frame_slice(
                start_frame=0, end_frame=int(DEBUG_DURATION * recording.sampling_frequency)
            )
            recording_list.append(recording_one)
        recording = si.append_recordings(recording_list)

    if CONCAT:
        recordings = [recording]
    else:
        recordings = si.split_recording(recording)

    for i_r, recording in enumerate(recordings):
        if CONCAT:
            recording_name = recording_name_stem
        else:
            recording_name = f"{recording_name_stem}{i_r + 1}"
        preprocessing_vizualization_data[recording_name] = {}
        preprocessing_vizualization_data[recording_name]["timeseries"] = {}
        recording_names.append(recording_name)
        print(f"Preprocessing recording: {recording_name}")
        print(f"\tDuration: {np.round(recording.get_total_duration(), 2)} s")

        preprocessing_vizualization_data[recording_name]["timeseries"]["full"] = dict(raw=recording)
        # maybe a recording is from a different source and it doesn't need to be phase shifted
        if "inter_sample_shift" in recording.get_property_keys():
            recording_ps_full = spre.phase_shift(recording, **preprocessing_params["phase_shift"])
            preprocessing_vizualization_data[recording_name]["timeseries"]["full"].update(
                dict(phase_shift=recording_ps_full)
            )
        else:
            recording_ps_full = recording

        recording_hp_full = spre.highpass_filter(recording_ps_full, **preprocessing_params["highpass_filter"])
        preprocessing_vizualization_data[recording_name]["timeseries"]["full"].update(dict(highpass=recording_hp_full))

        if recording.get_total_duration() < preprocessing_params["min_preprocessing_duration"] and not DEBUG:
            print(f"\tRecording is too short ({recording.get_total_duration()}s). Skipping further processing")
            preprocessing_notes += (
                f"\n- Recording is too short ({recording.get_total_duration()}s). Skipping further processing\n"
            )
            skip_processing = True
        if not recording.has_channel_location():
            print(f"\tRecording does not have channel locations. Skipping further processing")
            preprocessing_notes += f"\n- Recording does not have channel locations. Skipping further processing\n"
            skip_processing = True

        if not skip_processing:
            # IBL bad channel detection
            _, channel_labels = spre.detect_bad_channels(
                recording_hp_full, **preprocessing_params["detect_bad_channels"]
            )
            dead_channel_mask = channel_labels == "dead"
            noise_channel_mask = channel_labels == "noise"
            out_channel_mask = channel_labels == "out"
            print(f"\tBad channel detection:")
            print(
                f"\t\t- dead channels - {np.sum(dead_channel_mask)}\n\t\t- noise channels - {np.sum(noise_channel_mask)}\n\t\t- out channels - {np.sum(out_channel_mask)}"
            )
            dead_channel_ids = recording_hp_full.channel_ids[dead_channel_mask]
            noise_channel_ids = recording_hp_full.channel_ids[noise_channel_mask]
            out_channel_ids = recording_hp_full.channel_ids[out_channel_mask]

            all_bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids, out_channel_ids))

            skip_processing = False
            max_bad_channel_fraction_to_remove = preprocessing_params["max_bad_channel_fraction_to_remove"]
            if len(all_bad_channel_ids) >= int(max_bad_channel_fraction_to_remove * recording.get_num_channels()):
                print(
                    f"\tMore than {max_bad_channel_fraction_to_remove * 100}% bad channels ({len(all_bad_channel_ids)}). "
                    f"Skipping further processing for this recording."
                )
                preprocessing_notes += (
                    f"\n- Found {len(all_bad_channel_ids)} bad channels. Skipping further processing\n"
                )
                skip_processing = True
            else:
                if preprocessing_params["remove_out_channels"]:
                    print(f"\tRemoving {len(out_channel_ids)} out channels")
                    recording_rm_out = recording_hp_full.remove_channels(out_channel_ids)
                    preprocessing_notes += f"{recording_name}:\n- Removed {len(out_channel_ids)} outside of the brain."
                else:
                    recording_rm_out = recording_hp_full

                recording_processed_cmr = spre.common_reference(
                    recording_rm_out, **preprocessing_params["common_reference"]
                )

                bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids))
                recording_interp = spre.interpolate_bad_channels(recording_rm_out, bad_channel_ids)
                recording_hp_spatial = spre.highpass_spatial_filter(
                    recording_interp, **preprocessing_params["highpass_spatial_filter"]
                )
                preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = dict(
                    highpass=recording_rm_out,
                    cmr=recording_processed_cmr,
                    highpass_spatial=recording_hp_spatial,
                )

                preproc_strategy = preprocessing_params["preprocessing_strategy"]
                if preproc_strategy == "cmr":
                    recording_processed = recording_processed_cmr
                else:
                    recording_processed = recording_hp_spatial

                if preprocessing_params["remove_bad_channels"]:
                    print(f"\tRemoving {len(bad_channel_ids)} channels after {preproc_strategy} preprocessing")
                    recording_processed = recording_processed.remove_channels(bad_channel_ids)
                    preprocessing_notes += f"\n- Removed {len(bad_channel_ids)} bad channels after preprocessing.\n"
                recording_saved = recording_processed.save(folder=preprocessed_output_folder / recording_name)
                recording_drift = recording_saved
        if skip_processing:
            # in this case, processed timeseries will not be visualized
            preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = None
            recording_drift = recording_hp_full
        # store recording for drift visualization
        preprocessing_vizualization_data[recording_name]["drift"] = dict(recording=recording_drift)

    t_preprocessing_end = time.perf_counter()
    elapsed_time_preprocessing = np.round(t_preprocessing_end - t_preprocessing_start, 2)
    print(f"PREPROCESSING time: {elapsed_time_preprocessing}s")

    ####### SPIKESORTING ########
    print("\n\nSPIKE SORTING")
    spikesorting_notes = ""
    sorting_params = None

    datetime_start_sorting = datetime.now()
    t_sorting_start = time.perf_counter()
    preprocessed_folder = preprocessed_output_folder

    # try results here
    spikesorted_raw_output_folder = scratch_folder / "spikesorted_raw"
    for recording_name in recording_names:
        sorting_output_folder = results_folder / "spikesorted" / recording_name

        recording_folder = preprocessed_folder / recording_name
        if not recording_folder.is_dir():
            print(f"Skipping sorting for recording: {recording_name}")
            spikesorting_notes += f"{recording_name}:\n- Skipped spike sorting.\n"
            sorting_params = {}
            continue
        print(f"Sorting recording: {recording_name}")
        recording = si.load_extractor(recording_folder)
        print(recording)

        # we need to concatenate segments for KS
        if CONCAT:
            if recording.get_num_segments() > 1:
                recording = si.concatenate_recordings([recording])

        # run ks2.5
        try:
            sorting = ss.run_sorter(
                sorter_name,
                recording,
                output_folder=spikesorted_raw_output_folder / recording_name,
                verbose=False,
                delete_output_folder=True,
                **sorter_params,
            )
        except Exception as e:
            # save log to results
            sorting_output_folder.mkdir()
            shutil.copy(spikesorted_raw_output_folder / "spikeinterface_log.json", sorting_output_folder)
        print(f"\tRaw sorting output: {sorting}")
        spikesorting_notes += f"{recording_name}:\n- KS2.5 found {len(sorting.unit_ids)} units, "
        if sorting_params is None:
            sorting_params = sorting.sorting_info["params"]

        # remove empty units
        sorting = sorting.remove_empty_units()
        # remove spikes beyond num_Samples (if any)
        sorting = sc.remove_excess_spikes(sorting=sorting, recording=recording)
        print(f"\tSorting output without empty units: {sorting}")
        spikesorting_notes += f"{len(sorting.unit_ids)} after removing empty templates.\n"

        if CONCAT:
            # split back to get original segments
            if recording.get_num_segments() > 1:
                sorting = si.split_sorting(sorting, recording)

        # save results
        print(f"\tSaving results to {sorting_output_folder}")
        sorting = sorting.save(folder=sorting_output_folder)

    t_sorting_end = time.perf_counter()
    elapsed_time_sorting = np.round(t_sorting_end - t_sorting_start, 2)
    print(f"SPIKE SORTING time: {elapsed_time_sorting}s")

    ###### POSTPROCESSING ########
    print("\n\nPOSTPROCESSING")
    postprocessing_notes = ""
    datetime_start_postprocessing = datetime.now()
    t_postprocessing_start = time.perf_counter()

    spikesorted_folder = results_folder / "spikesorted"

    # loop through block-streams
    for recording_name in recording_names:
        recording_folder = preprocessed_folder / recording_name
        if not recording_folder.is_dir():
            print(f"Skipping postprocessing for recording: {recording_name}")
            postprocessing_notes += f"{recording_name}:\n- Skipped post-processsing.\n"
            continue
        print(f"Postprocessing recording: {recording_name}")

        recording = si.load_extractor(recording_folder)

        # make sure we have spikesorted output for the block-stream
        recording_sorted_folder = spikesorted_folder / recording_name
        assert recording_sorted_folder.is_dir(), f"Could not find spikesorted output for {recording_name}"
        sorting = si.load_extractor(recording_sorted_folder.absolute().resolve())

        # first extract some raw waveforms in memory to deduplicate based on peak alignment
        wf_dedup_folder = tmp_folder / "postprocessed" / recording_name
        we_raw = si.extract_waveforms(
            recording, sorting, folder=wf_dedup_folder, **postprocessing_params["waveforms_deduplicate"]
        )
        # de-duplication
        sorting_deduplicated = sc.remove_redundant_units(
            we_raw, duplicate_threshold=curation_params["duplicate_threshold"]
        )
        print(
            f"\tNumber of original units: {len(we_raw.sorting.unit_ids)} -- Number of units after de-duplication: {len(sorting_deduplicated.unit_ids)}"
        )
        postprocessing_notes += f"{recording_name}:\n- Removed {len(sorting.unit_ids) - len(sorting_deduplicated.unit_ids)} duplicated units.\n"
        deduplicated_unit_ids = sorting_deduplicated.unit_ids
        # use existing deduplicated waveforms to compute sparsity
        sparsity_raw = si.compute_sparsity(we_raw, **sparsity_params)
        sparsity_mask = sparsity_raw.mask[sorting.ids_to_indices(deduplicated_unit_ids), :]
        sparsity = si.ChannelSparsity(
            mask=sparsity_mask, unit_ids=deduplicated_unit_ids, channel_ids=recording.channel_ids
        )
        shutil.rmtree(wf_dedup_folder)
        del we_raw

        wf_sparse_folder = results_folder / "postprocessed" / recording_name

        # now extract waveforms on de-duplicated units
        print(f"\tSaving sparse de-duplicated waveform extractor folder")
        we = si.extract_waveforms(
            recording,
            sorting_deduplicated,
            folder=wf_sparse_folder,
            sparsity=sparsity,
            sparse=True,
            overwrite=True,
            **postprocessing_params["waveforms"],
        )
        print("\tComputing spike amplitides")
        amps = spost.compute_spike_amplitudes(we, **postprocessing_params["spike_amplitudes"])
        print("\tComputing unit locations")
        unit_locs = spost.compute_unit_locations(we, **postprocessing_params["locations"])
        print("\tComputing spike locations")
        spike_locs = spost.compute_spike_locations(we, **postprocessing_params["locations"])
        print("\tComputing correlograms")
        corr = spost.compute_correlograms(we, **postprocessing_params["correlograms"])
        print("\tComputing ISI histograms")
        tm = spost.compute_isi_histograms(we, **postprocessing_params["isis"])
        print("\tComputing template similarity")
        sim = spost.compute_template_similarity(we, **postprocessing_params["similarity"])
        print("\tComputing template metrics")
        tm = spost.compute_template_metrics(we, **postprocessing_params["template_metrics"])
        print("\tComputing PCA")
        pc = spost.compute_principal_components(we, **postprocessing_params["principal_components"])
        print("\tComputing quality metrics")
        qm = sqm.compute_quality_metrics(we, **postprocessing_params["quality_metrics"])

    t_postprocessing_end = time.perf_counter()
    elapsed_time_postprocessing = np.round(t_postprocessing_end - t_postprocessing_start, 2)
    print(f"POSTPROCESSING time: {elapsed_time_postprocessing}s")

    ###### CURATION ##############
    print("\n\nCURATION")
    curation_notes = ""
    datetime_start_curation = datetime.now()
    t_curation_start = time.perf_counter()

    # curation query
    isi_violations_ratio_thr = curation_params["isi_violations_ratio_threshold"]
    presence_ratio_thr = curation_params["presence_ratio_threshold"]
    amplitude_cutoff_thr = curation_params["amplitude_cutoff_threshold"]

    curation_query = f"isi_violations_ratio < {isi_violations_ratio_thr} and presence_ratio > {presence_ratio_thr} and amplitude_cutoff < {amplitude_cutoff_thr}"
    print(f"Curation query: {curation_query}")
    curation_notes += f"Curation query: {curation_query}\n"

    postprocessed_folder = results_folder / "postprocessed"

    # loop through block-streams
    for recording_name in recording_names:
        recording_folder = postprocessed_folder / recording_name
        if not recording_folder.is_dir():
            print(f"Skipping curation for recording: {recording_name}")
            curation_notes += f"{recording_name}:\n- Skipped curation.\n"
            continue
        print(f"Curating recording: {recording_name}")

        we = si.load_waveforms(recording_folder)

        # get quality metrics
        qm = we.load_extension("quality_metrics").get_data()
        qm_curated = qm.query(curation_query)
        curated_unit_ids = qm_curated.index.values

        # flag units as good/bad depending on QC selection
        qc_quality = [True if unit in curated_unit_ids else False for unit in we.sorting.unit_ids]
        sorting_precurated = we.sorting
        sorting_precurated.set_property("default_qc", qc_quality)
        sorting_precurated.save(folder=results_folder / "sorting_precurated" / recording_name)
        curation_notes += (
            f"{recording_name}:\n- {np.sum(qc_quality)}/{len(sorting_precurated.unit_ids)} passing default QC.\n"
        )

    t_curation_end = time.perf_counter()
    elapsed_time_curation = np.round(t_curation_end - t_curation_start, 2)
    print(f"CURATION time: {elapsed_time_curation}s")

    ###### VISUALIZATION #########
    print("\n\nVISUALIZATION")
    t_visualization_start = time.perf_counter()
    datetime_start_visualization = datetime.now()

    postprocessed_folder = results_folder / "postprocessed"

    # loop through block-streams
    for recording_name in recording_names:
        recording_folder = postprocessed_folder / recording_name
        print(f"Visualizing recording: {recording_name}")

        if recording_name not in visualization_output:
            visualization_output[recording_name] = {}

        # drift
        cmap = plt.get_cmap(visualization_params["drift"]["cmap"])
        norm = Normalize(
            vmin=visualization_params["drift"]["vmin"], vmax=visualization_params["drift"]["vmax"], clip=True
        )
        n_skip = visualization_params["drift"]["n_skip"]
        alpha = visualization_params["drift"]["alpha"]

        # use spike locations
        if recording_folder.is_dir():
            print(f"\tVisualizing drift maps using spike sorted data")
            we = si.load_waveforms(recording_folder)
            recording = we.recording
            peaks = we.sorting.to_spike_vector()
            peak_locations = we.load_extension("spike_locations").get_data()
            peak_amps = np.concatenate(we.load_extension("spike_amplitudes").get_data())
        # otherwise detect peaks
        else:
            from spikeinterface.core.node_pipeline import ExtractDenseWaveforms, run_node_pipeline
            from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive
            from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass

            print(f"\tVisualizing drift maps using detected peaks (no spike sorting available)")
            # locally_exclusive + pipeline steps LocalizeCenterOfMass + PeakToPeakFeature
            drift_data = preprocessing_vizualization_data[recording_name]["drift"]
            recording = drift_data["recording"]

            if not recording.has_channel_location():
                print(
                    f"\tSkipping drift and timeseries visualization for recording: {recording_name}. No channel locations."
                )
                continue

            # Here we use the node pipeline implementation
            peak_detector_node = DetectPeakLocallyExclusive(recording, **visualization_params["drift"]["detection"])
            extract_dense_waveforms_node = ExtractDenseWaveforms(
                recording,
                ms_before=visualization_params["drift"]["localization"]["ms_before"],
                ms_after=visualization_params["drift"]["localization"]["ms_after"],
                parents=[peak_detector_node],
                return_output=False,
            )
            localize_peaks_node = LocalizeCenterOfMass(
                recording,
                radius_um=visualization_params["drift"]["localization"]["radius_um"],
                parents=[peak_detector_node, extract_dense_waveforms_node],
            )
            pipeline_nodes = [peak_detector_node, extract_dense_waveforms_node, localize_peaks_node]
            peaks, peak_locations = run_node_pipeline(
                recording, pipeline_nodes=pipeline_nodes, job_kwargs=si.get_global_job_kwargs()
            )
            print(f"\tDetected {len(peaks)} peaks")
            peak_amps = peaks["amplitude"]

        y_locs = recording.get_channel_locations()[:, 1]
        ylim = [np.min(y_locs), np.max(y_locs)]

        fig_drift, axs_drift = plt.subplots(
            ncols=recording.get_num_segments(), figsize=visualization_params["drift"]["figsize"]
        )
        for segment_index in range(recording.get_num_segments()):
            segment_mask = peaks["segment_index"] == segment_index
            x = peaks[segment_mask]["sample_index"] / recording.sampling_frequency
            y = peak_locations[segment_mask]["y"]
            # subsample
            x_sub = x[::n_skip]
            y_sub = y[::n_skip]
            a_sub = peak_amps[::n_skip]
            colors = cmap(norm(a_sub))

            if recording.get_num_segments() == 1:
                ax_drift = axs_drift
            else:
                ax_drift = axs_drift[segment_index]
            ax_drift.scatter(x_sub, y_sub, s=1, c=colors, alpha=alpha)
            ax_drift.set_xlabel("time (s)", fontsize=12)
            ax_drift.set_ylabel("depth ($\mu$m)", fontsize=12)
            ax_drift.set_xlim(0, recording.get_num_samples(segment_index=segment_index) / recording.sampling_frequency)
            ax_drift.set_ylim(ylim)
            ax_drift.spines["top"].set_visible(False)
            ax_drift.spines["right"].set_visible(False)
        fig_drift_folder = results_folder / "drift_maps"
        fig_drift_folder.mkdir(exist_ok=True)
        fig_drift.savefig(fig_drift_folder / f"{recording_name}_drift.png", dpi=300)

        # make a sorting view View
        v_drift = vv.TabLayoutItem(
            label=f"Drift map", view=vv.Image(image_path=str(fig_drift_folder / f"{recording_name}_drift.png"))
        )

        # timeseries
        if not visualization_params["timeseries"]["skip"]:
            timeseries_tab_items = []
            print(f"\tVisualizing timeseries")

            timeseries_data = preprocessing_vizualization_data[recording_name]["timeseries"]
            recording_full_dict = timeseries_data["full"]
            recording_proc_dict = timeseries_data["proc"]

            # get random chunks to estimate clims
            clims_full = {}
            for layer, rec in recording_full_dict.items():
                chunk = si.get_random_data_chunks(rec)
                max_value = np.quantile(chunk, 0.99) * 1.2
                clims_full[layer] = (-max_value, max_value)
            clims_proc = {}
            if recording_proc_dict is not None:
                for layer, rec in recording_proc_dict.items():
                    chunk = si.get_random_data_chunks(rec)
                    max_value = np.quantile(chunk, 0.99) * 1.2
                    clims_proc[layer] = (-max_value, max_value)
            else:
                print(f"\tPreprocessed timeseries not avaliable")

            fs = recording.get_sampling_frequency()
            n_snippets_per_seg = visualization_params["timeseries"]["n_snippets_per_segment"]
            try:
                for segment_index in range(recording.get_num_segments()):
                    segment_duration = recording.get_num_samples(segment_index) / fs
                    # evenly distribute t_starts across segments
                    t_starts = np.linspace(0, segment_duration, n_snippets_per_seg + 2)[1:-1]
                    for t_start in t_starts:
                        time_range = np.round(
                            np.array([t_start, t_start + visualization_params["timeseries"]["snippet_duration_s"]]), 1
                        )
                        w_full = sw.plot_timeseries(
                            recording_full_dict,
                            order_channel_by_depth=True,
                            time_range=time_range,
                            segment_index=segment_index,
                            clim=clims_full,
                            backend="sortingview",
                            generate_url=False,
                        )
                        if recording_proc_dict is not None:
                            w_proc = sw.plot_timeseries(
                                recording_proc_dict,
                                order_channel_by_depth=True,
                                time_range=time_range,
                                segment_index=segment_index,
                                clim=clims_proc,
                                backend="sortingview",
                                generate_url=False,
                            )
                            view = vv.Splitter(
                                direction="horizontal",
                                item1=vv.LayoutItem(w_full.view),
                                item2=vv.LayoutItem(w_proc.view),
                            )
                        else:
                            view = w_full.view
                        v_item = vv.TabLayoutItem(
                            label=f"Timeseries - Segment {segment_index} - Time: {time_range}", view=view
                        )
                        timeseries_tab_items.append(v_item)
                # add drift map
                timeseries_tab_items.append(v_drift)

                v_timeseries = vv.TabLayout(items=timeseries_tab_items)
                try:
                    url = v_timeseries.url(label=f"{session_name} - {recording_name}")
                    print(f"\n{url}\n")
                    visualization_output[recording_name]["timeseries"] = url
                except Exception as e:
                    print("KCL error", e)
            except Exception as e:
                print(f"Something wrong when visualizing timeseries: {e}")

        # sorting summary
        if not recording_folder.is_dir():
            print(f"\tSkipping sorting summary visualization for recording: {recording_name}. No sorting data.")
            continue
        print(f"\tVisualizing sorting summary")
        we = si.load_waveforms(recording_folder)
        sorting_precurated = si.load_extractor(results_folder / "sorting_precurated" / recording_name)
        # set waveform_extractor sorting object to have pass_qc property
        we.sorting = sorting_precurated

        if len(we.sorting.unit_ids) > 0:
            # tab layout with Summary and Quality Metrics
            v_qm = sw.plot_quality_metrics(
                we,
                skip_metrics=["isi_violations_count", "rp_violations"],
                include_metrics_data=True,
                backend="sortingview",
                generate_url=False,
            ).view
            v_sorting = sw.plot_sorting_summary(
                we, unit_table_properties=["default_qc"], curation=True, backend="sortingview", generate_url=False
            ).view

            v_summary = vv.TabLayout(
                items=[
                    vv.TabLayoutItem(label="Sorting summary", view=v_sorting),
                    vv.TabLayoutItem(label="Quality Metrics", view=v_qm),
                ]
            )

            try:
                # pre-generate gh for curation
                url = v_summary.url(label=f"{session_name} - {recording_name} - {sorter_name} - Sorting Summary")
                print(f"\n{url}\n")
                visualization_output[recording_name]["sorting_summary"] = url

            except Exception as e:
                print("KCL error", e)
        else:
            print("No units after curation!")

    # save params in output
    visualization_notes = json.dumps(visualization_output, indent=4)
    # replace special characters
    visualization_notes = visualization_notes.replace('\\"', "%22")
    visualization_notes = visualization_notes.replace("#", "%23")

    # save vizualization output
    visualization_output_file = results_folder / "visualization_output.json"
    # remove escape characters
    visualization_output_file.write_text(visualization_notes)

    # save vizualization output
    t_visualization_end = time.perf_counter()
    elapsed_time_visualization = np.round(t_visualization_end - t_visualization_start, 2)
    print(f"VISUALIZATION time: {elapsed_time_visualization}s")

    # save processing files to output
    processing_params = dict(
        preprocessing=preprocessing_params,
        sorting=sorting_params,
        postprocessing=postprocessing_params,
        curation=curation_params,
        visualization=visualization_params,
    )
    with (results_folder / "processing_params.json").open("w") as f:
        json.dump(processing_params, f, indent=4)

    # remove tmp_folder
    shutil.rmtree(tmp_folder)

    t_global_end = time.perf_counter()
    elapsed_time_global = np.round(t_global_end - t_global_start, 2)
    print(f"\n\nFULL PIPELINE time:  {elapsed_time_global}s")