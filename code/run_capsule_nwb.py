import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import os
import argparse
import numpy as np
from pathlib import Path
import shutil
import json
import time
from datetime import datetime, timedelta
from packaging.version import parse

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

matplotlib.use("agg")

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


### ARGPARSE ###
parser = argparse.ArgumentParser(description="AIND processing ephys pipeline")

debug_group = parser.add_mutually_exclusive_group()
debug_help = "Whether to run in DEBUG mode"
debug_group.add_argument("--debug", action="store_true", help=debug_help)
debug_group.add_argument("static_debug", nargs="?", default="false", help=debug_help)

concat_group = parser.add_mutually_exclusive_group()
concat_help = "Whether to concatenate recordings (segments) or not. Default: False"
concat_group.add_argument("--concatenate", action="store_true", help=concat_help)
concat_group.add_argument("static_concatenate", nargs="?", default="false", help=concat_help)

denoising_group = parser.add_mutually_exclusive_group()
denoising_help = "Which denoising strategy to use. Can be 'cmr' or 'destripe'"
denoising_group.add_argument("--denoising", choices=["cmr", "destripe"], help=denoising_help)
denoising_group.add_argument("static_denoising", nargs="?", default="cmr", help=denoising_help)

remove_out_channels_group = parser.add_mutually_exclusive_group()
remove_out_channels_help = "Whether to remove out channels"
remove_out_channels_group.add_argument("--no-remove-out-channels", action="store_true", help=remove_out_channels_help)
remove_out_channels_group.add_argument(
    "static_remove_out_channels", nargs="?", default="true", help=remove_out_channels_help
)

remove_bad_channels_group = parser.add_mutually_exclusive_group()
remove_bad_channels_help = "Whether to remove bad channels"
remove_bad_channels_group.add_argument("--no-remove-bad-channels", action="store_true", help=remove_bad_channels_help)
remove_bad_channels_group.add_argument(
    "static_remove_bad_channels", nargs="?", default="true", help=remove_bad_channels_help
)

max_bad_channel_fraction_group = parser.add_mutually_exclusive_group()
max_bad_channel_fraction_help = (
    "Maximum fraction of bad channels to remove. If more than this fraction, processing is skipped"
)
max_bad_channel_fraction_group.add_argument(
    "--max-bad-channel-fraction", default=0.5, help=max_bad_channel_fraction_help
)
max_bad_channel_fraction_group.add_argument(
    "static_max_bad_channel_fraction", nargs="?", default="0.5", help=max_bad_channel_fraction_help
)

debug_duration_group = parser.add_mutually_exclusive_group()
debug_duration_help = (
    "Duration of clipped recording in debug mode. Default is 30 seconds. Only used if debug is enabled"
)
debug_duration_group.add_argument("--debug-duration", default=30, help=debug_duration_help)
debug_duration_group.add_argument("static_debug_duration", nargs="?", default="30", help=debug_duration_help)

parser.add_argument("--data-folder", default="../data", help="Custom data folder (default ../data)")
parser.add_argument("--results-folder", default="../results", help="Custom results folder (default ../results)")
parser.add_argument("--scratch-folder", default="../scratch", help="Custom scratch folder (default ../scratch)")

n_jobs_help = "Number of jobs to use for parallel processing. Default is -1 (all available cores). It can also be a float between 0 and 1 to use a fraction of available cores"
parser.add_argument("--n-jobs", default="-1", help=n_jobs_help)

params_group = parser.add_mutually_exclusive_group()
params_group.add_argument("--params-file", default=None, help="Optional json file with parameters")
params_group.add_argument("--params-str", default=None, help="Optional json string with parameters")


if __name__ == "__main__":
    datetime_now = datetime.now()
    t_global_start = time.perf_counter()

    args = parser.parse_args()

    DEBUG = args.debug or args.static_debug == "true"
    CONCAT = args.concatenate or args.static_concatenate.lower() == "true"
    DENOISING_STRATEGY = args.denoising or args.static_denoising
    REMOVE_OUT_CHANNELS = False if args.no_remove_out_channels else args.static_remove_out_channels == "true"
    REMOVE_BAD_CHANNELS = False if args.no_remove_bad_channels else args.static_remove_bad_channels == "true"
    MAX_BAD_CHANNEL_FRACTION = float(args.max_bad_channel_fraction or args.static_max_bad_channel_fraction)
    DEBUG_DURATION = float(args.debug_duration or args.static_debug_duration)
    DATA_FOLDER = Path(args.data_folder)
    RESULTS_FOLDER = Path(args.results_folder)
    SCRATCH_FOLDER = Path(args.scratch_folder)
    N_JOBS = int(args.n_jobs) if not args.n_jobs.startswith("0.") else float(args.n_jobs)
    PARAMS_FILE = args.params_file
    PARAMS_STR = args.params_str

    print(f"Running preprocessing with the following parameters:")
    print(f"\tCONCATENATE: {CONCAT}")
    print(f"\tDENOISING_STRATEGY: {DENOISING_STRATEGY}")
    print(f"\tREMOVE_OUT_CHANNELS: {REMOVE_OUT_CHANNELS}")
    print(f"\tREMOVE_BAD_CHANNELS: {REMOVE_BAD_CHANNELS}")
    print(f"\tMAX BAD CHANNEL FRACTION: {MAX_BAD_CHANNEL_FRACTION}")
    print(f"\tDATA_FOLDER: {DATA_FOLDER}")
    print(f"\tRESULTS_FOLDER: {RESULTS_FOLDER}")
    print(f"\tSCRATCH_FOLDER: {SCRATCH_FOLDER}")
    print(f"\tN_JOBS: {N_JOBS}")

    if PARAMS_FILE is not None:
        print(f"\nUsing custom parameter file: {PARAMS_FILE}")
        with open(PARAMS_FILE, "r") as f:
            processing_params = json.load(f)
    elif PARAMS_STR is not None:
        processing_params = json.loads(PARAMS_STR)
    else:
        with open("processing_params.json", "r") as f:
            processing_params = json.load(f)

    job_kwargs = processing_params["job_kwargs"]
    preprocessing_params = processing_params["preprocessing"]
    spikesorting_params = processing_params["spikesorting"]
    postprocessing_params = processing_params["postprocessing"]
    quality_metrics_params = processing_params["quality_metrics"]
    curation_params = processing_params["curation"]
    visualization_params = processing_params["visualization"]

    if DEBUG:
        print(f"\nDEBUG ENABLED - Only running with {DEBUG_DURATION} seconds\n")
        # when debug is enabled let's shorten some steps
        postprocessing_params["waveforms"]["max_spikes_per_unit"] = 200
        visualization_params["timeseries"]["n_snippets_per_segment"] = 1
        visualization_params["timeseries"]["snippet_duration_s"] = 0.1
        visualization_params["timeseries"]["skip"] = False
        # do not use presence ratio for short durations
        curation_params["presence_ratio_threshold"] = 0.1

    preprocessing_params["denoising_strategy"] = DENOISING_STRATEGY
    preprocessing_params["remove_out_channels"] = REMOVE_OUT_CHANNELS
    preprocessing_params["remove_bad_channels"] = REMOVE_BAD_CHANNELS
    preprocessing_params["max_bad_channel_fraction"] = MAX_BAD_CHANNEL_FRACTION

    # set paths
    data_folder = DATA_FOLDER
    scratch_folder = SCRATCH_FOLDER
    results_folder = RESULTS_FOLDER

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

    # this is the main try-except clause to clean up the output folders in case of failures
    try:
        # SET DEFAULT JOB KWARGS
        job_kwargs["n_jobs"] = N_JOBS
        si.set_global_job_kwargs(**job_kwargs)

        kachery_zone = os.getenv("KACHERY_ZONE", None)
        print(f"Kachery Zone: {kachery_zone}")

        ### DATA LOADING SECTION ###

        ## NWB data loader ##
        input_format = "nwb"
        ecephys_nwb_files = [p for p in data_folder.iterdir() if ".nwb" in p.name]
        assert len(ecephys_nwb_files) == 1, "Provide one NWB file at a time"
        ecephys_nwb_file = ecephys_nwb_files[0]

        print(f"Global job kwargs: {si.get_global_job_kwargs()}")

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

            skip_processing = False
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
                max_bad_channel_fraction = preprocessing_params["max_bad_channel_fraction"]
                if len(all_bad_channel_ids) >= int(max_bad_channel_fraction * recording.get_num_channels()):
                    print(
                        f"\tMore than {max_bad_channel_fraction * 100}% bad channels ({len(all_bad_channel_ids)}). "
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

                    denoising_strategy = preprocessing_params["denoising_strategy"]
                    if denoising_strategy == "cmr":
                        recording_processed = recording_processed_cmr
                    else:
                        recording_processed = recording_hp_spatial

                    if preprocessing_params["remove_bad_channels"]:
                        print(f"\tRemoving {len(bad_channel_ids)} channels after {denoising_strategy} preprocessing")
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
                    spikesorting_params["sorter_name"],
                    recording,
                    output_folder=spikesorted_raw_output_folder / recording_name,
                    verbose=False,
                    delete_output_folder=True,
                    **spikesorting_params["sorter_params"],
                )
            except Exception as e:
                # save log to results
                sorting_output_folder.mkdir(parents=True)
                shutil.copy(spikesorted_raw_output_folder / "spikeinterface_log.json", sorting_output_folder)
            print(f"\tRaw sorting output: {sorting}")
            spikesorting_notes += f"{recording_name}:\n- KS2.5 found {len(sorting.unit_ids)} units, "

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
                we_raw, duplicate_threshold=postprocessing_params["duplicate_threshold"]
            )
            print(
                f"\tNumber of original units: {len(we_raw.sorting.unit_ids)} -- Number of units after de-duplication: {len(sorting_deduplicated.unit_ids)}"
            )
            postprocessing_notes += f"{recording_name}:\n- Removed {len(sorting.unit_ids) - len(sorting_deduplicated.unit_ids)} duplicated units.\n"
            deduplicated_unit_ids = sorting_deduplicated.unit_ids
            # use existing deduplicated waveforms to compute sparsity
            sparsity_raw = si.compute_sparsity(we_raw, **postprocessing_params["sparsity"])
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

            # QUALITY METRICS
            print("\tComputing quality metrics")
            qm = sqm.compute_quality_metrics(we, **quality_metrics_params)

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
            sorting_curated = we.sorting
            sorting_curated.set_property("default_qc", qc_quality)
            sorting_curated.save(folder=results_folder / "curated" / recording_name)
            curation_notes += (
                f"{recording_name}:\n- {np.sum(qc_quality)}/{len(sorting_curated.unit_ids)} passing default QC.\n"
            )

        t_curation_end = time.perf_counter()
        elapsed_time_curation = np.round(t_curation_end - t_curation_start, 2)
        print(f"CURATION time: {elapsed_time_curation}s")

        ###### VISUALIZATION #########
        print("\n\nVISUALIZATION")
        t_visualization_start = time.perf_counter()
        datetime_start_visualization = datetime.now()
        postprocessed_folder = results_folder / "postprocessed"
        visualization_output = {}

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
                peaks, peak_locations = run_node_pipeline(recording, nodes=pipeline_nodes, job_kwargs=job_kwargs)
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
            sorting_curated = si.load_extractor(results_folder / "curated" / recording_name)
            # set waveform_extractor sorting object to have pass_qc property
            we.sorting = sorting_curated

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
                    sorter_name = spikesorting_params["sorter_name"]
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
        with (results_folder / "processing_params.json").open("w") as f:
            json.dump(processing_params, f, indent=4)

        # if we got here, it means everything went well
        status = "success"

    except Exception as e:
        import traceback

        error = "".join(traceback.format_exc())
        print(f"\nPIPELINE ERROR:\n\n{error}")

        # cleanup
        if tmp_folder.is_dir():
            shutil.rmtree(tmp_folder)
        for p in results_folder.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        # create an error txt
        error_file = results_folder / "error.log"
        error_file.write_text(f"PIPELINE ERROR:\n{error}")
        status = "error"

    t_global_end = time.perf_counter()
    elapsed_time_global = np.round(t_global_end - t_global_start, 2)
    print(f"\n\nFULL PIPELINE time:  {elapsed_time_global}s - STATUS: {status}\n\n")
