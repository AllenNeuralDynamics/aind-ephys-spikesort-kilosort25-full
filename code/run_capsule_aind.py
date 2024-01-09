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


# AIND
from aind_data_schema.core.data_description import (
    DataDescription,
    DerivedDataDescription,
    Institution,
    Modality,
    Modality,
    Platform,
    Funding,
    DataLevel,
)
from aind_data_schema.core.processing import DataProcess, Processing, PipelineProcess
from aind_data_schema.schema_upgrade.data_description_upgrade import DataDescriptionUpgrade
from aind_data_schema.schema_upgrade.processing_upgrade import ProcessingUpgrade, DataProcessUpgrade

# LOCAL
from version import version as __version__


GH_CURATION_REPO = "gh://AllenNeuralDynamics/ephys-sorting-manual-curation/main"
PIPELINE_URL = "https://github.com/AllenNeuralDynamics/aind-capsule-ephys-spikesort-kilosort25-full.git"
PIPELINE_MAINTAINER = "Alessio Buccino"
PIPELINE_VERSION = __version__


### PARAMS ###
n_jobs = os.cpu_count()
job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=False)


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

# TODO: add motion correction
# motion_correction_group = parser.add_mutually_exclusive_group()
# motion_correction_help = "How to deal with motion correction. Can be 'skip', 'compute', or 'apply'"
# motion_correction_group.add_argument("--motion", choices=["skip", "compute", "apply"], help=motion_correction_help)
# motion_correction_group.add_argument("static_motion", nargs="?", default="compute", help=motion_correction_help)

n_jobs_help = "Number of jobs to use for parallel processing. Default is -1 (all available cores). It can also be a float between 0 and 1 to use a fraction of available cores"
parser.add_argument("--n-jobs", default="-1", help=n_jobs_help)
parser.add_argument("--params-str", default=None, help="Optional json string with parameters")


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
    N_JOBS = int(args.n_jobs) if not args.n_jobs.startswith("0.") else float(args.n_jobs)
    PARAMS_STR = args.params_str

    # TODO: add motion correction
    # motion_arg = args.motion or args.static_motion
    # COMPUTE_MOTION = True if motion_arg != "skip" else False
    # APPLY_MOTION = True if motion_arg == "apply" else False

    print(f"Running preprocessing with the following parameters:")
    print(f"\tCONCATENATE: {CONCAT}")
    print(f"\tDENOISING_STRATEGY: {DENOISING_STRATEGY}")
    print(f"\tREMOVE_OUT_CHANNELS: {REMOVE_OUT_CHANNELS}")
    print(f"\tREMOVE_BAD_CHANNELS: {REMOVE_BAD_CHANNELS}")
    print(f"\tMAX BAD CHANNEL FRACTION: {MAX_BAD_CHANNEL_FRACTION}")
    print(f"\tN_JOBS: {N_JOBS}")
    # TODO: add motion correction
    # print(f"\tCOMPUTE_MOTION: {COMPUTE_MOTION}")
    # print(f"\tAPPLY_MOTION: {APPLY_MOTION}")

    if PARAMS_STR is not None:
        print(f"\nUsing custom params JSON string")
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
    # TODO: add motion correction
    # preprocessing_params["motion_correction"]["compute"] = COMPUTE_MOTION
    # preprocessing_params["motion_correction"]["apply"] = APPLY_MOTION

    # set paths
    data_folder = Path("../data")
    scratch_folder = Path("../scratch")
    results_folder = Path("../results")

    tmp_folder = results_folder / "tmp"
    if tmp_folder.is_dir():
        shutil.rmtree(tmp_folder)
    tmp_folder.mkdir()

    # SET DEFAULT JOB KWARGS
    si.set_global_job_kwargs(**job_kwargs)
    print(f"Global job kwargs: {si.get_global_job_kwargs()}")

    # MOVE this to top and check
    kachery_zone = os.getenv("KACHERY_ZONE", None)
    print(f"Kachery Zone: {kachery_zone}")

    ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower()]
    assert len(ecephys_sessions) == 1, f"Attach one session at a time {ecephys_sessions}"
    session = ecephys_sessions[0]
    session_name = session.name

    # propagate existing metadata files to results
    metadata_json_files = [
        p
        for p in session.iterdir()
        if p.is_file() and p.suffix == ".json" and "processing" not in p.name and "data_description" not in p.name
    ]
    for json_file in metadata_json_files:
        shutil.copy(json_file, results_folder)

    if (session / "processing.json").is_file():
        with open(session / "processing.json", "r") as processing_file:
            processing_dict = json.load(processing_file)
        # Allow for parsing earlier versions of Processing files
        processing = Processing.model_construct(**processing_dict)
    else:
        processing = None

    if (session / "data_description.json").is_file():
        with open(session / "data_description.json", "r") as data_description_file:
            data_description_dict = json.load(data_description_file)
        # Allow for parsing earlier versions of Processing files
        data_description = DataDescription.model_construct(**data_description_dict)
    else:
        data_description = None

    if (session / "subject.json").is_file():
        with open(session / "subject.json", "r") as subject_file:
            subject_info = json.load(subject_file)
        subject_id = subject_info["subject_id"]
    elif len(session_name.split("_")) > 1:
        subject_id = session_name.split("_")[1]
    else:
        subject_id = "000000"  # unknown

    ecephys_full_folder = session / "ecephys"
    ecephys_compressed_folder = session / "ecephys_compressed"
    compressed = False
    if ecephys_compressed_folder.is_dir():
        compressed = True
        ecephys_folder = session / "ecephys_clipped"
    else:
        ecephys_folder = ecephys_full_folder

    # get blocks/experiments and streams info
    num_blocks = se.get_neo_num_blocks("openephys", ecephys_folder)
    stream_names, stream_ids = se.get_neo_streams("openephys", ecephys_folder)

    # load first stream to map block_indices to experiment_names
    rec_test = se.read_openephys(ecephys_folder, block_index=0, stream_name=stream_names[0])
    record_node = list(rec_test.neo_reader.folder_structure.keys())[0]
    experiments = rec_test.neo_reader.folder_structure[record_node]["experiments"]
    exp_ids = list(experiments.keys())
    experiment_names = [experiments[exp_id]["name"] for exp_id in sorted(exp_ids)]

    print(f"Session: {session_name} - Num. Blocks {num_blocks} - Num. streams: {len(stream_names)}")

    ####### PREPROCESSING #######
    print("\n\nPREPROCESSING")
    preprocessed_tmp_folder = tmp_folder / "preprocessed"
    preprocessed_output_folder = results_folder / "preprocessed"
    preprocessed_output_folder.mkdir(exist_ok=True)

    datetime_start_preproc = datetime.now()
    t_preprocessing_start = time.perf_counter()

    recording_names = []
    preprocessing_notes = ""
    preprocessing_vizualization_data = {}
    for block_index in range(num_blocks):
        for stream_name in stream_names:
            # skip NIDAQ and NP1-LFP streams
            if "NI-DAQ" not in stream_name and "LFP" not in stream_name and "Rhythm" not in stream_name:
                experiment_name = experiment_names[block_index]
                exp_stream_name = f"{experiment_name}_{stream_name}"

                if not compressed:
                    recording = se.read_openephys(ecephys_folder, stream_name=stream_name, block_index=block_index)
                else:
                    recording = si.read_zarr(ecephys_compressed_folder / f"{exp_stream_name}.zarr")

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
                    skip_processing = False
                    if CONCAT:
                        recording_name = f"{exp_stream_name}_recording"
                    else:
                        recording_name = f"{exp_stream_name}_recording{i_r + 1}"

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

                    recording_hp_full = spre.highpass_filter(
                        recording_ps_full, **preprocessing_params["highpass_filter"]
                    )
                    preprocessing_vizualization_data[recording_name]["timeseries"]["full"].update(
                        dict(highpass=recording_hp_full)
                    )

                    if (
                        recording.get_total_duration() < preprocessing_params["min_preprocessing_duration"]
                        and not DEBUG
                    ):
                        print(
                            f"\tRecording is too short ({recording.get_total_duration()}s). Skipping further processing"
                        )
                        preprocessing_notes += f"\n- Recording is too short ({recording.get_total_duration()}s). Skipping further processing\n"
                        skip_processing = True
                    if not recording.has_channel_location():
                        print(f"\tRecording does not have channel locations. Skipping further processing")
                        preprocessing_notes += (
                            f"\n- Recording does not have channel locations. Skipping further processing\n"
                        )
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
                                preprocessing_notes += (
                                    f"{recording_name}:\n- Removed {len(out_channel_ids)} outside of the brain."
                                )
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
                                print(
                                    f"\tRemoving {len(bad_channel_ids)} channels after {denoising_strategy} preprocessing"
                                )
                                recording_processed = recording_processed.remove_channels(bad_channel_ids)
                                preprocessing_notes += (
                                    f"\n- Removed {len(bad_channel_ids)} bad channels after preprocessing.\n"
                                )
                            recording_saved = recording_processed.save(folder=preprocessed_tmp_folder / recording_name)
                            recording_processed.dump_to_json(
                                preprocessed_output_folder / f"{recording_name}.json", relative_to=data_folder
                            )
                            recording_drift = recording_saved

                    if skip_processing:
                        # in this case, processed timeseries will not be visualized
                        preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = None
                        recording_drift = recording_hp_full
                    # store recording for drift visualization
                    preprocessing_vizualization_data[recording_name]["drift"] = dict(recording=recording_drift)

    t_preprocessing_end = time.perf_counter()
    elapsed_time_preprocessing = np.round(t_preprocessing_end - t_preprocessing_start, 2)

    # save params in output
    preprocessing_process = DataProcess(
        name="Ephys preprocessing",
        software_version=PIPELINE_VERSION,  # either release or git commit
        start_date_time=datetime_start_preproc,
        end_date_time=datetime_start_preproc + timedelta(seconds=np.floor(elapsed_time_preprocessing)),
        input_location=str(data_folder),
        output_location=str(results_folder),
        code_url=PIPELINE_URL,
        parameters=preprocessing_params,
        notes=preprocessing_notes,
    )
    print(f"PREPROCESSING time: {elapsed_time_preprocessing}s")

    ####### SPIKESORTING ########
    print("\n\nSPIKE SORTING")
    spikesorting_notes = ""
    sorting_params = None

    datetime_start_sorting = datetime.now()
    t_sorting_start = time.perf_counter()
    preprocessed_folder = preprocessed_tmp_folder

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

    # save params in output
    spikesorting_process = DataProcess(
        name="Spike sorting",
        software_version=PIPELINE_VERSION,  # either release or git commit
        start_date_time=datetime_start_sorting,
        end_date_time=datetime_start_sorting + timedelta(seconds=np.floor(elapsed_time_sorting)),
        input_location=str(data_folder),
        output_location=str(results_folder),
        code_url=PIPELINE_URL,
        parameters=sorting_params,
        notes=spikesorting_notes,
    )
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
        qm = sqm.compute_quality_metrics(we, **postprocessing_params["quality_metrics"])

    t_postprocessing_end = time.perf_counter()
    elapsed_time_postprocessing = np.round(t_postprocessing_end - t_postprocessing_start, 2)

    # save params in output
    postprocessing_process = DataProcess(
        name="Ephys postprocessing",
        software_version=PIPELINE_VERSION,  # either release or git commit
        start_date_time=datetime_start_postprocessing,
        end_date_time=datetime_start_postprocessing + timedelta(seconds=np.floor(elapsed_time_postprocessing)),
        input_location=str(data_folder),
        output_location=str(results_folder),
        code_url=PIPELINE_URL,
        parameters=postprocessing_params,
        notes=postprocessing_notes,
    )
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
        print(f"\t{np.sum(qc_quality)}/{len(sorting_curated.unit_ids)} units passing default QC")

    t_curation_end = time.perf_counter()
    elapsed_time_curation = np.round(t_curation_end - t_curation_start, 2)

    # save params in output
    curation_process = DataProcess(
        name="Ephys curation",
        software_version=PIPELINE_VERSION,  # either release or git commit
        start_date_time=datetime_start_curation,
        end_date_time=datetime_start_curation + timedelta(seconds=np.floor(elapsed_time_curation)),
        input_location=str(data_folder),
        output_location=str(results_folder),
        code_url=PIPELINE_URL,
        parameters=curation_params,
        notes=curation_notes,
    )
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
                gh_path = f"{GH_CURATION_REPO}/{session_name}/{recording_name}/{sorter_name}/curation.json"
                state = dict(sortingCuration=gh_path)
                url = v_summary.url(
                    label=f"{session_name} - {recording_name} - {sorter_name} - Sorting Summary", state=state
                )
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

    visualization_process = DataProcess(
        name="Ephys visualization",
        software_version=PIPELINE_VERSION,  # either release or git commit
        start_date_time=datetime_start_visualization,
        end_date_time=datetime_start_visualization + timedelta(seconds=np.floor(elapsed_time_visualization)),
        input_location=str(data_folder),
        output_location=str(results_folder),
        code_url=PIPELINE_URL,
        parameters=visualization_params,
        notes=visualization_notes,
    )
    print(f"VISUALIZATION time: {elapsed_time_visualization}s")

    # construct processing.json
    ephys_data_processes = [
        preprocessing_process,
        spikesorting_process,
        postprocessing_process,
        curation_process,
        visualization_process,
    ]

    if (session / "processing.json").is_file():
        with open(session / "processing.json", "r") as processing_file:
            processing_dict = json.load(processing_file)
        # Allow for parsing earlier versions of Processing files
        processing_old = Processing.model_construct(**processing_dict)
        processing = ProcessingUpgrade(processing_old).upgrade(processor_full_name=PIPELINE_MAINTAINER)
        processing.processing_pipeline.data_processes.append(ephys_data_processes)
    else:
        processing_pipeline = PipelineProcess(
            data_processes=ephys_data_processes,
            processor_full_name=PIPELINE_MAINTAINER,
            pipeline_url=PIPELINE_URL,
            pipeline_version=PIPELINE_VERSION,
        )
        processing = Processing(processing_pipeline=processing_pipeline)

    # save processing files to output
    with (results_folder / "processing.json").open("w") as f:
        f.write(processing.model_dump_json(indent=3))

    process_name = "sorted"
    if data_description is not None:
        upgrader = DataDescriptionUpgrade(old_data_description_model=data_description)
        upgraded_data_description = upgrader.upgrade(platform=Platform.ECEPHYS)
        derived_data_description = DerivedDataDescription.from_data_description(
            upgraded_data_description, process_name=process_name
        )
    else:
        # make from scratch:
        data_description_dict = {}
        data_description_dict["creation_time"] = datetime.now()
        data_description_dict["name"] = session_name
        data_description_dict["institution"] = Institution.AIND
        data_description_dict["data_level"] = DataLevel.RAW
        data_description_dict["investigators"] = [""]
        data_description_dict["funding_source"] = [Funding(funder="AIND")]
        data_description_dict["modality"] = [Modality.ECEPHYS]
        data_description_dict["platform"] = Platform.ECEPHYS
        data_description_dict["subject_id"] = subject_id
        data_description = DataDescription(**data_description_dict)

        derived_data_description = DerivedDataDescription.from_data_description(
            data_description=data_description, process_name=process_name
        )

    # save processing files to output
    with (results_folder / "data_description.json").open("w") as f:
        f.write(derived_data_description.model_dump_json(indent=3))

    # remove tmp_folder
    shutil.rmtree(tmp_folder)

    t_global_end = time.perf_counter()
    elapsed_time_global = np.round(t_global_end - t_global_start, 2)
    print(f"\n\nFULL PIPELINE time:  {elapsed_time_global}s")
