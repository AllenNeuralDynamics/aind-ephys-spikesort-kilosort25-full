# Ephys processing pipeline Kilosort2.5

Electrophysiology analysis pipeline using [Kilosort2.5](https://github.com/MouseLand/Kilosort/tree/v2.5) via [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).

The pipeline includes:

- preprocessing: phase_shift, highpass filter, and 1. common median reference ("cmr") or 2. destriping (bad channel interpolation + highpass spatial filter - "destripe")
- spike sorting: with KS2.5
- postprocessing: remove duplicate units, compute amplitudes, spike/unit locations, PCA, correlograms, template similarity, template metrics, and quality metrics
- curation based on ISI violation ratio, presence ratio, and amplitude cutoff
- visualization of timeseries, drift maps, and sorting output in sortingview

# Usage

## Input parameters

The `run_capsule_*.py` scripts in the `code` folder accept positional or optional arguments.

When using positional argument, up to 7 arguments can be passed, in this STRICT order:

1. "debug": Whether to run in DEBUG mode (`false` or `true`, default `false`)
2. "concatenate": Whether to concatenate recordings/segments (`false` or `true`. default `false`)
3. "denoising strategy": Which denoising strategy to use. Can be `cmr` (default) or `destripe`
4. "remove out channels": Whether to remove out channels (`false` or `true`, default `true`)
5. "remove bad channels": Whether to remove bad channels (`false` or `true`, default `true`)
6. "max bad channel fraction": Maximum fraction of bad channels to remove. If more than this fraction, processing is skipped (default 0.5)
7. "debug duration": Duration of clipped recording in debug mode. Default is 30 seconds. Only used if debug is enabled


The scripts also support the same options as follows:

- `--debug`: Whether to run in DEBUG mode. Default: False
- `--concatenate`: Whether to concatenate recordings (segments) or not. Default: False
- `--denoising` {cmr,destripe}: Which denoising strategy to use. Can be 'cmr' or 'destripe'
- `--no-remove-out-channels`: Whether to remove out channels
- `--no-remove-bad-channels`: Whether to remove bad channels
- `--max-bad-channel-fraction`: Maximum fraction of bad channels to remove. If more than this fraction, processing is skipped
- `--debug-duration`: Duration of clipped recording in debug mode. Default is 30 seconds. Only used if debug is enabled

In addition, the scripts accept the following configuration parameters:
- `--data-folder`: option to modify the path of the data (by default `../data`) 
- `--results-folder`: option to modify the path of the results (by default `../results`) 
- `--scratch-folder`: option to modify the path of the scratch (by default `../scratch`), used to store temporary files
- `--n-jobs`: parameter to control the maximum number of jobs used for parallelization.
- `--params-file`: path to a JSON file to specify parameters
- `--params-str`: JSON-formatted string with custom parameters

The NWB script also accepts the following parameter:
- `--electrical-series-path`: path to the electrical series to process, e.g. `acquisision/ElectricalSeriesAP`

This parameter is required if multiple electrical series are avaialable in the NWB file (otherwise an error is thrown
with the available options).

> **_NOTES ON PARAMETERS:_** In case `--params-file`/`--params-str` are not specified, default parameters are used 
(see `code/processing_params.json` file).

For example, one could run:
```bash
python run_capsule_*.py true false destripe true false 0.8 30
```
Or:
```bash
python run_capsule_*.py --debug --denoising destripe --no-remove-bad-channels \
                      --max-bad-channel-fraction 0.8 --debug-duration 30
```

## Results organization

The script produces the following output files in the `results` folder:

- `drift_maps`: raster maps for each *stream*
- `postprocessed`: postprocessing output for each stream with waveforms, correlograms, isi histograms, principal components, quality metrics, similarity, spike amplitudes, spike and unit locations and template metrics. Each folder can be loaded with: `we = si.load_waveforms("postprocessed/{stream_name}", with_recording=False)`
- `spikesorted`: *raw* spike sorting output from KS2.5 for each stream. Each sorting output can be loaded with: `sorting_raw = si.load_extractor("spikesorted/{stream_name}")`
- `curated`: *pre-curated* spike sorting output, with an additional `default_qc` property (`True`/`False`) for each unit. Each pre-curated sorting output can be loaded with: `sorting_raw = si.load_extractor("curated/{stream_name}")`
- `processing_params.json`: the processing parameter following the [aind-data-schema](https://github.com/AllenNeuralDynamics/aind-data-schema) metadata schema.
- `visualization_output.json`: convenient file with [FigURL](https://github.com/flatironinstitute/figurl) links for cloud visualization

## Notes on visualization

The processing pipeline assumes that [FigURL](https://github.com/flatironinstitute/figurl) is correctly set up.
If you are planning to use this pipeline extensively, please consider providing your own cloud resources (see [Create Kachery Zone](https://github.com/flatironinstitute/kachery-cloud/blob/main/doc/create_kachery_zone.md))


# Local deployment

This pipeline is currently used at AIND on the Code Ocean platform. 

The `main` branch includes includes scripts and resources to run the pipeline locally.
In particular, the `code/run_capsule_spikeglx.py` is designed to run on SpikeGLX datasets.
The `code/run_capsule_nwb.py` is designed to run on an NWB file.

First, let's clone the repo:

```bash
git clone https://github.com/AllenNeuralDynamics/aind-capsule-ephys-spikesort-kilosort25-full
cd aind-capsule-ephys-spikesort-kilosort25-full
```


Next, we need to move the dataset to analyze in the `data` folder. 
For example, we can download an NWB file from [DANDI](https://dandiarchive.org/) (e.g. [this dataset](https://dandiarchive.org/dandiset/000028/draft/files?location=sub-mouse412804)) and 
move it to the `data` folder:

```bash
mkdir data
mv path-to-download-folder/sub-mouse412804_ses-20200803T115732_ecephys.nwb data
```

Finally, we can start the container (`ghcr.io/allenneuraldynamics/aind-ephys-spikesort-kilosort25-full:latest`) 
from the repo base folder (`aind-ephys-spikesort-kilosort25-full`):
```bash
chmod +x ./code/run_nwb
docker run -it --gpus all -v .:/capsule --shm-size 8G \
    --env KACHERY_ZONE --env KACHERY_CLOUD_CLIENT_ID --env KACHERY_CLOUD_PRIVATE_KEY \
    ghcr.io/allenneuraldynamics/aind-ephys-spikesort-kilosort25-full:latest
```

and run the pipeline:
```bash
cd /capsule/code
./run_nwb # + optional parameters (e.g., --debug)
```

> **_NOTES ON DOCKER RUN:_**  
> The `--gpu all` flag is required to make the GPU available to the container (and Kilosort).  
> The `--shm-size 8G` flag is required to increase the shared memory size (default is 64M), which is used internally for parallel processing.  
> The `-v .:/capsule` option mounts the current folder `.` to the `/capsule` folder in the container, so that the data and scripts are available.  
> **THE FOLDER IS NOT MOUNTED IN READ-ONLY MODE, so be careful when deleting files in the container.**  
> The `--env KACHERY_ZONE --env KACHERY_CLOUD_CLIENT_ID --env KACHERY_CLOUD_PRIVATE_KEY` flags are required to set up the cloud visualization with FigURL (see [Notes on visualization](#notes-on-visualization) for more details)  


# Code Ocean deployment

Use the `aind` branch for a Code Ocean-ready version.

The `environment` folder contains a `Dockerfile` to build the container with all required packages.

The `code` folder contains the scripts to run the analysis (`run_capsule_aind.py`).

The script assumes that the data in the `data` folder is organized as follows:

- there is only one "session" folder
- the "session" folder contains either:
  - the `ecephys` folder, with a valid `Open Ephys` folder
  - the `ecephys_compressed` and the `ecephys_clipped` folders, created by the [openephys_job](https://github.com/AllenNeuralDynamics/aind-data-transfer/blob/main/src/aind_data_transfer/jobs/openephys_job.py) script in the [aind-data-transfer](https://github.com/AllenNeuralDynamics/aind-data-transfer) repo.

For instructions for local deployment, refer to the **Local Deployment** section at the end of the page.



# Differences between `main` (local) and `aind` (Code Ocean) branches

Here is a list of the key changes that are needed:

### 1. Base Docker image

Code Ocean uses an [internal registry of base Docker](https://github.com/AllenNeuralDynamics/aind-capsule-ephys-spikesort-kilosort25-full/blob/84eab15e52d2ae24d2035b97e42d593c6cbfac52/environment/Dockerfile#L2) images. To use the same pipeline locally, 
the base Docker image in the `environment/Dockerfile` of the `aind` branch is changed to:

```Dockerfile
FROM registry.codeocean.allenneuraldynamics.org/codeocean/kilosort2_5-compiled-base:latest
```

### 2. Reading of the data

The first part of the `code/run_capsule.py` script is dealing with loading the data. 
This part is clearly tailored to the way we store the data at AIND (see [this section](https://github.com/AllenNeuralDynamics/aind-capsule-ephys-spikesort-kilosort25-full/blob/84eab15e52d2ae24d2035b97e42d593c6cbfac52/code/run_capsule.py#L240-L286)).
In the `main` branch, we included two extra `run_capsule_*` scripts, one for SpikeGLX (`run_capsule_spikeglx`) and one for NWB files (`run_capsule_nwb`).

In both cases, we assume that the data folder includes a single dataset (either a SpikeGLX generated folder or 
a single NWB file).

### 3. Metadata handling

At AIND, we use [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) to deal with metadata. 
The scripts in the `main` do not have metadata logging using the `aind-data-schema`.
