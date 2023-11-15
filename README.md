# Ephys processing pipeline Kilosort2.5

Electrophysiology analysis pipeline using [Kilosort2.5](https://github.com/MouseLand/Kilosort/tree/v2.5) via [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).

The pipeline includes:

- preprocessing: phase_shift, highpass filter, and 1. common median reference ("cmr") or 2. destriping (bad channel interpolation + highpass spatial filter - "destripe")
- spike sorting: with KS2.5
- postprocessing: remove duplicate units, compute amplitudes, spike/unit locations, PCA, correlograms, template similarity, template metrics, and quality metrics
- curation based on ISI violation ratio, presence ratio, and amplitude cutoff
- visualization of timeseries, drift maps, and sorting output in sortingview


## How to run locally

This pipeline is currently used at AIND on the Code Ocean platform. 

The `main` branch includes includes scripts and resources to run the pipeline locally.
In particular, the `code/run_capsule_spikeglx.py` is designed to run on SpikeGLX datasets.
The `code/run_capsule_nwb.py` is designed to run on an NWB file.

First, let's clone the repo:

```bash
git clone https://github.com/AllenNeuralDynamics/aind-capsule-ephys-spikesort-kilosort25-full
cd aind-capsule-ephys-spikesort-kilosort25-full
```

Next, first we need to build the docker image:

```bash
cd environment
docker build -t ephys-pipeline-container:latest .
cd ..
```

Next, we need to move the dataset to analyze in the `data` folder. 
For example, we can download an NWB file from [DANDI](https://dandiarchive.org/) (e.g. [this dataset](https://dandiarchive.org/dandiset/000028/draft/files?location=sub-mouse412804)) and 
move it to the `data` folder:

```bash
mkdir data
mv path-to-download-folder/sub-mouse412804_ses-20200803T115732_ecephys.nwb data
```

Finally, we can start the container:
```bash
chmod +x ./code/run_nwb
docker run -it --gpus all -v .:/capsule ephys-pipeline-container:latest 
```

and run the pipeline:
```bash
cd /capsule/code
./run_nwb
```


## How to run it on Code Ocean?

Use the `aind` branch for a Code Ocean-ready version.

The `environment` folder contains a `Dockerfile` to build the container with all required packages.

The `code` folder contains the scripts to run the analysis (`run_capsule.py`). 

The script assumes that the data in the `data` folder is organized as follows:

- there is only one "session" folder
- the "session" folder contains either:
  - the `ecephys` folder, with a valid `Open Ephys` folder
  - the `ecephys_compressed` and the `ecephys_clipped` folders, created by the [openephys_job](https://github.com/AllenNeuralDynamics/aind-data-transfer/blob/main/src/aind_data_transfer/jobs/openephys_job.py) script in the [aind-data-transfer](https://github.com/AllenNeuralDynamics/aind-data-transfer) repo.

For instructions for local deployment, refer to the **Local Deployment** section at the end of the page.


## Input parameters

The `run_capsule.py` script optionally accepts 4 arguments:

1. "preprocessing_strategy": `cmr` (default) or `destripe`
2. "debug": `false` or `true`
3. "debug duration s": number of seconds to use in debug mode
4. "concatenate recordings": `false` or `true`. If `false`, different segments from the same recordings are spike sorted separately. If `true`, they are concatenated and spike sorted together.

For example, one could run:
```
python run_capsule.py destripe true 60 false
```

## Results organization

The script produces the following output files in the `results` folder:

- `drift_maps`: raster maps for each *stream*
- `postprocessed`: postprocessing output for each stream with waveforms, correlograms, isi histograms, principal components, quality metrics, similarity, spike amplitudes, spike and unit locations and template metrics. Each folder can be loaded with: `we = si.load_waveforms("postprocessed/{stream_name}", with_recording=False)`
- `spikesorted`: *raw* spike sorting output from KS2.5 for each stream. Each sorting output can be loaded with: `sorting_raw = si.load_extractor("spikesorted/{stream_name}")`
- `sorting_precurated`: *pre-curated* spike sorting output, with an additional `default_qc` property (`True`/`False`) for each unit. Each pre-curated sorting output can be loaded with: `sorting_raw = si.load_extractor("sorting_precurated/{stream_name}")`
- `processing.json`: the processing parameter following the [aind-data-schema](https://github.com/AllenNeuralDynamics/aind-data-schema) metadata schema.
- `visualization_output.json`: convenient file with [FigURL](https://github.com/flatironinstitute/figurl) links for cloud visualization

## Notes on visualization

The processing pipeline assumes that [FigURL](https://github.com/flatironinstitute/figurl) is correctly set up. 
If you are planning to use this pipeline extensively, please consider providing your own cloud resources (see [Create Kachery Zone](https://github.com/flatironinstitute/kachery-cloud/blob/main/doc/create_kachery_zone.md))



## List of changes between `main` (local deployment) and `aind` (Code Ocean) branches

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


