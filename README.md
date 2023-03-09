# Ephys processing pipeline Kilosort2.5

Electrophysiology analysis pipeline using [Kilosort2.5](https://github.com/MouseLand/Kilosort/tree/v2.5) via [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).

The pipeline includes:

- preprocessing: phase_shift, highpass filter, and 1. common median reference ("cmr") or 2. destriping (bad channel interpolation + highpass spatial filter - "destripe")
- spike sorting: with KS2.5
- postprocessing: remove duplicate units, compute amplitudes, spike/unit locations, PCA, correlograms, template similarity, template metrics, and quality metrics
- curation based on ISI violation ratio, presence ratio, and amplitude cutoff
- visualization of timeseries, drift maps, and sorting output in sortingview

## How to run it?

The `environment` folder contains a `Dockerfile` to build the container with all required packages.

The `code` folder contains the scripts to run the analysis (`run_capsule.py`). 

The script assumes that the data in the `data` folder is organized as follows:

- there is only one "session" folder
- the "session" folder contains either:
  - the `ecephys` folder, with a valid `Open Ephys` folder
  - the `ecephys_compressed` and the `ecephys_clipped` folders, created by the [openephys_job](https://github.com/AllenNeuralDynamics/aind-data-transfer/blob/main/src/aind_data_transfer/jobs/openephys_job.py) script in the [aind-data-transfer](https://github.com/AllenNeuralDynamics/aind-data-transfer) repo.

**Note** that the data loading section (L236-270) can be easily customized to use other data formats compatible with SpikeInterface.

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
