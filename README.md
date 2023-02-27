# aind-capsule-ephys-spikesort-kilosort25-full

Electrophysiology analysis pipeline using Kilosort2.5 via SpikeInterface.

The pipeline includes:

- preprocessing: phase_shift, highpass filter, and common median reference ("cmr") or destriping
- spike sorting: with KS2.5
- postprocessing: remove duplicate units, compute amplitudes, spike/unit locations, PCA, correlograms, template similarity, template metrics, and quality metrics
- curation based on ISI violation ratio, presence ratio, and amplitude cutoff
- visualization of timeseries and sorting output in sortingview