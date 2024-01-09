# Release notes

## v3.0 - Jan 10, 2024

- update to `aind-data-schema==0.22.2`
- change path to precurated output from the `sorting_precurated` to `curated` folder
- add ArgumentParser to be able to pass discrete command lines
- extended available CLI arguments to control removal of out/bad channels and max fraction of bad channels
- some renaming:
    - `preprocessing_strategy` -> `denoising_strategy`
    - `max_bad_channel_fraction_to_remove` - > `max_bad_channel_fraction`

### v2.1 - Dec 7, 2023

- fix visualization of drift map for short recordings using peak pipeline

## v2.0 - Nov 17, 2023

- update to `spikeinterface==0.99.1`
- update to `aind-data-schema==0.17.1`
- update to `wavpack-numcodecs==0.1.5`
- add `min_processing_duration` (default 120), to skip processing for short (test) recordings
- added `amplitude_median`, `amplitude_cv`, `firing_range`, `synchrony_metrics`, `nearest-neighbor`
- include multi_channel template metrics (`velocity_above`, `velocity_below`, `spread`, `exp_decay`)
- change default correlograms params: 1ms bin, 50ms window
- add protection against recordings without channel locations
- dump preprocessed recording JSON to `preprocessed/` to easily reload processed object
- use `UpgradeProcessing` to upgrade old processing schema models
- Add nearest neighbor (and fix parameters) and silhouette metrics

### v1.9 - Oct 20, 2023

- Update to `aind-data-schema==0.15.12`
- Update to `spikeinterface==0.98.2`
- Use new `Processing` schema

### v1.8 - Sep 25, 2023

- Update data description name from 'Spike Sorting' to 'sorted'

### v1.7 - Jul 27, 2023

- Upgrade to `wavpack-numcodecs==0.1.4`

### v1.6 - Jul 18, 2023

- Use `aind-data-schema.DataDescriptionUpgrade`
- Update to `spikeinterface==0.98.1`
  
### v1.5 - Apr 26, 2023

- Fix bug in drift map path

### v1.4 - Apr 12, 2023

- updates `Processing`` construction


### v1.3 - Mar 29, 2023

- Fix bug in `detect_peaks`


### v1.2 - Mar 28, 2023

- Fix `matplotlib.Normalize` for drift plot


### v1.1 - Mar 24, 2023

- clean up verbosity
- install fonts for sortingview


## v1.0 - Mar 14, 2023

- first release, `spikeinterface==0.97.1`