# Release notes

## v2.0 - Nov 10, 2023

- update to `spikeinterface==0.99`
- update to `aind-data-schema==0.15.25`
- update to `wavpack-numcodecs==0.1.5`
- add `min_processing_duration` (default 120), to skip processing for short (test) recordings
- added `amplitude_median`, `amplitude_cv`, `firing_range`, `synchrony_metrics`
- include multi_channel template metrics (`velocity_above`, `velocity_below`, `spread`, `exp_decay`)
- change default correlograms params: 1ms bin, 50ms window
- add protection against recordings without channel locations
- dump preprocessed recording JSON to `preprocessed/` to easily reload processed object

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