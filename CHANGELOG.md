# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

## [0.2.0] — 2025-08-11

### Added

- Wrapper layer for coordinate conversions.

- `crosses_dateline`: function to check if coordinates cross the dateline.

### Changed

- `fill_great_circle`: unified input handling; unwrap on periodic axis.

### Fixed

- `_cart2sph` and `_cart2geo`: safe division when r=0.
