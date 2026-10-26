# Occulted flares

Reusable Python tools for solar-flare geometry, STIX instrument diagnostics,
lightcurve analysis and spectroscopy with Solar Orbiter/STIX and STEREO/EUVI.

This is a code-only export. It includes source, command-line helpers, synthetic
tests, dependency files and static instrument/atomic-reference metadata.
It excludes observations, event selections, research notebooks, interpretations,
results, documents, credentials and third-party IDL/SolarSoft installations.
It does not reproduce a particular research event out of the box.

## Installation

Use Python 3.12 or 3.13 in a virtual environment:

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e .
python scripts/run_tests.py
```

All 109 synthetic tests passed for this export using the existing Python 3.12
environment. This checks calculations and storage behavior without observations;
it does not validate complete event analysis or external IDL execution.

For spectroscopy, install `python -m pip install -e ".[spectroscopy]"`.
The requirements lock files are historical environment snapshots; they do not
guarantee compatibility on every platform. A fresh dependency installation has
not been verified for this export.

## Layout

- `src/occulted_flares/geometry/`: coordinates, occultation, observer geometry and timing.
- `src/occulted_flares/instrument/`: products, detector selection, calibration and livetime.
- `src/occulted_flares/lightcurves/`: backgrounds, binning and uncertainty calculations.
- `src/occulted_flares/spectroscopy/`: spectral components, fitting and sensitivity.
- `scripts/` and `tools/`: command-line entry points and environment helpers.
- `tests/`: synthetic analytical and regression checks.
- `config/`: static instrument reference values and atomic-file checksums.

## Using your own observations

Import calculation functions from `occulted_flares`, or run a script with
`--help` to inspect its arguments. Some pipeline helpers expect an event directory
and generated input products; these are intentionally absent. Scripts retaining
the historical `SOL2026-07-07` default require your own corresponding inputs or
an explicitly supplied event path where supported.

Geometry commands with `--config` accept your own configuration file. For default
event lookup, create a local `config/default_event.json` containing
`{"geometry_config": "../local-event/Geometry/selection_groups.json"}`.
Keep your observations and event configurations outside this public repository.
The interactive `scripts/select_points.py` helper reads a local
`scripts/select_points.json` with `input_file` and `output_root` settings.

Spectroscopy requiring IDL exports needs a separately installed, licensed IDL
and SolarSoft/STIX environment. Those programs and exports are not distributed.

## Contributing

Keep algorithms separate from event choices. Add a synthetic regression check
for scientific calculation changes and run `python scripts/run_tests.py`.
Before committing, inspect `git diff --cached` and `git status --short`.
Publish only reviewed source and supporting developer files. Never add local
observations, generated results, secrets or research notes.

## License

Distributed under the [MIT License](LICENSE).
