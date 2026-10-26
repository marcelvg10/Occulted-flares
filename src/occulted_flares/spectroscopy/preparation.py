"""Read-only inventory for the spectroscopy workpackage.

No response generation, calibration application, IDL invocation or physical fit.
Source inspection describes installed code rather than proving runtime behavior.
"""
import hashlib
import json
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd

from occulted_flares.instrument.products import read_product


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_setup(work, event):
    work, event = Path(work).resolve(), Path(event).resolve()
    stage = event/'Spectroscopy'
    settings = json.loads((stage/'spectroscopy.json').read_text(encoding='utf-8'))
    instrument_path = (stage/settings['instrument_config']).resolve()
    instrument = json.loads(instrument_path.read_text(encoding='utf-8'))
    paths = {key:(instrument_path.parent/instrument[key]).resolve()
             for key in ('cpd', 'housekeeping', 'calibration')}
    if not all(p.is_relative_to(event) for p in paths.values()):
        raise ValueError('Science input escapes event directory')
    return dict(work=work, event=event, stage=stage, settings=settings,
                instrument=instrument, paths=paths)


def input_inventory(setup):
    manifest = json.loads((setup['event']/'data_manifest.json').read_text(encoding='utf-8'))
    recorded = {(setup['event']/r['path']).resolve():r for r in manifest['files']}
    rows = []
    for role, path in setup['paths'].items():
        record = recorded.get(path)
        if record is None:
            raise ValueError(f'Unrecorded scientific input: {path.name}')
        checksum = file_sha256(path)
        if checksum != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError(f'Input differs from manifest: {path.name}')
        with fits.open(path) as hdus:
            header = hdus[0].header
            rows.append(dict(role=role, file=path.name, level=header.get('LEVEL',''),
                start=header.get('DATE-BEG',''), end=header.get('DATE-END',''),
                manifest_match=True, sha256=checksum))
    return pd.DataFrame(rows)


def native_inventory(setup):
    """Inspect native timing and nominal channels, without correcting counts."""
    cpd = read_product(setup['paths']['cpd'])
    with fits.open(setup['paths']['cpd']) as hdus:
        energy = hdus['ENERGIES']
        columns = {c.name.lower():c.name for c in energy.columns}
        lo, hi = columns['e_low'], columns['e_high']
        channels = pd.DataFrame(dict(channel_index_0_based=np.arange(len(energy.data)),
            nominal_low_keV=np.asarray(energy.data[lo],float),
            nominal_high_keV=np.asarray(energy.data[hi],float)))
        channels['finite_bounds'] = np.isfinite(channels.nominal_low_keV)&np.isfinite(channels.nominal_high_keV)
        count_shape = tuple(hdus['DATA'].data['counts'].shape)
    summary = pd.DataFrame([dict(native_bins=len(cpd['duration_s']),
        first_native_edge_UTC=cpd['start'][0].isot, last_native_edge_UTC=cpd['end'][-1].isot,
        minimum_duration_s=float(cpd['duration_s'].min()),
        maximum_duration_s=float(cpd['duration_s'].max()),
        native_0_3_s_bins=int(np.isclose(cpd['duration_s'],.3,atol=1e-8,rtol=0).sum()),
        count_array_shape=str(count_shape))])
    return summary, channels


def routine_inventory(setup):
    rows=[]
    for name, relative in setup['settings']['local_routines'].items():
        path=(setup['work']/relative).resolve()
        if not path.is_relative_to(setup['work']):
            raise ValueError('Routine path escapes Work')
        present=path.is_file()
        rows.append(dict(routine=name, present=present, path=relative,
            sha256=file_sha256(path) if present else None))
    return pd.DataFrame(rows)


def inspect_source(setup, routine, fragments):
    """Return actual one-based source lines for explicit audit fragments."""
    path=setup['work']/setup['settings']['local_routines'][routine]
    rows=[]
    for line_no, line in enumerate(path.read_text(encoding='utf-8',errors='replace').splitlines(),1):
        if any(fragment in line for fragment in fragments):
            rows.append(dict(line=line_no, code=line.strip()))
    return pd.DataFrame(rows, columns=['line','code'])
