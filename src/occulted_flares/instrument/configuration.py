"""Configuration evidence at its native sampling; never interpolate HK flags.

An unchanged pair of bracketing samples is evidence, not a guarantee about
the interval between them. RCR and request masks describe different things:
an omitted small pixel is not evidence of an onboard RCR switch.
"""
import hashlib
from pathlib import Path

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import numpy as np
import pandas as pd

from .products import time_seconds


HK_FIELDS = ['instrument_mode', 'att_ab_flag-open', 'att_bc_flag-closed',
             'att_m1-moving', 'att_m2-moving', 'attenuator_motions',
             'hv01-16-enabled_status', 'hv17-32-enabled_status',
             'hv1_depolar_in_progress', 'hv2_depolar_in_progress']
SCIENCE_FIELDS = ['rcr', 'detector_masks', 'pixel_masks', 'num_pixel_sets']


def _matrix(product, fields):
    return np.concatenate([np.asarray(product['states'][name]).reshape(len(product['time']), -1)
                           for name in fields], axis=1).astype(str)


def configuration_segments(cpd):
    """Continuous CPD runs split at any mask/RCR change or non-touching edge."""
    matrix = _matrix(cpd, SCIENCE_FIELDS)
    discontinuity = np.abs((cpd['start'][1:] - cpd['end'][:-1]).to_value(u.s)) > 1e-6
    breaks = np.flatnonzero(np.any(matrix[1:] != matrix[:-1], axis=1) | discontinuity) + 1
    rows = []
    for first, stop in zip(np.r_[0, breaks], np.r_[breaks, len(matrix)]):
        rows.append(dict(first_bin=int(first), last_bin=int(stop-1), bins=int(stop-first),
            start_native_utc=cpd['start'][first].isot, end_native_utc=cpd['end'][stop-1].isot,
            rcr=int(cpd['states']['rcr'][first]),
            detector_entries_set=int(np.count_nonzero(cpd['states']['detector_masks'][first])),
            pixel_entries_set=int(np.count_nonzero(cpd['states']['pixel_masks'][first])),
            shortest_integration_s=float(np.min(cpd['duration_s'][first:stop]))))
    return pd.DataFrame(rows)


def state_changes(hk, fields=HK_FIELDS):
    """Observed changes are bracketed by two samples, not dated to an instant."""
    rows = []
    for name in fields:
        values = np.asarray(hk['states'][name])
        matrix = values.reshape(len(values), -1)
        for index in np.flatnonzero(np.any(matrix[1:] != matrix[:-1], axis=1)) + 1:
            rows.append(dict(field=name, before_utc=hk['time'][index-1].isot,
                after_utc=hk['time'][index].isot, before=str(values[index-1].tolist()),
                after=str(values[index].tolist())))
    return pd.DataFrame(rows, columns=['field','before_utc','after_utc','before','after'])


def bracket_housekeeping(cpd, hk, fields=HK_FIELDS):
    """Report evidence around each integration, without certifying its state.

    Use the last sample <= the start and the first sample >= the end. Record
    their ages, values' agreement, and how many HK samples fall inside. Missing
    boundary samples remain unavailable; do not clip indices or extrapolate.
    """
    epoch = hk['time'][0].tai
    t = (hk['time'].tai-epoch).to_value(u.s)
    a = (cpd['start'].tai-epoch).to_value(u.s)
    b = (cpd['end'].tai-epoch).to_value(u.s)
    if not np.all(np.diff(t) > 0) or np.any(b <= a):
        raise ValueError('Ordered samples and positive integration windows required')
    before = np.searchsorted(t, a, side='right')-1
    after = np.searchsorted(t, b, side='left')
    available = (before >= 0) & (after < len(t))
    agree = np.zeros(len(a), dtype=bool)
    prior_age = np.full(len(a), np.nan)
    next_age = np.full(len(a), np.nan)
    matrix = _matrix(hk, fields)
    agree[available] = np.all(matrix[before[available]] == matrix[after[available]], axis=1)
    prior_age[available] = a[available]-t[before[available]]
    next_age[available] = t[after[available]]-b[available]
    samples_inside = np.searchsorted(t,b,side='right')-np.searchsorted(t,a,side='left')
    return pd.DataFrame(dict(native_bin=np.arange(len(a)),
        start_native_utc=cpd['start'].isot, end_native_utc=cpd['end'].isot,
        bracketing_samples_available=available,
        bracketing_states_agree=agree,
        previous_sample_to_start_s=prior_age, end_to_next_sample_s=next_age,
        hk_samples_inside_integration=samples_inside))


def sampled_hk_assessment(hk):
    """Point checks of decoded HK fields; no voltage/temperature health limits."""
    s = hk['states']
    checks = {
        'mode_nominal': s['instrument_mode'] == 'NOMINAL',
        'attenuator_open_flags': s['att_ab_flag-open'] & ~s['att_bc_flag-closed'],
        'both_motors_still': (s['att_m1-moving']=='Still') & (s['att_m2-moving']=='Still'),
        'both_hv_groups_enabled': (s['hv01-16-enabled_status']=='Enabled') & (s['hv17-32-enabled_status']=='Enabled'),
        'no_hv_depolarization_flag': ~s['hv1_depolar_in_progress'] & ~s['hv2_depolar_in_progress'],
    }
    return pd.DataFrame(dict(sample_native_utc=hk['time'].isot, **checks))


def read_calibration(path):
    """Inspect processed CAL energy product, without applying its corrections."""
    path = Path(path).resolve()
    with fits.open(path) as hdus:
        header = hdus[0].header.copy()
        if header.get('LEVEL') != 'CAL' or header.get('INSTRUME') != 'STIX':
            raise ValueError('Expected processed STIX CAL product')
        if header.get('TIMESYS') != 'UTC':
            raise ValueError('Explicit UTC calibration time system required')
        data, control = hdus['DATA'], hdus['CONTROL']
        if len(data.data) != 1 or len(control.data) != 1:
            raise ValueError('Inspect multi-row calibration selection explicitly')
        offset = time_seconds(data,'time')[0]
        duration = time_seconds(data,'timedel')[0]
        live = time_seconds(data,'live_time')[0]
        edges = np.asarray(data.data['e_edges_actual'][0], dtype=float).copy()
        if edges.shape != (32,12,33):
            raise ValueError('Unsupported calibration dimensions')
        if not (0 < live <= duration):
            raise ValueError('Invalid calibration duration/livetime')
        start = Time(header['DATE-BEG'],scale='utc') + (offset-duration/2)*u.s
        end = Time(header['DATE-BEG'],scale='utc') + (offset+duration/2)*u.s
        result = dict(path=path, header=header, start=start, end=end, livetime_s=live,
            elut_name=str(control.data['ob_elut_name'][0]).strip(),
            ecc_config=str(control.data['ecc_config_name'][0]).strip(),
            gain_range_ok=bool(data.data['gain_range_ok'][0]),
            finite_internal_edges=bool(np.isfinite(edges[...,1:-1]).all()),
            strictly_ordered_edges=bool(np.all(np.diff(edges,axis=-1)>0)),
            energy_edges_keV=edges,
            columns=pd.DataFrame([dict(column=c.name,unit=c.unit or '',dimensions=c.dim or '')
                                  for c in data.columns]))
    with path.open('rb') as stream:
        result['sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
    return result


def elut_index_match(index_path, start, end):
    """Installed index evidence only; an open end does not prove currency.

    Require one indexed ELUT over the entire closed query interval. A change
    exactly at end fails conservatively, so an observation never straddles it.
    """
    path = Path(index_path)
    table = pd.read_csv(path,skipinitialspace=True,keep_default_na=False)
    table.columns = table.columns.str.strip()
    if end < start:
        raise ValueError('Reversed query interval')
    matches = []
    for _, row in table.iterrows():
        a = Time(row['start_date'].strip(),scale='utc')
        raw_end = row['end_date'].strip()
        b = None if raw_end.lower()=='none' else Time(raw_end,scale='utc')
        if start >= a and (b is None or end < b):
            matches.append(dict(elut_name=Path(row['elut_file'].strip()).stem,
                indexed_start_utc=a.isot,indexed_end_utc=None if b is None else b.isot))
    if len(matches) != 1:
        raise ValueError('No single unambiguous ELUT index interval covers the observation')
    with path.open('rb') as stream:
        matches[0]['index_sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
    return matches[0]
