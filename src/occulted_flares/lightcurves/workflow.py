"""Read explicit event light-curve settings using existing instrument modules."""
from pathlib import Path
import json
import numpy as np
from astropy.io import fits

from occulted_flares.instrument.products import read_product
from occulted_flares.instrument.detectors import load_reference, reduce_pixels, detector_groups
from occulted_flares.instrument.livetime import livetime_fraction, map_groups_to_detectors
from .baseline import group_profiles


def load_context(work, event, config='signal.json'):
    stage = Path(event)/'Lightcurves'
    settings = json.loads((stage/config).read_text(encoding='utf8'))
    instrument_path = (stage/settings['instrument_config']).resolve()
    instrument = json.loads(instrument_path.read_text(encoding='utf8'))
    cpd = read_product(instrument_path.parent/instrument['cpd'])
    ref = load_reference(Path(work)/instrument['instrument_reference'])
    bands = settings['bands_keV']
    reduced = reduce_pixels(cpd['path'], bands)
    groups = detector_groups(ref)
    indices = groups[settings['detector_group']]
    parameters = {key: ref['livetime'][key] for key in ['tau_s', 'eta_s', 'beta']}
    model = livetime_fraction(reduced['triggers'], reduced['trigger_sigma'], cpd['duration_s'], **parameters)
    live = map_groups_to_detectors(model['live_s'], ref)
    profiles = group_profiles(reduced['counts'], cpd['duration_s'], live, {settings['detector_group']: indices})
    gaps = (cpd['start'][1:]-cpd['end'][:-1]).to_value('s')
    if np.any(np.abs(gaps) > 1e-6):
        raise ValueError('This event overview expects continuous native coverage; handle gaps explicitly')
    return dict(settings=settings, stage=stage, instrument=instrument,
                instrument_stage=instrument_path.parent, cpd=cpd, reference=ref,
                bands=bands, reduced=reduced, groups=groups, indices=indices,
                live=live, model=model, profile=profiles[settings['detector_group']],
                edges=np.r_[cpd['start'].plot_date, cpd['end'][-1].plot_date])


def native_spectrogram(path, live_s, indices):
    """Complete finite nominal channels, summed live-rate per nominal keV.

    Output time × energy. This is neither area-normalized count flux nor
    response-corrected photon flux. Exclude the unbounded overflow channel.
    """
    with fits.open(path) as hdus:
        lo = np.asarray(hdus['ENERGIES'].data['e_low'], float)
        hi = np.asarray(hdus['ENERGIES'].data['e_high'], float)
    finite = np.isfinite(hi)
    bands = list(zip(lo[finite], hi[finite]))
    reduced = reduce_pixels(path, bands)
    if not np.array_equal(lo[finite][1:], hi[finite][:-1]):
        raise ValueError('Spectrogram channels are not contiguous')
    p = group_profiles(reduced['counts'], np.ones(len(live_s)), live_s, dict(chosen=indices))['chosen']
    return dict(rate_per_keV=p['live_rate']/(hi[finite]-lo[finite]),
                energy_edges=np.r_[lo[finite], hi[finite][-1]], omitted_overflow_channels=int((~finite).sum()))
