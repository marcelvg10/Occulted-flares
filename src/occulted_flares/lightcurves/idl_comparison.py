"""Read controlled actual OSPEX outputs without treating them as validated fits."""
import numpy as np
from astropy.time import Time
from scipy.io import readsav


def _text(value):
    return (value.decode() if isinstance(value, bytes) else str(value)).strip()


def load_controlled_idl(path, product, indices, bands):
    saved = readsav(path, python_dict=True)
    if _text(saved['source_file']) != product['path'].name:
        raise ValueError('Controlled IDL source file differs')
    np.testing.assert_array_equal(saved['detector_indices'], indices)
    np.testing.assert_array_equal(saved['pixel_indices'], np.arange(8))
    np.testing.assert_array_equal(saved['energy_edges'], [bands[0][0]]+[b[1] for b in bands])
    start = Time([_text(v) for v in saved['start_utc']], scale='utc')
    end = Time([_text(v) for v in saved['end_utc']], scale='utc')
    if len(start) == 0 or np.any(end <= start) or np.any(np.diff(start.jd) <= 0):
        raise ValueError('Invalid exported IDL edges')
    rate_struct, flux_struct = saved['rate_lc'][0], saved['flux_lc'][0]
    for s in [rate_struct, flux_struct]:
        if float(s['TIME_SHIFT']) != 0:
            raise ValueError('Controlled IDL time shift must be zero')
        np.testing.assert_allclose(s['DURATION'], (end-start).to_value('s'), rtol=0, atol=.0011)
        np.testing.assert_allclose(s['ENERGY_BANDS'], bands, rtol=0, atol=0)
    np.testing.assert_allclose(rate_struct['UT'], flux_struct['UT'], rtol=0, atol=1e-6)
    if _text(rate_struct['DATA_TYPE']) != 'rate' or _text(flux_struct['DATA_TYPE']) != 'flux':
        raise ValueError('Unexpected controlled IDL unit conventions')
    rate, flux = np.asarray(rate_struct['DATA'], float), np.asarray(flux_struct['DATA'], float)
    if rate.shape != (len(start), len(bands)) or flux.shape != rate.shape or not np.isfinite(rate).all() or not np.isfinite(flux).all():
        raise ValueError('Invalid controlled IDL array shape or nonfinite data')
    return dict(start=start, end=end, rate=rate, flux=flux, saved=saved,
        settings=[dict(setting='IDL source file', value=_text(saved['source_file'])),
            dict(setting='Loaded wrapper path', value=_text(saved['routine_file'][0]['PATH'])),
            dict(setting='Background', value=_text(saved['background'])),
            dict(setting='Time convention', value=_text(saved['time_convention'])),
            dict(setting='Requested minimum duration [s]', value=float(saved['minimum_duration_s'])),
            dict(setting='Returned IDL bins', value=len(start)),
            dict(setting='Rate math-status flags', value=int(saved['rate_math_flags'])),
            dict(setting='Flux math-status flags', value=int(saved['flux_math_flags'])),
            dict(setting='Uncertainty limitation', value=_text(saved['uncertainty_note']))])
