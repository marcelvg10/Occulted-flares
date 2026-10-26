"""Neutral reader for GSW OSPEX diagnostics; preserve signs and stored errors.

OSPEX MATRIX is per measured keV. Multiply by channel width and GEOAREA
exactly once to obtain the rate response to integrated incident photon bins.
No fitting-package defaults, clipping or systematic terms are added here.
"""
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.time import Time,TimeDelta
from scipy.io import readsav


def effective_response(matrix,channel_edges,geoarea):
    matrix=np.asarray(matrix,float)
    edges=np.asarray(channel_edges,float)
    if (matrix.ndim!=2 or edges.shape!=(matrix.shape[1],2)
            or not np.isfinite(matrix).all() or np.any(matrix<0)
            or not np.isfinite(edges).all() or np.any(edges[:,1]<=edges[:,0])
            or not np.isfinite(geoarea) or geoarea<=0):
        raise ValueError('Finite nonnegative photon-by-channel matrix, valid channels and positive area required')
    return matrix*np.diff(edges,axis=1).ravel()[None,:]*geoarea


def read_products(spectrum_file,response_file):
    with fits.open(spectrum_file) as h:
        d=h['RATE'].data
        header=h['RATE'].header.copy()
        if h['RATE'].columns['RATE'].unit.strip().lower() not in ['counts/s','count/s','counts s-1']:
            raise ValueError('Explicit counts/s expected; do not silently convert count flux')
        rate=np.array(d['RATE'],float)
        error=np.array(d['STAT_ERR'],float)
        fraction=np.array(d['LIVETIME'],float)
        duration=np.array(d['TIMEDEL'],float)
        relative_time=np.array(d['TIME'],float)
        channels=np.column_stack([h['ENEBAND'].data['E_MIN'],h['ENEBAND'].data['E_MAX']]).astype(float)
        systematic=np.array(d['SYS_ERR'],float)
        base=Time(float(header['MJDREF'])+float(header['TIMEZERO']),format='mjd',scale='utc')
        center=base+TimeDelta(relative_time,format='sec')
        start=center-TimeDelta(duration/2,format='sec')
        end=center+TimeDelta(duration/2,format='sec')
    with fits.open(response_file) as h:
        r=h['SPECRESP MATRIX'].data
        rh=h['SPECRESP MATRIX'].header.copy()
        ec=np.column_stack([h['EBOUNDS'].data['E_MIN'],h['EBOUNDS'].data['E_MAX']]).astype(float)
        if (not np.array_equal(channels,ec) or not np.all(r['N_GRP']==1)
                or not np.all(r['F_CHAN']==0) or not np.all(r['N_CHAN']==len(channels))):
            raise ValueError('Expected matched channels and one full zero-based response group')
        photon_edges=np.column_stack([r['ENERG_LO'],r['ENERG_HI']]).astype(float)
        matrix=np.array(r['MATRIX'],float)
        area=float(rh['GEOAREA'])
    if (rate.shape!=error.shape or rate.shape!=(len(duration),len(channels))
            or fraction.shape!=duration.shape or not np.isfinite(rate).all()
            or not np.isfinite(error).all() or np.any(error<0)
            or not np.isfinite(duration).all() or np.any(duration<=0)
            or not np.isfinite(fraction).all() or np.any(fraction<=0) or np.any(fraction>1)):
        raise ValueError('Invalid spectrum/exposure/error dimensions or values')
    response=effective_response(matrix,channels,area)
    return dict(rate=rate,error=error,start=start,end=end,duration_s=duration,
        fraction=fraction,live_s=duration*fraction,systematic=systematic,channels=channels,
        photon_edges=photon_edges,matrix=matrix,response=response,geoarea=area,
        header=header,response_header=rh,spectrum_file=Path(spectrum_file),response_file=Path(response_file))


def integrate_rows(product,indices):
    """Sum exported pseudo-counts, then divide by summed mean live exposure.

    A GSW summed spectrum uses mean detector exposure. These pseudo-counts
    differ from raw detector counts and are not an independent Poisson sample.
    Errors are the stored reader errors, with unresolved propagation limits.
    """
    ids=np.asarray(indices)
    if (ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or not len(ids)
            or np.any(np.diff(ids)!=1) or np.any(ids<0) or np.any(ids>=len(product['rate']))):
        raise ValueError('Nonempty consecutive valid row indices required')
    live=product['live_s'][ids]
    exposure=live.sum()
    counts=(product['rate'][ids]*live[:,None]).sum(axis=0)
    variance=((product['error'][ids]*live[:,None])**2).sum(axis=0)
    return dict(rate=counts/exposure,variance=variance/exposure**2,
        pseudo_counts=counts,live_s=exposure,rows=ids)


def subtract_candidate(source,background):
    if np.intersect1d(source['rows'],background['rows']).size:
        raise ValueError('Source and background overlap; shared covariance required')
    return dict(rate=source['rate']-background['rate'],
        sigma=np.sqrt(source['variance']+background['variance']))


def compare_saved_reader(product,reader_file):
    r=readsav(reader_file,python_dict=True)
    # Saved IDL arrays have energy as first dimension; scipy reverses it.
    for name,target in [('rates',product['rate']),('errors',product['error'])]:
        data=np.asarray(r[name],float)
        if data.shape!=target.shape:raise ValueError('Unexpected saved reader orientation')
        np.testing.assert_allclose(data,target,rtol=2e-6,atol=1e-8)
    phi=np.asarray(r['photon_integrals'],float)
    folded=phi@product['response']
    np.testing.assert_allclose(folded,r['folded'],rtol=2e-6,atol=1e-8)
    return dict(reader_rate_max_absolute_difference=float(np.max(np.abs(r['rates']-product['rate']))),
        reader_error_max_absolute_difference=float(np.max(np.abs(r['errors']-product['error']))),
        folded_max_absolute_difference=float(np.max(np.abs(folded-r['folded']))))
