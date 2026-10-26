"""Explicit spectral sensitivity hypotheses, not a calibrated error budget."""
from copy import copy
import numpy as np
from .bridge import read_products,integrate_rows
from .reference import fit_data


def case_context(context,case):
    """Use already checked actual GSW case products; no synthetic response."""
    stage=context['stage'];answer=copy(context)
    name='nominal_spectrum.fits' if case=='nominal' else f'{case}_prepared_spectrum.fits'
    product=read_products(stage/'outputs'/name,stage/'outputs'/f'{case}_response.fits')
    for edge in ('start','end'):
        if len(product[edge])!=len(context['product'][edge]) or np.max(np.abs((product[edge]-context['product'][edge]).sec))>1e-5:
            raise ValueError('Cases must preserve identical native edges')
    if not np.array_equal(product['channels'],context['product']['channels']):raise ValueError('Case channels differ')
    answer['product']=product
    answer['backgrounds']={k:integrate_rows(product,b['rows']) for k,b in context['backgrounds'].items()}
    return answer


def integrate_variant(context,item,rows):
    """Reintegrate a consecutive source variant with the same quiet means."""
    source=integrate_rows(context['product'],np.asarray(rows,dtype=int))
    result=copy(item);result['rows']=source['rows'];result['source']=source
    result['start']=context['cpd']['start'][rows[0]];result['end']=context['cpd']['end'][rows[-1]]
    result['data']={(bg,policy):fit_data(context['product'],source,b,policy)
        for bg,b in context['backgrounds'].items() for policy in ['stored_only','file_sys_diagonal']}
    return result


def shifted_rows(item,shift,nrows):
    """Translate both interval edges by an integer number of native rows.

    Actual elapsed shifts depend on cadence; callers must show actual edges.
    This changes the selected observations, not any instrument timestamp.
    """
    if not isinstance(shift,(int,np.integer)):raise ValueError('Integer row shift required')
    ids=np.asarray(item['rows'],int)+shift
    if np.any(ids<0) or np.any(ids>=nrows):raise ValueError('Shift extends outside observations')
    return ids


def systematic_covariance(stored_data,systematic_vector,rho):
    """Same marginal SYS errors, varying assumed cross-channel coherence.

    C=diag(stored_sigma^2+(1-rho)*s^2)+rho*s*s.T. rho is an
    illustrative assumption, not an estimated detector/calibration correlation.
    Background errors inside stored_sigma remain channel-independent here.
    """
    y=np.asarray(stored_data['rate'],float);s=np.asarray(systematic_vector,float)
    sigma=np.asarray(stored_data['sigma'],float)
    if s.shape!=y.shape or sigma.shape!=y.shape or not np.isfinite(s).all() or np.any(s<0) or not 0<=rho<=1:
        raise ValueError('Matched nonnegative systematic vector and 0<=rho<=1 required')
    covariance=np.diag(sigma**2+(1-rho)*s**2)+rho*np.outer(s,s)
    return {**stored_data,'sigma':np.sqrt(np.diag(covariance)),'covariance':covariance}


def shared_background_covariance(source_variances,background_variance):
    """Phase covariance for net BAND RATES sharing one background mean.

    Assumes independent stored channel errors within each integrated band;
    not a covariance of nonlinear fitted parameters or a complete noise model.
    """
    v=np.asarray(source_variances,float);b=float(background_variance)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all() or np.any(v<0) or not np.isfinite(b) or b<0:
        raise ValueError('Nonnegative finite variance inputs required')
    return np.diag(v)+b*np.ones((len(v),len(v)))
