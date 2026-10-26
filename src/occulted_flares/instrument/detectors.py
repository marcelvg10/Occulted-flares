"""Band-reduced native pixel diagnostics; no detector is auto-rejected."""
import json
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd


def load_reference(path):
    reference = json.loads(Path(path).read_text())
    detectors = reference['detectors']
    if [d['detector_id'] for d in detectors] != list(range(1,33)):
        raise ValueError('Reference must list detectors 1–32 in order')
    groups = np.array([d['trigger_group'] for d in detectors])
    if not np.array_equal(np.bincount(groups,minlength=17)[1:],np.full(16,2)):
        raise ValueError('Each of 16 trigger groups must map to two detectors')
    return reference


def reduce_pixels(path, bands):
    """Read selected complete nominal bands, preserving detector/pixel axes.

    Current implementation requires all 32 detector columns and individually
    telemetered first eight large pixels, constant masks and RCR=0. Fail rather
    than silently infer a different compressed pixel-set layout.
    """
    with fits.open(path) as h:
        data=h['DATA'].data
        counts=data['counts']
        expected=np.r_[np.ones(8),np.zeros(4)]
        if (counts.shape[1:3] != (32,8) or not np.all(data['pixel_masks']==expected)
                or not np.all(data['detector_masks']==1) or not np.all(data['rcr']==0)
                or not np.all(data['num_pixel_sets']==8)):
            raise ValueError('Unsupported masks/pixel grouping/RCR: inspect and extend reader explicitly')
        energies=h['ENERGIES'].data
        lo,hi=energies['e_low'],energies['e_high']
        cb=np.empty((*counts.shape[:3],len(bands)))
        vb=np.empty_like(cb)
        compression_variance=np.empty_like(cb)
        for k,(low,high) in enumerate(bands):
            selected=(lo>=low)&(hi<=high)&np.isfinite(hi)
            if (not selected.any() or lo[selected][0]!=low or hi[selected][-1]!=high
                    or np.any(lo[selected][1:]!=hi[selected][:-1])):
                raise ValueError('Band must use complete contiguous nominal energy channels')
            c=np.asarray(counts[...,selected],float)
            err=np.asarray(data['counts_comp_err'][...,selected],float)
            if not np.isfinite(c).all() or np.any(c<0) or not np.isfinite(err).all() or np.any(err<0):
                raise ValueError('Non-finite/negative counts or compression errors')
            cb[...,k]=c.sum(axis=-1)
            vb[...,k]=(c+err**2).sum(axis=-1)
            compression_variance[...,k]=(err**2).sum(axis=-1)
        triggers=np.asarray(data['triggers'],float).copy()
        comp=np.asarray(data['triggers_comp_err'],float)
        trigger_sigma=np.sqrt(triggers+comp**2)
        if (triggers.shape!=(len(data),16) or not np.isfinite(triggers).all()
                or np.any(triggers<0) or not np.isfinite(trigger_sigma).all()):
            raise ValueError('Invalid trigger data')
    return dict(counts=cb,variance=vb,triggers=triggers,trigger_sigma=trigger_sigma,
                compression_variance=compression_variance,bands=bands,pixel_ids=list(range(8)))


def detector_groups(reference):
    """Zero-based indices; TOP24 is the installed GSW's baseline, not acceptance."""
    det=reference['detectors']
    imaging=[i for i,d in enumerate(det) if d['label'] not in ('bkg','cfl')]
    top24=[i for i in imaging if int(det[i]['label'][:-1])>=3]
    return dict(all32=list(range(32)),imaging30=imaging,top24=top24,
                fine6=[i for i in imaging if i not in top24],
                cfl=[i for i,d in enumerate(det) if d['label']=='cfl'],
                bkg=[i for i,d in enumerate(det) if d['label']=='bkg'])


def contribution_table(reduced,reference,selection,duration_s,band=0):
    """Exposure-weighted observed rates over complete selected integrations."""
    chosen=np.asarray(selection,bool)
    if not chosen.any():
        raise ValueError('No complete integrations selected')
    exposure=float(np.sum(np.asarray(duration_s)[chosen]))
    counts=reduced['counts'][chosen,:,:,band].sum(axis=(0,2))
    variance=reduced['variance'][chosen,:,:,band].sum(axis=(0,2))
    total=counts.sum()
    rows=[]
    for i,d in enumerate(reference['detectors']):
        rows.append(dict(detector_id=i+1,label=d['label'],trigger_group=d['trigger_group'],
            counts=counts[i],observed_rate_s=counts[i]/exposure,
            approximate_rate_sigma_s=np.sqrt(variance[i])/exposure,
            fraction_all32_counts=counts[i]/total if total else np.nan,
            all_pixels_zero=bool(counts[i]==0)))
    return pd.DataFrame(rows)
