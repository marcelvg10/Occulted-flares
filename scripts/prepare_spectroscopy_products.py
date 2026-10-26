"""Verify retained IDL bridge exports and write separately named repaired spectra.

Run once after both IDL drivers. Existing derived products are protected.
Notebooks never call this acquisition/preparation script automatically.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.io import readsav

WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'src'))
from occulted_flares.spectroscopy.bridge import read_products,compare_saved_reader
from occulted_flares.lightcurves.workflow import load_context
from occulted_flares.spectroscopy.channels import prepare_channels


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event',type=Path,default=WORK/'SOL2026-07-07')
    args=parser.parse_args()
    event=args.event.resolve();out=event/'Spectroscopy/outputs'
    metadata=out/'bridge_metadata.json'
    if metadata.exists():
        m=json.loads(metadata.read_text(encoding='utf-8'))
        for name,r in m['products'].items():
            if sha(out/name)!=r['sha256']:raise ValueError('Saved bridge product changed: '+name)
        print('Existing verified bridge products retained. No export or overwrite.')
        return
    names=['nominal','cal20','cal4','cal_onaxis']
    products={};checks=[]
    for name in names:
        products[name]=read_products(out/f'{name}_spectrum.fits',out/f'{name}_response.fits')
        checks.append(dict(case=name,**compare_saved_reader(products[name],out/f'{name}_reader.sav')))
    ctx=load_context(WORK,event)
    bands,prepared=prepare_channels(ctx)
    native_rate=(prepared['counts']/prepared['live_s'][...,None]).sum(axis=1)
    nominal=products['nominal']
    np.testing.assert_allclose(nominal['rate'],native_rate,rtol=2e-6,atol=1e-8)
    np.testing.assert_allclose(nominal['channels'],bands,rtol=0,atol=0)
    for p in products.values():
        for field in ['start','end']:
            if np.max(np.abs((p[field]-ctx['cpd'][field]).to_value('s')))>.0011:
                raise ValueError('Exported/native time edges differ')
    edge=readsav(out/'calibration_edge_rows.sav',python_dict=True)
    control,last=map(int,edge['native_indices'])
    if (last!=len(native_rate)-1 or control!=last-1
            or not np.allclose(edge['native_duration_s'],20,rtol=0,atol=1e-8)):
        raise ValueError('Edge adapter expects the two final complete 20 s integrations')
    np.testing.assert_allclose(edge['original_rates'],nominal['rate'][[control,last]],rtol=2e-6,atol=1e-8)
    if not np.all(edge['rates'][-1]>0):raise ValueError('Invalid independently calibrated terminal rates')
    repairs=[]
    for name in names[1:]:
        p=products[name]
        zero=np.flatnonzero(np.all(p['rate']==0,axis=1))
        np.testing.assert_array_equal(zero,[last])
        np.testing.assert_allclose(edge['rates'][0],p['rate'][control],rtol=2e-6,atol=1e-8)
        np.testing.assert_allclose(edge['errors'][0],p['error'][control],rtol=2e-6,atol=1e-8)
        destination=out/f'{name}_prepared_spectrum.fits'
        if destination.exists():raise FileExistsError('Prepared output protected: '+str(destination))
        original=out/f'{name}_spectrum.fits'
        with fits.open(original) as hdus:
            hdus['RATE'].data['RATE'][last]=edge['rates'][-1]
            hdus['RATE'].data['STAT_ERR'][last]=edge['errors'][-1]
            hdus[0].header.add_history('Local adapter: repair terminal converter CAL omission, row '+str(last))
            hdus[0].header.add_history('Original GSW export retained; calibration_edge_rows.sav supplies control.')
            hdus[0].header.add_history('Penultimate complete 20 s row verified at rtol=2e-6, atol=1e-8.')
            hdus.writeto(destination,checksum=True)
        fixed=read_products(destination,out/f'{name}_response.fits')
        np.testing.assert_array_equal(fixed['rate'][:-1],p['rate'][:-1])
        np.testing.assert_array_equal(fixed['error'][:-1],p['error'][:-1])
        np.testing.assert_allclose(fixed['rate'][last],edge['rates'][-1],rtol=2e-6,atol=1e-8)
        repairs.append(dict(case=name,row=last,prepared_file=destination.name,
            control_rate_max_absolute_difference=float(np.max(np.abs(edge['rates'][0]-p['rate'][control])))))
    context=readsav(out/'bridge_context.sav',python_dict=True)
    m=dict(date='2026-10-21',scope='Actual installed GSW preparation and neutral Python validation; no physical fit',
        reader_comparisons=checks,
        independent_native_rate_max_absolute_difference=float(np.max(np.abs(nominal['rate']-native_rate))),
        native_rows=len(native_rate),source_interval_UTC=[s.decode() for s in context['source_interval']],
        preliminary_source_hpc_arcsec=context['xy_hpc'].tolist(),
        preliminary_source_stix_arcsec=context['xy_stix'].tolist(),
        sidelobes_ratio=float(context['sidelobes_ratio']),distance_AU=float(context['distance']),
        unused_header_Earth_arrival_shift_s=float(context['header_earth_shift']),applied_time_shift_s=0,
        normalization='Standard OSPEX SRM: MATRIX times measured-channel width times GEOAREA. GEOAREA includes 1/distance_AU^2; incident model flux at 1 AU.',
        background_subtracted_in_exports=False,source_counts_kind='Live-normalized sums represented as pseudo-counts at mean detector exposure, not raw Poisson counts',
        converter_slope_context_s=dict(cal20=20,cal4=4),native_output_grid_preserved=True,
        terminal_CAL_omission_repair=repairs,SSW_sources_modified=False,
        systematic_file_fractions=[.07,.05,.03],OSPEX_spex_uncert=0,
        error_policy='Stored STAT_ERR retained and reader-checked. File SYS_ERR kept separate; no extra systematic added by neutral Python reader. Physical covariance/model and previous reader discrepancy remain open.',
        IDL_math_flags=context['math_flags'].tolist(),
        loader_audit='Current online sunkit-spex STIXLoader source clips negative input rates and adds 7/5/3% errors. Version-specific runtime behavior must be audited before fit. Not used/installed in this bridge.',
        products={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.suffix in ['.fits','.sav']})
    metadata.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
    print('Verified four IDL cases; preserved raw exports; prepared three terminal-repaired spectra.')


if __name__=='__main__':main()
