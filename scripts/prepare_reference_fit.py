"""Write protected one-integration net spectra for matched OSPEX controls.

Run explicitly once; default notebook execution never exports or launches IDL.
Input spectra keep signed rates, supplied errors and actual whole-bin times.
"""
from pathlib import Path
import json,sys
import numpy as np
from astropy.io import fits
from astropy.time import Time
WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'src'))
from occulted_flares.spectroscopy.reference import load_reference,ATOMIC_FILES
from occulted_flares.spectroscopy.preparation import file_sha256


def main():
    ctx=load_reference(WORK,WORK/'SOL2026-07-07');out=ctx['stage']/'outputs'
    metadata=out/'reference_inputs.json'
    if metadata.exists():
        old=json.loads(metadata.read_text(encoding='utf-8'))
        for name,h in old['products'].items():
            if file_sha256(out/name)!=h:raise ValueError('Reference input changed: '+name)
        print('Existing reference inputs verified; no overwrite.')
        return
    start,end=ctx['native_start'],ctx['native_end'];duration=(end-start).to_value('s')
    records={}
    for (name,policy),data in ctx['data'].items():
        destination=out/f'reference_{name}_{policy}.fits'
        if destination.exists():raise FileExistsError(destination)
        with fits.open(out/'cal20_prepared_spectrum.fits') as original:
            h=fits.HDUList([hdu.copy() for hdu in original])
            table=fits.BinTableHDU(data=original['RATE'].data[[ctx['source']['rows'][0]]].copy(),
                header=original['RATE'].header.copy(),name='RATE')
            base=Time(float(table.header['MJDREF'])+float(table.header['TIMEZERO']),format='mjd',scale='utc')
            table.data['TIME'][0]=((start-base).to_value('s')+duration/2)
            table.data['TIMEDEL'][0]=duration
            table.data['LIVETIME'][0]=data['live_s']/duration
            table.data['RATE'][0]=data['rate'];table.data['STAT_ERR'][0]=data['sigma']
            table.data['SYS_ERR'][0]=0 # selected policy already in STAT_ERR; no extra default
            h['RATE']=table
            for header in [h[0].header,table.header]:
                header['DATE-BEG']=start.isot;header['DATE-END']=end.isot
                header.add_history('Local one-integration reference fixture; source minus '+name)
                header.add_history('Supplied errors: '+policy+'; background mean variance added once.')
                header.add_history('Signed corrected rates/pseudo-counts; conditional chi2 only.')
                header.add_history('SYS_ERR=0 here: do not add systematic defaults to supplied errors.')
            h.writeto(destination,checksum=True)
        records[destination.name]=file_sha256(destination)
    db=WORK/'ssw/packages/xray/dbase/chianti'
    metadata.write_text(json.dumps(dict(date='2026-10-22',native_start_UTC=start.isot,
        native_end_UTC=end.isot,source_native_bins=len(ctx['source']['rows']),
        windows=ctx['windows'],no_additional_background_in_OSPEX=True,
        atomic_files={name:dict(path=str((db/file).relative_to(WORK)),sha256=file_sha256(db/file)) for name,file in ATOMIC_FILES.items()},
        products=records),indent=2)+'\n',encoding='utf-8')
    print('Wrote four one-row signed net fixtures, explicit supplied errors and provenance.')


if __name__=='__main__':main()
