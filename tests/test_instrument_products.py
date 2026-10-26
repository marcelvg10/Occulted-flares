"""Independent checks of unit conversion and conservative band/state selection."""
import unittest

from astropy.io import fits
from astropy.time import Time
import numpy as np
import pandas as pd

from occulted_flares.instrument.products import time_seconds, band_rate, state_summary


class InstrumentProductTests(unittest.TestCase):
    def test_scaled_unsigned_centiseconds(self):
        # Exercise actual FITS unsigned storage scaling, not just floating data.
        from io import BytesIO
        hdu = fits.BinTableHDU.from_columns([fits.Column(name="time", format="J",
            unit="cs", bzero=2147483648, array=np.array([0, 30, 1960], dtype=np.uint32))])
        stream = BytesIO()
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(stream)
        stream.seek(0)
        with fits.open(stream) as hdus:
            np.testing.assert_allclose(time_seconds(hdus[1], "time"), [0, .3, 19.6])

    def test_units_required_and_seconds_supported(self):
        for unit in (None, 's'):
            hdu = fits.BinTableHDU.from_columns([fits.Column(name='time',format='D',unit=unit,array=[3.])])
            if unit is None:
                with self.assertRaises(ValueError):
                    time_seconds(hdu,'time')
            else:
                np.testing.assert_allclose(time_seconds(hdu,'time'),[3.])

    def test_band_uses_counts_and_native_exposure(self):
        product=dict(energies=pd.DataFrame(dict(e_low=[4.,5.,6.],e_high=[5.,6.,7.])),
                     counts_by_energy=np.array([[2.,4.,100.],[20.,40.,100.]]),
                     duration_s=np.array([.3,3.]))
        np.testing.assert_allclose(band_rate(product,4,6),[20,20])
        with self.assertRaises(ValueError):
            band_rate(product,4.5,6)
        product['energies'].loc[1,'e_low']=5.5
        with self.assertRaises(ValueError):
            band_rate(product,4,6)

    def test_state_changes_detect_mask_swap_at_constant_population(self):
        product=dict(time=Time(['2026-07-07','2026-07-08','2026-07-09']),
                     states=dict(mask=np.array([[1,0],[0,1],[0,1]])))
        summary=state_summary(product,['mask'])
        self.assertEqual(summary.iloc[0].changes_between_samples,1)
        with self.assertRaises(ValueError):
            state_summary(product,['mask'],[False,False,False])


if __name__ == '__main__':
    unittest.main()
