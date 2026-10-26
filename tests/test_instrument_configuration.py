"""Boundary tests: do not turn sampled state evidence into continuous truth."""
from pathlib import Path
import tempfile
import unittest

from astropy.time import Time
import astropy.units as u
import numpy as np

from occulted_flares.instrument.configuration import (
    bracket_housekeeping, configuration_segments, elut_index_match, state_changes)


class ConfigurationTests(unittest.TestCase):
    def test_missing_brackets_and_intermediate_change(self):
        epoch=Time('2026-07-07')
        hk=dict(time=epoch+np.array([0,10,20])*u.s,states=dict(flag=np.array([1,0,1])))
        cpd=dict(start=epoch+np.array([-1,1,1,21])*u.s,
                 end=epoch+np.array([1,9,19,22])*u.s)
        table=bracket_housekeeping(cpd,hk,['flag'])
        self.assertEqual(table.bracketing_samples_available.tolist(),[False,True,True,False])
        self.assertFalse(table.iloc[1].bracketing_states_agree)
        # Equal outer values do not erase the observed intermediate change.
        self.assertTrue(table.iloc[2].bracketing_states_agree)
        self.assertEqual(table.iloc[2].hk_samples_inside_integration,1)
        self.assertEqual(len(state_changes(hk,['flag'])),2)

    def test_segments_detect_mask_swap_and_gap(self):
        epoch=Time('2026-07-07')
        p=dict(time=epoch+np.arange(4)*u.s, start=epoch+np.array([0,1,2,4])*u.s,
               end=epoch+np.array([1,2,3,5])*u.s,duration_s=np.ones(4),
               states=dict(rcr=np.zeros(4),num_pixel_sets=np.ones(4),
                  detector_masks=np.array([[1,0],[1,0],[0,1],[0,1]]),
                  pixel_masks=np.ones((4,1))))
        runs=configuration_segments(p)
        self.assertEqual(runs.bins.tolist(),[2,1,1])
        self.assertEqual(runs.first_bin.tolist(),[0,2,3])

    def test_elut_boundary_overlap_and_open_ended_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'index.csv'
            p.write_text('index,start_date,end_date,elut_file\n'
                         '0,2026-01-01,2026-07-01,old.csv\n'
                         '1,2026-07-01,none,new.csv\n')
            match=elut_index_match(p,Time('2026-07-07'),Time('2026-07-08'))
            self.assertEqual(match['elut_name'],'new')
            self.assertIsNone(match['indexed_end_utc'])
            self.assertEqual(len(match['index_sha256']),64)
            with self.assertRaises(ValueError):
                elut_index_match(p,Time('2026-06-30'),Time('2026-07-01'))
            p.write_text(p.read_text()+'2,2026-07-01,none,conflict.csv\n')
            with self.assertRaises(ValueError):
                elut_index_match(p,Time('2026-07-07'),Time('2026-07-08'))


if __name__=='__main__':
    unittest.main()
