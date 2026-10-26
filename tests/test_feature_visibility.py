"""Interpretive guards: interval logic and evidence must not identify sources."""
import unittest
from occulted_flares.geometry.feature_visibility import band_visibility, validate_reviews


class FeatureAssessmentTest(unittest.TestCase):
    def test_trial_intervals_and_tangent_are_distinct(self):
        self.assertEqual(band_visibility(40, True, 0, 2), "all_trial_heights_hidden")
        self.assertEqual(band_visibility(40, True, 30, 100), "straddles_or_touches_threshold")
        self.assertEqual(band_visibility(1, True, 5, 30), "all_trial_heights_clear")
        self.assertEqual(band_visibility(1, True, 0, 1), "straddles_or_touches_threshold")
        self.assertEqual(band_visibility(0, False, 0, 2), "all_trial_heights_clear")
        self.assertEqual(band_visibility(float("inf"), True, 0, 100), "all_trial_heights_hidden")

    def test_invalid_height_band_rejected(self):
        for lo, hi in ((-1, 2), (3, 2), (0, float("nan"))):
            with self.assertRaises(ValueError):
                band_visibility(1, True, lo, hi)

    def test_evidence_and_alternative_required(self):
        validate_reviews(["point_1"], {})
        with self.assertRaises(ValueError):
            validate_reviews(["point_1"], {"point_1": {"classification": "candidate_low_atmosphere"}})
        validate_reviews(["point_1"], {"point_1": dict(classification="ambiguous", evidence="compact brightening", alternative="projected loop")})
        with self.assertRaises(ValueError):
            validate_reviews(["point_1"], {"old_label": {}})


if __name__ == "__main__":
    unittest.main()
