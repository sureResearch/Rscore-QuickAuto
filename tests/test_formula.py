import math
import unittest

import rscore_quickauto as rq


def mean_vector():
    return {name: spec[1] for name, spec in rq.FEATURE_SPECS.items()}


class LockedFormulaTests(unittest.TestCase):
    def test_locked_model_metadata(self):
        self.assertEqual(rq.MODEL_TARGET, "CMRRI_z")
        self.assertEqual(rq.PRIMARY_OUTCOME, "Rscore_z")
        self.assertEqual(rq.MODEL_ID, "Rscore-CMRRI-v1")
        self.assertEqual(rq.PIPELINE_ID, "CineSAX-Rad-v1")
        self.assertEqual(len(rq.FEATURE_SPECS), 23)
        rq.validate_locked_model()

    def test_training_mean_vector_returns_intercept(self):
        result = rq.calculate_rscore(mean_vector())
        self.assertTrue(math.isclose(
            result["raw_Rscore"],
            rq.INTERCEPT,
            rel_tol=0.0,
            abs_tol=1e-12,
        ))
        expected_z = (
            rq.INTERCEPT - rq.OOF_RSCORE_MEAN
        ) / rq.OOF_RSCORE_SD
        self.assertTrue(math.isclose(
            result["Rscore_z"],
            expected_z,
            abs_tol=1e-12,
        ))

    def test_one_sd_change_equals_coefficient(self):
        for feature_name in rq.FEATURE_SPECS:
            with self.subTest(feature=feature_name):
                values = mean_vector()
                coefficient, mean, sd = rq.FEATURE_SPECS[feature_name]
                values[feature_name] = mean + sd
                result = rq.calculate_rscore(values)
                self.assertTrue(math.isclose(
                    result["raw_Rscore"],
                    rq.INTERCEPT + coefficient,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ))

    def test_missing_feature_fails(self):
        values = mean_vector()
        values.pop(next(iter(values)))
        with self.assertRaisesRegex(RuntimeError, "Missing locked-model"):
            rq.calculate_rscore(values)

    def test_nonfinite_feature_fails(self):
        values = mean_vector()
        values[next(iter(values))] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            rq.calculate_rscore(values)


if __name__ == "__main__":
    unittest.main()
