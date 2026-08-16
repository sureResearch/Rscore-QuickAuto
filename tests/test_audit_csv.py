import csv
import math
import unittest
from pathlib import Path

import rscore_quickauto as rq


class AuditCsvTests(unittest.TestCase):
    def test_audit_csv_matches_embedded_model(self):
        csv_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "Final_model_coefficients.csv"
        )
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 24)
        intercept_rows = [
            row for row in rows if row["Term"] == "(Intercept)"
        ]
        self.assertEqual(len(intercept_rows), 1)
        self.assertTrue(math.isclose(
            float(intercept_rows[0]["Coefficient_on_standardized_scale"]),
            rq.INTERCEPT,
            rel_tol=0.0,
            abs_tol=0.0,
        ))
        feature_rows = {
            row["Term"]: row
            for row in rows
            if row["Term"] != "(Intercept)"
        }
        self.assertEqual(set(feature_rows), set(rq.FEATURE_SPECS))
        for name, (coefficient, mean, sd) in rq.FEATURE_SPECS.items():
            row = feature_rows[name]
            self.assertEqual(
                float(row["Coefficient_on_standardized_scale"]),
                coefficient,
            )
            self.assertEqual(
                float(row["Training_mean_for_standardization"]),
                mean,
            )
            self.assertEqual(
                float(row["Training_SD_for_standardization"]),
                sd,
            )


if __name__ == "__main__":
    unittest.main()
