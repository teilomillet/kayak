from __future__ import annotations

import unittest

from kayak_bridge.usl_scaling import UslObservation, fit_usl


class UslScalingTests(unittest.TestCase):
    def test_fits_linearized_usl_from_exact_observations(self) -> None:
        base = 100.0
        alpha = 0.05
        beta = 0.01
        observations = [
            UslObservation(
                scale=scale,
                throughput=base
                * float(scale)
                / (1.0 + alpha * float(scale - 1) + beta * scale * (scale - 1)),
            )
            for scale in (1, 2, 4, 8, 16)
        ]

        fit = fit_usl(observations, max_predicted_scale=16)

        self.assertAlmostEqual(fit.base_throughput, base)
        self.assertAlmostEqual(fit.alpha, alpha)
        self.assertAlmostEqual(fit.beta, beta)
        self.assertAlmostEqual(fit.r_squared, 1.0)
        self.assertEqual(fit.fit_status, "ok")

    def test_rejects_duplicate_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_usl(
                [
                    UslObservation(scale=1, throughput=1.0),
                    UslObservation(scale=1, throughput=2.0),
                    UslObservation(scale=2, throughput=3.0),
                ]
            )


if __name__ == "__main__":
    unittest.main()
