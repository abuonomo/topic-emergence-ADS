import logging
import unittest
from unittest import TestCase

import pandas as pd

from model import VizPrepper

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class TestVizPrepper(TestCase):

    def setUp(self) -> None:
        self.time_series1 = pd.Series([0, 1, 2, 3, 4, 5])
        self.time_series2 = pd.Series([1, 10, 0, 1, 2])
        self.viz_prepper = VizPrepper()

    def test_cagr(self):
        val1 = self.viz_prepper.cagr(self.time_series1)
        true_val1 = 0.4953487812212205
        self.assertAlmostEqual(true_val1, val1)

        val2 = self.viz_prepper.cagr(self.time_series2)
        true_val2 = 0.18920711500272103
        self.assertAlmostEqual(true_val2, val2)

    def test_cagr3(self):
        val1 = self.viz_prepper.cagr(self.time_series1, n_mean=3)
        true_val1 = 0.18920711500272103
        self.assertAlmostEqual(true_val1, val1)

        val2 = self.viz_prepper.cagr(self.time_series2, n_mean=3)
        true_val2 = -0.27734311885439467
        self.assertAlmostEqual(true_val2, val2)


if __name__ == "__main__":
    unittest.main()


