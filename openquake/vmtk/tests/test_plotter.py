import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Stub openseespy before plotter is imported — openseespy is only needed by
# the OpenSees-dependent animation/mode methods, which are skipped in CI.
sys.modules.setdefault('openseespy', MagicMock())
sys.modules.setdefault('openseespy.opensees', MagicMock())

import matplotlib  # noqa: E402
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from openquake.vmtk.plotter import plotter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pl():
    """Return a fresh plotter instance."""
    return plotter()


def _cloud_dict():
    """Minimal cloud_dict matching the structure expected by plot_mca_analysis
    and plot_fragility_from_mca."""
    n = 30
    rng = np.random.default_rng(0)
    imls = np.geomspace(0.05, 2.0, n)
    edps = np.exp(rng.normal(np.log(0.01), 0.5, n))
    n_boot = 10
    n_ds = 2
    intensities = np.geomspace(0.05, 2.0, 50)
    return {
        'cloud inputs': {
            'imls': imls,
            'edps': edps,
            'upper_limit': 0.10,
        },
        'raw_data': {
            'im_nc': imls[:20],
            'edp_nc': edps[:20],
            'im_c': imls[20:],
            'edp_c': edps[20:],
        },
        'regression': {
            'b0': np.log(0.018),   # stored as log(a)
            'b1': 1.2,
            'sigma': 0.4,
        },
        'bootstraps': {
            # plot_mca_analysis reads boot['a'], boot['b1'], boot['sigma_rr']
            'a': rng.uniform(0.01, 0.03, n_boot),
            'b1': rng.normal(1.2, 0.05, n_boot),
            'sigma_rr': rng.uniform(0.3, 0.5, n_boot),
            'alpha0': rng.normal(-5.0, 0.2, n_boot),
            'alpha1': rng.normal(2.0, 0.1, n_boot),
            'poes_all': rng.uniform(0.0, 1.0, (n_boot, len(intensities), n_ds)),
        },
        'fragility': {
            'intensities': intensities,
            'poes': np.clip(rng.uniform(0.0, 1.0, (len(intensities), n_ds)), 0, 1),
            'medians': [0.3, 0.8],
            'betas_total': [0.4, 0.5],
        },
    }


def _ida_dict():
    """Minimal ida_dict for plot_ida_analysis and plot_fragility_from_ida."""
    rng = np.random.default_rng(1)
    n_gm = 5
    n_im = 8
    n_ds = 2
    intensities = np.linspace(0.1, 2.0, 50)
    # raw_curves: list of dicts with 'im' and 'edp' keys
    raw_curves = [
        {
            'im': np.sort(rng.uniform(0.1, 2.0, n_im)),
            'edp': np.sort(rng.uniform(0.001, 0.08, n_im)),
        }
        for _ in range(n_gm)
    ]
    return {
        'ida_inputs': {
            'raw_curves': raw_curves,
            'damage_thresholds': [0.01, 0.025],
            'imt_key': 'Sa(T1) [g]',
        },
        'stats': {
            'fitted_edps': np.linspace(0.001, 0.08, 50),
            'median_im': [0.3, 0.6],
            'p16_im': [0.2, 0.45],
            'p84_im': [0.4, 0.75],
        },
        'fragility': {
            'intensities': intensities,
            'poes': np.clip(rng.uniform(0.0, 1.0, (len(intensities), n_ds)), 0, 1),
            'medians': [0.35, 0.7],
            'betas_total': [0.4, 0.5],
        },
    }


def _msa_dict():
    """Minimal msa_dict for plot_fragility_from_msa."""
    rng = np.random.default_rng(2)
    n_ds = 2
    intensities = np.geomspace(0.05, 2.0, 50)
    stripe_levels = np.array([0.1, 0.3, 0.5, 0.8, 1.2])
    return {
        'fragility': {
            'intensities': intensities,
            'poes': np.clip(rng.uniform(0.0, 1.0, (len(intensities), n_ds)), 0, 1),
            'medians': [0.4, 0.9],
            'betas_total': [0.35, 0.45],
        },
        'metadata': {
            'stripe_levels': stripe_levels,
            'observed_fractions': [
                rng.uniform(0.0, 1.0, len(stripe_levels))
                for _ in range(n_ds)
            ],
        },
    }


def _slf_out_cache():
    """Minimal out/cache dicts for plot_slf_model.

    plot_slf_model iterates over cache.keys() and accesses
    cache[key]['total_loss_storey'] (shape: n_real x n_edp),
    cache[key]['empirical_16th/84th/median'], and out[key]['edp_range']/['slf'].
    """
    rng = np.random.default_rng(3)
    edp_range = np.linspace(0.0, 0.1, 50)
    n_real = 20
    group_key = 'psd_ns_1'
    out = {
        group_key: {
            'edp_range': edp_range,
            'slf': np.clip(
                np.cumsum(rng.uniform(0.0, 0.02, len(edp_range))), 0, 1),
        }
    }
    cache = {
        group_key: {
            # shape (n_real, n_edp) — each row is one realisation
            'total_loss_storey': rng.uniform(0.0, 0.3, (n_real, len(edp_range))),
            'empirical_16th': rng.uniform(0.0, 0.15, len(edp_range)),
            'empirical_84th': rng.uniform(0.15, 0.3, len(edp_range)),
            'empirical_median': rng.uniform(0.07, 0.2, len(edp_range)),
        }
    }
    return out, cache


# ---------------------------------------------------------------------------
# __init__ attribute tests
# ---------------------------------------------------------------------------

class TestPlotterInit(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()

    def test_figsize_default(self):
        self.assertEqual(self.pl.figsize, (10, 7))

    def test_figsize_anim_default(self):
        self.assertEqual(self.pl.figsize_anim, (10, 7))

    def test_resolution_default(self):
        self.assertEqual(self.pl.resolution, 400)

    def test_font_name_default(self):
        self.assertEqual(self.pl.font_name, 'Arial')

    def test_font_sizes_keys(self):
        for key in ('title', 'labels', 'ticks', 'legend'):
            self.assertIn(key, self.pl.font_sizes)

    def test_line_widths_keys(self):
        for key in ('thick', 'medium', 'thin'):
            self.assertIn(key, self.pl.line_widths)

    def test_marker_sizes_keys(self):
        for key in ('large', 'medium', 'small'):
            self.assertIn(key, self.pl.marker_sizes)

    def test_colors_keys(self):
        for key in ('fragility', 'damage_states', 'gem'):
            self.assertIn(key, self.pl.colors)

    def test_attributes_are_mutable(self):
        self.pl.resolution = 200
        self.assertEqual(self.pl.resolution, 200)


# ---------------------------------------------------------------------------
# _set_plot_style tests
# ---------------------------------------------------------------------------

class TestSetPlotStyle(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close('all')

    def test_title_is_set(self):
        self.pl._set_plot_style(self.ax, title='Test Title')
        self.assertEqual(self.ax.get_title(), 'Test Title')

    def test_xlabel_is_set(self):
        self.pl._set_plot_style(self.ax, xlabel='X Label')
        self.assertEqual(self.ax.get_xlabel(), 'X Label')

    def test_ylabel_is_set(self):
        self.pl._set_plot_style(self.ax, ylabel='Y Label')
        self.assertEqual(self.ax.get_ylabel(), 'Y Label')

    def test_no_title_when_none(self):
        self.pl._set_plot_style(self.ax, title=None)
        self.assertEqual(self.ax.get_title(), '')

    def test_grid_disabled(self):
        self.pl._set_plot_style(self.ax, grid=False)
        # The axes still exists — no exception raised


# ---------------------------------------------------------------------------
# _save_plot tests
# ---------------------------------------------------------------------------

class TestSavePlot(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()

    def tearDown(self):
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_save_plot_no_directory(self, mock_show):
        """Calling _save_plot with None should call plt.show() without saving."""
        fig, ax = plt.subplots()
        self.pl._save_plot(None, 'test_label')
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_save_plot_with_directory(self, mock_show):
        """Calling _save_plot with a directory should write a PNG file."""
        fig, ax = plt.subplots()
        with tempfile.TemporaryDirectory() as tmp:
            self.pl._save_plot(tmp, 'my_plot')
            expected = os.path.join(tmp, 'my_plot.png')
            self.assertTrue(os.path.isfile(expected))
        mock_show.assert_called_once()


# ---------------------------------------------------------------------------
# duplicate_for_drift tests
# ---------------------------------------------------------------------------

class TestDuplicateForDrift(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()

    def test_output_lengths_match(self):
        drifts = [0.01, 0.02]
        nodes = [0.0, 2.8, 5.6]
        x, y = self.pl.duplicate_for_drift(drifts, nodes)
        self.assertEqual(len(x), len(y))

    def test_trailing_zero_appended(self):
        drifts = [0.01, 0.02]
        nodes = [0.0, 2.8, 5.6]
        x, _ = self.pl.duplicate_for_drift(drifts, nodes)
        self.assertAlmostEqual(x[-1], 0.0)

    def test_length_formula(self):
        # Each storey contributes 2 to x/y, then +1 trailing
        n_storeys = 3
        drifts = [0.01] * n_storeys
        nodes = list(range(n_storeys + 1))
        x, y = self.pl.duplicate_for_drift(drifts, nodes)
        self.assertEqual(len(x), 2 * n_storeys + 1)

    def test_drift_values_correct(self):
        drifts = [0.05, 0.10]
        nodes = [0.0, 3.0, 6.0]
        x, _ = self.pl.duplicate_for_drift(drifts, nodes)
        # First two entries should both equal drifts[0]
        self.assertAlmostEqual(x[0], 0.05)
        self.assertAlmostEqual(x[1], 0.05)

    def test_single_storey(self):
        x, y = self.pl.duplicate_for_drift([0.03], [0.0, 3.0])
        self.assertEqual(len(x), 3)
        self.assertAlmostEqual(x[-1], 0.0)


# ---------------------------------------------------------------------------
# plot_demand_profiles tests
# ---------------------------------------------------------------------------

class TestPlotDemandProfiles(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()
        rng = np.random.default_rng(4)
        n_st = 3
        n_gm = 4
        self.peak_drift_list = [
            np.column_stack([rng.uniform(0.001, 0.03, n_st),
                             np.arange(1, n_st + 1)])
            for _ in range(n_gm)
        ]
        self.peak_accel_list = [
            np.column_stack([rng.uniform(0.5, 5.0, n_st + 1),
                             np.arange(n_st + 1)])
            for _ in range(n_gm)
        ]
        self.control_nodes = list(range(n_st + 1))

    def tearDown(self):
        plt.close('all')

    def test_runs_without_error(self):
        self.pl.plot_demand_profiles(
            self.peak_drift_list,
            self.peak_accel_list,
            self.control_nodes,
            pFlag=False,
        )

    def test_runs_with_title(self):
        self.pl.plot_demand_profiles(
            self.peak_drift_list,
            self.peak_accel_list,
            self.control_nodes,
            title='My Title',
            pFlag=False,
        )

    @patch('matplotlib.pyplot.show')
    def test_export_path_creates_file(self, _mock_show):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'demand.png')
            self.pl.plot_demand_profiles(
                self.peak_drift_list,
                self.peak_accel_list,
                self.control_nodes,
                pFlag=True,
                export_path=path,
            )
            self.assertTrue(os.path.isfile(path))


# ---------------------------------------------------------------------------
# plot_mca_analysis tests
# ---------------------------------------------------------------------------

class TestPlotMCAAnalysis(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()
        self.cd = _cloud_dict()

    def tearDown(self):
        plt.close('all')

    def test_runs_without_error(self):
        self.pl.plot_mca_analysis(
            self.cd, 'PGA [g]', 'PSD [-]', pFlag=False
        )

    def test_runs_with_title(self):
        self.pl.plot_mca_analysis(
            self.cd, 'PGA [g]', 'PSD [-]',
            title='MCA', pFlag=False
        )


# ---------------------------------------------------------------------------
# plot_ida_analysis tests
# ---------------------------------------------------------------------------

class TestPlotIDAAnalysis(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()
        self.id = _ida_dict()

    def tearDown(self):
        plt.close('all')

    def test_runs_without_error(self):
        self.pl.plot_ida_analysis(
            self.id, 'Sa(T1) [g]', 'PSD [-]',
            xlims=(0.0, 0.1), ylims=(0.0, 2.5),
            pFlag=False,
        )


# ---------------------------------------------------------------------------
# plot_msa_analysis tests
# ---------------------------------------------------------------------------

class TestPlotMSAAnalysis(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()
        rng = np.random.default_rng(5)
        n_gm = 10
        n_stripes = 4
        stripe_imls_val = np.array([0.1, 0.3, 0.6, 1.0])
        self.stripe_imls = np.tile(stripe_imls_val, (n_gm, 1))
        self.stripe_edps = rng.uniform(0.001, 0.05, (n_gm, n_stripes))

    def tearDown(self):
        plt.close('all')

    def test_runs_without_error(self):
        self.pl.plot_msa_analysis(
            self.stripe_imls, self.stripe_edps,
            'Sa(T1) [g]', 'PSD',
            xlims=(0.0, 5.0), ylims=(0.0, 1.5),
            pFlag=False,
        )


# ---------------------------------------------------------------------------
# plot_fragility_from_mca tests
# ---------------------------------------------------------------------------

class TestPlotFragilityFromMCA(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()
        self.cd = _cloud_dict()

    def tearDown(self):
        plt.close('all')

    def test_runs_without_error(self):
        self.pl.plot_fragility_from_mca(
            self.cd, 'PGA [g]',
            xlims=(0.0, 2.0), ylims=(0.0, 1.0),
            pFlag=False,
        )

    def test_runs_with_bootstrap(self):
        self.pl.plot_fragility_from_mca(
            self.cd, 'PGA [g]',
            xlims=(0.0, 2.0), ylims=(0.0, 1.0),
            plot_bootstrap=True,
            pFlag=False,
        )


# ---------------------------------------------------------------------------
# plot_fragility_from_ida tests
# ---------------------------------------------------------------------------

class TestPlotFragilityFromIDA(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()
        self.id = _ida_dict()

    def tearDown(self):
        plt.close('all')

    def test_runs_without_error(self):
        self.pl.plot_fragility_from_ida(
            self.id, 'Sa(T1) [g]',
            xlims=(0.0, 2.0), ylims=(0.0, 1.0),
            pFlag=False,
        )


# ---------------------------------------------------------------------------
# plot_fragility_from_msa tests
# ---------------------------------------------------------------------------

class TestPlotFragilityFromMSA(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()
        self.md = _msa_dict()

    def tearDown(self):
        plt.close('all')

    def test_runs_without_error(self):
        self.pl.plot_fragility_from_msa(
            self.md, 'Sa(T1) [g]',
            xlims=(0.0, 2.0), ylims=(0.0, 1.0),
            pFlag=False,
        )


# ---------------------------------------------------------------------------
# plot_slf_model tests
# ---------------------------------------------------------------------------

class TestPlotSLFModel(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()
        self.out, self.cache = _slf_out_cache()

    def tearDown(self):
        plt.close('all')

    def test_runs_without_error(self):
        self.pl.plot_slf_model(
            self.out, self.cache,
            edp_label='PSD [-]', loss_label='Loss Ratio',
            xlims=(0.0, 0.1), ylims=(0.0, 1.0),
            pFlag=False,
        )


# ---------------------------------------------------------------------------
# plot_vulnerability_function tests
# ---------------------------------------------------------------------------

class TestPlotVulnerabilityFunction(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()
        self.intensities = [0.1, 0.3, 0.5, 0.8, 1.0]
        self.loss = [0.02, 0.08, 0.15, 0.30, 0.45]
        self.cov = [0.5, 0.4, 0.35, 0.30, 0.25]

    def tearDown(self):
        plt.close('all')

    def test_runs_without_error(self):
        self.pl.plot_vulnerability_function(
            self.intensities, self.loss, self.cov,
            imt_label='PGA [g]', loss_label='Mean Loss Ratio',
            pFlag=False,
        )

    def test_runs_with_title(self):
        self.pl.plot_vulnerability_function(
            self.intensities, self.loss, self.cov,
            imt_label='PGA [g]', loss_label='Mean Loss Ratio',
            title='Vulnerability',
            pFlag=False,
        )


# ---------------------------------------------------------------------------
# OpenSees-dependent methods — skipped in CI
# ---------------------------------------------------------------------------

@unittest.skip("Requires a compiled OpenSees model — run manually")
class TestOpenSeesDependent(unittest.TestCase):

    def setUp(self):
        self.pl = _pl()

    def test_plot_modes(self):
        self.pl.plot_modes([], [], [], export_path=None)

    def test_animate_spo(self):
        pass

    def test_animate_cpo(self):
        pass

    def test_animate_nrha(self):
        pass

    def test_animate_model_run(self):
        pass


if __name__ == '__main__':
    unittest.main()
