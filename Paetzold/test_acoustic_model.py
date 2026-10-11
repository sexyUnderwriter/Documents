import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from acoustic_model import (
    AcousticModel, Measurement, ROOT, calibrate, cents_difference,
    load_measurements, sound_speed, target_frequency,
)


def tube_config(length_mm=1000, side_mm=40):
    return {
        "schema_version": 1,
        "tuning": {"a4_hz": 440},
        "physical_geometry_confirmed": True,
        "tonehole_inventory_confirmed": True,
        "head_correction_mm": 0,
        "bell_correction_mm": 0,
        "segments": [{
            "id": "test_tube", "length_mm": length_mm,
            "width_start_mm": side_mm, "height_start_mm": side_mm,
            "width_end_mm": side_mm, "height_end_mm": side_mm,
        }],
        "toneholes": [],
        "fingerings": {"all_closed": []},
    }


class AcousticTests(unittest.TestCase):
    def test_cli_targets_and_missing_geometry_guard(self):
        script = str(ROOT / "acoustic_model.py")
        targets = subprocess.run([sys.executable, script, "targets"], capture_output=True, text=True)
        self.assertEqual(targets.returncode, 0, targets.stderr)
        rows = list(csv.DictReader(targets.stdout.splitlines()))
        self.assertEqual(len(rows), 13)
        self.assertEqual(rows[0]["note"], "F2")
        self.assertEqual(rows[-1]["note"], "F3")
        self.assertAlmostEqual(float(rows[0]["target_hz"]), target_frequency(41), places=6)
        self.assertIn("NOT physical tonehole coordinates", targets.stderr)
        incomplete = subprocess.run([sys.executable, script, "resonances"],
                                    capture_output=True, text=True)
        self.assertEqual(incomplete.returncode, 2)
        self.assertEqual(incomplete.stdout, "")
        self.assertIn("Physical geometry is unconfirmed", incomplete.stderr)

    def test_equal_temperament_and_cents(self):
        self.assertEqual(target_frequency(69), 440)
        self.assertEqual(target_frequency(57), 220)
        self.assertAlmostEqual(target_frequency(41), 87.30705785825097)
        self.assertAlmostEqual(target_frequency(42) / target_frequency(41), 2 ** (1 / 12))
        self.assertEqual(cents_difference(880, 440), 1200)
        self.assertEqual(cents_difference(220, 440), -1200)

    def test_invalid_numbers(self):
        for value in (None, math.nan, math.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sound_speed(value)
        for midi in (True, 41.5, -1, 128):
            with self.subTest(midi=midi), self.assertRaises(ValueError):
                target_frequency(midi)
        with self.assertRaises(ValueError):
            cents_difference(0, 440)

    def test_uniform_open_open_tube_analytic_modes(self):
        model = AcousticModel(tube_config())
        roots = model.resonances(high_hz=700)
        speed = sound_speed(20)
        self.assertEqual(len(roots), 4)
        for n, actual in enumerate(roots, 1):
            self.assertAlmostEqual(actual, n * speed / 2, places=7)

    def test_explicit_end_corrections(self):
        config = tube_config()
        config.update(head_correction_mm=25, bell_correction_mm=12)
        roots = AcousticModel(config).resonances()
        self.assertAlmostEqual(roots[0], sound_speed(20) / (2 * 1.037), places=7)

    def test_temperature_scaling(self):
        model = AcousticModel(tube_config())
        f20 = model.resonances(temperature_c=20)[0]
        f30 = model.resonances(temperature_c=30)[0]
        self.assertAlmostEqual(f30 / f20, sound_speed(30) / sound_speed(20), places=9)

    def test_uniform_segmentation_invariance(self):
        config = tube_config()
        first = config["segments"][0]
        config["segments"] = [
            dict(first, id="one", length_mm=333),
            dict(first, id="two", length_mm=667),
        ]
        split = AcousticModel(config, cell_mm=7).resonances()
        single = AcousticModel(tube_config(), cell_mm=13).resonances()
        for a, b in zip(split, single):
            self.assertAlmostEqual(a, b, places=8)

    def test_taper_mesh_convergence(self):
        config = tube_config(length_mm=1800, side_mm=60)
        config["segments"][0].update(width_end_mm=38, height_end_mm=38)
        medium = AcousticModel(config, cell_mm=5).resonances()[0]
        fine = AcousticModel(config, cell_mm=2.5).resonances()[0]
        self.assertLess(abs(cents_difference(medium, fine)), 0.05)

    def test_hole_opening_and_closed_cavity(self):
        config = tube_config()
        config["toneholes"] = [{
            "id": "h1", "x_mm": 603.3, "diameter_mm": 12,
            "chimney_mm": 9, "end_correction_mm": 8,
        }]
        config["fingerings"]["h1_open"] = ["h1"]
        model = AcousticModel(config, cell_mm=7)
        bare = AcousticModel(tube_config()).resonances()[0]
        closed = model.resonances()[0]
        opened = model.resonances("h1_open")[0]
        self.assertLess(closed, bare)
        self.assertGreater(opened, bare)
        self.assertLess(opened, sound_speed(20) / (2 * 0.6033))
        self.assertTrue(any(abs(right - 603.3) < 1e-7 for right, _, _ in model.cells))

    def test_hole_at_segment_join(self):
        config = tube_config()
        first = config["segments"][0]
        config["segments"] = [dict(first, id="a", length_mm=600),
                              dict(first, id="b", length_mm=400)]
        config["toneholes"] = [{
            "id": "h1", "x_mm": 600, "diameter_mm": 10,
            "chimney_mm": 9, "end_correction_mm": 8,
        }]
        config["fingerings"]["opened"] = ["h1"]
        actual = AcousticModel(config).resonances("opened")[0]
        single = tube_config()
        single.update(toneholes=config["toneholes"], fingerings=config["fingerings"])
        self.assertAlmostEqual(actual, AcousticModel(single).resonances("opened")[0], places=8)

    def test_repository_model_cannot_emit_unearned_predictions(self):
        config = json.loads((ROOT / "instrument_model.json").read_text())
        self.assertEqual(config["tuning"]["a4_hz"], 440)
        self.assertEqual(config["tuning"]["lowest_midi"], 41)
        with self.assertRaisesRegex(ValueError, "unconfirmed"):
            AcousticModel(config)
        config["physical_geometry_confirmed"] = True
        config["tonehole_inventory_confirmed"] = True
        config["head_correction_mm"] = 0
        config["bell_correction_mm"] = 0
        with self.assertRaisesRegex(ValueError, "head_limb_from_window.length_mm"):
            AcousticModel(config)

    def test_invalid_holes_and_searches(self):
        config = tube_config()
        config["toneholes"] = [{
            "id": "h", "x_mm": 1000, "diameter_mm": 10,
            "chimney_mm": 9, "end_correction_mm": 8,
        }]
        with self.assertRaises(ValueError):
            AcousticModel(config)
        model = AcousticModel(tube_config())
        for options in ({"fingering": "unknown"}, {"step_hz": 100},
                        {"high_hz": 10000}, {"high_hz": 10}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                model.resonances(**options)

    def test_one_parameter_fit_against_analytic_data_and_holdout(self):
        model = AcousticModel(tube_config(length_mm=1800), cell_mm=20)
        known_mm = 32.0
        data = [
            Measurement("fit1", "all_closed", 1, sound_speed(20) / (2 * 1.832), 20, 300, True, 41),
            Measurement("heldout", "all_closed", 2, sound_speed(25) / 1.832, 25, 300, False, 53),
        ]
        fit = calibrate(model, data, upper_mm=80)
        self.assertAlmostEqual(fit["head_correction_mm"], known_mm, places=3)
        self.assertLess(fit["fit_rmse_cents"], 0.001)
        self.assertLess(fit["held_out_rmse_cents"], 0.001)
        self.assertFalse(fit["at_bound"])
        self.assertEqual(fit["residuals"][0]["target_hz"], target_frequency(41))
        self.assertAlmostEqual(fit["residuals"][0]["measured_tuning_cents"],
                               cents_difference(data[0].frequency_hz, target_frequency(41)))
        changed = [data[0], Measurement("heldout", "all_closed", 2, 210, 25, 300, False, 53)]
        changed_fit = calibrate(model, changed, upper_mm=80)
        self.assertEqual(changed_fit["head_correction_mm"], fit["head_correction_mm"])
        self.assertGreater(changed_fit["held_out_rmse_cents"], 100)

    def test_fit_bound_and_missing_training(self):
        model = AcousticModel(tube_config(length_mm=1800), cell_mm=30)
        data = [Measurement("a", "all_closed", 1, sound_speed(20) / (2 * 1.9), 20, None, True, 41)]
        fit = calibrate(model, data, upper_mm=20)
        self.assertTrue(fit["at_bound"])
        self.assertEqual(fit["head_correction_mm"], 20)
        self.assertIsNone(fit["held_out_rmse_cents"])
        with self.assertRaisesRegex(ValueError, "No rows"):
            calibrate(model, [Measurement("b", "all_closed", 1, 100, 20, None, False, 41)])

    def test_measurement_csv_empty_missing_and_valid(self):
        model = AcousticModel(tube_config())
        with self.assertRaisesRegex(ValueError, "header-only"):
            load_measurements(ROOT / "pitch_measurements.csv", model)
        fields = ["measurement_id", "fingering_id", "mode", "target_midi", "frequency_hz",
                  "temperature_c", "pressure_pa", "use_for_fit", "notes"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.csv"
            valid = ["m1", "all_closed", "1", "41", "171.6", "20", "", "true", "test"]
            for row, expected_error in (
                (valid, None), ([*valid[:4], "", *valid[5:]], "frequency"),
                ([*valid[:5], "", *valid[6:]], "temperature_c"),
                ([*valid[:7], "yes", valid[8]], "use_for_fit"),
                ([*valid[:3], "128", *valid[4:]], "MIDI"),
                (valid[:-1], "fields"),
            ):
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(fields)
                    writer.writerow(row)
                if expected_error:
                    with self.assertRaisesRegex(ValueError, expected_error):
                        load_measurements(path, model)
                else:
                    measurements = load_measurements(path, model)
                    self.assertEqual(measurements[0].frequency_hz, 171.6)
                    self.assertIsNone(measurements[0].pressure_pa)


if __name__ == "__main__":
    unittest.main()
