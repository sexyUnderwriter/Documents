"""Exploratory passive acoustic model; predictions are not drilling coordinates."""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def number(value, name, minimum=None, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def sound_speed(temperature_c):
    """Dry-air approximation, restricted to normal instrument temperatures."""
    temperature_c = number(temperature_c, "temperature_c")
    if not 0 <= temperature_c <= 40:
        raise ValueError("temperature_c must be in 0..40 C for this approximation")
    return 331.3 * math.sqrt(1 + temperature_c / 273.15)


def target_frequency(midi, a4_hz=440.0):
    if isinstance(midi, bool) or not isinstance(midi, int) or not 0 <= midi <= 127:
        raise ValueError("MIDI note must be an integer in 0..127")
    return number(a4_hz, "a4_hz", strictly_positive=True) * 2 ** ((midi - 69) / 12)


def cents_difference(actual_hz, reference_hz):
    actual_hz = number(actual_hz, "actual_hz", strictly_positive=True)
    reference_hz = number(reference_hz, "reference_hz", strictly_positive=True)
    return 1200 * math.log2(actual_hz / reference_hz)


@dataclass(frozen=True)
class Segment:
    id: str
    length_mm: float
    width_start_mm: float
    height_start_mm: float
    width_end_mm: float
    height_end_mm: float

    def area(self, fraction):
        width = self.width_start_mm + fraction * (self.width_end_mm - self.width_start_mm)
        height = self.height_start_mm + fraction * (self.height_end_mm - self.height_start_mm)
        return width * height * 1e-6


@dataclass(frozen=True)
class Tonehole:
    id: str
    x_mm: float
    diameter_mm: float
    chimney_mm: float
    end_correction_mm: float

    @property
    def area(self):
        return math.pi * (self.diameter_mm * 0.001 / 2) ** 2


@dataclass(frozen=True)
class Measurement:
    id: str
    fingering: str
    mode: int
    frequency_hz: float
    temperature_c: float
    pressure_pa: float | None
    use_for_fit: bool
    target_midi: int


class AcousticModel:
    def __init__(self, config, cell_mm=5.0):
        if config.get("schema_version") != 1:
            raise ValueError("schema_version must be 1")
        if config.get("physical_geometry_confirmed") is not True:
            raise ValueError("Physical geometry is unconfirmed: measure the head path and confirm body assembly first")
        if config.get("tonehole_inventory_confirmed") is not True:
            raise ValueError("Tonehole inventory is unconfirmed: record every hole, or explicitly confirm an unperforated body")
        self.a4_hz = number(config["tuning"]["a4_hz"], "a4_hz", strictly_positive=True)
        self.cell_mm = number(cell_mm, "cell_mm", strictly_positive=True)
        self.head_correction_mm = number(config.get("head_correction_mm"), "head_correction_mm", minimum=0)
        self.bell_correction_mm = number(config.get("bell_correction_mm"), "bell_correction_mm", minimum=0)
        self.segments = []
        for item in config["segments"]:
            values = {key: number(item.get(key), f"{item['id']}.{key}", strictly_positive=True)
                      for key in ("length_mm", "width_start_mm", "height_start_mm",
                                  "width_end_mm", "height_end_mm")}
            self.segments.append(Segment(id=item["id"], **values))
        if not self.segments or len({s.id for s in self.segments}) != len(self.segments):
            raise ValueError("segments must be nonempty with unique IDs")
        self.length_mm = sum(s.length_mm for s in self.segments)
        self.reference_area = self.segments[0].area(0)
        self.holes = {}
        for item in config["toneholes"]:
            values = {key: number(item.get(key), f"{item['id']}.{key}",
                                  strictly_positive=key in ("diameter_mm", "chimney_mm"),
                                  minimum=0)
                      for key in ("x_mm", "diameter_mm", "chimney_mm", "end_correction_mm")}
            hole = Tonehole(id=item["id"], **values)
            if hole.id in self.holes or not 0 < hole.x_mm < self.length_mm:
                raise ValueError("Toneholes require unique IDs and positions strictly inside the measured path")
            self.holes[hole.id] = hole
        self.fingerings = config["fingerings"]
        if not isinstance(self.fingerings, dict) or not self.fingerings:
            raise ValueError("fingerings must be a nonempty mapping from IDs to lists of open holes")
        for name, opened in self.fingerings.items():
            if not isinstance(opened, list) or any(not isinstance(h, str) for h in opened):
                raise ValueError(f"{name}: fingering must be a list of hole IDs")
            if len(opened) != len(set(opened)) or set(opened) - self.holes.keys():
                raise ValueError(f"{name}: duplicate or unknown open hole IDs")
        self.cells = self._build_cells()

    def _build_cells(self):
        cells = []
        offset = 0.0
        for segment in self.segments:
            count = math.ceil(segment.length_mm / self.cell_mm)
            cuts = {segment.length_mm * i / count for i in range(count + 1)}
            cuts.update(h.x_mm - offset for h in self.holes.values()
                        if offset < h.x_mm < offset + segment.length_mm)
            cuts = sorted(cuts)
            for left, right in zip(cuts, cuts[1:]):
                area = segment.area((left + right) / (2 * segment.length_mm))
                cells.append((offset + right, (right - left) * 0.001, area))
            offset += segment.length_mm
        return cells

    def _tube(self, state, k, length_m, area):
        pressure, flow = state
        angle = k * length_m
        cosine, sine = math.cos(angle), math.sin(angle)
        ratio = area / self.reference_area
        result = (cosine * pressure + sine / ratio * flow,
                  -ratio * sine * pressure + cosine * flow)
        scale = max(abs(result[0]), abs(result[1]))
        return result[0] / scale, result[1] / scale

    def _boundary(self, frequency_hz, temperature_c, fingering, head_correction_mm):
        k = 2 * math.pi * frequency_hz / sound_speed(temperature_c)
        opened = set(self.fingerings[fingering])
        state = self._tube((0.0, 1.0), k, self.bell_correction_mm * 0.001,
                           self.segments[-1].area(1))
        holes = sorted(self.holes.values(), key=lambda h: h.x_mm, reverse=True)
        index = 0
        for right, length, area in reversed(self.cells):
            while index < len(holes) and math.isclose(
                    holes[index].x_mm, right, rel_tol=0, abs_tol=1e-7):
                hole = holes[index]
                if hole.id in opened:
                    # Low-frequency open-hole inertance; retain the downstream bore.
                    coefficient = hole.area / self.reference_area / (
                        k * (hole.chimney_mm + hole.end_correction_mm) * 0.001)
                else:
                    coefficient = -hole.area / self.reference_area * math.tan(k * hole.chimney_mm * 0.001)
                pressure, flow = state
                flow += coefficient * pressure
                scale = max(abs(pressure), abs(flow))
                state = pressure / scale, flow / scale
                index += 1
            state = self._tube(state, k, length, area)
        state = self._tube(state, k, head_correction_mm * 0.001, self.reference_area)
        return state[0]

    def resonances(self, fingering="all_closed", temperature_c=20.0,
                   low_hz=20.0, high_hz=500.0, step_hz=0.5, head_correction_mm=None):
        if fingering not in self.fingerings:
            raise ValueError(f"Unknown fingering: {fingering}")
        speed = sound_speed(temperature_c)
        low_hz = number(low_hz, "low_hz", strictly_positive=True)
        high_hz = number(high_hz, "high_hz", strictly_positive=True)
        step_hz = number(step_hz, "step_hz", strictly_positive=True)
        if high_hz <= low_hz:
            raise ValueError("high_hz must exceed low_hz")
        correction = self.head_correction_mm if head_correction_mm is None else number(
            head_correction_mm, "head_correction_mm", minimum=0)
        max_side = max(max(s.width_start_mm, s.width_end_mm, s.height_start_mm,
                           s.height_end_mm) for s in self.segments) * 0.001
        if high_hz >= speed / (2 * max_side):
            raise ValueError("Search reaches the first rectangular transverse-mode cutoff")
        if any(high_hz >= speed / (4 * h.chimney_mm * 0.001) for h in self.holes.values()):
            raise ValueError("Search reaches a closed-hole stub pole; this low-frequency model is invalid")
        if any(high_hz * 2 * math.pi / speed *
               (h.chimney_mm + h.end_correction_mm) * 0.001 > 0.3 for h in self.holes.values()):
            raise ValueError("Open-hole k*effective_chimney exceeds 0.3; use a fuller tonehole model")
        phase_length = (self.length_mm + correction + self.bell_correction_mm) * 0.001
        if step_hz > speed / (20 * phase_length):
            raise ValueError("step_hz too coarse for the acoustic path; decrease it")
        roots = []
        left = low_hz
        value_left = self._boundary(left, temperature_c, fingering, correction)
        while left < high_hz:
            right = min(left + step_hz, high_hz)
            value_right = self._boundary(right, temperature_c, fingering, correction)
            if value_left == 0:
                roots.append(left)
            elif value_left * value_right < 0:
                a, b, fa = left, right, value_left
                for _ in range(35):
                    middle = (a + b) / 2
                    fm = self._boundary(middle, temperature_c, fingering, correction)
                    if fa * fm <= 0:
                        b = middle
                    else:
                        a, fa = middle, fm
                roots.append((a + b) / 2)
            left, value_left = right, value_right
        if value_left == 0:
            roots.append(high_hz)
        return roots

    def predict(self, measurement, head_correction_mm=None):
        roots = self.resonances(measurement.fingering, measurement.temperature_c,
                                head_correction_mm=head_correction_mm)
        if len(roots) < measurement.mode:
            raise ValueError(f"{measurement.id}: mode {measurement.mode} not found in 20..500 Hz")
        return roots[measurement.mode - 1]


def load_measurements(path, model):
    result = []
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"measurement_id", "fingering_id", "mode", "target_midi", "frequency_hz",
                    "temperature_c", "pressure_pa", "use_for_fit", "notes"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"Measurement CSV must have columns: {', '.join(sorted(required))}")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("wrong number of CSV fields")
                for field in ("measurement_id", "fingering_id", "mode", "target_midi",
                              "frequency_hz", "temperature_c", "use_for_fit"):
                    if not row[field].strip():
                        raise ValueError(f"{field} is required")
                identifier = row["measurement_id"].strip()
                fingering = row["fingering_id"].strip()
                if not identifier or fingering not in model.fingerings:
                    raise ValueError("missing measurement ID or unknown fingering")
                mode = int(row["mode"])
                if mode < 1:
                    raise ValueError("mode must be a positive integer")
                target_midi = int(row["target_midi"])
                target_frequency(target_midi, model.a4_hz)
                frequency = number(float(row["frequency_hz"]), "frequency_hz", strictly_positive=True)
                temperature = float(row["temperature_c"])
                sound_speed(temperature)
                pressure = (number(float(row["pressure_pa"]), "pressure_pa", minimum=0)
                            if row["pressure_pa"].strip() else None)
                flag = row["use_for_fit"].strip().lower()
                if flag not in ("true", "false"):
                    raise ValueError("use_for_fit must be true or false")
                result.append(Measurement(identifier, fingering, mode, frequency,
                                          temperature, pressure, flag == "true", target_midi))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error
    if not result:
        raise ValueError("No measured pitches: CSV is header-only")
    if len({m.id for m in result}) != len(result):
        raise ValueError("Measurement IDs must be unique")
    return result


def calibrate(model, measurements, lower_mm=0.0, upper_mm=150.0):
    lower_mm = number(lower_mm, "lower_mm", minimum=0)
    upper_mm = number(upper_mm, "upper_mm", minimum=0)
    if upper_mm <= lower_mm:
        raise ValueError("Calibration upper bound must exceed lower bound")
    training = [m for m in measurements if m.use_for_fit]
    if not training:
        raise ValueError("No rows marked use_for_fit=true")

    def objective(correction):
        return sum(cents_difference(model.predict(m, correction), m.frequency_hz) ** 2
                   for m in training) / len(training)

    # Coarse bracket avoids relying on a globally unimodal objective.
    grid = [lower_mm + (upper_mm - lower_mm) * i / 12 for i in range(13)]
    scores = [objective(x) for x in grid]
    best = min(range(len(grid)), key=scores.__getitem__)
    a, b = grid[max(0, best - 1)], grid[min(12, best + 1)]
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = b - ratio * (b - a), a + ratio * (b - a)
    y1, y2 = objective(x1), objective(x2)
    for _ in range(28):
        if y1 < y2:
            b, x2, y2 = x2, x1, y1
            x1 = b - ratio * (b - a)
            y1 = objective(x1)
        else:
            a, x1, y1 = x1, x2, y2
            x2 = a + ratio * (b - a)
            y2 = objective(x2)
    candidates = [lower_mm, upper_mm, (a + b) / 2]
    correction = min(candidates, key=objective)
    residuals = []
    for measurement in measurements:
        prediction = model.predict(measurement, correction)
        target = target_frequency(measurement.target_midi, model.a4_hz)
        residuals.append({
            "measurement_id": measurement.id,
            "use_for_fit": measurement.use_for_fit,
            "measured_hz": measurement.frequency_hz,
            "target_midi": measurement.target_midi,
            "target_hz": target,
            "measured_tuning_cents": cents_difference(measurement.frequency_hz, target),
            "predicted_passive_hz": prediction,
            "residual_cents": cents_difference(prediction, measurement.frequency_hz),
            "pressure_pa": measurement.pressure_pa
        })
    held_out = [r for r in residuals if not r["use_for_fit"]]
    return {
        "head_correction_mm": correction,
        "fit_rmse_cents": math.sqrt(objective(correction)),
        "held_out_rmse_cents": (
            math.sqrt(sum(r["residual_cents"] ** 2 for r in held_out) / len(held_out))
            if held_out else None),
        "at_bound": abs(correction - lower_mm) < 0.001 or abs(correction - upper_mm) < 0.001,
        "warnings": [
            "Effective fitted correction can absorb jet, bend and geometry error; it is not a measured head dimension.",
            "Pressure is recorded but not modeled. Compare recordings at consistent blowing conditions.",
            *([] if held_out else ["No held-out recordings: fit quality is not independent validation."])
        ],
        "residuals": residuals
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "instrument_model.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory", help="Show source evidence and unresolved inputs")
    targets = sub.add_parser("targets", help="Equal-tempered F2..F3 reference; not hole positions")
    targets.add_argument("--temperature-c", type=float, default=20)
    resonances = sub.add_parser("resonances", help="Requires complete measured geometry")
    resonances.add_argument("--fingering", default="all_closed")
    resonances.add_argument("--temperature-c", type=float, default=20)
    resonances.add_argument("--cell-mm", type=float, default=5)
    fit = sub.add_parser("calibrate", help="Fit one effective head correction to recorded pitches")
    fit.add_argument("--measurements", type=Path, default=ROOT / "pitch_measurements.csv")
    fit.add_argument("--lower-mm", type=float, default=0)
    fit.add_argument("--upper-mm", type=float, default=150)
    fit.add_argument("--cell-mm", type=float, default=5)
    args = parser.parse_args()
    try:
        with args.config.open() as stream:
            config = json.load(stream)
        if args.command == "inventory":
            print(json.dumps({
                "tuning": config["tuning"],
                "physical_geometry_confirmed": config["physical_geometry_confirmed"],
                "tonehole_inventory_confirmed": config["tonehole_inventory_confirmed"],
                "missing_segment_inputs": [
                    f"{s['id']}.{key}" for s in config["segments"]
                    for key, value in s.items() if value is None
                ],
                "missing_boundary_inputs": [key for key in ("head_correction_mm", "bell_correction_mm")
                                            if config[key] is None],
                "evidence": config["evidence"]
            }, indent=2))
        elif args.command == "targets":
            speed = sound_speed(args.temperature_c)
            base = config["tuning"]["lowest_midi"]
            target_frequency(base, config["tuning"]["a4_hz"])
            if base + 12 > 127:
                raise ValueError("Lowest MIDI note must permit an octave within 0..127")
            print("note,midi,target_hz,ideal_half_wave_mm")
            for midi in range(base, base + 13):
                frequency = target_frequency(midi, config["tuning"]["a4_hz"])
                note = f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"
                print(f"{note},{midi},{frequency:.6f},{speed / (2 * frequency) * 1000:.3f}")
            print("Ideal half-wave lengths are NOT physical tonehole coordinates.", file=sys.stderr)
        else:
            model = AcousticModel(config, args.cell_mm)
            if args.command == "resonances":
                print(json.dumps({
                    "kind": "passive, lossless exploratory resonances; not played pitches",
                    "fingering": args.fingering,
                    "temperature_c": args.temperature_c,
                    "resonances_hz": model.resonances(args.fingering, args.temperature_c)
                }, indent=2))
            else:
                measurements = load_measurements(args.measurements, model)
                result = calibrate(model, measurements, args.lower_mm, args.upper_mm)
                print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
