# Initial acoustic model: folded contrabass recorder clone

## Status and tuning

This is an original, exploratory acoustic model for the supplied instrument,
not a reproduction of a manufacturer's design. The target is **A4 = 440 Hz,
12-tone equal temperament**, with **F2 as the lowest sounding note**, confirmed
by the builder. F2 is 87.307057858 Hz; F3 is 174.614115717 Hz. These are sounding,
not octave-transposed written pitches.

There are currently **no verified blown pitches** in the measurement record.
Headjoint path lengths and cross-sectional depths are unknown. The executable
therefore produces a reference tuning table but deliberately refuses to predict
this instrument's resonances until geometry and hole inventory are completed.
No drilling coordinates have been established.

## What the supplied evidence establishes

| Evidence | Useful information | Qualification |
| --- | --- | --- |
| `trapezoid_4panel_box_assembly.obj` | 490 mm axial length; 70/78 mm outer square ends; 9 mm walls; derived 52/60 mm inner square ends | CAD, not an as-built measurement |
| `trapezoid_785_4panel_box_rabbeted.obj` | 785 mm axial length; 56/70 mm outer square ends; 9 mm walls; derived 38/52 mm inner square ends | CAD, not an as-built measurement |
| Matching SVG/DXF panels | Panel outlines corroborate intended construction dimensions | Mating widths 66.6/74.6 and 52.6/66.6 mm are not bore dimensions |
| Gasket SVGs | 51.7 and 59.7 mm square apertures | Compressed thickness, installation and local restriction remain unknown |
| `front_dimensions.svg`, `back_dimenstions.svg`, `combined.svg`, `IMG_0849.png` | Annotated 1520 mm exterior height; 260 mm top-to-window-region dimension; several other external labels | Reference image; dimensions are not verified clone bore lengths or exact acoustic datums |
| `IMG_0838.png` | Longitudinal view of open head, divider, corner blocks and ruler | Rough visible span 320-340 mm, not a measured acoustic limb length |
| `IMG_0850.png` | Removed cover and divider; confirms a folded internal passage rather than a straight head tube | Hidden/assembled interfaces remain to be measured |
| `IMG_0851.png` | Ruler across one visible passage | Approximate visible width 58-63 mm; perspective and ruler zero limit precision; depth unknown |
| `IMG_0731.png` | Mouthpiece and removable head components | Does not establish windway/window dimensions or assembled acoustic boundaries |

The photo ranges are rough visual readings, **not statistical confidence
intervals**, and are not inserted into solver geometry. All source filenames,
labels and qualifications are retained in [instrument_model.json](./instrument_model.json).
The CAD's 52 mm ends suggest a join between the two bodies. The provisional
ordered body path is 60 -> 52 mm over 490 mm, then 52 -> 38 mm over 785 mm,
but **assembly direction, overlap and actual bore must be verified**.

The two nominal body lengths sum to 1275 mm. That is not the total air path:
the folded head adds path, and joints can add or subtract axial length.
Neither the 1520 mm exterior height nor twice the visible head height is a
valid substitute for measuring that path.

## Model and coordinate convention

Use `x = 0` at the chosen acoustic window/labium datum, and measure `x` in mm
along the **unfolded passage centerline**, through the head's first limb, turn,
return limb, body sections and terminal outlet. The wind-supply mouthpiece and
windway are not automatically counted as resonator length: the window couples
the jet to the bore. Record exactly which physical edge defines the datum.

Input units are mm, Hz, degrees C and Pa. Internally lengths and areas are
converted to metres and square metres.

### 1. Target pitches and sanity-check length

For MIDI note number `m`:

```text
f_target(m) = 440 * 2^((m - 69)/12)
cents(actual, reference) = 1200 * log2(actual/reference)
c(T) = 331.3 * sqrt(1 + T/273.15) metres/second
L_ideal(m,T) = c(T) / (2*f_target(m))
```

The speed approximation is for dry air, restricted here to 0-40 C. Bore air
can be warmer and wetter than room air; breath humidity and temperature
gradients are not yet modeled.

At 20 C, F2 requires roughly **1.966 m of ideal open-open effective length**.
This is a sanity check, not a prediction of physical length for the tapered,
folded instrument. In particular, do not subtract ideal note lengths to obtain
hole coordinates.

### 2. Rectangular, tapered bore

Each measured section has length `l` and endpoint internal width/height.
Width and height vary linearly within a section:

```text
S(x) = w(x) * h(x)
(1/S) * d/dx(S * dp/dx) + k^2*p = 0
k = 2*pi*f/c(T)
```

This is the one-dimensional lossless Webster description. The implementation
approximates the taper with short constant-area cells, default maximum length
5 mm, retaining pressure and volume-flow continuity at area transitions.
It does not incorrectly assign one cylindrical matrix to a whole tapered bore.

For a constant-area cell, with upstream/downstream state `(p,U)` and the
`exp(i*omega*t)` convention:

```text
[p_up]   [ cos(kl)       i*rho*c/S * sin(kl) ] [p_down]
[U_up] = [ i*S/(rho*c) * sin(kl)   cos(kl)   ] [U_down]
```

The solver uses the real transformed flow `v = i*(rho*c/S_ref)*U`.
Density consequently cancels for these lossless resonance frequencies.
The cells are cascaded backwards from a pressure-release bell. Pressure
release at the window gives the scalar root condition `p_window(f) = 0`.
For a uniform tube with no corrections this recovers `f_n = n*c/(2L)`.
These are **passive resonance frequencies**, not guaranteed blown pitches.
Open-input flute-family instruments should not be modeled as a clarinet-style
closed-open quarter-wave pipe.

### 3. Boundary and fold corrections

Two explicit equivalent lengths terminate the measured geometry:

- `head_correction_mm`: constant-area extension at the window end.
- `bell_correction_mm`: constant-area extension at the terminal outlet.

They are initially `null`, not silently zero. A zero is a deliberate ideal
boundary assumption; a nonzero value must have a documented origin.
The familiar approximately `0.6*r` unflanged circular-pipe correction is only
a rough prior. A rectangular window, corner blocks, head turn and square outlet
do not inherit an exact circular end correction.

The fold is represented initially by equivalent one-dimensional passage
segments with measured centerline lengths and areas. Corner blocks, divider
ends, three-dimensional bend scattering and jet feedback require later
refinement. Keep physical path measurement separate from empirical head
correction; do not fit an unknown entire head as a fictitious end correction.
Split the turn into more segments if its cross section changes appreciably.
If the terminal opening is a side port with a closed tail, model that port and
tail explicitly rather than assuming an unobstructed axial open bell.

### 4. Toneholes and fingerings

For each hole record:

```json
{
  "id": "h1",
  "x_mm": 1400.0,
  "diameter_mm": 12.0,
  "chimney_mm": 9.0,
  "end_correction_mm": 8.0
}
```

**This example is invented schema data, not an instrument measurement or a
recommended location.** `x_mm` is the hole center on the unfolded coordinate.
The bore is split at every hole. Each fingering lists all open hole IDs;
unlisted holes are closed. Example: `"h1_open": ["h1"]`.

For a circular hole of area `S_h`, the initial open-hole load is:

```text
t_eff = chimney + combined inner/outer end correction
Z_open = i*omega*rho*t_eff/S_h
Y_open = 1/Z_open
p_up = p_down
U_up = U_down + Y_open*p
```

For a sealed hole, its chimney is a closed side cavity:

```text
Z_closed = -i*(rho*c/S_h)*cot(k*chimney)
```

This captures closed-hole compliance and finite open-hole inertance while
**retaining the downstream bore and all other holes**. It is better than simply
cutting the tube off at the first open hole, but is still a low-frequency model.
It omits tonehole series corrections, mutual radiation, pad cavities/leaks,
undercuts, noncircular holes and viscothermal/radiation losses.
The solver rejects searches reaching the first transverse rectangular bore
mode, a closed-hole stub pole, or `k*t_eff > 0.3` for any hole. Those checks
are applicability bounds, not a claim of accuracy inside them.

Hole position and diameter are coupled: one frequency does not identify both,
nor head geometry, bend correction and blowing-pressure shift simultaneously.
Start with independently measured dimensions and calibrate few parameters.
Before future position optimization, specify target fingerings, permitted
diameters, wall thickness, physical mounting/key constraints and the complete
register to be tuned.

## Preparing geometry and running

No third-party Python dependencies are needed. On this machine `python3`
is available; the previously suggested Desktop virtual environment is absent.

```sh
cd /Users/chrisarehart/Documents/Paetzold
python3 acoustic_model.py inventory
python3 acoustic_model.py targets --temperature-c 20
python3 -m unittest -v test_acoustic_model
```

To enable predictions, update [instrument_model.json](./instrument_model.json):

1. Measure and enter all head segment lengths and internal width/depth endpoints.
   Add segments for restrictions, gaskets and joints where appropriate.
2. Verify the ordered body dimensions, orientation and assembled lengths,
   including joint overlap. The current numeric body sections are hypotheses.
3. Record every existing hole and the terminal outlet geometry. If the body
   truly has no holes, explicitly confirm that fact.
4. Enter explicit head and bell correction assumptions, initially zero only
   if intentionally testing ideal boundaries.
5. Set both confirmation flags to `true` only after the above inventory.

Then:

```sh
python3 acoustic_model.py resonances --fingering all_closed --temperature-c 20
python3 acoustic_model.py resonances --fingering all_closed --cell-mm 2.5
```

Compare 5 and 2.5 mm meshes. Numerical convergence does not validate geometry
or excitation physics. Discontinuities are allowed but should be documented,
not automatically smoothed away.

## Recording and calibrating verified pitches

Enter actual recordings in [pitch_measurements.csv](./pitch_measurements.csv).
It intentionally contains only headers: no pitch observations are fabricated.

| Field | Meaning |
| --- | --- |
| `measurement_id` | Unique recording ID |
| `fingering_id` | ID in the model's fingering map |
| `mode` | Positive resonance index; normally 1 for first-register readings, not blindly the nearest harmonic |
| `target_midi` | Intended equal-tempered sounding pitch: 41 for F2, 53 for F3; explicitly specified per recording/mode |
| `frequency_hz` | Verified sounding fundamental, not a tuner-selected overtone |
| `temperature_c` | Measured representative bore-air temperature; document ambient-only readings in notes |
| `pressure_pa` | Mouth pressure if measured; may be blank, never silently imputed |
| `use_for_fit` | `true` for calibration or `false` for independent validation |
| `notes` | Player, take, date, microphone/tuner method, duration, humidity, assembly state and uncertainty |

Minimum recording procedure:

1. Fix the assembled head setting and verify airtight joints/pads.
2. Warm the instrument consistently. Record temperature and, where possible,
   pressure; distinguish steady playing from attack and overblowing.
3. Confirm the fundamental using a spectrum/audio recording as well as a tuner.
4. Make at least three sustained independent recordings of all-closed F2.
   Retain repeats instead of retaining only the best-looking tuner result.
5. Once available, record several known fingerings and higher resonances.
   Record soft/normal/firm pressure separately to expose pressure-induced pitch
   shifts, rather than pooling them as geometry evidence.
6. Mark independent later takes `false` for validation **before fitting**.
   If several rows come from the same take/session, keep the group together.

The initial calibration fits **only one nonnegative effective head correction**,
with geometry, bell correction and toneholes held fixed:

```text
minimize mean[1200*log2(f_passive(theta)/f_measured)]^2 over fit rows
```

```sh
python3 acoustic_model.py calibrate --lower-mm 0 --upper-mm 150
```

This uses measured temperatures per recording. Outputs are JSON: fitted
correction, signed model residuals, measured tuning error relative to the
440 Hz equal-tempered target, fit RMSE in cents, held-out RMSE when available,
pressure metadata and whether the fit hit a bound. Calibration fits measured
frequencies, not target frequencies; otherwise an out-of-tune instrument would
be mistakenly treated as in tune. No held-out readings means
validation RMSE is `null`, not zero. Missing frequency/temperature, unknown
fingerings, malformed records and empty data produce explicit errors.
The resonance index used in fitting is counted from the first root in the
20-500 Hz search band; this assumes the fundamental is above 20 Hz.

The command reports a candidate calibration but does not overwrite geometry
or observations. After inspecting residuals and independent validation, record
the accepted correction in the configuration with the recording IDs and date
in an additional metadata field. Archive the raw measurements unchanged.
An optimum on a bound flags a possibly wrong geometry/boundary/pressure
assumption; do not simply enlarge the bound and declare success.

With only one all-closed frequency, the correction may be fit exactly but is
not independently validated and cannot establish tonehole positions. Systematic
residuals by fingering, mode or pressure mean the model needs refinement, not
automatic drilling.

## Verification and next milestone

The tests compare the engine to analytic uniform open-open tube resonances,
explicit end lengths and temperature scaling; test taper mesh convergence,
hole splitting at arbitrary positions and joins, closed-hole compliance,
open-hole pitch raising, known-parameter recovery from analytic synthetic
recordings, held-out exclusion and missing-input failure.
All synthetic test data are clearly separated from the empty measurement log.

Next milestone: a completed unfolded head/body survey plus repeated measured
all-closed F2 readings at known conditions. Then check calibrated first and
second modes and measured temporary-hole/fingering responses. Only after that
should an inverse solver propose position/diameter combinations with geometry
and pitch uncertainty, followed by reversible tests on a sacrificial section.

## Public physics references

- UNSW, [Flute acoustics](https://newt.phys.unsw.edu.au/jw/fluteacoustics.html):
  open-input resonator, pressure nodes, end corrections, jet feedback,
  fingerholes and difference between passive resonances and playing frequency.
- UNSW, [Introduction to woodwind acoustics](https://newt.phys.unsw.edu.au/jw/woodwind.html):
  open-open versus reed-instrument boundaries, effective length and harmonics.

The transfer and shunt equations above are stated explicitly so assumptions
can be checked; they are not claimed to be a verified manufacturer-specific
Paetzold head model. Respect the source drawing's ownership/distribution notice.
