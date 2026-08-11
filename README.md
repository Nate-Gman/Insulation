# INSULATRIX

**A single-file, physics-first digital twin of a near-zero heat transfer insulation system — plus a hypervelocity impact shield, a self-healing hull, and a live economic-impact estimator, all in one interactive pygame app.**

> Educational simulation. Not a real, manufactured, or commercially costed product. Every claim below is honestly scored by the physics actually computed at runtime — see [Honesty, by design](#honesty-by-design).

---

## What this is

`Insulatrix.py` is a ~3,200-line, dependency-light Python program that models:

1. **11 thermal-insulation formulations** — from plain PU foam up to **Insulatrix Pro**, a 6-stage combined stack (gradient spray → phase-change material → endothermic chemical absorber → thermoelectric engine → MLI vacuum jacket → a cosmetic aerogel-glitter topcoat) — simulated with real heat-transfer physics (Fourier conduction, Knudsen gas suppression, Rosseland radiative diffusion, PCM latent heat, TEG/Peltier active recovery).
2. **A parallel class-based scientific-model framework** (`SLayer`, `SInsulationStack`, `OptimizationFramework`) for steady-state and transient (finite-difference) analysis, independent of the pygame viewer.
3. **5 boundary-condition scenarios**, from a forge hot face down to a literal near-absolute-zero cold face (1 microkelvin) — and a live `--proof` report showing Insulatrix Pro holds **100.000% suppression / 0 W/m² transmitted** even there.
4. **A durability/weathering model** — rated service envelope, thermal-cycle fatigue, hydrophobic/UV/abrasion resistance, and a *computed* (not hard-coded) service-life estimate per formulation.
5. **An impact-shield pod** — a second "tab" modeling a spacecraft hull combining a Whipple shield, pulsed electromagnetic (eddy-current) debris deflection, and a self-healing auto-patching rear wall, wrapped around the same Insulatrix Pro thermal core.
6. **A live economic-impact estimator** (`--economics`) that computes real dollar savings from the actual simulated physics (not hand-typed numbers) across houses, refrigeration, space habitats, and polar research facilities.

All of it lives in **one file**, runs with no build step, and every headline number is backed by a `--selftest` you can run yourself.

## Quick start

```bash
pip install pygame numpy
python Insulatrix.py              # launch the interactive 3D app
python Insulatrix.py --selftest   # headless sanity check of every subsystem
```

Python 3.9+ recommended. `pygame` is optional at import time (the CLI report flags all work without a display), but required for the interactive viewer.

## The interactive app

Launch with no arguments to open a 3D viewer with two "tabs":

- **Thermal tab** (default): a cutaway panel showing the selected formulation's layer stack, live temperature gradient, and suppression/transmission over time.
- **Pod tab** (press `M`): a cutaway spacecraft hull with an animated hypervelocity-impact test sequence.

Press **`TAB`** / **`SHIFT+TAB`** at any time to step through **every individual model in the app — all 12 (11 formulations + the pod) — in one flat sequence**, wrapping around in both directions. A `MODEL n/12` indicator under the title shows where you are.

### Key controls

| Key | Thermal tab | Pod tab |
|---|---|---|
| `TAB` / `SHIFT+TAB` | Next / previous model (all 12) | same |
| `M` | Switch to pod tab | Switch to thermal tab |
| `1`–`9`, `0`, `C` | Select formulation (11 total) | — |
| `1`–`8` | — | Select threat scenario |
| `SPACE` / `B` | Toggle simulation | Launch impact test |
| `A` | Auto-cycle formulations | Auto-cycle threats (loops) |
| `T` | Auto-rotate camera | same |
| `P` | Toggle particle effects | same |
| `V` | Cycle boundary scenario (5) | — |
| `W` | Durability/weathering panel | — |
| `D` | Material diagram | — |
| `+` / `-` | Add/remove layers (thickness) | — |
| `[` / `]` | Hot-side temperature up/down | — |
| `;` / `'` | Cold-side temperature up/down | — |
| `BACKSPACE` | Reset simulation | Reset current test |
| `R` | Reset camera | same |
| `L` | Toggle labels | same |
| `X` | Cross-section cutaway | same |
| `I` | Full specification | same |
| `H` | Help overlay | same |
| Mouse drag / right-drag / wheel | Orbit / pan / zoom | same |
| `ESC` / `Q` | Quit | — |

## CLI reference

Every report below is generated live from the same physics engine the interactive app uses — nothing is pre-baked text.

| Flag | What it prints |
|---|---|
| `--selftest` | Headless build + physics + render + economics sanity check of every subsystem (exit code reflects pass/fail) |
| `--compare` | Side-by-side comparison table of all 11 formulations |
| `--sim APPROACH` | Detailed simulation report for one formulation |
| `--feasibility` | Full honest physics/feasibility report with performance tiers |
| `--durability` | Durability/weathering service-life report |
| `--proof` | Live proof that Insulatrix Pro achieves near-zero heat transfer at the absolute-zero-limit scenario |
| `--scenario NAME` | Apply a named boundary-condition preset to any report (see below) |
| `--T-hot K` / `--T-cold K` | Override boundary temperatures directly |
| `--pod-report` | Impact-shield assessment across 8 representative hypervelocity threats |
| `--pod-proof` | A/B proof that magnetic deflection works + step-by-step self-healing decompression race |
| `--economics` | Global economic-impact estimate (see [Economics](#economics) below) |
| `--sm-compare` | Scientific-model-framework comparison table |
| `--sm-sim APPROACH` | Scientific-model steady-state simulation for one stack |
| `--sm-transient APPROACH` | Transient (finite-difference) simulation, 3600s |
| `--sm-sweep-thickness APPROACH` | Sweep thickness vs. performance |
| `--sm-sweep-mli` | Sweep MLI reflective-layer count vs. performance |
| `--sm-config FILE` | Load a custom stack from a JSON config |
| `--export-obj` | Write OBJ+MTL 3D mesh files to `./export/` |
| `--export-csv FILE` | Export comparison/sweep results to CSV |
| `--export-config FILE` | Export a scientific-model stack config to JSON |

**Boundary scenarios** (`--scenario NAME`): `FORGE + CRYOGENIC` (default, 1000°C → -196°C), `FORGE + AMBIENT`, `HABITAT / DEEP SPACE` (25°C → 3K), `HABITAT / BOOMERANG EXTREME` (25°C → 1K), `FORGE + ABSOLUTE ZERO` (1000°C → 1 microkelvin).

Example:

```bash
python Insulatrix.py --scenario "absolute zero" --compare
python Insulatrix.py --pod-report
python Insulatrix.py --economics
```

## Physics models

- **Fourier conduction**: `q = -k_eff * dT/dx` through each layer
- **Knudsen gas conduction**: suppressed when pore size ≪ mean free path (`k_gas = k_0 / (1 + 2*beta*Kn)`)
- **Rosseland radiative diffusion**: `k_rad ≈ 16/3 * sigma * T³ * l_mfp`, suppressed 1/N by reflective MLI layers
- **Opacifier absorption**: SiC/TiO₂/carbon reduce the radiative mean free path
- **PCM latent heat**: absorption at constant phase-change temperature
- **Endothermic chemical storage**: dehydration reactions store heat as bond energy
- **Thermoelectric + Peltier**: Seebeck conversion of ΔT to electricity, partially re-pumped as active cooling
- **Whipple-shield fragmentation**: areal-density-ratio efficacy, debris-cloud dilution over a standoff gap
- **Eddy-current deflection**: Lenz's-law impulse, capped by a finite frame-reaction-impulse budget, zero for non-conductive threats
- **Choked-orifice decompression**: standard compressible critical-flow equation for cabin pressure vs. time during a hull breach

Every model is intentionally **simplified but honestly labeled** — same rigor level throughout, citations included in code comments, never presented as more precise than it is.

## Economics

`python Insulatrix.py --economics` computes real dollar figures from the actual simulated thermal performance (STANDARD FOAM baseline vs. INSULATRIX PRO), then layers on stated, cited macro assumptions (housing stock, energy prices, fridge counts, etc.) to reach a global total. The full methodology, category-by-category breakdown, and heavily-caveated valuation are in **[OVERVIEW.md](OVERVIEW.md)**. Headline result, as of this build:

- **~$825 billion/year** global full-adoption ceiling (buildings + refrigeration + space habitats + polar facilities)
- **~$11.8 trillion** illustrative capitalized value (7% perpetuity)

These are Fermi-estimate ceilings for an educational simulation, not a forecast — read the caveats printed with every run before quoting either number.

## Honesty, by design

This project's guiding rule, stated explicitly throughout its own output: **near-zero is achievable; true zero is not**, and every report says so. Notable examples the code itself surfaces:

- Plain MLI actually **outperforms** Insulatrix Pro for a dedicated ~90K cryogenic propellant tank — the economics module reports this rather than forcing a win.
- The impact-shield pod explicitly labels threats that exceed its self-healing patch rating as unsurvivable, instead of silently claiming success.
- `--proof` and `--pod-proof` close with an explicit caveat on what was (and wasn't) actually demonstrated.
- `--economics` prints its full list of assumptions and limitations every time it runs.

## Project layout

Everything lives in `Insulatrix.py`, organized into commented sections: physical constants → formulations → durability model → impact-shield model → economics model → 3D engine → geometry builders → thermal simulation → scientific-model framework → particle system → scene renderer → interactive app → self-test/reports → CLI entry point.

`Goal.md` is the original design brief this project grew from. `ReferenceCode/` holds unrelated sibling projects by the same author (other single-file pygame simulations) and is not part of this project.

## Requirements

- Python 3.9+
- `numpy`
- `pygame` (only required for the interactive viewer; all `--` report flags work without it)

## License

No license specified — personal/educational project.
