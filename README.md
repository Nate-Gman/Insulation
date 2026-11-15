# INSULATRIX

**A single-file, physics-first digital twin of near-zero heat transfer insulation systems — plus a hypervelocity impact shield, a self-healing hull, full fabrication protocols, and a live economic-impact estimator, all in one interactive pygame app.**

> Educational simulation. Not a real, manufactured, or commercially costed product. Every claim below is honestly scored by the physics actually computed at runtime — see [Honesty, by design](#honesty-by-design).

---

## What this is

`Insulatrix.py` is a ~4,900-line, dependency-light Python program that models:

1. **12 thermal-insulation formulations** — from plain PU foam up to **ULTRA CASCADE**, a 35-layer, ~246 mm, six-zone optimized stack (cascaded refractory→aerogel spray passes → evacuated carbon-MLI hot shields → vacuum-microsphere foam + VIP core → staged Bi₂Te₃ TEG energy engines → dual Kapton-film vacuum MLI jackets → low-e cold cap + ceramifiable weather seal) — simulated with real heat-transfer physics (Fourier conduction, Knudsen gas suppression, Rosseland radiative diffusion, PCM latent heat, TEG/Peltier active recovery). ULTRA is the optimized sprayable layering built per `goal2.md`'s PerfectVoid recipe on top of `ReferenceCode/RTP2.0.py`'s `_spray_pass` architecture, and verified by `--ultra-proof` at **99.997% suppression, 4.49 W/m² transmitted, R=283.7 m²K/W** under forge+cryogenic boundary conditions — near-zero, honestly, not literal zero.
2. **A parallel class-based scientific-model framework** (`SLayer`, `SInsulationStack`, `OptimizationFramework`) for steady-state and transient analysis, independent of the pygame viewer — including an **implicit backward-Euler finite-volume solver** (`_implicit_transient`) used as the independent cross-check in the proof (explicit FDM is impractical at sub-mm coatings).
3. **5 boundary-condition scenarios**, from a forge hot face down to a literal near-absolute-zero cold face (1 microkelvin) — ULTRA CASCADE passes all five at ≥99.98% suppression, and Insulatrix Pro holds **100.000% suppression / 0 W/m² transmitted** at the absolute-zero limit via `--proof`.
4. **A durability/weathering model** — rated service envelope, thermal-cycle fatigue, hydrophobic/UV/abrasion resistance, and a *computed* (not hard-coded) service-life estimate per formulation.
5. **Detailed fabrication protocols** — a `FABRICATION` spec per formulation: substrate prep, slurry solids-loading recipes, spray parameters (nozzle, pressure, standoff, overlap, flash times), cure/fire-in schedules, vacuum-bagging and leak-test procedures, TEG integration, QC gates, and safety notes — all keyed to the same layer data the thermal model uses. In-app on `F`, on the CLI via `--fabrication`.
6. **An impact-shield pod** — a second "tab" modeling a spacecraft hull combining a Whipple shield, pulsed electromagnetic (eddy-current) debris deflection, and a self-healing auto-patching rear wall, wrapped around the same Insulatrix Pro thermal core.
7. **A live economic-impact estimator** (`--economics`) that computes real dollar savings from the actual simulated physics (not hand-typed numbers) across houses, refrigeration, space habitats, and polar research facilities.

All of it lives in **one file**, runs with no build step, and every headline number is backed by a `--selftest` you can run yourself.

## Quick start

```bash
pip install pygame numpy
python Insulatrix.py              # launch the interactive 3D app
python Insulatrix.py --selftest   # headless sanity check of every subsystem
python Insulatrix.py --ultra-proof  # the 8-step ULTRA CASCADE stacking proof
```

Python 3.9+ recommended. `pygame` is optional at import time (the CLI report flags all work without a display), but required for the interactive viewer.

## The interactive app

Launch with no arguments to open a 3D viewer with two "tabs":

- **Thermal tab** (default): a cutaway panel showing the selected formulation's layer stack, live temperature gradient, and suppression/transmission over time.
- **Pod tab** (press `M`): a cutaway spacecraft hull with an animated hypervelocity-impact test sequence.

Press **`TAB`** / **`SHIFT+TAB`** at any time to step through **every individual model in the app — all 13 (12 formulations + the pod) — in one flat sequence**, wrapping around in both directions. A `MODEL n/13` indicator under the title shows where you are.

### What's on screen (thermal tab)

- **Stats panel**: live q_incoming / q_transmitted, suppression %, k_eff overall, PCM absorbed, TEG converted, Peltier pumped — plus a **STACK PHYSICS** column (R_total, U-value, areal mass, areal heat capacity, lumped τ, structure counts, live T_max-violation count).
- **Layer Breakdown**: every layer in the stack (all 35 for ULTRA) — scrollable, zone-grouped, with per-layer thickness, R contribution, T_in→T_out, and flags (VAC, foil count, PCM %, over-T_max warnings).
- **Hover tooltips**: hovering a layer row opens a full detail card — all material DB fields, live k_s+g / k_rad / k_eff split, R, T profile, T_max check, the material's literature reference, and the layer's `ROLE:` annotation explaining why it sits where it sits.
- **3D cross-highlighting**: the hovered (or pinned) layer's slab tints and outlines in the cutaway view; **click a row to pin it** (card stays open, `J`/`K` step it through the stack).
- **Suppression/transmission time plot**, temperature-gradient bar with numeric endpoints, formulation list with live suppression % per entry.

### Overlay surfaces

| Key | Panel | Contents |
|---|---|---|
| `N` | **Physics Inspector** | Live metrics, conduction-mode split, T_max service check, stacking synergy (S_ordering/S_material/S_gradient/S_total vs. the scientific twin at live BCs), scenario sweep across all 5 boundaries, full per-layer table, three live charts (T(x) profile, log-R per layer, log-k_eff per layer), energy-fate bar, materials-DB dump |
| `O` | **Live Proof Checklist** | 7 checks evaluated live (suppression, finite R, synergy>1, T_max, all scenarios, implicit-FV transient cold-flux, finite results) → PROVEN/NOT PROVEN, with the honest caveat |
| `F` | **Fabrication Protocol** | Full step-by-step build spec for the active formulation — equipment, materials, numbered phases, QC gates, safety (scrollable) |
| `I` | **Full Specification** | About, physics models, every formulation's complete recipe card + per-layer role annotations, durability data, controls |
| `W` | **Durability panel** | Service-life model at the live boundary condition |
| `D` | **Material diagram** | Reaction schematic + deep mechanism text for the active formulation |
| `H` | Help | Key reference |

### Key controls

| Key | Thermal tab | Pod tab |
|---|---|---|
| `TAB` / `SHIFT+TAB` | Next / previous model (all 13) | same |
| `M` | Switch to pod tab | Switch to thermal tab |
| `1`–`9`, `0`, `C`, `U` | Select formulation (12 total; `U`=ULTRA CASCADE) | — |
| `1`–`8` | — | Select threat scenario |
| `SPACE` / `B` | Toggle simulation | Launch impact test |
| `A` | Auto-cycle formulations | Auto-cycle threats (loops) |
| `T` | Auto-rotate camera | same |
| `P` | Toggle particle effects | same |
| `V` | Cycle boundary scenario (5) | — |
| `N` / `O` / `F` | Inspector / proof checklist / fabrication | — |
| `E` | Export live layer table to CSV | — |
| `F12` | Save screenshot PNG | same |
| `J` / `K` | Step pinned layer through stack | — |
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
| Mouse drag / right-drag / wheel | Orbit / pan / zoom (wheel also scrolls panels) | same |
| `ESC` / `Q` | Quit | — |

## CLI reference

Every report below is generated live from the same physics engine the interactive app uses — nothing is pre-baked text.

| Flag | What it prints |
|---|---|
| `--selftest` | Headless build + physics + render + economics sanity check of every subsystem (exit code reflects pass/fail) |
| `--compare` | Side-by-side comparison table of all 12 formulations |
| `--sim APPROACH` | Detailed simulation report for one formulation |
| `--feasibility` | Full honest physics/feasibility report with performance tiers |
| `--durability` | Durability/weathering service-life report |
| `--fabrication [APPROACH]` | Step-by-step build protocol (spray params, cures, vacuum steps, QC, safety; default `ultra_cascade`) |
| `--proof` | Live proof that Insulatrix Pro achieves near-zero heat transfer at the absolute-zero-limit scenario |
| `--ultra-proof` | Live 8-step proof for ULTRA CASCADE: per-layer attenuation, stacking synergy (ordering/material/gradient), MLI 1/N law, all-scenario validation, T_max feasibility, implicit-FV transient cross-check |
| `--scenario NAME` | Apply a named boundary-condition preset to any report (see below) |
| `--T-hot K` / `--T-cold K` | Override boundary temperatures directly |
| `--pod-report` | Impact-shield assessment across 8 representative hypervelocity threats |
| `--pod-proof` | A/B proof that magnetic deflection works + step-by-step self-healing decompression race |
| `--economics` | Global economic-impact estimate (see [Economics](#economics) below) |
| `--sm-compare` | Scientific-model-framework comparison table |
| `--sm-sim APPROACH` | Scientific-model steady-state simulation for one stack |
| `--sm-transient APPROACH` | Transient (explicit finite-difference) simulation; per-layer adaptive grid keeps sub-mm coatings cheap |
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
python Insulatrix.py --ultra-proof
python Insulatrix.py --fabrication          # full ULTRA build spec
python Insulatrix.py --pod-report
python Insulatrix.py --economics
```

## ULTRA CASCADE — the optimized formulation

Formulation 12 is the deliverable of `goal2.md`'s brief ("sprayable layering for near-zero heat loss via accumulative stacking") built on `ReferenceCode/RTP2.0.py`'s spray-pass architecture:

- **Zone A — 4 cascaded spray passes** (5 sublayers × 2 mm each): YSZ/Al₂O₃-ZrO₂ refractory-rich → zirconia-aerogel hybrid → SiC/C-opacified aerogel → pure zirconia aerogel → aerogel + embedded low-e foil cap. Zirconia, not silica, aerogel — the hot face is ~1270 K, over silica's 923 K rating.
- **Zone B — 5 evacuated carbon-MLI hot shields** (60 refractory foils each) + deep jacket (80 foils): does the MLI job at ~900–1100 K where any polymer film fails.
- **Zone C — PerfectVoid vacuum-microsphere foam + fumed-silica VIP**: the goal2 recipe core (96% porosity, each microsphere carrying its own vacuum pocket).
- **Zone D — 2 staged Bi₂Te₃ TEG energy engines**: solid-state recovery — PCM dams were deliberately omitted because at ~600 K they exceed decomposition limits and a phase-pinned dam leaks against a ~0 K sink.
- **Zone E — 2 Kapton-film vacuum MLI jackets** (120 + 80 foils at 1e-6 torr): only feasible because zones A–D pulled local T under Kapton's 673 K limit.
- **Zone F — aerogel-MLI cold cap + TiN low-e topcoat + ceramifiable weather seal**: graceful-degradation outer armor.

`--ultra-proof` verifies: **99.9971%** steady-state suppression, **R = 283.7 m²K/W**, k_eff ≈ 0.87 mW/mK, **0/35 layers over T_max**, stacking synergy **S_total = 1.43**, all 5 boundary scenarios ≥99.98%, and an independent implicit finite-volume transient that sees the cold face stay at ~7×10⁻¹⁶ W/m² after a simulated hour. Every layer also carries a `why` annotation (visible in tooltips, the info panel, and CSV export) explaining its role in the cascade.

## Physics models

- **Fourier conduction**: `q = -k_eff * dT/dx` through each layer (temperature-dependent k)
- **Knudsen gas conduction**: suppressed when pore size ≪ mean free path (`k_gas = k_0 / (1 + 2*beta*Kn)`)
- **Rosseland radiative diffusion**: `k_rad ≈ 16/3 * sigma * T³ * l_mfp`, suppressed 1/N by reflective MLI layers
- **Opacifier absorption**: SiC/TiO₂/carbon reduce the radiative mean free path
- **PCM latent heat**: absorption at constant phase-change temperature (finite-capacity thermal diode)
- **Endothermic chemical storage**: dehydration reactions store heat as bond energy
- **Thermoelectric + Peltier**: Seebeck conversion of ΔT to electricity, partially re-pumped as active cooling
- **Ordered-gradient synergy**: per-layer evaluation at the real T profile, not a uniform-T approximation
- **Implicit backward-Euler finite-volume transient**: the independent cross-check engine (unconditionally stable — sub-mm coatings don't force micro-timesteps)
- **Whipple-shield fragmentation**: areal-density-ratio efficacy, debris-cloud dilution over a standoff gap
- **Eddy-current deflection**: Lenz's-law impulse, capped by a finite frame-reaction-impulse budget, zero for non-conductive threats
- **Choked-orifice decompression**: standard compressible critical-flow equation for cabin pressure vs. time during a hull breach

Every model is intentionally **simplified but honestly labeled** — same rigor level throughout, citations included in code comments, never presented as more precise than it is.

## Economics

`python Insulatrix.py --economics` computes real dollar figures from the actual simulated thermal performance (STANDARD FOAM baseline vs. INSULATRIX PRO), then layers on stated, cited macro assumptions (housing stock, energy prices, fridge counts, etc.) to reach a global total. The full methodology, category-by-category breakdown, and heavily-caveated valuation are in **[OVERVIEW.md](OVERVIEW.md)**. Headline result, as of this build:

- **~$827 billion/year** global full-adoption ceiling (buildings + refrigeration + space habitats + polar facilities)
- **~$11.8 trillion** illustrative capitalized value (7% perpetuity)

These are Fermi-estimate ceilings for an educational simulation, not a forecast — read the caveats printed with every run before quoting either number.

## Honesty, by design

This project's guiding rule, stated explicitly throughout its own output: **near-zero is achievable; true zero is not**, and every report says so. Notable examples the code itself surfaces:

- ULTRA CASCADE's proof reports **99.9971%** — not 100% — and names residual conduction, vacuum retention, supports, and seams as the real-world parasitic paths it does not model.
- PCM dams were **removed** from ULTRA's design when the physics showed they exceeded T_max and a pinned dam would leak against a cryo sink — the honest answer beat the impressive-sounding one.
- Plain MLI actually **outperforms** Insulatrix Pro for a dedicated ~90K cryogenic propellant tank — the economics module reports this rather than forcing a win.
- The impact-shield pod explicitly labels threats that exceed its self-healing patch rating as unsurvivable, instead of silently claiming success.
- `--proof`, `--ultra-proof`, and `--pod-proof` close with an explicit caveat on what was (and wasn't) actually demonstrated.
- `--economics` prints its full list of assumptions and limitations every time it runs.

## Project layout

Everything lives in `Insulatrix.py`, organized into commented sections: physical constants → formulations (with per-layer `why` annotations) → fabrication protocols → durability model → impact-shield model → economics model → 3D engine → geometry builders → thermal simulation → scientific-model framework → particle system → scene renderer → interactive app → self-test/reports → CLI entry point.

`Goal.md` is the original design brief; `goal2.md` is the sprayable-insulation brief ULTRA CASCADE answers. `ReferenceCode/` holds the author's source references — notably `RTP2.0.py`, whose `_spray_pass` compositional-gradient architecture is the direct basis of ULTRA's Zone A.

## Requirements

- Python 3.9+
- `numpy`
- `pygame` (only required for the interactive viewer; all `--` report flags work without it)

## License

No license specified — personal/educational project.
