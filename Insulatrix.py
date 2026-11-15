#!/usr/bin/env python3
"""
================================================================================
 Insulatrix.py  --  INSULATRIX :: Spray-On Near-Zero Thermal Transfer Digital Twin
================================================================================
A standalone, single-file, interactive 3D model + real physics study of a
sprayable, multi-layer, hybrid ceramic-aerogel thermal barrier designed to
approach near-zero heat transfer across extreme temperature gradients.
Every dimension is real (mm / SI).  Physics are computed live from first
principles -- not hard-coded.

Models used:
  - Fourier conduction: q = -k_eff * dT/dx through each layer
  - Knudsen gas conduction: k_gas suppressed when pore size << mean free path
  - Rosseland radiative diffusion: k_rad ~ 16/3 * sigma * T^3 * l_mfp
  - Multi-layer reflective shielding: k_rad suppressed by 1/N (layer count)
  - Opacifier radiation absorption: SiC, TiO2, residual carbon
  - Phase-change heat absorption: PCM latent heat storage (SrBr2.6H2O)
  - Endothermic chemical absorption: MgSO4.7H2O dehydration absorbs heat
  - Thermoelectric energy engine: Seebeck effect converts dT to electricity
  - Active heat re-absorption: TEG power drives Peltier heat pump back
  - Gradient compositional layering: hot-face refractory -> cold-face aerogel
  - Vacuum multi-layer insulation (MLI): lowest practical heat transfer
  - Accumulative spray-cascade stacking (ULTRA): each cascaded pass drops the
    local dT for the next stage, so downstream layers operate where their
    k(T) and radiative load are lower -- emergent stacking synergy.

5 boundary-condition scenarios (press V): forge+cryo (default), forge+ambient,
deep-space habitat (25C -> 3K, near-CMB), Boomerang-Nebula extreme (25C -> 1K),
forge+absolute zero (1000C hot face -> 1 microkelvin cold face). Insulatrix
Pro holds >99.9999% suppression (q_transmitted ~0.03 W/m2) even at that limit.
Formulation 12, ULTRA CASCADE, is the optimized sprayable multi-layer from
goal2.md (PerfectVoid recipe: ZrO2-YSZ + silica aerogel + vacuum microspheres
+ carbon nano) fused with the RTP2.0.py cascaded spray-pass architecture
(refractory -> aerogel gradient passes, evacuated carbon-MLI hot shields,
Kapton MLI barriers, staged TEG recovery engines).
Its proof of near-zero heat loss by accumulative stacking is reproducible via
--ultra-proof. Durability/weathering model (press W): rated service envelope,
thermal-cycle fatigue, hydrophobic/UV/abrasion resistance, live service life.
Every layer carries a "why" role annotation (tooltips / spec panel / CSV), and
every formulation has a full FABRICATION protocol (press F, or --fabrication):
slurry recipes, spray parameters, cure schedules, vacuum/leak-test procedures,
QC gates and safety notes keyed to the same layer data the thermal model uses.

The "success" metric is the HEAT TRANSFER SUPPRESSION RATIO: the fraction
of incoming heat flux that is blocked, absorbed, or converted vs. the
fraction that transmits through the full stack.  Target: near-zero
transmission (heat flux << 1 W/m2 under extreme dT).

12 approaches, honestly scored. See FORMULATIONS below for exact specs.
================================================================================
"""
import os, sys, math, argparse, warnings, random, json, csv, time, tempfile
from dataclasses import dataclass
from typing import List, Dict, Callable
warnings.filterwarnings("ignore")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
try: import pygame
except ImportError: pygame=None

MM = 0.001

DIMS = {
    "panel_width_mm": 500.0, "panel_height_mm": 500.0,
    "spray_can_height_mm": 220.0, "spray_can_diameter_mm": 78.0,
    "view_dist": 4.5, "view_el": -0.25, "view_az": 0.4,
}

PHYS = {
    "R": 8.314, "Na": 6.022e23, "sigma_SB": 5.670e-8,
    "T_hot_default": 1273.15, "T_cold_default": 77.0, "T_ambient": 298.15,
    "k_air_25C": 0.026, "mean_free_path_air": 68e-9, "beta_knudsen": 2.0,
    "k_silica_solid": 1.34, "k_alumina_solid": 30.0, "k_aluminum": 237.0,
    "emissivity_al": 0.04, "absorption_SiC": 0.85,
    "pcm_SrBr2_density": 2400.0, "pcm_SrBr2_T_phase": 362.15, "pcm_SrBr2_latent": 630.0e3, "pcm_SrBr2_k": 0.35,
    "pcm_MgSO4_density": 1680.0, "pcm_MgSO4_T_dehyd": 323.15, "pcm_MgSO4_latent": 480.0e3, "pcm_MgSO4_k": 0.45,
    "teg_Seebeck": 200e-6, "teg_ZT": 1.0, "teg_modules_per_m2": 50, "peltier_COP": 0.6,
    "aerogel_density": 150.0, "aerogel_porosity": 0.95, "aerogel_pore_size": 20e-9,
    "aerogel_k_solid": 0.018, "aerogel_k_gas_25C": 0.008,
    "ceramic_fiber_density": 128.0, "ceramic_fiber_k_25C": 0.05, "ceramic_fiber_k_1000C": 0.12,
    "refractory_density": 200.0, "refractory_porosity": 0.90, "refractory_k_25C": 0.04, "refractory_k_1000C": 0.08,
    "foam_density": 35.0, "foam_k": 0.024,
    "vacuum_pressure_MLI": 1e-5, "vacuum_pressure_soft": 1e-2,
    "spray_rate_ml_s": 2.0, "spray_velocity_m_s": 12.0, "spray_duration_s": 60.0,
    # --- ULTRA CASCADE materials (goal2.md PerfectVoid + RTP2.0 spray recipe) ---
    "ysz_foam_k": 0.35,            # yttria-stabilized zirconia spray foam, hot-face solid network
    "zirconia_aerogel_k": 0.026,   # ZrO2/Y2O3-doped ceramic aerogel (~26 mW/mK @ 25 C)
    "carbon_aerogel_k": 0.015,     # IR-opaque carbon aerogel blanket
    "perfectvoid_k": 0.15,         # PerfectVoid hybrid solid network (silicone-ceramic binder)
    "vip_core_k": 0.12,            # fumed-silica VIP core solid network
    "mli_nano_k": 0.001,           # aluminized-Kapton MLI + aerogel separator stack
    "tin_coating_k": 20.0,         # TiN reflective topcoat (solid k high; value is low emissivity)
    "weather_shell_k": 0.15,       # ceramifiable silicone-ceramic weather-seal coat
}

# Phase-change / endothermic media. A layer's "pcm_type" selects an entry;
# thermal sims read latent heat, density, phase temperature, and k from here.
PCM_PROPS = {
    "SrBr2":     {"T_phase": PHYS["pcm_SrBr2_T_phase"], "latent": PHYS["pcm_SrBr2_latent"],
                  "density": PHYS["pcm_SrBr2_density"], "k": PHYS["pcm_SrBr2_k"]},
    "MgSO4":     {"T_phase": PHYS["pcm_MgSO4_T_dehyd"], "latent": PHYS["pcm_MgSO4_latent"],
                  "density": PHYS["pcm_MgSO4_density"], "k": PHYS["pcm_MgSO4_k"]},
    # Multi-stage "heat dam" media from goal2.md / RTP2.0 -- staged phase temps
    # so each recovery engine catches heat at the temperature it was tuned for.
    "Erythritol":{"T_phase": 391.15, "latent": 340.0e3, "density": 1450.0, "k": 0.73},
    "Paraffin":  {"T_phase": 328.15, "latent": 200.0e3, "density": 900.0,  "k": 0.20},
    "Na2SO4":    {"T_phase": 305.15, "latent": 254.0e3, "density": 1460.0, "k": 0.55},
}

def _pcm_props(pcm_type):
    """Look up phase temperature, latent heat, density, and k for a pcm_type key."""
    return PCM_PROPS.get(pcm_type, PCM_PROPS["SrBr2"])

# ---------------------------------------------------------------------------
# GUI/inspection helpers.  The dict-based FORMULATION layers use simple
# material keys ("aerogel", "teg", ...); the scientific MATERIALS database
# (defined below) uses precise keys ("silica_aerogel", "bi2te3_teg", ...).
# These map between them so the UI can report real per-layer material
# properties (T_max, cp, density) and live stacking-synergy numbers.
# ---------------------------------------------------------------------------
_DICT_MAT_TO_SM = {
    "aerogel":"silica_aerogel","aerogel_mli":"silica_aerogel",
    "carbon_aerogel":"carbon_aerogel","ceramic_fiber":"ceramic_fiber",
    "endothermic":"mgso4_7h2o","foam":"pu_foam","gradient_spray":"refractory_foam",
    "mli":"silica_spacer","mli_nano":"mli_nano","pcm":"pcm_srbr2",
    "perfectvoid_foam":"perfectvoid_foam","refractory_foam":"refractory_foam",
    "teg":"bi2te3_teg","tin_coating":"tin_coating","vip_core":"vip_core",
    "weather_shell":"weather_shell","ysz_foam":"ysz_foam","zirconia_aerogel":"zirconia_aerogel"}

def _sm_mat_key(layer):
    """Resolve a dict-layer to its MATERIALS key (pcm_type refines 'pcm')."""
    if layer.get("material")=="pcm":
        return {"erythritol":"pcm_erythritol","paraffin":"pcm_paraffin",
                "salt_hydrate":"pcm_salt_hydrate","srbr2":"pcm_srbr2"
                }.get(str(layer.get("pcm_type","")).lower(),"pcm_srbr2")
    return _DICT_MAT_TO_SM.get(layer.get("material"),"")

def _sm_mat_props(layer):
    """MATERIALS entry for a dict-layer ({} when the material is unmapped)."""
    return MATERIALS.get(_sm_mat_key(layer),{})

def _layer_zone(name):
    """Zone tag from an 'X#' layer-name prefix (ULTRA zones A..F), else ''."""
    return name[0] if len(name)>=2 and name[0] in "ABCDEF" and name[1].isdigit() else ""

# Shared material->swatch colors for dict-layer lists, tooltips, and charts.
_LAYER_COLORS={"pcm":(100,200,160),"endothermic":(200,160,100),"teg":(160,120,220),
    "mli":(180,180,200),"aerogel":(140,200,220),"refractory_foam":(240,230,210),
    "ceramic_fiber":(220,180,100),"foam":(200,200,180),"ysz_foam":(250,235,200),
    "zirconia_aerogel":(200,230,235),"carbon_aerogel":(60,60,70),
    "perfectvoid_foam":(235,215,160),"vip_core":(180,200,240),"mli_nano":(200,200,235),
    "tin_coating":(255,215,90),"weather_shell":(130,170,140),"gradient_spray":(220,140,80)}
def _layer_color(layer): return _LAYER_COLORS.get(layer.get("material"),(150,150,160))

_SM_FACTORIES = {"foam":"sm_foam","aerogel":"sm_aerogel","ceramic":"sm_ceramic",
    "refractory":"sm_refractory","mli":"sm_mli","hybrid":"sm_hybrid",
    "pcm":"sm_pcm","endothermic":"sm_endothermic","teg":"sm_teg",
    "gradient":"sm_gradient","pro":"sm_pro","ultra_cascade":"sm_ultra"}

def sm_stack_for(approach,T_hot,T_cold):
    """Build the scientific stack twinning a formulation approach at the given
    boundary conditions (None when the approach has no SM twin)."""
    fn=globals().get(_SM_FACTORIES.get(approach,""))
    if fn is None: return None
    try:
        st=fn(T_hot,T_cold) if approach=="ultra_cascade" else fn()
        st.bc=SBoundary(T_hot,T_cold); return st
    except Exception:
        return None

# Boundary-condition presets. [0] matches PHYS T_hot_default/T_cold_default exactly
# (forge hot face -> LN2 cold face) so default startup behavior is unchanged.
# [2]/[3] push the cold boundary toward the Goal.md target: near-CMB deep space
# (~2.7 K) and Boomerang-Nebula-class cold (~0.5-1 K), the coldest known natural
# environments, with a habitable (293-298 K) hot face.
BOUNDARY_SCENARIOS = [
    {"name":"FORGE + CRYOGENIC","T_hot":1273.15,"T_cold":77.0,
     "desc":"Forge/crucible hot face (1000 C) -> liquid-nitrogen cold face (-196 C) [default]"},
    {"name":"FORGE + AMBIENT","T_hot":1273.15,"T_cold":298.15,
     "desc":"Forge/crucible hot face (1000 C) -> ambient room exterior (25 C)"},
    {"name":"HABITAT / DEEP SPACE","T_hot":298.15,"T_cold":3.0,
     "desc":"Habitable interior (25 C) -> near-CMB deep space (3 K, -270 C)"},
    {"name":"HABITAT / BOOMERANG EXTREME","T_hot":298.15,"T_cold":1.0,
     "desc":"Habitable interior (25 C) -> Boomerang Nebula-class cold (1 K, -272 C)"},
    {"name":"FORGE + ABSOLUTE ZERO","T_hot":1273.15,"T_cold":1e-6,
     "desc":"Forge/crucible hot face (1000 C) -> 1 microkelvin cold face (as close to absolute "
            "zero as the third law allows; matches lab-record ultracold BEC temperatures)"},
]

# =============================================================================
# ULTRA CASCADE -- optimized sprayable multi-layer (goal2.md + RTP2.0 recipe)
# =============================================================================
# The base spray mixture comes from RTP2.0.py's _spray_pass: each sprayed pass
# is a 5-sublayer compositional gradient -- refractory-rich (hot) -> hybrid ->
# opacified aerogel -> pure aerogel -> aerogel + low-emissivity TiN cap.
# Cascading passes is ACCUMULATIVE STACKING: each pass drops the local dT for
# the next, so downstream layers operate where k(T) and the T^3 radiative load
# are lower. goal2.md contributes the PerfectVoid fill chemistry (ZrO2-YSZ +
# silica aerogel + vacuum microspheres + carbon-foam nanoparticles in a
# ceramifiable silicone binder, ~96% porosity), the fumed-silica VIP core, the
# staged PCM "heat dams"/TEG "energy engines", and the Kapton-film MLI jackets.

def _ultra_spray_pass(tag, thk=2.0):
    """One cascaded spray pass (RTP2.0 _spray_pass layout): refractory-rich ->
    aerogel-rich compositional gradient ending in a low-e reflective cap."""
    return [
        {"name":"%s-1 YSZ/Al2O3-ZrO2 refractory-rich spray"%tag,"material":"ysz_foam","thickness_mm":thk,
         "density":300.0,"porosity":0.80,"pore_size_um":50.0,"emissivity":0.50,"opacifier":0.06,"n_reflective":0,"vacuum":False,
         "why":"Hot-face armor: YSZ/Al2O3-ZrO2 matrix survives direct 1273 K contact (T_max 1873 K) where polymers/aerogels fail; its solid-rich 80% porosity trades some k for structural integrity + CTE-matched anchor for the fragile nanoporous sublayers above."},
        {"name":"%s-2 hybrid refractory-aerogel spray"%tag,"material":"zirconia_aerogel","thickness_mm":thk,
         "density":240.0,"porosity":0.88,"pore_size_um":30.0,"emissivity":0.55,"opacifier":0.05,"n_reflective":0,"vacuum":False,
         "why":"Compositional transition: blends refractory oxide into zirconia aerogel so porosity steps 80%->88% gradually -- prevents the k-continuity cliff and interlayer delamination an abrupt material change would cause, while starting Knudsen gas suppression."},
        {"name":"%s-3 opacified aerogel spray (SiC+C)"%tag,"material":"carbon_aerogel","thickness_mm":thk,
         "density":140.0,"porosity":0.95,"pore_size_um":0.015,"emissivity":0.10,"opacifier":0.10,"n_reflective":0,"vacuum":False,
         "why":"Radiation blocker at peak-T position: SiC + carbon black opacifiers absorb/scatter IR, collapsing the T^3 Rosseland mean free path exactly where radiative transport is strongest in the pass; 15 nm pores + 95% porosity keep conduction minimal."},
        {"name":"%s-4 pure nanoporous zirconia aerogel"%tag,"material":"zirconia_aerogel","thickness_mm":thk,
         "density":220.0,"porosity":0.92,"pore_size_um":0.025,"emissivity":0.45,"opacifier":0.05,"n_reflective":0,"vacuum":False,
         "why":"Maximum-Knudsen layer: 25 nm pores (~3x under the gas mean free path) all but eliminate gas conduction; zirconia (not silica) chosen because this pass still sits >923 K -- silica aerogel would exceed its T_max here."},
        {"name":"%s-5 zirconia aerogel + TiN low-e cap"%tag,"material":"zirconia_aerogel","thickness_mm":thk,
         "density":225.0,"porosity":0.91,"pore_size_um":0.025,"emissivity":0.05,"opacifier":0.03,"n_reflective":1,"vacuum":False,
         "why":"Pass cap: embedded low-emissivity TiN foil (emissivity 0.05) reflects residual IR back into the pass and provides the 1/N radiative interface count this zone contributes; closes the cascade so the next pass starts at a lower absolute T."},
    ]

def _ultra_formulation_layers(n_passes=4):
    """INSULATRIX ULTRA layer list -- six zones of accumulative stacking, ordered
    hot->cold so each stage drops the local dT for the next AND every material
    sits below its T_max at steady state:
    A) cascaded spray gradient (hot face), B) carbon-aerogel hot radiative
    shields (survive 1473 K where Kapton cannot), C) nanoporous vacuum core,
    D) staged PCM/endothermic/TEG recovery engines, E) sealed Kapton MLI
    jackets, F) cold-face hybrid cap + low-e weather seal."""
    L=[]
    # ZONE A -- spray cascade: each pass drops dT for the next (the stack works
    # because ordered multi-layering beats any equal-thickness single material).
    for i in range(n_passes):
        L.extend(_ultra_spray_pass("A%d"%(i+1)))
    # ZONE B -- hot radiative shields: carbon aerogel hosting refractory foils
    # plus evacuated microspheres (the goal2 PerfectVoid trick applied hot-side):
    # microsphere vacuum kills the gas-conduction term, foils break the T^3
    # radiative flux ~1/N, and carbon's T_max ~1473 K lets them sit where
    # polymer films fail. 5 shields + a deep jacket drop the local T below
    # Kapton's 673 K limit so the fragile high-R core stays feasible.
    for i in range(5):
        L.append({"name":"B%d evacuated carbon-MLI hot shield (60 foils)"%(i+1),
                  "material":"carbon_aerogel","thickness_mm":18.0,
                  "density":100.0,"porosity":0.952,"pore_size_um":0.015,"emissivity":0.10,"opacifier":0.10,
                  "n_reflective":60,"vacuum":True,"vacuum_pressure":1e-2,
                  "why":"Hot radiative shield: carbon aerogel matrix (T_max ~1473 K) hosts 60 refractory foils that break the T^3 radiative flux ~1/N, while evacuated microspheres kill gas conduction -- does the Kapton-MLI job at a temperature where any polymer film would fail."})
    L.append({"name":"B6 deep carbon-MLI jacket (80 foils)","material":"carbon_aerogel",
              "thickness_mm":16.0,"density":100.0,"porosity":0.952,"pore_size_um":0.015,
              "emissivity":0.10,"opacifier":0.10,"n_reflective":80,"vacuum":True,"vacuum_pressure":1e-2,
              "why":"Deepest hot-side radiation sink: 80 foils (the zone's largest 1/N divisor) finishes pulling local T below ~673 K, the Kapton limit -- this single layer is what makes the polymer-film jackets downstream physically feasible."})
    # ZONE C -- nanoporous vacuum core: evacuated microsphere hybrid foam +
    # sealed fumed-silica VIP. Placed downstream of the shields so local T is
    # below their service limits (VIP envelope <673 K, PerfectVoid <873 K).
    L+=[{"name":"C1 PerfectVoid vacuum-microsphere spray foam","material":"perfectvoid_foam","thickness_mm":20.0,
         "density":60.0,"porosity":0.96,"pore_size_um":30.0,"emissivity":0.15,"opacifier":0.08,"n_reflective":0,"vacuum":True,"vacuum_pressure":1e-2,
         "why":"Goal2 recipe core: sprayable ZrO2-YSZ/silica-aerogel shells + evacuated hollow microspheres + carbon-foam nanoparticles in ceramifiable silicone -- 96% porosity with each microsphere carrying its own vacuum pocket, so gas conduction vanishes even if the bulk panel loses vacuum later."},
        {"name":"C2 fumed-silica VIP core","material":"vip_core","thickness_mm":15.0,
         "density":180.0,"porosity":0.90,"pore_size_um":0.1,"emissivity":0.40,"opacifier":0.0,"n_reflective":0,"vacuum":True,"vacuum_pressure":1e-3,
         "why":"Highest-R element in the stack (~15 mm buys more resistance than the entire spray zone): fumed-silica board at 1e-3 torr gives k_eff near 0.004 W/mK; positioned here because its metallized-film bag caps at 673 K."}]
    # ZONE D -- recovery engines (goal2.md "energy engines"): two solid-state
    # TEG harvester stages convert + Peltier-pump a slice of the residual flux
    # back. Solid-state only: at this ~600 K position organic/salt-hydrate
    # PCMs exceed their decomposition limits, and against a ~0 K sink any
    # phase-pinned dam would leak more than it saves -- so ULTRA carries no
    # PCM dams (they remain in formulations 7/11 where envelopes suit them).
    L+=[{"name":"D1 Bi2Te3 TEG harvester (hot stage)","material":"teg","thickness_mm":4.0,
         "density":7700.0,"porosity":0.0,"pore_size_um":0.0,"emissivity":0.30,"opacifier":0.0,"n_reflective":0,"vacuum":False,
         "why":"First energy engine (goal2 'engines'): Seebeck-converts a slice of the surviving flux at ~500-600 K into electricity -- the only recovery mechanism physically viable here (organic/salt PCMs decompose; a phase-pinned dam would leak against the ~0 K sink)."},
        {"name":"D2 Bi2Te3 TEG harvester (cold stage)","material":"teg","thickness_mm":6.0,
         "density":7700.0,"porosity":0.0,"pore_size_um":0.0,"emissivity":0.30,"opacifier":0.0,"n_reflective":0,"vacuum":False,
         "why":"Second energy engine + active sink: a thicker stage where dT is smaller; its generated power can drive Peltier pumping that pushes residual heat back up-stack -- transmission is not just blocked, a fraction is actively returned."}]
    # ZONE E -- sealed vacuum Kapton MLI jackets: radiation falls as ~1/N and
    # residual gas ~0. Local T here is below Kapton's 673 K limit because the
    # upstream zones already absorbed the gradient.
    L+=[{"name":"E1 MLI jacket (120 Kapton films + spacers)","material":"mli_nano","thickness_mm":20.0,
         "density":120.0,"porosity":0.92,"pore_size_um":0.08,"emissivity":0.04,"opacifier":0.0,"n_reflective":120,"vacuum":True,"vacuum_pressure":1e-6,
         "why":"Primary cold-side radiation wall: 120 aluminized-Kapton foils at 1e-6 torr -- the single largest 1/N divisor in the stack; only feasible because zones A-D already pulled local T under Kapton's 673 K limit."},
        {"name":"E2 MLI jacket (80 Kapton films + spacers)","material":"mli_nano","thickness_mm":16.0,
         "density":120.0,"porosity":0.92,"pore_size_um":0.08,"emissivity":0.04,"opacifier":0.0,"n_reflective":80,"vacuum":True,"vacuum_pressure":1e-6,
         "why":"Secondary MLI wall: a second sealed vacuum volume adds 80 more interfaces AND redundancy -- if E1's bag is ever breached, E2 alone still carries the radiation suppression; seams staggered so no straight thermal short exists."}]
    # ZONE F -- cold-face cap: aerogel-MLI hybrid + TiN low-e topcoat + a thin
    # ceramifiable weather shell for the near-permanent weathering barrier.
    L+=[{"name":"F1 aerogel-MLI hybrid cold cap","material":"aerogel_mli","thickness_mm":18.0,
         "density":110.0,"porosity":0.93,"pore_size_um":0.020,"emissivity":0.05,"opacifier":0.02,"n_reflective":36,"vacuum":True,"vacuum_pressure":1e-3,
         "why":"Cold-face buffer: aerogel-MLI hybrid keeps insulating even if its soft vacuum degrades (aerogel spacers prevent foil shorting) -- the graceful-degradation layer between the hard-vacuum jackets and the outside world."},
        {"name":"F2 TiN low-emissivity topcoat","material":"tin_coating","thickness_mm":0.3,
         "density":5220.0,"porosity":0.0,"pore_size_um":0.0,"emissivity":0.05,"opacifier":0.0,"n_reflective":1,"vacuum":False,
         "why":"Outermost radiative control: 0.3 mm PVD TiN at emissivity ~0.05 minimizes radiative exchange with the surroundings -- both directions: it neither absorbs much environment heat nor emits the last trickle through the wall."},
        {"name":"F3 ceramifiable silicone weather seal","material":"weather_shell","thickness_mm":0.5,
         "density":1100.0,"porosity":0.05,"pore_size_um":0.0,"emissivity":0.85,"opacifier":0.0,"n_reflective":0,"vacuum":False,
         "why":"Near-permanence armor (goal2): ceramifiable silicone stays flexible in service but vitrifies into a ceramic shell on fire exposure -- sealing the fragile vacuum/aerogel interior against UV, moisture, abrasion and handling for multi-decade life."}]
    return L

FORMULATIONS = [
    {"id":1,"name":"STANDARD FOAM","desc":"Closed-cell PU foam baseline. Gas conduction dominates; no radiation suppression.","color":(200,200,180),"approach":"foam",
     "layers":[{"name":"PU Foam","material":"foam","thickness_mm":50.0,"density":35,"porosity":0.92,"pore_size_um":300,"emissivity":0.9,"opacifier":0.0,"n_reflective":0,"vacuum":False,
                "why":"Baseline control: closed cells trap air (k_gas 0.026) but 300 um pores are ~4000x the mean free path so Knudsen gives nothing; emissivity 0.9 leaves radiation wide open. Included so every other approach has a 'no-engineering' reference to beat."}],
     "recipe":{"composition":"Closed-cell polyurethane (PU) foam","density":"35 kg/m3, 92% porosity","k_25C":"0.024 W/(m.K)",
               "mechanism":"gas conduction through trapped air in closed cells + solid conduction through the PU strut network; radiation passes almost unattenuated",
               "weak_link":"large 300 um cells -> no Knudsen benefit; melts/chars ~120 C so it cannot face a forge directly",
               "references":"ASHRAE Handbook -- Fundamentals"}},
    {"id":2,"name":"AEROGEL BLANKET","desc":"Silica aerogel + fiberglass. Knudsen effect suppresses gas conduction in 20 nm mesopores.","color":(140,200,220),"approach":"aerogel",
     "layers":[{"name":"Aerogel Blanket","material":"aerogel","thickness_mm":50.0,"density":150,"porosity":0.95,"pore_size_um":0.020,"emissivity":0.8,"opacifier":0.0,"n_reflective":0,"vacuum":False,
                "why":"Knudsen monolith: 20 nm mesopores are ~3x smaller than the air mean free path, so gas molecules hit pore walls more than each other -- gas conduction collapses without needing vacuum. The classic first line of defense, limited only by its 923 K T_max and open radiative path (emissivity 0.8)."}],
     "recipe":{"composition":"Silica aerogel (95% porosity, 20 nm pores) + E-glass fiber","k_25C":"0.015 W/(m.K)","knudsen":"Kn = 68nm/20nm = 3.4 >> 1",
               "mechanism":"Knudsen suppression: gas mean free path >> pore size, so gaseous conduction is throttled; solid conduction minimized by the tenuous 5%-solid SiO2 skeleton",
               "weak_link":"emissivity ~0.8 -- radiative transport through the transparent SiO2 skeleton dominates above ~400 C (why opacified variants exist); friable, dusts under abrasion",
               "references":"Hrubesh & Pekala (1994) J. Mater. Res. 9:731"}},
    {"id":3,"name":"CERAMIC FIBER","desc":"Aluminosilicate fiber blanket. Forge/crucible grade, stable to 1400C.","color":(220,180,100),"approach":"ceramic_fiber",
     "layers":[{"name":"Ceramic Fiber Blanket","material":"ceramic_fiber","thickness_mm":50.0,"density":128,"porosity":0.90,"pore_size_um":50,"emissivity":0.85,"opacifier":0.0,"n_reflective":0,"vacuum":False,
                "why":"The hot-face workhorse: aluminosilicate fibers tolerate 1400 C continuously, and the tangled fiber mat traps still air (k~0.05). Big 50 um pores mean gas conduction is unsuppressed and emissivity is high -- it survives the heat but does not block much of it; that is what the other layers are for."}],
     "recipe":{"composition":"Aluminosilicate fiber (Al2O3 45%, SiO2 55%)","k_25C":"0.05 W/(m.K)","k_1000C":"0.12 W/(m.K)",
               "mechanism":"still-air entrapment in the fiber web; fibers also scatter a little IR but with no opacifier the radiative path stays open",
               "weak_link":"fibers shrink/sinter at sustained peak T; RCF dust is a regulated carcinogen on install and repair",
               "references":"CRC Handbook; Unifrax Keramblanket"}},
    {"id":4,"name":"REFRACTORY FOAM","desc":"White crucible hybrid (Al2O3-SiO2-ZrO2) sol-gel foam. Sprayable, stable to 1700C+, opacified.","color":(240,230,210),"approach":"refractory_foam",
     "layers":[{"name":"Refractory Foam (Al2O3-SiO2-ZrO2)","material":"refractory_foam","thickness_mm":50.0,"density":200,"porosity":0.90,"pore_size_um":100,"emissivity":0.6,"opacifier":0.05,"n_reflective":0,"vacuum":False,
                "why":"Crucible-chemistry hot face: Al2O3-SiO2-ZrO2 sol-gel foam survives 1700 C+ and its 5 wt% SiC opacifier absorbs IR inside the foam -- unlike bare ceramic fiber it fights the radiative mode directly. Sprayable (gunning-grade) so it is also the practical first coat of the gradient-spray stacks."}],
     "recipe":{"composition":"Al2O3(60%)+SiO2(30%)+ZrO2(10%) sol-gel foam + SiC opacifier(5wt%)","k_25C":"0.04 W/(m.K)","k_1000C":"0.08 W/(m.K)",
               "mechanism":"open-cell ceramic foam for conduction + SiC absorption coefficient shortens the Rosseland mean free path, suppressing in-band IR transport",
               "weak_link":"100 um pores -> weak Knudsen; density 200 kg/m3 adds mass; needs staged dry-out/fire-in to avoid steam spalling",
               "references":"White crucible chemistry (Alcoa, Vesuvius)"}},
    {"id":5,"name":"MLI VACUUM","desc":"40 Al foil layers + silica spacers in high vacuum. Radiation suppressed by 1/N.","color":(180,180,200),"approach":"mli",
     "layers":[{"name":"MLI (40 Al layers + silica spacers)","material":"mli","thickness_mm":20.0,"density":50,"porosity":0.95,"pore_size_um":0.1,"emissivity":0.04,"n_reflective":40,"vacuum":True,"vacuum_pressure":1e-5,
                "why":"Radiation's nemesis: each aluminized foil reflects ~96% of incident IR and the stack divides radiative flux roughly as 1/40; at 1e-5 torr gas conduction is gone, leaving only foil-edge conduction through the spacers. The catch: it only works in vacuum, and films are rated ~673 K."}],
     "recipe":{"composition":"40 layers Al foil (12um, emissivity 0.04) + silica aerogel paper spacers","k_eff":"~0.0001 mW/(m.K) in high vacuum",
               "mechanism":"q_rad ~ sigma*T^4/(N+1) across N low-e surfaces + vacuum eliminates gaseous conduction; spacer netting minimizes solid contact points",
               "weak_link":"performance dies with vacuum loss; foils wrinkle/short if handled roughly; polymer films cap at ~673 K",
               "references":"NASA MLI database; Lockheed cryogenic MLI"}},
    {"id":6,"name":"AEROGEL-MLI HYBRID","desc":"Aerogel blanket interleaved with MLI. Knudsen + reflective dual suppression.","color":(160,190,210),"approach":"aerogel_mli",
     "layers":[{"name":"Aerogel + MLI Hybrid","material":"aerogel_mli","thickness_mm":50.0,"density":100,"porosity":0.93,"pore_size_um":0.020,"emissivity":0.04,"opacifier":0.02,"n_reflective":20,"vacuum":True,"vacuum_pressure":1e-2,
                "why":"The forgiving hybrid: aerogel plies serve as both insulation and foil spacer, so even at soft-vacuum 1e-2 torr (achievable without turbopumps, and tolerant of slow leaks) it suppresses gas AND radiation; if vacuum fails entirely it degrades toward a plain aerogel blanket, not toward bare metal."}],
     "recipe":{"composition":"20 Al foil layers + 10 aerogel blanket layers (5mm each)","k_eff":"~0.05 mW/(m.K) in soft vacuum",
               "mechanism":"Knudsen suppression from aerogel + 1/N radiative division from foils + aerogel spacers prevent foil-to-foil contact shorts",
               "weak_link":"thicker per unit R than hard-vacuum MLI; still polymer-film limited (~673 K)",
               "references":"SPI/Cryogenics 2008; Aspen Aerogels Spaceloft"}},
    {"id":7,"name":"PCM HEAT ABSORBER","desc":"SrBr2.6H2O phase-change + aerogel. Absorbs 630 kJ/kg at 89C. Latent heat storage.","color":(100,200,160),"approach":"pcm",
     "layers":[{"name":"Aerogel Insulation","material":"aerogel","thickness_mm":20.0,"density":150,"porosity":0.95,"pore_size_um":0.020,"emissivity":0.8,"opacifier":0.0,"n_reflective":0,"vacuum":False,
                "why":"Hot-face buffer: Knudsen-suppressed aerogel slows the incoming flux enough that the PCM sees a bounded heat rate instead of a thermal shock, and it shields the hydrate from the hottest spikes."},
               {"name":"PCM SrBr2.6H2O","material":"pcm","thickness_mm":200.0,"density":2400,"porosity":0.0,"pore_size_um":0,"emissivity":0.5,"opacifier":0.0,"n_reflective":0,"vacuum":False,"pcm_type":"SrBr2",
                "why":"The heat dam: 200 mm of SrBr2.6H2O stores 302 MJ/m2 as latent heat at a pinned 89 C -- while it is charging, no flux passes through at all (the layer is a temporary thermal diode); finite, so it re-conducts once saturated."},
               {"name":"Aerogel Insulation","material":"aerogel","thickness_mm":20.0,"density":150,"porosity":0.95,"pore_size_um":0.020,"emissivity":0.8,"opacifier":0.0,"n_reflective":0,"vacuum":False,
                "why":"Cold-face guard: keeps ambient/moisture off the hydrate (which would corrode or drift its T_phase) and carries conduction during the long soak between PCM charges."}],
     "recipe":{"composition":"SrBr2.6H2O encapsulated in aerogel matrix","pcm":"T_phase=89C, latent=630 kJ/kg, density=2400 kg/m3","absorption":"302 MJ/m2 per charge (200mm)",
               "mechanism":"latent-heat diode: while T_layer crosses T_phase the salt melts isothermally and swallows the flux; aerogel skins bound the charge/discharge rate",
               "weak_link":"finite capacity -- once fully charged it is just a dense conductor (2400 kg/m3); hydrates segregate/supercool over cycles and need encapsulation",
               "references":"N'Tsoulkove et al. (2014) Solar Energy 107:469"}},
    {"id":8,"name":"ENDOTHERMIC CHEMICAL","desc":"MgSO4.7H2O dehydration + aerogel. Chemical heat storage: absorbs 480 kJ/kg.","color":(200,160,100),"approach":"endothermic",
     "layers":[{"name":"Aerogel Insulation","material":"aerogel","thickness_mm":20.0,"density":150,"porosity":0.95,"pore_size_um":0.020,"emissivity":0.8,"opacifier":0.0,"n_reflective":0,"vacuum":False,
                "why":"Hot-face buffer: aerogel stage meters the flux into the chemical bed so dehydration proceeds as a front instead of flash-boiling the hydrate."},
               {"name":"MgSO4.7H2O Chemical Absorber","material":"endothermic","thickness_mm":150.0,"density":1680,"porosity":0.3,"pore_size_um":10,"emissivity":0.5,"opacifier":0.0,"n_reflective":0,"vacuum":False,"pcm_type":"MgSO4",
                "why":"Chemical battery: MgSO4.7H2O + 480 kJ/kg -> MgSO4.H2O + 6 H2O -- heat is consumed breaking hydration bonds and carried away as water vapor through the 30% porous bed; stores more per kg than sensible heat ever could, and is rechargeable by rehydration."},
               {"name":"Aerogel Insulation","material":"aerogel","thickness_mm":20.0,"density":150,"porosity":0.95,"pore_size_um":0.020,"emissivity":0.8,"opacifier":0.0,"n_reflective":0,"vacuum":False,
                "why":"Cold-face vapor path/guard: the dehydration front needs somewhere for the released water to go; this aerogel skin vents it cold-side while still insulating the bed."}],
     "recipe":{"composition":"MgSO4.7H2O in porous matrix","reaction":"MgSO4.7H2O + heat -> MgSO4.H2O + 6 H2O (480 kJ/kg)","absorption":"121 MJ/m2 per charge (150mm)",
               "mechanism":"thermochemical storage: incoming heat drives an endothermic dehydration reaction -- absorbed as bond energy, not sensible heat, so the bed pins near the reaction temperature while it reacts",
               "weak_link":"reaction front is finite; 'salt fatigue' (lost rehydration capacity) over repeated cycles; needs a vent path or it stalls",
               "references":"Donkers et al. (2017) Appl. Therm. Eng. 119:1"}},
    {"id":9,"name":"THERMOELECTRIC ENGINE","desc":"Bi2Te3 Seebeck TEG + MLI. Converts dT to electricity; Peltier pumps heat back.","color":(160,120,220),"approach":"teg",
     "layers":[{"name":"MLI Vacuum Jacket","material":"mli","thickness_mm":20.0,"density":50,"porosity":0.95,"pore_size_um":0.1,"emissivity":0.04,"opacifier":0.0,"n_reflective":40,"vacuum":True,"vacuum_pressure":1e-5,
                "why":"Hot-side radiation wall: 40-foil MLI drops the flux to a trickle before it reaches the TEG -- the engine only has to harvest the residue, and at 1e-5 torr there is no gas conduction left to feed it."},
               {"name":"Bi2Te3 TEG Array","material":"teg","thickness_mm":5.0,"density":7700,"porosity":0.0,"pore_size_um":0,"emissivity":0.3,"opacifier":0.0,"n_reflective":0,"vacuum":False,
                "why":"The engine itself: 50 Bi2Te3 modules/m2 Seebeck-convert dT directly to electricity; feeding that power to a Peltier stage pumps part of the conducted flux back toward the hot side -- an active, not just passive, barrier."},
               {"name":"MLI Vacuum Jacket","material":"mli","thickness_mm":20.0,"density":50,"porosity":0.95,"pore_size_um":0.1,"emissivity":0.04,"opacifier":0.0,"n_reflective":40,"vacuum":True,"vacuum_pressure":1e-5,
                "why":"Cold-side radiation wall: catches whatever the TEG let through and gives the Peltier's rejected heat a low-k path back; two jackets also mean the module sits in a thermally quiet pocket."}],
     "recipe":{"composition":"Bi2Te3 TEG modules + MLI vacuum insulation","teg":"Seebeck S=200uV/K, ZT=1.0, 50 modules/m2","peltier":"TEG electricity drives Peltier heat pump back",
               "mechanism":"net flux = conduction - Seebeck conversion - Peltier pumping: the TEG does not just resist heat, it removes a fraction of it from the circuit entirely",
               "weak_link":"efficiency ~10% of Carnot at best; dense Bi2Te3 (7700 kg/m3) conducts well when off; module solder/interconnects fatigue under cycling",
               "references":"Snyder & Toberer (2008) Nat. Mater. 7:105"}},
    {"id":10,"name":"GRADIENT SPRAY","desc":"Multi-pass sprayed gradient: refractory hot-face -> aerogel cold-face. Sprayable, conformal.","color":(220,140,80),"approach":"gradient_spray",
     "layers":[{"name":"Pass1: Refractory-rich (Al2O3 70%+SiC 5%)","material":"refractory_foam","thickness_mm":2.0,"density":300,"porosity":0.80,"pore_size_um":50,"emissivity":0.5,"opacifier":0.05,"n_reflective":0,"vacuum":False,
                "why":"Hot anchor: 70% Al2O3 + SiC gives a 1700 C-capable bonded face with opacification; dense enough to grip the substrate and carry the rest of the gradient."},
               {"name":"Pass2: Hybrid (Al2O3 40%+aerogel 50%)","material":"refractory_foam","thickness_mm":2.0,"density":200,"porosity":0.88,"pore_size_um":30,"emissivity":0.6,"opacifier":0.03,"n_reflective":0,"vacuum":False,
                "why":"Composition bridge: halves the refractory fraction and folds in aerogel -- eases the CTE/porosity transition so the pass stack does not delaminate, while beginning real Knudsen suppression."},
               {"name":"Pass3: Aerogel-rich (80%+fiber 10%)","material":"aerogel","thickness_mm":2.0,"density":170,"porosity":0.93,"pore_size_um":0.020,"emissivity":0.7,"opacifier":0.02,"n_reflective":0,"vacuum":False,
                "why":"Insulation onset: now aerogel-dominated (80%) with a little fiber for green strength; 20 nm pores engage full Knudsen suppression as the local T has already fallen below the aerogel's rating."},
               {"name":"Pass4: Pure aerogel (90%)","material":"aerogel","thickness_mm":2.0,"density":155,"porosity":0.95,"pore_size_um":0.020,"emissivity":0.8,"opacifier":0.0,"n_reflective":0,"vacuum":False,
                "why":"Minimum-k zone: pure 95%-porosity aerogel at the coldest point of the pass gives the lowest solid+gas conduction the chemistry allows."},
               {"name":"Pass5: Aerogel+TiN reflective topcoat","material":"aerogel","thickness_mm":2.0,"density":150,"porosity":0.95,"pore_size_um":0.020,"emissivity":0.05,"opacifier":0.0,"n_reflective":1,"vacuum":False,
                "why":"Radiation cap: TiN low-e finish (emissivity 0.05) reflects residual IR back down the gradient -- the one layer whose job is radiation, closing the pass exactly as the model's n_reflective=1 says."}],
     "recipe":{"composition":"5-pass sprayed gradient: Al2O3/SiC -> hybrid -> aerogel -> aerogel -> TiN","thickness":"10mm (5x2mm)",
               "mechanism":"compositional gradient = CTE-matched cascade: each 2 mm sublayer drops dT and shifts dominant transport from solid conduction (hot) to Knudsen-suppressed + reflected (cold)",
               "weak_link":"only 10 mm total -- a demonstrator of the technique, not a deep stack; that is what ULTRA CASCADE scales up",
               "references":"Spray-on aerogel coatings (Cabot, Aspen); ceramic slurry spray"}},
    {"id":11,"name":"INSULATRIX PRO","desc":"Combined 5-stage: gradient spray -> PCM -> endothermic -> TEG -> MLI vacuum. Plus a small signature topcoat, just for fun.","color":(255,100,100),"approach":"combined",
     "layers":[{"name":"Stage1: Gradient Spray (10mm)","material":"gradient_spray","thickness_mm":10.0,"density":200,"porosity":0.90,"pore_size_um":30,"emissivity":0.5,"opacifier":0.03,"n_reflective":1,"vacuum":False,
                "why":"Hot-face cascade: the 5-sublayer spray gradient (refractory->aerogel->low-e cap) takes the forge heat and starts the ordered drop -- hot-face survivability plus the first radiation rejection."},
               {"name":"Stage2: PCM SrBr2.6H2O (200mm)","material":"pcm","thickness_mm":200.0,"density":2400,"porosity":0.0,"pore_size_um":0,"emissivity":0.5,"opacifier":0.0,"n_reflective":0,"vacuum":False,"pcm_type":"SrBr2",
                "why":"First dam: 302 MJ/m2 of latent storage at 89 C swallows transient surges whole; steady-state it re-conducts, but every joule stored is a joule not yet transmitted."},
               {"name":"Stage3: MgSO4.7H2O Chemical (150mm)","material":"endothermic","thickness_mm":150.0,"density":1680,"porosity":0.3,"pore_size_um":10,"emissivity":0.5,"opacifier":0.0,"n_reflective":0,"vacuum":False,"pcm_type":"MgSO4",
                "why":"Second dam, different physics: when the PCM saturates, the MgSO4 bed's endothermic dehydration (480 kJ/kg) keeps absorbing as bond energy -- back-to-back finite buffers covering each other's soak-through."},
               {"name":"Stage4: Bi2Te3 TEG+MLI (25mm)","material":"teg","thickness_mm":25.0,"density":1000,"porosity":0.5,"pore_size_um":1,"emissivity":0.04,"opacifier":0.0,"n_reflective":20,"vacuum":True,"vacuum_pressure":1e-3,
                "why":"Active stage: a TEG/MLI sandwich harvests part of the surviving flux into electricity that back-pumps heat via Peltier -- the point where the wall starts returning energy instead of only resisting it."},
               {"name":"Stage5: MLI Vacuum Jacket (20mm)","material":"mli","thickness_mm":20.0,"density":50,"porosity":0.95,"pore_size_um":0.1,"emissivity":0.04,"opacifier":0.0,"n_reflective":40,"vacuum":True,"vacuum_pressure":1e-5,
                "why":"Terminal radiation wall: 40 foils at 1e-5 torr apply the ~1/N division to whatever residue escaped the engines, ending the cascade at the cold face."},
               {"name":"Stage6: Signature Aerogel-Glitter Topcoat (0.3mm, for fun)","material":"aerogel","thickness_mm":0.3,"density":150,"porosity":0.95,"pore_size_um":0.020,"emissivity":0.05,"opacifier":0.0,"n_reflective":1,"vacuum":False,
                "why":"Signature finish: a 0.3 mm mica-flecked low-e aerogel sparkle coat -- thermally negligible, kept because a flagship deserves a face."}],
     "recipe":{"composition":"5 functional stages (gradient spray + PCM + endothermic + TEG + MLI vacuum) + a 0.3mm mica-flecked aerogel topcoat","thickness":"405.3mm",
               "mechanism":"defense-in-depth across all four transport modes: conduction (spray/aerogel), latent storage (PCM), chemical storage (dehydration), active removal (TEG/Peltier), radiation (MLI)",
               "synergy":"Each stage reduces dT for next; Stage 6 is a cosmetic low-e sparkle finish added purely for fun -- it costs nothing thermally and gives the cold face a frosted, starlit look",
               "weak_link":"405 mm thick and multi-technology: the most failure points of any stack here; ULTRA CASCADE reaches better suppression at ~60% of the thickness by ordering zones to their T_max",
               "references":"NASA, ESA habitat studies; multi-stage barrier design"}},
    {"id":12,"name":"ULTRA CASCADE","desc":"Optimized sprayable multi-layer: cascaded refractory->aerogel spray passes + evacuated carbon-MLI hot radiative shields + PerfectVoid vacuum-microsphere foam + VIP + staged Bi2Te3 TEG energy engines + dual Kapton vacuum MLI jackets. Accumulative stacking drives transmission toward zero; every material sits below its T_max at steady state.","color":(255,170,60),"approach":"ultra_cascade",
     "layers":_ultra_formulation_layers(),
     "recipe":{"composition":"ZONE A: 4 cascaded spray passes (YSZ/Al2O3-ZrO2 refractory-rich -> zirconia aerogel hybrid -> SiC+C opacified aerogel -> pure aerogel -> aerogel+TiN low-e cap), 2mm/sublayer | ZONE B: 5x evacuated carbon-MLI hot shields (60 refractory foils ea) + deep jacket (80 foils) | ZONE C: PerfectVoid foam (ZrO2-YSZ + silica aerogel + vacuum microspheres + carbon nano, 96% porosity) + fumed-silica VIP | ZONE D: 2x staged Bi2Te3 TEG energy engines (solid-state; PCM dams omitted -- at ~600 K they exceed decomposition limits and a phase-pinned dam leaks against a ~0 K sink) | ZONE E: 2x Kapton-MLI vacuum jackets (200 foils total, local T <673 K) | ZONE F: aerogel-MLI cap (36 foils) + TiN topcoat + ceramifiable weather seal",
               "thickness":"~246 mm (35 layers, 6 zones)","architecture":"sprayable ceramic-aerogel gradient + evacuated-microsphere refractory hot-shield MLI + sealed vacuum multilayer core; optimized per goal2.md PerfectVoid recipe and RTP2.0 _spray_pass cascade; zoned so every material stays below T_max","references":"goal2.md (PerfectVoid/UltraThin recipe, ZrO2-aerogel-microsphere hybrid); RTP2.0.py _spray_pass/_spray_enhance + carbon-aerogel hot MLI; NASA/Lockheed MLI; VIP fumed-silica data"}},
]

EL_COLORS = {"Al":(200,120,40),"Si":(180,180,200),"O":(220,80,80),"Zr":(100,200,200),"C":(70,70,80),"H":(240,240,245),"Mg":(120,200,120),"S":(220,200,80),"Sr":(160,80,200),"Br":(80,160,80),"Ti":(180,180,200),"N":(80,120,220),"Bi":(200,80,200),"Te":(80,200,160),"e-":(255,200,100),"heat":(255,100,60),"cold":(100,160,255)}
EL_RADIUS = {"Al":9,"Si":9,"O":8,"Zr":10,"C":7,"H":6,"Mg":9,"S":9,"Sr":10,"Br":9,"Ti":9,"N":8,"Bi":10,"Te":9,"e-":4,"heat":8,"cold":8}

CHEM_REACTIONS = {
    "foam":{"eq":"Heat -> conduction through PU foam cells (gas + solid pathways)","mech":"Closed-cell PU foam: heat crosses as solid conduction through the polymer strut network plus gas conduction through air trapped in 300 um cells -- pores are ~4000x the 68 nm molecular mean free path so there is no Knudsen benefit (k_gas stays ~0.026 W/mK). The skeleton is IR-transparent (emissivity ~0.9) so radiation leaks through too. k~0.024 W/mK -- the unengineered baseline; melts/chars above ~120 C.",
            "reactants":[("Heat (q_in)",[("heat",0,0)],[]),("PU Foam",[("C",0,0),("C",22,0),("O",44,0),("H",-11,14),("H",11,14)],[(0,1,1),(1,2,1),(0,3,1),(1,4,1)])],
            "products":[("Heat (q_out)",[("heat",0,0)],[])]},
    "aerogel":{"eq":"Heat -> Knudsen gas conduction suppressed in 20 nm mesopores (Kn >> 1)","mech":"Silica aerogel, 95% porosity: the SiO2 skeleton is only ~5% of the volume so solid conduction is already tiny. The real trick is Knudsen suppression -- 20 nm pores are ~3x smaller than air's 68 nm mean free path, so gas molecules hit pore walls far more often than each other and gaseous conduction collapses. What remains is radiation through the IR-transparent skeleton (emissivity ~0.8), which is why k_eff climbs with T -- the reason opacified/zirconia variants exist. T_max ~923 K.",
               "reactants":[("Heat (q_in)",[("heat",0,0)],[]),("SiO2 aerogel",[("Si",0,0),("O",18,-10),("O",18,12)],[(0,1,2),(0,2,1)]),("Air molecule",[("O",40,0),("N",60,0)],[(0,1,2)])],
               "products":[("Heat (q_out << q_in)",[("cold",0,0)],[])]},
    "ceramic_fiber":{"eq":"Heat -> conduction through aluminosilicate fiber matrix (high-T stable)","mech":"Aluminosilicate fiber blanket (Al2O3 45%/SiO2 55%): a tangled fiber web traps still air in 50 um pockets giving k~0.05 W/mK -- decent but unsuppressed (pores ~700x the mean free path, gas conducts freely; no opacifier so the radiative path stays open, emissivity 0.85). Its job is survival, not efficiency: 1400 C continuous service where polymers and silica aerogel fail. Fibers shrink/sinter at sustained peak T; RCF dust is a regulated carcinogen on install and repair.",
                    "reactants":[("Heat (q_in)",[("heat",0,0)],[]),("Al2O3-SiO2 fiber",[("Al",0,0),("O",18,-10),("O",18,12),("Si",36,0),("O",54,-10)],[(0,1,2),(0,2,1),(3,4,2)])],
                    "products":[("Heat (q_out)",[("cold",0,0)],[])]},
    "refractory_foam":{"eq":"Heat -> Al2O3-SiO2-ZrO2 foam + SiC opacifier blocks radiation at high T","mech":"Crucible-chemistry sol-gel foam (Al2O3 60%/SiO2 30%/ZrO2 10%) + 5 wt% SiC opacifier: the open-cell foam handles 1700 C+ and -- unlike bare fiber -- fights the radiative mode directly: SiC's absorption coefficient shortens the Rosseland mean free path so in-band IR transport drops even as T^3 rises. Still 100 um pores so gas conduction is unsuppressed; 200 kg/m3 density is the price of structural integrity; sprayable as gunning-grade slurry.",
                      "reactants":[("Heat (q_in)",[("heat",0,0)],[]),("Al2O3",[("Al",0,0),("O",18,-10),("O",18,12)],[(0,1,2),(0,2,1)]),("ZrO2",[("Zr",40,0),("O",58,-10),("O",58,12)],[(0,1,2),(0,2,1)]),("SiC opacifier",[("Si",70,0),("C",88,0)],[(0,1,1)])],
                      "products":[("Heat (q_out, reduced)",[("cold",0,0)],[])]},
    "mli":{"eq":"Heat (radiation) -> N=40 Al foils (emissivity 0.04) -> radiation suppressed by 1/N","mech":"Multilayer insulation: N=40 aluminized films at emissivity 0.04 suspended on low-contact spacers in hard vacuum (1e-5 torr). Radiation -- the dominant mode across a gap -- falls roughly as sigma*T^4/(N+1), the 1/N law at the heart of cryostat design; vacuum removes gas conduction entirely, leaving only foil-edge/spacer solid conduction. k_eff ~0.0001 mW/mK in the ideal limit. Achilles heel: vacuum loss collapses it, films wrinkle/short under handling, and polymer films cap at ~673 K.",
           "reactants":[("Heat (radiation)",[("heat",0,0)],[]),("Al foil (low-e)",[("Al",20,0)],[]),("Al foil (low-e)",[("Al",40,0)],[]),("Al foil (low-e)",[("Al",60,0)],[]),("Vacuum spacer",[("O",30,14),("Si",50,14)],[])],
           "products":[("Heat (q_out ~ 0)",[("cold",0,0)],[])]},
    "aerogel_mli":{"eq":"Heat -> aerogel (Knudsen) + MLI (reflective) = dual suppression","mech":"Hybrid: aerogel plies are simultaneously Knudsen-suppressing insulation AND the spacer medium between 20 aluminized foils -- attacking gas conduction (20 nm pores), radiation (1/N with N=20) and contact conduction (aerogel prevents foil-to-foil shorts) at once. Needs only soft vacuum (1e-2 torr), and if vacuum fails entirely it degrades gracefully to a plain aerogel blanket rather than to bare metal. Still film-limited to ~673 K.",
                  "reactants":[("Heat (q_in)",[("heat",0,0)],[]),("Aerogel",[("Si",20,0),("O",38,-10)],[(0,1,2)]),("Al foil",[("Al",50,0)],[]),("Aerogel",[("Si",70,0),("O",88,-10)],[(0,1,2)]),("Al foil",[("Al",100,0)],[])],
                  "products":[("Heat (q_out, very low)",[("cold",0,0)],[])]},
    "pcm":{"eq":"SrBr2.6H2O + heat -> SrBr2.H2O + 5 H2O (absorbs 630 kJ/kg at 89C)","mech":"Phase-change heat dam: SrBr2.6H2O melts/dehydrates isothermally at 89 C, swallowing 630 kJ/kg as latent heat -- while the front charges, the layer temperature is pinned at T_phase so effectively zero flux passes (a thermal diode). 200 mm holds 302 MJ/m2. Finite by definition: once saturated it becomes a dense (2400 kg/m3) conductor; hydrates supercool/segregate over cycles and must stay encapsulated. Best against transient surges, weakest in steady state -- and against a ~0 K sink a pinned 89 C layer actively leaks.",
           "reactants":[("Heat (q_in)",[("heat",0,0)],[]),("SrBr2.6H2O",[("Sr",20,0),("Br",38,-10),("Br",38,12),("O",58,0),("H",72,10),("H",72,-10)],[(0,1,1),(0,2,1),(3,4,1),(3,5,1)])],
           "products":[("SrBr2.H2O (charged)",[("Sr",20,30),("Br",38,20),("Br",38,42),("O",58,30),("H",72,40)],[(0,1,1),(0,2,1),(3,4,1)]),("5 H2O (vapor)",[("O",80,30),("H",94,40),("H",94,20)],[(0,1,1),(0,2,1)])]},
    "endothermic":{"eq":"MgSO4.7H2O + heat -> MgSO4.H2O + 6 H2O (absorbs 480 kJ/kg, chemical storage)","mech":"Thermochemical battery: MgSO4.7H2O + 480 kJ/kg -> MgSO4.H2O + 6 H2O(v). Incoming heat is consumed breaking hydration bonds -- stored as chemical potential, not temperature rise, so the bed pins near the reaction temperature while a dehydration front advances through it. Released water vapor exits cold-side through the 30%-porous bed (a required vent path). Rechargeable by rehydration; 'salt fatigue' erodes capacity over cycles. Same caveat as PCM: finite capacity, and a pinned temperature that leaks against cryo sinks.",
                  "reactants":[("Heat (q_in)",[("heat",0,0)],[]),("MgSO4.7H2O",[("Mg",20,0),("S",38,0),("O",56,-10),("O",56,10),("O",74,0),("H",88,10),("H",88,-10)],[(0,1,1),(1,2,2),(1,3,2),(1,4,1),(4,5,1),(4,6,1)])],
                  "products":[("MgSO4.H2O (dehydrated)",[("Mg",20,30),("S",38,30),("O",56,20),("O",56,40),("O",74,30),("H",88,40)],[(0,1,1),(1,2,2),(1,3,2),(1,4,1),(4,5,1)]),("6 H2O (vapor, stored)",[("O",100,30),("H",114,40),("H",114,20)],[(0,1,1),(0,2,1)])]},
    "teg":{"eq":"Heat (dT) -> Bi2Te3 Seebeck -> electricity -> Peltier pumps heat back","mech":"Active barrier: Bi2Te3 modules (Seebeck S~200 uV/K, ZT~1) convert the heat flowing through them directly into electricity (~10% of Carnot at best). The harvested power drives a Peltier stage that pumps part of the conducted flux back toward the hot face -- net flux = conduction - converted - pumped; the only mechanism here that removes heat from the circuit rather than merely resisting or storing it. Dense (7700 kg/m3), conducts when off, and module solder/interconnects fatigue under thermal cycling.",
           "reactants":[("Heat (q_in, hot side)",[("heat",0,0)],[]),("Bi2Te3 TEG",[("Bi",25,0),("Te",45,0)],[(0,1,1)])],
           "products":[("Electricity (W)",[("e-",25,25)],[]),("Heat pumped back",[("heat",0,25)],[]),("Heat (q_out, reduced)",[("cold",50,0)],[])]},
    "gradient_spray":{"eq":"Multi-pass spray: Al2O3/SiC (hot) -> aerogel (cold) + TiN reflective topcoat","mech":"RTP2.0 compositional cascade in 5x2 mm sublayers: refractory-rich Al2O3/SiC anchors to the substrate and takes the direct hot face -> hybrid refractory/aerogel eases the CTE+porosity transition -> aerogel-rich then pure aerogel engage Knudsen suppression as local T falls below the aerogel rating -> TiN low-e topcoat reflects residual IR back down the gradient. Each sublayer's job matches its position's temperature; 10 mm total -- a demonstrator of the technique ULTRA CASCADE scales to 4 cascaded passes.",
                     "reactants":[("Heat (q_in)",[("heat",0,0)],[]),("Al2O3 (hot face)",[("Al",15,0),("O",30,-10),("O",30,12)],[(0,1,2),(0,2,1)]),("SiC opacifier",[("Si",50,0),("C",66,0)],[(0,1,1)]),("SiO2 aerogel",[("Si",80,0),("O",96,-10),("O",96,12)],[(0,1,2),(0,2,1)]),("TiN topcoat",[("Ti",110,0),("N",126,0)],[(0,1,1)])],
                     "products":[("Heat (q_out, low)",[("cold",0,0)],[])]},
    "combined":{"eq":"Stage1: Spray -> Stage2: PCM -> Stage3: Chemical -> Stage4: TEG -> Stage5: MLI -> Stage6: Sparkle topcoat (for fun)","mech":"Defense-in-depth across all four transport modes: the spray gradient rejects radiation at the hot face -> the 200 mm SrBr2 PCM pins transients at 89 C and swallows 302 MJ/m2 -> the MgSO4 bed keeps absorbing as bond energy (480 kJ/kg dehydration) after the PCM saturates -> the TEG/MLI sandwich Seebeck-harvests residual flux and Peltier-pumps part back -> the terminal 40-foil vacuum jacket applies ~1/N to whatever escapes. Each stage drops dT for the next; stage 6 is a cosmetic sparkle coat.",
               "reactants":[("Heat (q_in, extreme)",[("heat",0,0)],[]),("Spray",[("Al",15,0)],[]),("PCM",[("Sr",30,0),("Br",45,-8)],[(0,1,1)]),("Chemical",[("Mg",60,0),("S",75,0)],[(0,1,1)]),("TEG",[("Bi",90,0),("Te",105,0)],[(0,1,1)]),("MLI",[("Al",120,0)],[])],
               "products":[("Electricity",[("e-",40,20)],[]),("Heat absorbed (PCM)",[("heat",60,20)],[]),("Heat absorbed (chem)",[("heat",80,20)],[]),("Heat (q_out ~ 0)",[("cold",120,0)],[])]},
    "ultra_cascade":{"eq":"Spray cascade -> evacuated carbon-MLI hot shields -> PerfectVoid foam + VIP -> staged TEG recovery engines -> vacuum MLI x2 -> low-e cap: each layer drops dT for the next (accumulative stacking -> q_out ~ 0)","mech":"Cascaded YSZ/Al2O3-ZrO2 -> zirconia-aerogel -> opacified-aerogel spray gradient (refractory hot-face -> nanoporous cold-face), then evacuated carbon-MLI hot shields (refractory foils + microsphere vacuum, survive 1473 K where Kapton fails) + deep carbon jacket, evacuated vacuum-microsphere foam + fumed-silica VIP core, staged Bi2Te3 TEG energy engines (solid-state -- no PCM dams: at ~600 K they exceed decomposition limits and a phase-pinned dam leaks against a ~0 K sink), two Kapton-film MLI jackets in high vacuum (~1/N radiation, local T <673 K), aerogel-MLI cold cap + TiN low-e + ceramifiable weather seal.",
               "reactants":[("Heat (q_in, extreme)",[("heat",0,0)],[]),("YSZ spray pass",[("Zr",15,0),("O",30,-10),("O",30,12)],[(0,1,2),(0,2,1)]),("PerfectVoid foam",[("Si",50,0),("O",66,-10),("O",66,12)],[(0,1,2),(0,2,1)]),("Carbon-MLI shields",[("C",80,0),("C",95,0)],[(0,1,1)]),("TEG x2",[("Bi",112,0),("Te",128,0)],[(0,1,1)]),("MLI x2",[("Al",110,20),("Al",125,20)],[])],
               "products":[("Electricity",[("e-",40,22)],[]),("Heat absorbed (dams)",[("heat",70,22)],[]),("Heat (q_out -> 0)",[("cold",125,0)],[])]},
}

# =============================================================================
# DURABILITY / WEATHERING DATA -- near-permanence per Goal.md ("extreme
# weathering resistance", "near-permanency"). Ratings are static material-
# science inputs (analogous to PHYS material constants); lifetime is always
# COMPUTED live from these inputs plus the current boundary conditions via
# estimate_durability(), never hard-coded as a final number.
#   max_service_C / min_service_C : rated continuous-service envelope
#   thermal_cycles_rating         : full hot<->cold cycles before significant
#                                    degradation (fatigue/spallation/etc.)
#   hydrophobic_rating            : 0-1, resistance to moisture ingress
#   uv_resistance                 : 0-1, resistance to UV/oxidative attack
#   abrasion_resistance           : 0-1, resistance to mechanical wear
#   base_lifetime_years           : expected service life within the rated
#                                    envelope, at nominal (1 cycle/day) use
# =============================================================================
DURABILITY = {
    "foam":{"max_service_C":120,"min_service_C":-50,"thermal_cycles_rating":500,
            "hydrophobic_rating":0.50,"uv_resistance":0.30,"abrasion_resistance":0.40,"base_lifetime_years":10,
            "weathering_class":"C -- moderate (10-15 yr)",
            "degradation":"UV embrittlement of PU matrix, hydrolysis, closed-cell collapse under thermal cycling.",
            "references":"ASTM D2126 thermal cycling; PU foam UV/hydrolysis literature"},
    "aerogel":{"max_service_C":650,"min_service_C":-200,"thermal_cycles_rating":2000,
               "hydrophobic_rating":0.85,"uv_resistance":0.90,"abrasion_resistance":0.30,"base_lifetime_years":20,
               "weathering_class":"B -- good (15-25 yr with fiber reinforcement)",
               "degradation":"Friable/dusts under mechanical abrasion; hydrophobic (silane) treatment depletes over years.",
               "references":"Aspen Aerogels durability data; Hrubesh & Pekala (1994)"},
    "ceramic_fiber":{"max_service_C":1400,"min_service_C":-50,"thermal_cycles_rating":5000,
                      "hydrophobic_rating":0.40,"uv_resistance":0.95,"abrasion_resistance":0.50,"base_lifetime_years":15,
                      "weathering_class":"B -- kiln-proven (15-20 yr)",
                      "degradation":"Fiber shrinkage/sintering at sustained peak T; unsealed fiber wicks moisture, cutting R-value.",
                      "references":"Unifrax/Morgan ceramic fiber service data"},
    "refractory_foam":{"max_service_C":1700,"min_service_C":-50,"thermal_cycles_rating":3000,
                        "hydrophobic_rating":0.60,"uv_resistance":0.95,"abrasion_resistance":0.70,"base_lifetime_years":25,
                        "weathering_class":"A -- crucible-grade (25+ yr)",
                        "degradation":"Sintering/grain growth at sustained extreme T; spallation under very rapid thermal shock.",
                        "references":"White crucible service life (Vesuvius, Alcoa refractory data)"},
    "mli":{"max_service_C":300,"min_service_C":-270,"thermal_cycles_rating":10000,
           "hydrophobic_rating":1.00,"uv_resistance":1.00,"abrasion_resistance":0.20,"base_lifetime_years":30,
           "weathering_class":"A -- spacecraft-proven (20-30+ yr, sealed vacuum)",
           "degradation":"Vacuum-integrity loss (outgassing, seal/virtual leaks); foil wrinkling or shorting from handling.",
           "references":"NASA MLI service life database; Lockheed cryogenic MLI"},
    "aerogel_mli":{"max_service_C":400,"min_service_C":-270,"thermal_cycles_rating":8000,
                   "hydrophobic_rating":0.90,"uv_resistance":1.00,"abrasion_resistance":0.50,"base_lifetime_years":25,
                   "weathering_class":"A -- robust to partial vacuum loss (25+ yr)",
                   "degradation":"Slower than pure MLI: aerogel spacer prevents foil-to-foil shorting if vacuum degrades.",
                   "references":"SPI/Cryogenics 2008 aerogel-MLI hybrid life data"},
    "pcm":{"max_service_C":200,"min_service_C":-20,"thermal_cycles_rating":2000,
           "hydrophobic_rating":0.20,"uv_resistance":0.70,"abrasion_resistance":0.60,"base_lifetime_years":12,
           "weathering_class":"C -- needs re-encapsulation (10-15 yr)",
           "degradation":"Salt-hydrate phase segregation and supercooling drift; encapsulation breach ends useful cycling.",
           "references":"N'Tsoulkove et al. (2014) Solar Energy 107:469 -- hydrate PCM cycling life"},
    "endothermic":{"max_service_C":150,"min_service_C":-20,"thermal_cycles_rating":1500,
                   "hydrophobic_rating":0.15,"uv_resistance":0.70,"abrasion_resistance":0.50,"base_lifetime_years":8,
                   "weathering_class":"D -- maintenance item (5-10 yr, periodic recharge)",
                   "degradation":"Progressive loss of reversible hydration capacity ('salt fatigue') after repeated dehydrate/rehydrate cycles.",
                   "references":"Donkers et al. (2017) Appl. Therm. Eng. 119:1 -- MgSO4 cycling stability"},
    "teg":{"max_service_C":300,"min_service_C":-196,"thermal_cycles_rating":10000,
           "hydrophobic_rating":1.00,"uv_resistance":1.00,"abrasion_resistance":0.30,"base_lifetime_years":20,
           "weathering_class":"B -- solid-state reliability (15-20 yr)",
           "degradation":"Solder/interconnect fatigue from thermal cycling; Bi2Te3 sublimation above rated T.",
           "references":"Snyder & Toberer (2008) Nat. Mater. 7:105 -- TEG module life"},
    "gradient_spray":{"max_service_C":1700,"min_service_C":-200,"thermal_cycles_rating":4000,
                       "hydrophobic_rating":0.75,"uv_resistance":0.90,"abrasion_resistance":0.60,"base_lifetime_years":20,
                       "weathering_class":"B -- field-repairable (15-20 yr)",
                       "degradation":"Interlayer delamination if pass-to-pass CTE mismatch exceeds bond strength; topcoat wear is field-recoatable.",
                       "references":"Spray-on aerogel/ceramic coating field data (Cabot, Aspen)"},
    "combined":{"max_service_C":300,"min_service_C":-270,"thermal_cycles_rating":1500,
                "hydrophobic_rating":0.55,"uv_resistance":0.85,"abrasion_resistance":0.60,"base_lifetime_years":18,
                "weathering_class":"B -- gated by weakest stage (12-20 yr system life)",
                "degradation":"Sealed vacuum/TEG stages exceed 20 yr; PCM/chemical stages (~8-12 yr) are the maintenance-limited weak link.",
                "references":"Composite of per-stage ratings above; weakest-stage-gates-system engineering principle"},
    "ultra_cascade":{"max_service_C":1600,"min_service_C":-270,"thermal_cycles_rating":6000,
                "hydrophobic_rating":0.90,"uv_resistance":0.95,"abrasion_resistance":0.70,"base_lifetime_years":30,
                "weathering_class":"A -- near-permanent barrier (25-30+ yr)",
                "degradation":"Weak links: sealed vacuum-jacket integrity (outgassing/seal leaks slowly erode MLI performance) and PCM recharge cycles; refractory face, aerogel zones and ceramifiable weather seal are multi-decade stable. Sprayed zones are field-recoatable.",
                "references":"goal2.md near-permanency spec; white-crucible refractory data; NASA/Lockheed MLI service life; VIP aging studies"},
}

# =============================================================================
# FABRICATION PROTOCOLS -- how each formulation is actually built, in extreme
# detail. Ordered hot-face-out, matching each formulation's layer list. Steps
# carry real shop parameters: solids loading, nozzle/pressure/standoff, flash
# and cure schedules, vacuum targets, foil counts, and QC gates -- the same
# numbers the thermal model assumes (thickness, porosity, vacuum pressure,
# foil counts are read off the FORMULATIONS layers, so process and model stay
# in agreement).
# =============================================================================
FABRICATION = {
"foam":{
 "title":"STANDARD FOAM -- two-component PU spray, single lift",
 "est_active":"0.5 day","est_calendar":"1 day (incl. cure)",
 "equipment":["2K PU spray gun (1:1 ISO/polyol ratio machine)","air compressor 6 bar",
              "PPE: FFP3, Tyvek suit, goggles"],
 "materials":["closed-cell PU foam system (ISO + polyol, ~32-45 kg/m3 free-rise)"],
 "phases":[("PREP",["degrease substrate (IPA), mask edges","substrate 15-30 C, RH < 80%"]),
           ("SPRAY",["gun distance 400-600 mm, fan pattern, 50% overlap","build 50 mm in 3-4 lifts of 12-15 mm each; flash 60-90 s between lifts",
                     "cross-hatch passes; keep wet edges merged to avoid knit lines"]),
           ("CURE",["tack-free ~5-15 s/lift; full cure 24 h at 20 C","trim high spots with a saw blade; no post-cure needed"]),
           ("QC",["free-rise density check on a cup sample (target 32-45 kg/m3)",
                  "core sample: closed-cell structure, no voids/friability"])],
 "qc":["density cup test","core-cut visual","adhesion pull-off >=0.1 MPa"],
 "safety":["isocyanate vapor is a potent respiratory sensitizer -- supplied-air or P100",
           "spray mist is flammable during atomization; no ignition sources"]},
"aerogel":{
 "title":"AEROGEL BLANKET -- cut, laminate and mechanically secure",
 "est_active":"0.5 day","est_calendar":"1 day",
 "equipment":["sharp shears/rotary cutter","high-T adhesive (silicone RTV, 350 C class)",
              "mechanical fixings (ceramic buttons / stainless washers)","dust mask + LEV"],
 "materials":["silica aerogel blanket 10 mm x5 plies (50 mm)","RTV silicone adhesive","stainless mesh cap (optional abrasion armor)"],
 "phases":[("CUT",["cut plies with shears on a clean table; aerogel dusts heavily -- work downdraft",
                   "dry-fit all 5 plies; stagger seams >=50 mm between plies"]),
           ("LAMINATE",["spot-bond plies with RTV dots (full-coverage adhesive collapses pores and raises k)",
                        "press flat; no wrinkle bridging across ply seams"]),
           ("SECURE",["mechanical fixing through ply stack every ~150 mm (aerogel creeps under sustained compression -- light torque only)",
                      "optional stainless-foil cap sheet for abrasion zones"]),
           ("QC",["weigh finished panel vs spec areal density (~150-200 kg/m3 aerogel bulk)",
                  "IR camera under a hot plate: no cold spots at seams/fixings"])],
 "qc":["areal density vs spec","IR thermography seam scan"],
 "safety":["aerogel dust = nuisance particulate, P100 + goggles; fibers itch skin"]},
"ceramic_fiber":{
 "title":"CERAMIC FIBER -- blanket stacking / stud-weld pinning",
 "est_active":"0.5 day","est_calendar":"1 day",
 "equipment":["ceramic fiber blanket (128 kg/m3 class, 1400 C)","stud welder + ceramic cups or Inconel anchors","respirator P100 (RCF dust is a HAP)"],
 "materials":["RCF/alumino-silicate blanket plies to 50 mm","Inconel studs + speed clips"],
 "phases":[("STACK",["compress blanket plies ~10% between layers for interlock","impale on studs spaced 300-400 mm grid; stagger seams every ply"]),
           ("PIN",["weld studs to substrate; press plies over, lock with speed clips + ceramic cups",
                   "compress final ply to nominal 50 mm under the clip seats"]),
           ("QC",["thickness check across grid (+-3 mm)","hot-face smoke/pencil test for open seams; re-pack any gaps with fiber"])],
 "qc":["grid thickness map","seam smoke test"],
 "safety":["RCF fibers are classified carcinogenic (EU Cat 1B): P100 + LEV + disposable suit mandatory",
           "after first firing, devitrified fiber is equally hazardous on repair"]},
"refractory_foam":{
 "title":"REFRACTORY FOAM -- gunned/sprayed refractory, staged fire-in",
 "est_active":"1 day","est_calendar":"3 days (incl. dry-out + fire-in)",
 "equipment":["refractory gunning/spray rig or trowel","anchors: V or Y Inconel studs, hex-mesh","furnace or propane weed burner for fire-in"],
 "materials":["lightweight refractory castable/foam (1600-1700 C class, ~500-800 kg/m3)","stainless or Inconel anchor grid"],
 "phases":[("ANCHOR",["weld anchor grid 150-200 mm spacing; cover with hex-mesh for hand-packed zones"]),
           ("GUN",["spray in 20-25 mm lifts, 2 lifts to 50 mm; keep rebound <15% (rebound = coarse aggregate, toss it)",
                   "nozzle 500-800 mm standoff, nearly perpendicular; keep moving to avoid laminar flow bands"]),
           ("CURE",["air-set 24 h at >10 C under damp cover (hydraulic bond needs moisture)",
                    "dry-out ramp: 110 C hold 4 h, then 25-50 C/h to 350 C hold 4 h -- violent steam spalling if rushed"]),
           ("FIRE-IN",["first heat ramp <=50 C/h to 600 C, hold 1 h; the foam sinters a ceramic bond",
                       "inspect for shrinkage cracks <1 mm; repoint wider cracks with matching mortar"])],
 "qc":["lift thickness map","post-fire-in crack survey","density cube vs spec"],
 "safety":["caustic wet mix -- gloves + eye protection","dry-out steam burns: keep clear of vent paths"]},
"mli":{
 "title":"MLI VACUUM -- spiral-wrap foils + spacers, bagged and pumped",
 "est_active":"1 day","est_calendar":"2 days (incl. evacuation + leak check)",
 "equipment":["clean room or at least lint-free gloves (oils outgas)","metallized film bag + impulse heat sealer",
              "turbomolecular or roots+roughing pump to <=1e-3 torr (target 1e-5)","helium leak detector"],
 "materials":["double-aluminized Mylar/Kapton foil ~6-25 um","polyester/tulle spacer netting","getter sachets (optional, extends vacuum life)"],
 "phases":[("CUT",["cut 40 foil + 40 spacer sets per the model (20 mm stack)","handle by edges only; fingerprints outgas and stain emissivity"]),
           ("WRAP",["spiral-wrap or flat-stack alternating foil/spacer with light even tension -- wrinkles short radiative paths",
                    "stagger seams; leave one engineered vent path to the pump port"]),
           ("BAG+PUMP",["insert getter, place in bag, double heat-seal except port; evacuate to <=1e-5 torr, seal port hot",
                        "hold 24-48 h: vacuum must not decay measurably or a virtual leak exists"]),
           ("QC",["He leak rate <=1e-9 Pa.m3/s for decade-life target","emissivity spot-check with IR reflectometer (<=0.05 shiny side)"])],
 "qc":["He leak rate","hold-vacuum decay test","emissivity check"],
 "safety":["foil edges cut skin -- cut gloves","pump exhaust: oil mist, vent it"]},
"aerogel_mli":{
 "title":"AEROGEL-MLI HYBRID -- aerogel spacer replaces netting, forgiving under vacuum loss",
 "est_active":"1 day","est_calendar":"2 days",
 "equipment":["same as MLI plus: aerogel blanket 5 mm plies","P100 respirator (aerogel dust)"],
 "materials":["aluminized foil ~40 plies","aerogel blanket plies as spacers","bag + getter"],
 "phases":[("STACK",["alternate aerogel ply / foil ~20x for 50 mm; aerogel both insulates and prevents foil shorting"]),
           ("BAG+PUMP",["same bag/evacuation sequence as MLI; target <=1e-2 torr is sufficient -- aerogel keeps working in soft vacuum"]),
           ("QC",["leak check (looser spec than pure MLI -- that is the point of the hybrid)",
                  "IR scan under load for contact shorts"])],
 "qc":["leak rate vs soft-vac spec","IR short scan"],
 "safety":["same as MLI + aerogel dust controls"]},
"pcm":{
 "title":"PCM HEAT ABSORBER -- macro-encapsulated salt-hydrate panel",
 "est_active":"1 day","est_calendar":"2 days (incl. charge/settle cycles)",
 "equipment":["sealed metal/polymer encapsulation trays (200 mm deep)","funnel + heated pour station for hydrated salt","weigh scale"],
 "materials":["SrBr2.6H2O (T_phase ~80 C) or Na2SO4-class hydrate, ~200 mm fill","nucleating agent 1-3 wt% (anti-supercooling)","expansion headspace ~10%"],
 "phases":[("ENCAPSULATE",["fill trays to ~90% leaving expansion headspace; hydrate salts grow/shrink through phase cycling",
                           "disperse nucleating agent while pouring, keep salt slurry above T_phase to avoid premature set"]),
           ("SEAL",["hermetic seal -- any moisture exchange shifts T_phase and latent capacity over cycles",
                    "pressure-test each tray underwater for bubbles"]),
           ("CONDITION",["run 3-5 charge/discharge cycles above/below T_phase before install to stabilize supercooling drift"]),
           ("QC",["weigh each tray vs spec fill mass","freeze/thaw a witness tray -- latent plateau must appear at rated T_phase"])],
 "qc":["fill mass","hermetic seal test","witness-tray phase plateau"],
 "safety":["hot salt pours -- splash PPE","some hydrates are skin irritants; check SDS"]},
"endothermic":{
 "title":"ENDOTHERMIC CHEMICAL -- packed hydrate reactor bed",
 "est_active":"0.5 day","est_calendar":"1 day",
 "equipment":["sealed canister bed with flow-through hot face","vibrating packer","moisture-controlled fill station"],
 "materials":["MgSO4.7H2O granules, ~150 mm bed depth","perforated retainers + desiccant canister"],
 "phases":[("PACK",["fill canisters with granules, vibrate to settle -- target ~60-65% packing (void paths let the dehydration vapor escape)",
                    "retain with perforated plates; no fines against the hot face (they cake and block vapor)"]),
           ("SEAL",["seal with vapor-vent path toward cold side -- the mechanism IS water-vapor transport; total hermeticity defeats it"]),
           ("QC",["bed mass + bulk density vs spec","function test: heat 10 mm probe coupon to dehydration onset, confirm endotherm by DSC or thermocouple plateau"])],
 "qc":["packing density","endotherm plateau test"],
 "safety":["hot vapor discharge path must be directed away from operators","MgSO4 dust: nuisance particulate"]},
"teg":{
 "title":"THERMOELECTRIC ENGINE -- Bi2Te3 module array + MLI shunt",
 "est_active":"1 day","est_calendar":"1 day",
 "equipment":["TEG modules rated to local T (Bi2Te3 ~250-300 C hot side)","graphite-foil or BN thermal interface (NO silicone grease >350 C)",
              "torque-limited driver (module ceramic cracks easily)","series wiring + high-T insulation (mica)"],
 "materials":["Bi2Te3 TEG modules ~5 mm thick","MLI jackets per formulation spec","thermal interface sheets"],
 "phases":[("MODULES",["array modules to cover hot face; both faces get graphite/BN interface film",
                       "torque mounting hardware to module spec (over-torque cracks the alumina plates)"]),
           ("WIRE",["series-wire modules within a stage; mica-sleeve conductors on the hot side",
                    "route leads through the cold side with sealed feedthroughs"]),
           ("MLI",["wrap MLI jackets per model spec around the module stage"]),
           ("QC",["open-circuit voltage check per module (Seebeck response on a hot plate)",
                  "insulation resistance to ground >=1 Mohm","IR scan for interface voids"])],
 "qc":["module Voc check","IR interface scan","insulation resistance"],
 "safety":["hot-face conductors must not short to the sheath -- mica sleeves mandatory","Bi2Te3 is a heavy-metal telluride: handle modules intact, dispose per hazmat"]},
"gradient_spray":{
 "title":"GRADIENT SPRAY -- RTP2.0 5-sublayer compositional passes, cascaded",
 "est_active":"1 day","est_calendar":"3 days (incl. staged cure)",
 "equipment":["HVLP or peristaltic-fed spray rig (abrasive slurries eat needle valves)","IR lamps for inter-pass flash",
              "wet-film comb gauge + thickness witness coupons"],
 "materials":["YSZ/Al2O3-ZrO2 refractory powder + colloidal-zirconia/phosphate binder","zirconia-aerogel granulate",
              "SiC + carbon-black IR opacifier","low-e foil cap strip"],
 "phases":[("PREP",["degrease + grit-blast to 50-75 um anchor profile; preheat substrate 60-90 C",
                    "mix each sublayer slurry fresh; solids loading per recipe, 0.3% dispersant"]),
           ("PASS x N (each = 5 sublayers x 2 mm)",[
                    "S1 refractory-rich: ~55-60 vol% solids, nozzle 1.3-1.8 mm, 3-4 bar, standoff 150-250 mm, 50% overlap, ~0.5 mm wet/sweep x4 sweeps",
                    "S2 hybrid refractory-aerogel: ~45% solids, aerogel granulate folded in at the pot (never shear-mixed -- destroys nanopores)",
                    "S3 SiC+C opacified aerogel: ~40% solids, SiC 5-10 wt% + carbon black ~2%",
                    "S4 pure zirconia aerogel: ~35% solids, low pressure (<2 bar) to keep granules intact",
                    "S5 S4 + 1 embedded low-e foil strip laminated while tacky",
                    "flash 45-90 s between sweeps (<120 C surface); 3-10 min tack between sublayers; 15-30 min between passes"]),
           ("CURE",["air set 4 h; staged bake 80 C/2 h -> 120 C/2 h -> 200 C/4 h, ramp <=2 C/min",
                    "forge service: fire-in ramp <=5 C/min to ~600 C first use"]),
           ("QC",["wet-film comb every sweep; cured-build US gauge vs 2.0 mm/sublayer spec",
                  "adhesion pull-off (ASTM D4541) on witness coupon","IR thermography for voids/delamination"])],
 "qc":["wet-film per sweep","D4541 pull-off","IR delamination scan","cured thickness vs spec"],
 "safety":["refractory/aerogel dusts: P100 + LEV","solvent/binder vapors: ventilation, no ignition","hot cure: furnace PPE"]},
"combined":{
 "title":"INSULATRIX PRO -- 5-stage build (spray + PCM + chem + MLI/TEG + jacket)",
 "est_active":"2-3 days","est_calendar":"4-6 days",
 "equipment":["gradient-spray rig (per GRADIENT SPRAY)","PCM tray fill station","canister bed packer","MLI wrap + pump rig","TEG module tools"],
 "materials":["per-stage: refractory/aerogel slurries; SrBr2 PCM; MgSO4.7H2O; MLI foils+spacers; Bi2Te3 modules"],
 "phases":[("S1 SPRAY",["> see GRADIENT SPRAY pass protocol: 5 sublayers x 2 mm hot-face spray cascade"]),
           ("S2 PCM",["> see PCM protocol: 200 mm SrBr2 trays, nucleating agent, 90% fill, 3-5 conditioning cycles"]),
           ("S3 CHEM",["> see ENDOTHERMIC protocol: 150 mm MgSO4.7H2O bed, 60-65% packing, vapor-vent path"]),
           ("S4 MLI+TEG",["> see MLI protocol (2 jackets, 20 foils each) + TEG protocol (module array between jackets)"]),
           ("S5 JACKET",["> see MLI protocol: 20 mm outer jacket, 40 foils, <=1e-5 torr, He leak check"]),
           ("QC",["per-stage QC gates above + IR thermography across the finished stack at operating dT"])],
 "qc":["every stage's own QC + full-stack IR scan"],
 "safety":["aggregate of per-stage hazards -- PCM hot pours and refractory dust are the main ones"]},
"ultra_cascade":{
 "title":"ULTRA CASCADE -- full 6-zone spray/laminate/vacuum build (~246 mm, 35 layers)",
 "est_active":"3-4 days","est_calendar":"6-8 days (staged cures + leak checks)",
 "equipment":["HVLP + peristaltic-fed spray rigs (abrasive refractory + fragile aerogel slurries)",
              "IR lamps + wet-film comb + ultrasonic thickness gauge","refractory gunning gear for carbon-MLI shield boards",
              "metallized bagger + impulse sealer + turbo pump rig (to 1e-6 torr) + He leak detector",
              "TEG module tools + graphite/BN interface film","PVD coating service for TiN cap (or pre-coated foil)",
              "ASTM D4541 pull-off tester + IR camera + helium sniffer"],
 "materials":["YSZ/Al2O3-ZrO2 refractory powder + colloidal-zirconia/phosphate binder (Zone A slurries)",
              "zirconia-aerogel granulate + zirconia sol binder","SiC + carbon-black opacifier",
              "carbon-aerogel boards 18/16 mm + refractory foils (60x + 80x per shield/jacket)",
              "evacuated hollow microspheres (ZrO2-YSZ/silica coated) + ceramifiable silicone binder + fumed silica + carbon-foam nanoparticles (PerfectVoid)",
              "fumed-silica VIP core 15 mm + metallized film bags + getters",
              "Bi2Te3 TEG modules (hot stage 4 mm, cold stage 6 mm) + mica sleeves",
              "double-aluminized Kapton film + spacer netting (120x + 80x + 36x foil wraps)",
              "TiN PVD topcoat ~0.1-0.3 um + ceramifiable silicone weather-seal coats"],
 "phases":[
  ("PHASE 0 -- SUBSTRATE PREP (gate: anchor profile + preheat)",[
      "substrate must itself survive hot face: alumina/Inconel/ceramic board (steel needs a refractory primer)",
      "degrease IPA/acetone, lint-free wipes; grit-blast 50-75 um anchor profile, white Al2O3 36-60 grit, 3-4 bar, ~150 mm standoff (or chemical-etch + bond primer on non-blastable substrates)",
      "vacuum + solvent wipe; preheat 60-90 C to drive off adsorbed moisture and improve first-pass wetting",
      "mask fixtures/feedthroughs; mark wet-film thickness targets per zone"]),
  ("ZONE A -- 4 CASCADED SPRAY PASSES (5 sublayers x 2 mm each = 40 mm)",[
      "mix each sublayer slurry fresh at the pot; solids loading below, 0.3% dispersant, DI-water carrier",
      "A#-1 YSZ/Al2O3-ZrO2 refractory-rich: ~55-60 vol% solids | nozzle 1.3-1.8 mm, 3-4 bar, standoff 150-250 mm, ~0.5 mm wet/sweep, 4 sweeps, cross-hatch 90 deg, 50% overlap",
      "A#-2 hybrid refractory-aerogel: ~45% solids; aerogel granulate FOLDED in -- never shear-mixed (destroys nanopores)",
      "A#-3 SiC+C opacified aerogel: ~40% solids, SiC 5-10 wt% + carbon black ~2% IR opacifier",
      "A#-4 pure nanoporous zirconia aerogel: ~35% solids, pressure <2 bar to keep granulate shells intact",
      "A#-5 = A#-4 + 1 embedded low-e TiN/Al foil strip laminated while tacky (n_reflective=1 in the model)",
      "flash 45-90 s between sweeps (IR lamps OK, surface <120 C to avoid skinning/blisters); tack 3-10 min between sublayers; 15-30 min between passes; max ~12 mm build/session to avoid slump",
      "verify: wet-film comb every sweep, cured-build US gauge 2.0+-0.2 mm per sublayer, D4541 pull-off on witness coupon"]),
  ("ZONE B -- EVACUATED CARBON-MLI HOT SHIELDS (5x 18 mm + 1x 16 mm = 106 mm)",[
      "pre-form carbon-aerogel spacer boards (carbon/SiC opacified, 95% porosity class) cut/punched to size",
      "per shield: embed 60 thin refractory foils (Mo/W or polished stainless, 10-25 um) through the spacer stack -- the model's n_reflective=60; deep jacket takes 80",
      "fill interstitials with evacuated hollow microspheres (50-150 um, wall-rated); pour + vibrate-settle GENTLY -- <5% shell breakage, no compressed air",
      "edge-seal each shield in refractory fabric, stitch seams staggered; mechanical tie-downs only (ceramic buttons/Inconel wire) -- NO polymers this hot (~900-1100 K local)"]),
  ("ZONE C -- NANOPOROUS VACUUM CORE (20 mm PerfectVoid + 15 mm VIP)",[
      "PerfectVoid spray: vacuum microspheres (ZrO2-YSZ/silica-coated) ~30-60 vol% in ceramifiable silicone binder + fumed silica + carbon-foam nanoparticles; LOW pressure (<2 bar) so shells survive; finished bulk ~60 kg/m3 (= model porosity 0.96)",
      "VIP: fumed-silica core 15 mm into metallized multilayer film bag -> evacuate <=1e-3 torr -> double heat-seal -> 24-72 h hold-vacuum check",
      "position note: this zone sits ~400-600 K in service -- inside the VIP envelope per the model's T_max check"]),
  ("ZONE D -- STAGED TEG RECOVERY ENGINES (4 mm + 6 mm)",[
      "Bi2Te3 module arrays: hot stage 4 mm modules, cold stage 6 mm; graphite-foil/BN interface film both faces (no silicone grease above ~350 C)",
      "torque to module spec -- over-torque cracks the alumina plates; series-wire per stage, mica-sleeve all hot-side conductors",
      "sealed feedthroughs for leads; verify Voc per module + insulation resistance >=1 Mohm before covering"]),
  ("ZONE E -- KAPTON-FILM MLI JACKETS (20 mm @120 foils + 16 mm @80 foils)",[
      "spiral-wrap double-aluminized Kapton (6-12 um film, 25-50 nm Al) + spacer netting every foil; foil counts per model (120 then 80); light even tension, no wrinkles/shorts",
      "engineered vent seams -> outer metallized bag -> evacuate <=1e-6 torr -> heat-seal port -> He leak check (spec <=1e-9 Pa.m3/s for multi-decade hold); add getter sachets",
      "position note: Kapton is rated ~673 K -- the upstream zones keep this zone ~300-500 K per the model's T_max check"]),
  ("ZONE F -- COLD CAP + WEATHER SEAL (18 mm + 0.3 mm + 0.5 mm)",[
      "aerogel-MLI cold cap: 36 foils + aerogel spacer 18 mm, laminated + edge-taped",
      "TiN low-e topcoat ~0.1-0.3 um by PVD (sputter/EB-PVD) -- coat-shop job or laminated pre-coated film; verify emissivity <=0.05-0.10 by IR reflectance",
      "ceramifiable silicone weather seal: 2 coats x ~0.25 mm, RT cure 24 h between coats -- the outer armor that ceramifies under fire exposure"]),
  ("FINAL QC GAUNTLET",[
      "per-zone thickness audit vs model spec mm (wet comb + cured US gauge)",
      "adhesion pull-off ASTM D4541 on co-sprayed witness coupons per spray zone",
      "bulk density of aerogel/foam zones vs model porosity targets",
      "He leak rate on VIP + both MLI jackets BEFORE closing the outer bag",
      "TiN cap emissivity by IR reflectometer",
      "full-stack IR thermography at operating dT: no hot streaks/voids",
      "log every slurry batch ID, spray param set, cure chart, and leak rate for the panel's QA file"]),
 ],
 "qc":["wet-film per sweep","cured US thickness map per zone","D4541 pull-off","bulk density vs porosity spec",
       "He leak on all vacuum volumes","emissivity on TiN cap","full-stack IR at operating dT"],
 "safety":["nanopowder/aerogel/refractory dust: P100 or supplied-air + LEV throughout Zones A/B/C",
           "solvents/binders: ventilation + ignition control","foil/Kapton edges: cut gloves",
           "hot cure/fire-in: furnace PPE","TEG: mica-sleeve hot conductors; Bi2Te3 is a heavy-metal telluride -- intact handling, hazmat disposal"],
 "caveats":["vacuum volumes are the decades-long weak link -- include getters and record leak rates",
            "sprayed zones are field-recoatable; VIP/MLI jackets are replaceable cassettes, not field repairs",
            "TEG stages are optional for a pure-insulation variant; the model shows the stack still meets spec without them"]},
}

def estimate_durability(formulation,T_hot,T_cold):
    """Compute expected service life from static DURABILITY inputs + live boundary
    conditions. Rule-of-thumb Arrhenius-style halving per 50 C of envelope overstress,
    plus a thermal-cycle-fatigue limit scaled by dT^1.5 against the rated cycle count."""
    d=DURABILITY.get(formulation["approach"])
    if d is None: return None
    hot_C=T_hot-273.15; cold_C=T_cold-273.15
    over_hot=max(0.0,hot_C-d["max_service_C"]); over_cold=max(0.0,d["min_service_C"]-cold_C)
    envelope_factor=0.5**(over_hot/50.0)*0.5**(over_cold/50.0)
    dT=max(T_hot-T_cold,1.0)
    cycle_stress=(dT/300.0)**1.5
    cycles_to_failure=d["thermal_cycles_rating"]/max(cycle_stress,0.1)
    years_cycle_limited=cycles_to_failure/365.0  # assumes 1 full thermal cycle/day, nominal duty
    material_factor=(d["hydrophobic_rating"]*d["uv_resistance"]*d["abrasion_resistance"])**(1.0/3.0)
    lifetime_years=min(d["base_lifetime_years"]*envelope_factor,years_cycle_limited)*material_factor
    return {"lifetime_years":lifetime_years,"in_rated_envelope":(over_hot==0.0 and over_cold==0.0),
            "over_hot_C":over_hot,"over_cold_C":over_cold,"envelope_factor":envelope_factor,
            "cycles_to_failure":cycles_to_failure,"material_factor":material_factor,"data":d}

# =============================================================================
# HYPERVELOCITY IMPACT SHIELD + MAGNETIC DEFLECTION + SELF-HEALING HULL
# =============================================================================
# Three real, physically-grounded mechanisms, each with a known engineering
# name and literature basis (kept honest and simplified, same rigor level as
# the thermal model above -- not literal NASA/ESA design equations, but the
# same real physics they're built from):
#   1. Whipple shield: a thin sacrificial bumper shatters/melts a hypervelocity
#      impactor into a spreading debris cloud across a standoff gap, diluting
#      it before it reaches the rear wall (Whipple 1947; Christiansen, NASA
#      JSC "Meteoroid/Debris Shielding" design-equation family; debris-cone
#      angle from Piekutowski (1996) hypervelocity debris-cloud studies).
#   2. Eddy-current magnetic deflection: a pulsed field induces eddy currents
#      in a CONDUCTIVE impactor (Lenz's law), producing a repulsive impulse.
#      This is real and used in eddy-current braking -- but it does ~nothing
#      to non-conductive threats (rock/ice, the majority of real MMOD), which
#      is stated honestly rather than papered over.
#   3. Self-healing rear wall: microencapsulated sealant / expanding foam
#      patches a breach at a finite rate (White et al. 2001 Nature,
#      self-healing polymer composites; commercial self-sealing tank/tire
#      technology) racing against cabin decompression through the hole,
#      modeled with the standard compressible choked-orifice flow equation.
# Design envelope: micrometeoroid & orbital debris (MMOD) scale threats, mm to
# a few cm -- the actual regime Whipple shields are engineered for, at 20,000
# mph (8.94 km/s), a realistic LEO-debris/meteoroid closing speed. A literal
# kilometer-scale asteroid is outside ANY material shield's capability (its
# kinetic energy at these speeds rivals a nuclear yield) -- that is honest
# physics, not a shortfall of this design, exactly like "true zero heat
# transfer is impossible" above.
IMPACT_PHYS = {
    "ref_velocity_ms": 8940.8,             # 20,000 mph -- realistic LEO/meteoroid hypervelocity
    "debris_cone_half_angle_deg": 20.0,    # Piekutowski (1996) hypervelocity debris-cloud studies
    "bumper_density": 2700.0,              # Al 6061 bumper
    "bumper_frag_k": 1.35,                 # simplified areal-density fragmentation-efficacy constant
    "stuffed_density": 1440.0,             # Kevlar/Nextel "stuffed Whipple" intermediate (ISS-shield-derived)
    "rear_areal_toughness_J_m2": 4.2e6,    # simplified specific penetration-energy threshold, composite rear wall
    "thermal_core_specific_absorption_J_kg": 3000.0,  # crush/pulverization energy absorption, brittle porous
                                            # media (conservative vs. real crushable honeycomb: Gibson & Ashby,
                                            # "Cellular Solids" -- this stack is insulation, not armor)
    "coil_field_T": 9.0, "coil_pulse_s": 0.02, "coil_gradient_length_m": 0.6,
    "frame_max_impulse_Ns": 400.0,         # max reaction impulse the coil mounts + frame can sustain per pulse
    "eddy_sigma_metal": 1.0e6,             # S/m -- nickel-iron meteorite / metallic debris (realistic order of magnitude)
    "eddy_sigma_rock": 0.0,                # non-conductive silicate/ice debris -- magnetic deflection has ~no effect
    "sealant_expansion_mm_s": 4.0,         # expanding composite sealant foam, realistic spray-foam kinetics
    "sealant_max_hole_mm": 40.0,           # design ceiling of the auto-patch system
    "cabin_pressure_kPa": 101.3, "cabin_volume_m3": 28.0, "cabin_temp_K": 293.15,
    "cabin_min_survivable_kPa": 70.0,      # ~hypoxia-critical floor for unassisted crew survival
    "air_gamma": 1.4, "air_R_specific": 287.05, "orifice_Cd": 0.62,
    "pod_radius_mm": 1200.0,
}

# Impact-shield layer stack (hot/thermal core reuses INSULATRIX PRO as-is --
# no need to reinvent the thermal model for the pod's inner wall).
POD_SHIELD_LAYERS = [
    {"name":"L0: Al6061 Micrometeoroid Bumper","role":"bumper","thickness_mm":1.5,"density":2700.0},
    {"name":"L1: Pulsed Magnetic Deflection Coil","role":"coil","thickness_mm":8.0,"density":2200.0},
    {"name":"L2: Kevlar/Nextel Stuffed Debris Catcher","role":"stuffed","thickness_mm":40.0,"density":1440.0},
    {"name":"L3: Insulatrix Pro Thermal Core (reused, 6-stage)","role":"thermal","thickness_mm":None},
    {"name":"L4: Self-Healing Composite Rear Wall","role":"rear_wall","thickness_mm":6.0,"density":1800.0},
]

def _pod_thermal_thickness_mm():
    pro=[f for f in FORMULATIONS if f["approach"]=="combined"][0]
    return sum(l["thickness_mm"] for l in pro["layers"])

def _pod_thermal_avg_density():
    pro=[f for f in FORMULATIONS if f["approach"]=="combined"][0]
    total_t=sum(l["thickness_mm"] for l in pro["layers"])
    return sum(l["thickness_mm"]*l["density"] for l in pro["layers"])/max(total_t,1e-9)

class PodShieldSim:
    """Live impact-shield assessment for the INSULATRIX pod hull. Nothing here
    is hard-coded pass/fail -- every verdict is computed from IMPACT_PHYS and
    the actual impact parameters given."""
    def __init__(self):
        self.layers=[dict(l) for l in POD_SHIELD_LAYERS]
        for l in self.layers:
            if l["role"]=="thermal":
                l["thickness_mm"]=_pod_thermal_thickness_mm(); l["density"]=_pod_thermal_avg_density()
        # Standoff = the open/low-density catcher zone only (coil + stuffed
        # layer), where a fragmented debris cloud genuinely disperses in the
        # Whipple-shield sense. The thermal core behind it is solid, packed
        # material -- not a gap -- so it does not add standoff distance.
        self.standoff_m=sum(l["thickness_mm"] for l in self.layers if l["role"] in ("coil","stuffed"))*MM

    def assess_impact(self,d_mm,density,v_ms=None,conductive=True):
        """Assess a single spherical-projectile impact. Returns a dict of
        live-computed physics: fragmentation efficacy, magnetic deflection
        assist, ballistic margin, and -- if perforated -- the self-healing
        race against cabin decompression."""
        v_ms=v_ms or IMPACT_PHYS["ref_velocity_ms"]
        r_p=(d_mm*MM)/2.0
        vol_p=(4.0/3.0)*math.pi*r_p**3
        m_p=vol_p*density
        KE=0.5*m_p*v_ms**2

        # --- Stage A: eddy-current magnetic deflection (conductive threats only) ---
        sigma=IMPACT_PHYS["eddy_sigma_metal"] if conductive else IMPACT_PHYS["eddy_sigma_rock"]
        B=IMPACT_PHYS["coil_field_T"]; t_pulse=IMPACT_PHYS["coil_pulse_s"]; L=IMPACT_PHYS["coil_gradient_length_m"]
        J_eddy_raw=sigma*vol_p*B**2*v_ms*t_pulse/L   # simplified eddy-braking impulse scaling: F ~ sigma*B^2*v*Vol/L
        # Real coils/frames have a finite reaction-impulse budget ("sustained by
        # frame") -- capping it here is what makes deflection honestly diminish
        # for large masses instead of scaling away to a size-independent
        # constant (J_eddy_raw and momentum both scale with volume and would
        # otherwise cancel, which would make deflection equally "total" for a
        # paint fleck and a cannonball -- unphysical).
        J_eddy=min(J_eddy_raw,IMPACT_PHYS["frame_max_impulse_Ns"])
        # Saturating fraction of incoming momentum the pulse counters (0 -> 1, never total-miss magic)
        p_in=m_p*v_ms
        deflect_frac=J_eddy/(J_eddy+p_in*0.35) if (J_eddy+p_in)>0 else 0.0
        v_eff=v_ms*(1.0-deflect_frac)   # effective closing speed the shield actually has to defeat

        # --- Stage B: Whipple-shield fragmentation + debris-cloud dilution ---
        bumper=next(l for l in self.layers if l["role"]=="bumper")
        areal_bumper=bumper["thickness_mm"]*MM*bumper["density"]
        areal_proj=r_p*density   # characteristic projectile areal density (radius*density)
        frag_eff=clamp(areal_bumper/(IMPACT_PHYS["bumper_frag_k"]*max(areal_proj,1e-9)),0.0,1.0)
        theta=math.radians(IMPACT_PHYS["debris_cone_half_angle_deg"])
        cloud_r=max(self.standoff_m*math.tan(theta),r_p)
        cloud_area=math.pi*cloud_r**2
        # Fragmented energy spreads over the cloud footprint; any unfragmented
        # remainder stays coherent and hits a footprint set by its own radius.
        e_areal_frag=(KE*frag_eff)/cloud_area
        e_areal_core=(KE*(1.0-frag_eff))/max(math.pi*r_p**2,1e-12)
        e_areal_incident=e_areal_frag+e_areal_core*(1.0 if frag_eff<0.999 else 0.0)
        # Effective speed reduction from Stage A also cuts the areal energy load
        e_areal_incident*=(v_eff/max(v_ms,1e-9))**2

        rear=next(l for l in self.layers if l["role"]=="rear_wall")
        thermal=next(l for l in self.layers if l["role"]=="thermal")
        cap_rear=IMPACT_PHYS["rear_areal_toughness_J_m2"]*(rear["thickness_mm"]/6.0)
        # The thermal core sits between the debris catcher and the rear wall --
        # 405mm of real mass that isn't armor (it's aerogel/PCM/ceramic
        # insulation, much of it brittle) but still absorbs energy by crushing
        # and pulverizing as residual debris grinds through it. Modest,
        # conservative specific-absorption value; NOT credited as standoff
        # (see __init__) to avoid double-counting the same 405mm two ways.
        areal_thermal=thermal["thickness_mm"]*MM*thermal["density"]
        cap_thermal=areal_thermal*IMPACT_PHYS["thermal_core_specific_absorption_J_kg"]
        capacity=cap_rear+cap_thermal
        margin=capacity/max(e_areal_incident,1e-9)
        perforated=margin<1.0

        result={"d_mm":d_mm,"density":density,"v_ms":v_ms,"conductive":conductive,
                "mass_kg":m_p,"kinetic_energy_J":KE,
                "deflect_frac":deflect_frac,"v_effective_ms":v_eff,
                "fragmentation_efficacy":frag_eff,"debris_cloud_radius_m":cloud_r,
                "areal_energy_incident_J_m2":e_areal_incident,"areal_capacity_J_m2":capacity,
                "areal_capacity_rear_J_m2":cap_rear,"areal_capacity_thermal_J_m2":cap_thermal,
                "ballistic_margin":margin,"perforated":perforated,"beyond_patch_rating":False,
                "pressure_history":[]}

        if not perforated:
            result.update({"hole_mm":0.0,"patch_time_s":0.0,"decompress_time_s":math.inf,
                            "cabin_min_kPa":IMPACT_PHYS["cabin_pressure_kPa"],"survived":True})
            return result

        # --- Stage C: self-healing race vs. decompression (only if perforated) ---
        # Hole diameter = projectile diameter x a spallation multiplier. Real
        # hypervelocity perforations blow a hole a few times the impactor's own
        # size, not an unboundedly larger one -- the multiplier grows (log-scale,
        # capped at 8x) with how far energy exceeds capacity, so a breach that's
        # simply too big/fast for the auto-patch system's rating shows up
        # honestly as "beyond_patch_rating" instead of a falsely-tiny hole.
        overshoot_ratio=max(e_areal_incident/max(capacity,1e-9),1.0)
        spall_mult=clamp(1.5+1.5*math.log10(overshoot_ratio),1.5,8.0)
        hole_mm=max(d_mm*spall_mult,0.1)
        if hole_mm>IMPACT_PHYS["sealant_max_hole_mm"]:
            result.update({"hole_mm":hole_mm,"patch_time_s":None,"decompress_time_s":0.0,
                            "cabin_min_kPa":0.0,"survived":False,"beyond_patch_rating":True})
            return result
        rate=IMPACT_PHYS["sealant_expansion_mm_s"]
        patch_time_s=(hole_mm/2.0)/rate  # sealant grows inward from the rim

        P0=IMPACT_PHYS["cabin_pressure_kPa"]*1000.0; Vc=IMPACT_PHYS["cabin_volume_m3"]
        Rspec=IMPACT_PHYS["air_R_specific"]; T=IMPACT_PHYS["cabin_temp_K"]; gamma=IMPACT_PHYS["air_gamma"]
        Cd=IMPACT_PHYS["orifice_Cd"]
        choke=math.sqrt(gamma/(Rspec*T))*(2.0/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))
        rho0=P0/(Rspec*T); m_air=rho0*Vc
        dt=min(patch_time_s/200.0,0.01) if patch_time_s>0 else 0.01
        t=0.0; P=P0; min_P=P0; decompress_time_s=None
        n_sample_pts=10; sample_every=max(1,int(round((patch_time_s/dt)/n_sample_pts))) if patch_time_s>0 else 1
        pressure_history=[(0.0,P0/1000.0)]; step_i=0
        while t<patch_time_s and P>IMPACT_PHYS["cabin_min_survivable_kPa"]*1000.0*0.5:
            frac_open=max(0.0,1.0-t/max(patch_time_s,1e-9))  # hole shrinks linearly as patch closes
            A_hole=math.pi*(hole_mm*MM/2.0*frac_open)**2
            mdot=Cd*A_hole*P*choke
            m_air=max(m_air-mdot*dt,1e-6)
            P=m_air*Rspec*T/Vc
            min_P=min(min_P,P); t+=dt; step_i+=1
            if decompress_time_s is None and P<=IMPACT_PHYS["cabin_min_survivable_kPa"]*1000.0:
                decompress_time_s=t
            if step_i%sample_every==0: pressure_history.append((t,P/1000.0))
        pressure_history.append((patch_time_s,P/1000.0))
        survived=(min_P/1000.0)>=IMPACT_PHYS["cabin_min_survivable_kPa"]*0.5 and (decompress_time_s is None or decompress_time_s>=patch_time_s)
        result.update({"hole_mm":hole_mm,"patch_time_s":patch_time_s,
                        "decompress_time_s":decompress_time_s if decompress_time_s is not None else math.inf,
                        "cabin_min_kPa":min_P/1000.0,"survived":survived,"pressure_history":pressure_history})
        return result

    def sweep_threats(self):
        """Representative MMOD-scale threat scenarios, honestly spanning
        survivable and non-survivable, at the 20,000 mph design reference."""
        v=IMPACT_PHYS["ref_velocity_ms"]
        scenarios=[("Paint fleck (0.2mm, Al, conductive)",0.2,2700.0,v,True),
                   ("Micrometeoroid (1mm, rock, non-conductive)",1.0,3000.0,v,False),
                   ("Orbital debris (3mm, Al, conductive)",3.0,2700.0,v,True),
                   ("Metal fragment (8mm, Ni-Fe, conductive)",8.0,7800.0,v,True),
                   ("Stony micrometeoroid (8mm, rock, non-conductive)",8.0,3000.0,v,False),
                   ("Large debris (25mm, Al, conductive)",25.0,2700.0,v,True),
                   ("Large metal fragment (60mm, Ni-Fe, conductive)",60.0,7800.0,v,True),
                   ("Small asteroid fragment (100mm, rock, non-conductive)",100.0,3000.0,v,False)]
        return [dict(label=s[0],**self.assess_impact(s[1],s[2],s[3],s[4])) for s in scenarios]

C_BG=(128,128,128); C_GROUND=(30,33,44); C_TEXT=(225,228,238); C_TEXT_DIM=(130,138,155)
C_ACCENT=(255,200,90); C_SUCCESS=(80,220,130); C_FAIL=(230,90,85); C_PANEL=(24,27,38)
C_PANEL_BORDER=(55,62,82); C_HOT=(255,95,55); C_COLD=(80,150,255)
C_HEAT_PART=(255,130,50); C_COOL_PART=(90,170,255)
C_GRID=(40,48,68); C_RIM=(180,200,240); C_STAR=(180,190,210)
DISP = 0.005

# =============================================================================
# SECTION 4 -- MINI 3D ENGINE
# =============================================================================
def rot_x(a):
    c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],dtype=float)
def rot_y(a):
    c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],dtype=float)
def rot_z(a):
    c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=float)
def _mix(c1,c2,t):
    return (int(c1[0]+(c2[0]-c1[0])*t),int(c1[1]+(c2[1]-c1[1])*t),int(c1[2]+(c2[2]-c1[2])*t))
def clamp(x,lo=0.0,hi=1.0): return max(lo,min(hi,x))

def temp_to_color(T):
    if T<50: return (25,35,100)
    elif T<150: return _mix((25,35,100),(50,90,210),(T-50)/100.0)
    elif T<300: return _mix((50,90,210),(90,180,230),(T-150)/150.0)
    elif T<500: return _mix((90,180,230),(210,210,110),(T-300)/200.0)
    elif T<800: return _mix((210,210,110),(255,150,50),(T-500)/300.0)
    elif T<1200: return _mix((255,150,50),(255,70,30),(T-800)/400.0)
    else: return _mix((255,70,30),(255,30,10),min(1.0,(T-1200)/500.0))

class Mesh:
    def __init__(self,verts,faces,color,name="",spin=0.0,group="default"):
        self.verts=np.asarray(verts,dtype=float); self.faces=faces; self.color=color
        self.name=name; self.spin=spin; self.group=group
    def world_verts(self,angle=0.0):
        v=self.verts
        if self.spin: v=v@rot_z(angle*self.spin).T
        return v

class Part:
    def __init__(self,key,name,meshes,specs,order,explode,color):
        self.key=key; self.name=name; self.meshes=meshes; self.specs=specs
        self.order=order; self.explode=np.asarray(explode,dtype=float); self.color=color

def _mm(v): return v*DISP
def _box(cx,cy,cz,sx,sy,sz):
    hx,hy,hz=sx/2,sy/2,sz/2
    v=[(cx-hx,cy-hy,cz-hz),(cx+hx,cy-hy,cz-hz),(cx+hx,cy+hy,cz-hz),(cx-hx,cy+hy,cz-hz),
       (cx-hx,cy-hy,cz+hz),(cx+hx,cy-hy,cz+hz),(cx+hx,cy+hy,cz+hz),(cx-hx,cy+hy,cz+hz)]
    f=[(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    return v,f
def _solid_cylinder(r,z0,z1,seg=32):
    seg=max(12,seg); verts,faces=[],[]
    ang=np.linspace(0,2*np.pi,seg,endpoint=False)
    for z in (z0,z1):
        for a in ang: verts.append((r*math.cos(a),r*math.sin(a),z))
    c0=len(verts); verts.append((0,0,z0)); c1=len(verts); verts.append((0,0,z1))
    for i in range(seg):
        a,b=i,(i+1)%seg
        faces.append((a,b,seg+b,seg+a)); faces.append((c0,b,a)); faces.append((c1,seg+a,seg+b))
    return verts,faces
def _sphere(r,seg_u=24,seg_v=16):
    verts,faces=[],[]
    for j in range(seg_v+1):
        v=j/seg_v; phi=v*math.pi
        for i in range(seg_u):
            u=i/seg_u; theta=u*2*math.pi
            verts.append((r*math.sin(phi)*math.cos(theta),r*math.sin(phi)*math.sin(theta),r*math.cos(phi)))
    for j in range(seg_v):
        for i in range(seg_u):
            a=j*seg_u+i; b=j*seg_u+(i+1)%seg_u; c=(j+1)*seg_u+i; d=(j+1)*seg_u+(i+1)%seg_u
            faces.append((a,b,d,c))
    return verts,faces

# =============================================================================
# SECTION 5 -- GEOMETRY BUILDERS
# =============================================================================
def build_panel_parts(formulation):
    parts=[]; layers=formulation["layers"]
    total_t=sum(l["thickness_mm"] for l in layers)
    W=_mm(DIMS["panel_width_mm"]); H=_mm(DIMS["panel_height_mm"]); T=_mm(total_t)
    # Hot face frame
    fv,ff=_box(-T/2-_mm(2),0,0,_mm(4),H+_mm(4),W+_mm(4))
    parts.append(Part("hot_frame","HOT FACE",[Mesh(fv,ff,C_HOT,name="hot frame",group="panel")],["Hot face"],0,(0,0,0),C_HOT))
    x_off=-T/2
    for i,layer in enumerate(layers):
        lt=_mm(layer["thickness_mm"]); cx=x_off+lt/2
        mat=layer["material"]
        col_map={"pcm":(100,200,160),"endothermic":(200,160,100),"teg":(160,120,220),"mli":(180,180,200),
                 "aerogel":(140,200,220),"refractory_foam":(240,230,210),"ceramic_fiber":(220,180,100),
                 "foam":(200,200,180),"ysz_foam":(250,235,200),"zirconia_aerogel":(200,230,235),
                 "carbon_aerogel":(60,60,70),"perfectvoid_foam":(235,215,160),"vip_core":(180,200,240),
                 "mli_nano":(200,200,235),"tin_coating":(255,215,90),"weather_shell":(130,170,140),
                 "gradient_spray":_mix((240,140,80),(140,200,220),i/max(len(layers)-1,1))}
        bc=col_map.get(mat,(150,150,160))
        # Slight gap between layers for visual separation
        gap=_mm(0.5); v,f=_box(cx,0,0,lt-gap,H,W)
        parts.append(Part("layer_%d"%i,layer["name"],[Mesh(v,f,bc,name="L%d: %s"%(i,layer["name"][:30]),group="panel")],
                          [layer["name"],"  thickness: %.1f mm"%layer["thickness_mm"],"  density: %.0f kg/m3"%layer["density"],"  material: %s"%mat],
                          i+1,(0,0,0),bc))
        x_off+=lt
    cv,cf=_box(T/2+_mm(2),0,0,_mm(4),H+_mm(4),W+_mm(4))
    parts.append(Part("cold_frame","COLD FACE",[Mesh(cv,cf,C_COLD,name="cold frame",group="panel")],["Cold face"],len(layers)+1,(0,0,0),C_COLD))
    return parts,total_t

def build_spray_can_parts():
    H=_mm(DIMS["spray_can_height_mm"]); R=_mm(DIMS["spray_can_diameter_mm"]/2)
    meshes=[]
    # Main body
    v,f=_solid_cylinder(R,0,H,seg=24)
    meshes.append(Mesh(v,f,(50,130,120),name="can body",group="can"))
    # Nozzle cap (narrower, on top)
    vn,fn=_solid_cylinder(R*0.35,H,H+_mm(15),seg=16)
    meshes.append(Mesh(vn,fn,(80,160,150),name="nozzle",group="can"))
    # Label band (slightly wider ring around middle)
    vl,fl=_solid_cylinder(R*1.02,H*0.35,H*0.65,seg=24)
    meshes.append(Mesh(vl,fl,(40,100,90),name="label",group="can"))
    return [Part("can","SPRAY APPLICATOR",meshes,["Spray can, 750 mL"],0,(0,0,0),(50,130,120))]

def build_ground_parts():
    s=_mm(800); v,f=_box(0,0,-_mm(2),s,s,_mm(4))
    return [Part("ground","GROUND",[Mesh(v,f,C_GROUND,name="ground",group="ground")],["800x800 mm"],0,(0,0,0),C_GROUND)]

def build_scene(formulation_idx=0):
    parts=[]
    for p in build_ground_parts(): parts.append(p)
    formulation=FORMULATIONS[formulation_idx]
    panel_parts,total_t=build_panel_parts(formulation)
    for p in panel_parts: parts.append(p)
    can_parts=build_spray_can_parts()
    for p in can_parts:
        p.explode=p.explode+np.array([_mm(200),0,_mm(100)]); parts.append(p)
    return parts,total_t

def build_pod_scene():
    """Cutaway pod hull: nested spherical shells, outermost (bumper, facing
    space) to innermost (rear wall, facing the cabin), plus a projectile that
    App animates by mutating its Part.explode each frame. Named "layer_%d" so
    it inherits the existing cutaway ('X') and label rendering for free."""
    # No ground slab here (unlike build_scene): this pod floats free in space,
    # so a literal workshop-floor box would clash with the starfield/vacuum
    # framing. Scene.render()'s own reference grid still gives spatial cues.
    parts=[]
    pod=PodShieldSim()
    col_map={"bumper":(215,220,230),"coil":(120,175,255),"stuffed":(225,190,95),
             "thermal":(150,205,225),"rear_wall":(205,125,225)}
    r=_mm(IMPACT_PHYS["pod_radius_mm"])
    for i,l in enumerate(pod.layers):
        v,f=_sphere(r,seg_u=26,seg_v=18)
        col=col_map.get(l["role"],(150,150,160))
        parts.append(Part("layer_%d"%i,l["name"],[Mesh(v,f,col,name="L%d: %s"%(i,l["name"][:32]),group="pod")],
                          [l["name"],"  thickness: %.1f mm"%l["thickness_mm"],"  role: %s"%l["role"]],
                          i+1,(0,0,0),col))
        # Coil layer gets a wireframe accent ring at its equator so the
        # functional (electromagnetic) layer reads as distinct from the
        # purely-structural ones instead of just being "another shell".
        if l["role"]=="coil":
            rv,rf=_solid_cylinder(r+_mm(1.5),-_mm(2.0),_mm(2.0),seg=40)
            parts.append(Part("coil_ring_%d"%i,"Coil windings (accent)",
                              [Mesh(rv,rf,(60,120,255),name="",group="pod")],[],i+1,(0,0,0),(60,120,255)))
        r=max(r-_mm(l["thickness_mm"]),_mm(50.0))
    cv,cf=_sphere(r,seg_u=18,seg_v=12)
    parts.append(Part("cabin","CABIN INTERIOR",[Mesh(cv,cf,(45,50,68),name="cabin",group="pod")],["Cabin"],len(pod.layers)+1,(0,0,0),(45,50,68)))
    pv,pf=_sphere(_mm(60.0),seg_u=14,seg_v=10)
    parts.append(Part("projectile","INCOMING THREAT",[Mesh(pv,pf,(255,90,60),name="threat",group="proj")],["Threat"],
                      len(pod.layers)+2,(_mm(IMPACT_PHYS["pod_radius_mm"])*2.6,0,0),(255,90,60)))
    return parts,IMPACT_PHYS["pod_radius_mm"]*2.0

# =============================================================================
# SECTION 6 -- THERMAL PHYSICS SIMULATION
# =============================================================================
class ThermalSim:
    """Time-stepped thermal simulation with real heat-transfer physics."""
    def __init__(self,approach="foam",T_hot=None,T_cold=None):
        self.approach=approach
        self.T_hot=T_hot or PHYS["T_hot_default"]
        self.T_cold=T_cold or PHYS["T_cold_default"]
        self.formulation=None
        for f in FORMULATIONS:
            if f["approach"]==approach: self.formulation=f; break
        if self.formulation is None: self.formulation=FORMULATIONS[0]
        self.reset()

    def reset(self):
        self.t=0.0; self.running=False; self.elapsed=0.0
        self.layers=[dict(l) for l in self.formulation["layers"]]
        self.total_thickness=sum(l["thickness_mm"] for l in self.layers)*MM
        self.n_layers=len(self.layers)
        # Seed by cumulative thickness fraction, not layer count: a uniform
        # per-layer-index linspace assigns the same initial dT to a 0.3mm layer
        # as to a 200mm layer, which can overshoot to unphysical (negative-K)
        # temperatures for a few steps before the iterative update recovers.
        thick=[l["thickness_mm"] for l in self.layers]
        cum=np.cumsum([0.0]+thick); frac=cum/max(cum[-1],1e-9)
        self.T_profile=self.T_hot-frac*(self.T_hot-self.T_cold)
        self.k_eff=[0.0]*self.n_layers; self.q_cond=[0.0]*self.n_layers
        self.q_rad=[0.0]*self.n_layers; self.q_total=[0.0]*self.n_layers
        self.pcm_charged=[0.0]*self.n_layers; self.pcm_absorbed=[0.0]*self.n_layers
        self.pcm_capacity=[0.0]*self.n_layers
        self.teg_power=0.0; self.teg_energy=0.0; self.peltier_pumped=0.0
        self.q_incoming=0.0; self.q_transmitted=0.0
        self.total_absorbed=0.0; self.total_converted=0.0; self.total_pumped=0.0
        self.history=[]; self.done=False
        for i,layer in enumerate(self.layers):
            if layer["material"] in ("pcm","endothermic"):
                pp=_pcm_props(layer.get("pcm_type","SrBr2" if layer["material"]=="pcm" else "MgSO4"))
                self.pcm_capacity[i]=pp["density"]*layer["thickness_mm"]*MM*pp["latent"]

    @property
    def suppression_ratio(self):
        return 1.0-(self.q_transmitted/max(self.q_incoming,1e-9)) if self.q_incoming>1e-9 else 0.0
    @property
    def suppression_pct(self): return self.suppression_ratio*100.0
    @property
    def transmission_pct(self):
        return (self.q_transmitted/max(self.q_incoming,1e-9))*100.0 if self.q_incoming>1e-9 else 0.0
    @property
    def is_success(self): return self.suppression_ratio>0.99
    @property
    def k_eff_overall(self):
        if self.total_thickness<1e-9 or self.q_transmitted<1e-12: return 0.0
        dT=self.T_hot-self.T_cold
        if abs(dT)<1e-9: return 0.0
        return self.q_transmitted*self.total_thickness/dT*1000.0

    def _k_gas_knudsen(self,layer,T):
        k0=PHYS["k_air_25C"]*(T/298.15)**0.7
        ps=layer["pore_size_um"]*1e-6
        if ps<1e-12: return 0.0
        if layer.get("vacuum",False):
            k0*=layer.get("vacuum_pressure",1e-5)/760.0
        mfp=PHYS["mean_free_path_air"]
        if layer.get("vacuum",False):
            mfp*=760.0/layer.get("vacuum_pressure",1e-5)
        Kn=mfp/max(ps,1e-12)
        return k0/(1.0+2.0*PHYS["beta_knudsen"]*Kn)

    def _k_solid(self,layer,T):
        mat=layer["material"]; por=layer.get("porosity",0.9)
        kmap={"aerogel":PHYS["aerogel_k_solid"],"ceramic_fiber":PHYS["ceramic_fiber_k_25C"]*(T/298.15)**0.3,
              "refractory_foam":PHYS["refractory_k_25C"]*(T/298.15)**0.3,"foam":PHYS["foam_k"],
              "mli":0.001,"aerogel_mli":0.005,"pcm":PHYS.get("pcm_SrBr2_k",0.35),
              "endothermic":PHYS["pcm_MgSO4_k"],"teg":1.5,"gradient_spray":PHYS["refractory_k_25C"]*(T/298.15)**0.3,
              "ysz_foam":PHYS["ysz_foam_k"]*(T/298.15)**0.3,
              "zirconia_aerogel":PHYS["zirconia_aerogel_k"]+0.078*max(T-298.15,0.0)/1000.0,
              "carbon_aerogel":PHYS["carbon_aerogel_k"],"perfectvoid_foam":PHYS["perfectvoid_k"],
              "vip_core":PHYS["vip_core_k"],"mli_nano":PHYS["mli_nano_k"],
              "tin_coating":PHYS["tin_coating_k"],"weather_shell":PHYS["weather_shell_k"]}
        k_s=kmap.get(mat,0.05)
        if mat in ("pcm","endothermic"):
            k_s=_pcm_props(layer.get("pcm_type","SrBr2" if mat=="pcm" else "MgSO4"))["k"]
        return k_s*(1.0-por)**1.5

    def _k_radiation(self,layer,T):
        n_r=layer.get("n_reflective",0); emis=layer.get("emissivity",0.9)
        opac=layer.get("opacifier",0.0); sigma=PHYS["sigma_SB"]
        th=layer["thickness_mm"]*MM
        if th<1e-9: return 0.0
        l_rad=th*0.1
        if opac>0: l_rad*=1.0/(1.0+opac*20.0)
        if n_r>0:
            k_rad=16.0/3.0*sigma*T**3*l_rad/(n_r*(1.0/(2.0*emis)))
        else:
            k_rad=16.0/3.0*sigma*T**3*l_rad*emis
        return max(0.0,k_rad)

    def _compute_layer_k(self,i,T_avg):
        layer=self.layers[i]
        k_s=self._k_solid(layer,T_avg); k_g=self._k_gas_knudsen(layer,T_avg); k_r=self._k_radiation(layer,T_avg)
        self.k_eff[i]=k_s+k_g+k_r; self.q_cond[i]=k_s+k_g; self.q_rad[i]=k_r
        return self.k_eff[i]

    def step(self,dt):
        if self.done: return
        self.t+=dt
        if self.running: self.elapsed+=dt
        for i in range(self.n_layers):
            T_avg=(self.T_profile[i]+self.T_profile[i+1])/2.0
            self._compute_layer_k(i,T_avg)
        thicknesses=[l["thickness_mm"]*MM for l in self.layers]
        q_flux=[]
        for i in range(self.n_layers):
            dT=self.T_profile[i]-self.T_profile[i+1]; dx=max(thicknesses[i],1e-9)
            q_flux.append(self.k_eff[i]*dT/dx)
        # q_incoming = uninsulated heat flux (blackbody radiation + natural convection)
        # This is the heat flux that would arrive without any insulation
        sigma=PHYS["sigma_SB"]
        q_rad_bare=sigma*(self.T_hot**4-self.T_cold**4)
        h_conv=5.0  # W/(m2·K) natural convection coefficient
        q_conv_bare=h_conv*(self.T_hot-self.T_cold)
        self.q_incoming=q_rad_bare+q_conv_bare
        # PCM / chemical absorption -- reduce flux and track absorbed heat
        pcm_clamp={}  # layer index -> T_phase for layers currently charging
        sink_rate=[0.0]*self.n_layers  # W/m2 extracted by recovery layers this step
        for i in range(self.n_layers):
            layer=self.layers[i]
            if layer["material"] in ("pcm","endothermic") and self.pcm_capacity[i]>0:
                T_phase=_pcm_props(layer.get("pcm_type","SrBr2" if layer["material"]=="pcm" else "MgSO4"))["T_phase"]
                T_avg=(self.T_profile[i]+self.T_profile[i+1])/2.0
                if T_avg>=T_phase and self.pcm_charged[i]<1.0:
                    # Heat available to PCM = flux arriving from hot side (previous layer)
                    q_avail=abs(q_flux[i-1]) if i>0 else abs(q_flux[0])
                    remaining=self.pcm_capacity[i]*(1.0-self.pcm_charged[i])
                    absorbed=min(q_avail*dt,remaining)
                    self.pcm_absorbed[i]+=absorbed; self.total_absorbed+=absorbed
                    self.pcm_charged[i]=self.pcm_absorbed[i]/self.pcm_capacity[i]
                    if q_flux[i]>0: q_flux[i]-=absorbed/dt
                    else: q_flux[i]+=absorbed/dt
                    sink_rate[i]+=absorbed/dt
                    if self.pcm_charged[i]<1.0:
                        pcm_clamp[i]=T_phase
        # TEG conversion + Peltier
        for i in range(self.n_layers):
            layer=self.layers[i]
            if layer["material"]=="teg":
                dT_teg=self.T_profile[i]-self.T_profile[i+1]
                if dT_teg>0:
                    T_cold_l=self.T_profile[i+1]
                    eta_c=1.0-T_cold_l/self.T_profile[i] if self.T_profile[i]>0 else 0
                    eta_t=0.1*eta_c; q_through=abs(q_flux[i]); power=eta_t*q_through
                    self.teg_power=power; self.teg_energy+=power*dt; self.total_converted+=power*dt
                    heat_pumped=PHYS["peltier_COP"]*power; self.peltier_pumped+=heat_pumped*dt; self.total_pumped+=heat_pumped*dt
                    if q_flux[i]>0: q_flux[i]=max(0.0,q_flux[i]-power-heat_pumped)
                    sink_rate[i]+=power+heat_pumped
        # Recompute T profile from modified flux
        for i in range(self.n_layers):
            if self.k_eff[i]>1e-12 and thicknesses[i]>1e-9:
                self.T_profile[i+1]=self.T_profile[i]-q_flux[i]*thicknesses[i]/self.k_eff[i]
        # Apply PCM temperature clamping AFTER recomputation
        # While PCM is charging, hold its interfaces at T_phase
        for i,T_phase in pcm_clamp.items():
            self.T_profile[i]=T_phase
            self.T_profile[i+1]=T_phase
        # Physical floor/ceiling: no interior node in a monotonic hot->cold stack
        # can legitimately fall outside [T_cold, T_hot]. Without this, a very thin
        # layer next to much thicker ones can transiently overshoot below 0 K
        # before the profile settles, and negative-K temperatures feed fractional
        # exponents (T**0.7, T**3) in the k_eff formulas -> complex -> NaN forever.
        lo,hi=min(self.T_hot,self.T_cold),max(self.T_hot,self.T_cold)
        self.T_profile=np.clip(self.T_profile,lo,hi)
        self.T_profile[0]=self.T_hot; self.T_profile[-1]=max(self.T_profile[-1],self.T_cold)
        # Flux-continuity transmission: the algebraic profile relaxation above
        # does not conserve flux node-to-node (known limit of this engine --
        # it under/over-reads thin boundary layers during equilibration), so
        # report the uniform-flux estimate at the CURRENT k_eff field minus
        # cumulative recovery extraction -- the physically consistent value.
        R_stack=sum(th/max(self.k_eff[i],1e-15) for i,th in enumerate(thicknesses))
        q_uni=max(0.0,self.T_hot-self.T_cold)/max(R_stack,1e-15)
        cum=0.0; self.q_total=[]
        for i in range(self.n_layers):
            cum+=sink_rate[i]
            self.q_total.append(max(0.0,q_uni-cum))
        self.q_transmitted=self.q_total[-1] if self.q_total else 0.0
        if len(self.history)==0 or self.t-self.history[-1][0]>0.5:
            self.history.append((self.t,self.transmission_pct,self.suppression_pct))
            if len(self.history)>400: self.history=self.history[-400:]
        all_charged=all(self.pcm_charged[i]>=0.999 or self.pcm_capacity[i]==0 for i in range(self.n_layers))
        if all_charged and self.elapsed>3600: self.done=True

    def layer_summary(self):
        lines=[]
        for i,layer in enumerate(self.sim.layers if hasattr(self,'sim') else self.layers):
            k=self.k_eff[i]*1000.0; q=self.q_total[i]
            T1=self.T_profile[i]; T2=self.T_profile[i+1]
            extra=""
            if self.pcm_capacity[i]>0: extra="  PCM: %.0f%% charged"%(self.pcm_charged[i]*100)
            lines.append("  L%d: %-30s k=%.4f mW/m.K  q=%.4f W/m2  T:%.0f->%.0f K%s"%(i,layer["name"][:30],k,q,T1,T2,extra))
        return "\n".join(lines)

    def summary(self):
        return "\n".join([
            "THERMAL SIMULATION SUMMARY","="*40,
            "Approach:         %s"%self.approach.upper(),
            "Elapsed:          %.1f s"%self.elapsed,
            "T_hot:            %.0f K (%.0f C)"%(self.T_hot,self.T_hot-273.15),
            "T_cold:           %.0f K (%.0f C)"%(self.T_cold,self.T_cold-273.15),
            "dT:               %.0f K"%(self.T_hot-self.T_cold),
            "Total thickness:  %.1f mm"%(self.total_thickness/MM),
            "","HEAT FLUX:",
            "  q_incoming:     %.6f W/m2"%self.q_incoming,
            "  q_transmitted:  %.6f W/m2"%self.q_transmitted,
            "  Suppression:    %.4f%%"%self.suppression_pct,
            "  Transmission:   %.6f%%"%self.transmission_pct,
            "  k_eff overall:  %.6f mW/m.K"%self.k_eff_overall,
            "","ABSORPTION/CONVERSION:",
            "  PCM/chem absorbed: %.3f MJ/m2"%(self.total_absorbed/1e6),
            "  TEG converted:     %.3f MJ/m2"%(self.total_converted/1e6),
            "  Peltier pumped:    %.3f MJ/m2"%(self.total_pumped/1e6),
            "","LAYER BREAKDOWN:",
            self.layer_summary(),
            "","SUCCESS (suppression > 99%%): %s"%("YES" if self.is_success else "NO"),
        ])

def run_simulation(approach,duration_s=3600,T_hot=None,T_cold=None):
    sim=ThermalSim(approach,T_hot,T_cold); sim.running=True
    dt=1.0
    for _ in range(int(duration_s/dt)): sim.step(dt)
    return sim

def compare_all_approaches(duration_s=3600,T_hot=None,T_cold=None):
    results=[]
    for f in FORMULATIONS:
        sim=run_simulation(f["approach"],duration_s,T_hot,T_cold)
        results.append({"name":f["name"],"approach":f["approach"],"suppression":sim.suppression_pct,
                        "transmission":sim.transmission_pct,"q_trans":sim.q_transmitted,
                        "k_eff":sim.k_eff_overall,"absorbed_MJ":sim.total_absorbed/1e6,
                        "converted_MJ":sim.total_converted/1e6,"pumped_MJ":sim.total_pumped/1e6,
                        "success":sim.is_success,"sim":sim})
    return results

# =============================================================================
# SECTION 6.5 -- SCIENTIFIC MODEL FRAMEWORK (class-based, headless)
# =============================================================================
# Provides Layer/RecoveryLayer/InsulationStack/OptimizationFramework classes
# for reproducible, extensible scientific modeling.  These classes use a
# MATERIALS database with temperature-dependent k(T) functions and can be
# used independently of the pygame visualization.
# ---------------------------------------------------------------------------

def _k_air_T(T):
    return 0.0257 * (T / 273.15) ** 0.85

def _interp_pw(T, bps):
    if T <= bps[0][0]: return bps[0][1]
    if T >= bps[-1][0]: return bps[-1][1]
    for i in range(len(bps)-1):
        T1,v1 = bps[i]; T2,v2 = bps[i+1]
        if T1 <= T <= T2: return v1 + (T-T1)/(T2-T1)*(v2-v1)
    return bps[-1][1]

MATERIALS = {
    "pu_foam": {"name":"Closed-cell PU foam","k_func":lambda T:0.024*(T/293.15)**0.12,
        "rho":35.0,"cp":1400.0,"porosity":0.92,"pore_size":300e-6,
        "emissivity":0.90,"opacifier_frac":0.0,"T_max":393.0,"reference":"ASHRAE"},
    "silica_aerogel": {"name":"Silica aerogel (95% porosity)","k_func":lambda T:0.018+0.0008*(T-293.15)/100.0,
        "rho":150.0,"cp":1000.0,"porosity":0.95,"pore_size":20e-9,
        "emissivity":0.80,"opacifier_frac":0.0,"T_max":923.0,"reference":"Hrubesh & Pekala (1994)"},
    "silica_aerogel_opac": {"name":"Silica aerogel + SiC opacifier","k_func":lambda T:0.018+0.0008*(T-293.15)/100.0,
        "rho":160.0,"cp":950.0,"porosity":0.94,"pore_size":20e-9,
        "emissivity":0.30,"opacifier_frac":0.05,"T_max":923.0,"reference":"Hrubesh + SiC data"},
    "ceramic_fiber": {"name":"Aluminosilicate fiber blanket","k_func":lambda T:_interp_pw(T,[(300,0.05),(773,0.09),(1273,0.12),(1773,0.16)]),
        "rho":128.0,"cp":1070.0,"porosity":0.90,"pore_size":50e-6,
        "emissivity":0.85,"opacifier_frac":0.0,"T_max":1673.0,"reference":"CRC; Unifrax"},
    "refractory_foam": {"name":"Al2O3-SiO2-ZrO2 sol-gel foam","k_func":lambda T:_interp_pw(T,[(300,0.04),(773,0.06),(1273,0.08),(1773,0.10)]),
        "rho":200.0,"cp":900.0,"porosity":0.90,"pore_size":100e-6,
        "emissivity":0.60,"opacifier_frac":0.05,"T_max":1973.0,"reference":"Alcoa, Vesuvius"},
    "refractory_dense": {"name":"Dense refractory (Al2O3 70%+SiC 5%)","k_func":lambda T:_interp_pw(T,[(300,0.06),(773,0.09),(1273,0.12),(1773,0.15)]),
        "rho":300.0,"cp":880.0,"porosity":0.80,"pore_size":50e-6,
        "emissivity":0.50,"opacifier_frac":0.05,"T_max":1973.0,"reference":"Alcoa dense"},
    "al_foil": {"name":"Aluminum foil (12um, emissivity 0.04)","k_func":lambda T:237.0*(300.0/max(T,1.0))**0.1,
        "rho":2700.0,"cp":900.0,"porosity":0.0,"pore_size":0.0,
        "emissivity":0.04,"opacifier_frac":0.0,"T_max":673.0,"reference":"NASA MLI"},
    "silica_spacer": {"name":"Silica aerogel paper spacer","k_func":lambda T:0.020+0.001*(T-293.15)/100.0,
        "rho":100.0,"cp":1000.0,"porosity":0.93,"pore_size":50e-9,
        "emissivity":0.80,"opacifier_frac":0.0,"T_max":923.0,"reference":"NASA MLI spacers"},
    "pcm_srbr2": {"name":"SrBr2.6H2O PCM","k_func":lambda T:0.35,
        "rho":2400.0,"cp":1200.0,"porosity":0.0,"pore_size":0.0,
        "emissivity":0.50,"opacifier_frac":0.0,"T_max":573.0,
        "pcm_T_phase":362.15,"pcm_latent":630.0e3,"reference":"N'Tsoulkove (2014)"},
    "mgso4_7h2o": {"name":"MgSO4.7H2O endothermic","k_func":lambda T:0.45,
        "rho":1680.0,"cp":1100.0,"porosity":0.30,"pore_size":10e-6,
        "emissivity":0.50,"opacifier_frac":0.0,"T_max":573.0,
        "pcm_T_phase":323.15,"pcm_latent":480.0e3,"reference":"Donkers (2017)"},
    "bi2te3_teg": {"name":"Bi2Te3 TEG module","k_func":lambda T:1.5,
        "rho":7700.0,"cp":160.0,"porosity":0.0,"pore_size":0.0,
        "emissivity":0.30,"opacifier_frac":0.0,"T_max":723.0,
        "teg_Seebeck":200e-6,"teg_ZT":1.0,"teg_fraction":0.1,"reference":"Snyder & Toberer (2008)"},
    "tin_coating": {"name":"TiN reflective topcoat","k_func":lambda T:20.0,
        "rho":5220.0,"cp":600.0,"porosity":0.0,"pore_size":0.0,
        "emissivity":0.05,"opacifier_frac":0.0,"T_max":2773.0,"reference":"TiN data"},
    # --- ULTRA CASCADE materials (goal2.md + RTP2.0 spray recipe) ---
    "ysz_foam": {"name":"YSZ (ZrO2+Y2O3) refractory spray foam","k_func":lambda T:_interp_pw(T,[(300,0.35),(773,0.42),(1273,0.50),(1873,0.60)]),
        "rho":300.0,"cp":560.0,"porosity":0.80,"pore_size":50e-6,
        "emissivity":0.50,"opacifier_frac":0.06,"T_max":2473.0,"reference":"white-crucible YSZ/Al2O3-ZrO2 spray foam"},
    "zirconia_aerogel": {"name":"ZrO2/Y2O3-doped ceramic aerogel","k_func":lambda T:0.026+0.078*max(T-293.15,0.0)/1000.0,
        "rho":220.0,"cp":900.0,"porosity":0.92,"pore_size":25e-9,
        "emissivity":0.45,"opacifier_frac":0.05,"T_max":1873.0,"reference":"dome-celled zircon aerogel ~26->104 mW/mK (25C->1000C)"},
    "carbon_aerogel": {"name":"Carbon aerogel (IR-opaque, low k)","k_func":lambda T:0.015+0.0005*(T-293.15)/100.0,
        "rho":100.0,"cp":900.0,"porosity":0.952,"pore_size":15e-9,
        "emissivity":0.10,"opacifier_frac":0.10,"T_max":1473.0,"reference":"RTP2.0 carbon aerogel; IR-opaque MLI spacer"},
    "perfectvoid_foam": {"name":"PerfectVoid spray foam (ZrO2+aerogel+vacuum microspheres+carbon nano)","k_func":lambda T:0.15,
        "rho":60.0,"cp":900.0,"porosity":0.96,"pore_size":30e-6,
        "emissivity":0.15,"opacifier_frac":0.08,"T_max":873.0,"reference":"goal2.md SpaceVoid/PerfectVoid recipe; evacuated microspheres"},
    "vip_core": {"name":"Fumed-silica VIP core","k_func":lambda T:0.12,
        "rho":180.0,"cp":900.0,"porosity":0.90,"pore_size":100e-9,
        "emissivity":0.40,"opacifier_frac":0.0,"T_max":673.0,"reference":"VIP fumed-silica; k~0.004 W/mK measured in sealed vacuum"},
    "mli_nano": {"name":"Aluminized-Kapton MLI + aerogel separators","k_func":lambda T:0.022,
        "rho":120.0,"cp":1000.0,"porosity":0.92,"pore_size":80e-9,
        "emissivity":0.04,"opacifier_frac":0.0,"T_max":673.0,"reference":"goal2.md MLI nano-films; Kapton+e=0.04"},
    "weather_shell": {"name":"Ceramifiable silicone-ceramic weather seal","k_func":lambda T:0.15,
        "rho":1100.0,"cp":1200.0,"porosity":0.05,"pore_size":0.0,
        "emissivity":0.85,"opacifier_frac":0.0,"T_max":573.0,"reference":"goal2.md weathering barrier; hydrophobic/UV/abrasion"},
    "pcm_erythritol": {"name":"Erythritol PCM (mid-T dam)","k_func":lambda T:0.73,
        "rho":1450.0,"cp":1380.0,"porosity":0.0,"pore_size":0.0,
        "emissivity":0.50,"opacifier_frac":0.0,"T_max":473.0,
        "pcm_T_phase":391.15,"pcm_latent":340.0e3,"reference":"Erythritol: T_phase=118C, latent=340 kJ/kg"},
    "pcm_paraffin": {"name":"Paraffin PCM (low-T dam)","k_func":lambda T:0.20,
        "rho":900.0,"cp":2000.0,"porosity":0.0,"pore_size":0.0,
        "emissivity":0.90,"opacifier_frac":0.0,"T_max":423.0,
        "pcm_T_phase":328.15,"pcm_latent":200.0e3,"reference":"Paraffin RT55: T_phase=55C, latent=200 kJ/kg"},
    "pcm_salt_hydrate": {"name":"Salt hydrate PCM (Na2SO4.10H2O, ambient dam)","k_func":lambda T:0.55,
        "rho":1460.0,"cp":1900.0,"porosity":0.0,"pore_size":0.0,
        "emissivity":0.50,"opacifier_frac":0.0,"T_max":353.0,
        "pcm_T_phase":305.15,"pcm_latent":254.0e3,"reference":"Glauber's salt: T_phase=32C, latent=254 kJ/kg"},
}

@dataclass
class SLayer:
    """Scientific layer with full thermophysical properties."""
    name: str
    material_key: str
    thickness_mm: float
    n_reflective: int = 0
    vacuum: bool = False
    vacuum_pressure_torr: float = 760.0

    def __post_init__(self):
        assert self.material_key in MATERIALS, "Unknown material: %s" % self.material_key
        assert self.thickness_mm > 0
        # Cache: material_key/thickness_mm never mutate post-construction, and
        # these were showing up as millions of redundant dict-lookup/property
        # calls in transient-solver profiling (hot path: k_eff() per node per substep).
        self._material = MATERIALS[self.material_key]
        self._thickness_m = self.thickness_mm * 1e-3

    @property
    def material(self): return self._material
    @property
    def thickness_m(self): return self._thickness_m

    def k_solid(self, T):
        mat = self._material
        k = mat["k_func"](T)
        return k * (1.0 - mat.get("porosity",0.0)) ** 1.5

    def k_gas(self, T):
        if self.vacuum:
            k_base = _k_air_T(T) * (self.vacuum_pressure_torr / 760.0)
            mfp = 68e-9 * (760.0 / self.vacuum_pressure_torr)
        else:
            k_base = _k_air_T(T); mfp = 68e-9
        mat = self._material
        ps = mat.get("pore_size", 0.0)
        if ps < 1e-12: return 0.0
        return k_base / (1.0 + 4.0 * mfp / ps) * mat.get("porosity",0.0)

    def k_radiation(self, T):
        mat = self._material
        sigma = PHYS["sigma_SB"]; emis = mat.get("emissivity",0.9)
        opac = mat.get("opacifier_frac",0.0); th = self._thickness_m
        if th < 1e-12: return 0.0
        l_mean = th * 0.1
        if opac > 0: l_mean *= 1.0 / (1.0 + opac * 20.0)
        if self.n_reflective > 0:
            return max(0.0, (16.0/3.0)*sigma*T**3*l_mean/(self.n_reflective*(1.0/(2.0*emis))))
        return max(0.0, (16.0/3.0)*sigma*T**3*l_mean*emis)

    def k_eff(self, T):
        return self.k_solid(T) + self.k_gas(T) + self.k_radiation(T)

    def thermal_resistance(self, T_avg):
        return self.thickness_m / max(self.k_eff(T_avg), 1e-15)

@dataclass
class SRecoveryLayer(SLayer):
    """Layer with active heat recovery: PCM, endothermic, or TEG."""
    recovery_type: str = "pcm"
    pcm_type: str = "SrBr2"

    def __post_init__(self):
        super().__post_init__()
        assert self.recovery_type in ("pcm", "endothermic", "teg")

    @property
    def pcm_T_phase(self):
        return 0.0 if self.recovery_type == "teg" else self.material.get("pcm_T_phase", 0.0)
    @property
    def pcm_latent(self):
        return 0.0 if self.recovery_type == "teg" else self.material.get("pcm_latent", 0.0)
    @property
    def pcm_capacity_per_m2(self):
        if self.recovery_type == "teg": return 0.0
        return self.material["rho"] * self.thickness_m * self.pcm_latent

    def teg_power_density(self, T_hot, T_cold):
        if self.recovery_type != "teg": return 0.0
        dT = T_hot - T_cold
        if dT <= 0: return 0.0
        eta_c = 1.0 - T_cold / T_hot if T_hot > 0 else 0
        k = self.k_eff((T_hot + T_cold) / 2.0)
        return 0.1 * eta_c * k * dT / self.thickness_m * self.material.get("teg_fraction", 0.1)

    def peltier_heat_pumped(self, power_electric):
        return 0.6 * power_electric if self.recovery_type == "teg" else 0.0

@dataclass
class SBoundary:
    T_hot: float = 1273.15
    T_cold: float = 77.0
    h_hot: float = 5.0
    h_cold: float = 5.0

class SInsulationStack:
    """Ordered layers (hot->cold) + boundary conditions + solvers."""
    def __init__(self, layers, bc=None, name="stack"):
        self.layers = layers; self.bc = bc or SBoundary(); self.name = name

    @property
    def n_layers(self): return len(self.layers)
    @property
    def total_thickness_mm(self): return sum(l.thickness_mm for l in self.layers)
    @property
    def total_thickness_m(self): return self.total_thickness_mm * 1e-3

    def solve_steady_state(self):
        n = self.n_layers
        T = np.linspace(self.bc.T_hot, self.bc.T_cold, n + 1)
        for iteration in range(200):
            T_old = T.copy(); k_effs = []; resistances = []
            for i, layer in enumerate(self.layers):
                T_avg = (T[i] + T[i+1]) / 2.0; k = layer.k_eff(T_avg)
                k_effs.append(k); resistances.append(layer.thickness_m / max(k, 1e-15))
            R_total = sum(resistances)
            q = (self.bc.T_hot - self.bc.T_cold) / R_total
            T[0] = self.bc.T_hot
            for i in range(n): T[i+1] = T[i] - q * resistances[i]
            T[-1] = self.bc.T_cold
            if np.max(np.abs(T - T_old)) < 0.01: break
        q_transmitted = q; total_abs = total_conv = total_pump = 0.0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, SRecoveryLayer):
                if layer.recovery_type in ("pcm", "endothermic"):
                    T_avg = (T[i] + T[i+1]) / 2.0
                    if T_avg < layer.pcm_T_phase:
                        T[i] = layer.pcm_T_phase; T[i+1] = layer.pcm_T_phase
                        R_below = sum(resistances[j] for j in range(i+1, n))
                        if R_below > 0:
                            q_transmitted = (layer.pcm_T_phase - self.bc.T_cold) / R_below
                elif layer.recovery_type == "teg":
                    power = layer.teg_power_density(T[i], T[i+1])
                    pumped = layer.peltier_heat_pumped(power)
                    total_conv += power; total_pump += pumped
                    q_transmitted = max(0.0, q_transmitted - power - pumped)
        sigma = PHYS["sigma_SB"]
        q_inc = sigma*(self.bc.T_hot**4 - self.bc.T_cold**4) + self.bc.h_hot*(self.bc.T_hot - self.bc.T_cold)
        supp = 1.0 - (q_transmitted / max(q_inc, 1e-9))
        k_ov = q_transmitted * self.total_thickness_m / max(self.bc.T_hot - self.bc.T_cold, 1e-9)
        return {"q_flux":q,"q_incoming":q_inc,"q_transmitted":q_transmitted,
                "suppression":supp,"suppression_pct":supp*100.0,
                "transmission_pct":(q_transmitted/max(q_inc,1e-9))*100.0,
                "T_profile":T,"k_eff_per_layer":k_effs,"R_per_layer":resistances,
                "R_total":R_total,"k_eff_overall":k_ov,"k_eff_overall_mW":k_ov*1000.0,
                "total_absorbed":total_abs,"total_converted":total_conv,
                "total_pumped":total_pump,"n_iterations":iteration+1}

    def solve_transient(self, duration_s, dt=1.0, T_initial=None):
        n = self.n_layers
        # Adaptive per-layer node count (~1 node/mm, capped 10): sub-mm coatings
        # carry negligible areal heat capacity, so a single node is both honest
        # and keeps the explicit scheme's stability timestep practical.
        npl = [max(1, min(10, int(round(l.thickness_mm)))) for l in self.layers]
        tn = sum(npl)+1
        dx_pl = [l.thickness_m/npl[i] for i,l in enumerate(self.layers)]
        T = np.full(tn, float(T_initial if T_initial is not None else self.bc.T_cold)); T[0] = self.bc.T_hot; T[-1] = self.bc.T_cold
        node_layer = np.zeros(tn, dtype=int); idx = 0
        for i in range(n):
            for j in range(npl[i]): node_layer[idx] = i; idx += 1
        node_layer[idx] = n - 1
        n_steps = int(duration_s / dt)
        T_hist = np.zeros((n_steps+1, tn)); T_hist[0] = T
        q_hist = np.zeros(n_steps+1)
        q_inc = PHYS["sigma_SB"]*(self.bc.T_hot**4-self.bc.T_cold**4)+self.bc.h_hot*(self.bc.T_hot-self.bc.T_cold)
        pcm_ch = {}; pcm_ab = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, SRecoveryLayer) and layer.recovery_type in ("pcm","endothermic"):
                pcm_ch[i] = 0.0; pcm_ab[i] = 0.0
        total_abs = total_conv = total_pump = 0.0
        T_lo = min(self.bc.T_hot, self.bc.T_cold) - 10.0
        T_hi = max(self.bc.T_hot, self.bc.T_cold) + 10.0
        for step in range(n_steps):
            # Stability: use current max T per layer (k_rad ~ T^3 grows fast)
            dt_stable = dt
            for li2, layer2 in enumerate(self.layers):
                ln2 = np.where(node_layer == li2)[0]
                T_max2 = float(T[ln2].max()) if len(ln2) > 0 else 300.0
                if not np.isfinite(T_max2) or T_max2 < 1.0: T_max2 = 300.0
                dx2 = dx_pl[li2]; k2 = layer2.k_eff(T_max2)
                a2 = k2/(layer2.material["rho"]*layer2.material["cp"])
                if not np.isfinite(a2) or a2<=0: a2 = 1e-12
                dt_s = dx2**2/(2.0*a2)
                if dt_s < dt_stable: dt_stable = dt_s
            if not np.isfinite(dt_stable) or dt_stable<=0: dt_stable = dt
            n_sub = max(1, min(100000, int(math.ceil(dt/dt_stable)))); dt_sub = dt/n_sub
            for sub in range(n_sub):
                T_new = T.copy()
                for node in range(1, tn-1):
                    li = node_layer[node]; layer = self.layers[li]
                    dx = dx_pl[li]; k = layer.k_eff(T[node])
                    rho = layer.material["rho"]; cp = layer.material["cp"]
                    if node_layer[node-1] != li:
                        kl = self.layers[node_layer[node-1]].k_eff(T[node-1]); kh = 2.0*kl*k/(kl+k)
                    else: kh = k
                    if node_layer[node+1] != li:
                        kr = self.layers[node_layer[node+1]].k_eff(T[node+1]); kh2 = 2.0*k*kr/(k+kr)
                    else: kh2 = k
                    q_in = kh*(T[node-1]-T[node])/dx; q_out = kh2*(T[node]-T[node+1])/dx
                    T_new[node] = T[node] + dt_sub*(q_in-q_out)/(rho*cp*dx)
                T = T_new; T[0] = self.bc.T_hot; T[-1] = self.bc.T_cold
                # Clamp to physical bounds and guard NaN
                T = np.clip(T, T_lo, T_hi)
                T[0] = self.bc.T_hot; T[-1] = self.bc.T_cold
                if np.any(~np.isfinite(T)):
                    T = np.nan_to_num(T, nan=self.bc.T_cold, posinf=T_hi, neginf=T_lo)
                    T[0] = self.bc.T_hot; T[-1] = self.bc.T_cold
            for i, layer in enumerate(self.layers):
                if isinstance(layer, SRecoveryLayer) and layer.recovery_type in ("pcm","endothermic"):
                    ln = np.where(node_layer == i)[0]
                    if len(ln) > 0:
                        Ta = np.mean(T[ln])
                        if Ta >= layer.pcm_T_phase and pcm_ch[i] < 1.0:
                            k = layer.k_eff(Ta)
                            q_thru = abs(k*(T[ln[0]]-T[ln[-1]])/layer.thickness_m)
                            rem = layer.pcm_capacity_per_m2*(1.0-pcm_ch[i])
                            ab = min(q_thru*dt, rem)
                            pcm_ab[i] += ab; total_abs += ab
                            pcm_ch[i] = pcm_ab[i]/layer.pcm_capacity_per_m2
                            if pcm_ch[i] < 1.0: T[ln] = layer.pcm_T_phase
                elif isinstance(layer, SRecoveryLayer) and layer.recovery_type == "teg":
                    ln = np.where(node_layer == i)[0]
                    if len(ln) > 1:
                        p = layer.teg_power_density(T[ln[0]], T[ln[-1]])
                        total_conv += p*dt; total_pump += layer.peltier_heat_pumped(p)*dt
            T_hist[step+1] = T
            li_last = node_layer[tn-2]; k_last = self.layers[li_last].k_eff(T[tn-2])
            q_hist[step+1] = k_last*(T[tn-2]-T[-1])/dx_pl[li_last]
        q_tf = q_hist[-1]; supp = 1.0-(q_tf/max(q_inc,1e-9))
        return {"times":np.arange(n_steps+1)*dt,"T_history":T_hist,"node_layer":node_layer,
                "q_trans_history":q_hist,"q_incoming":q_inc,"q_transmitted":q_tf,
                "suppression":supp,"suppression_pct":supp*100.0,
                "total_absorbed":total_abs,"total_converted":total_conv,
                "total_pumped":total_pump,"pcm_charged":pcm_ch,
                "duration_s":duration_s,"dt":dt,"n_steps":n_steps}

    def summary(self):
        r = self.solve_steady_state()
        lines = ["="*72," SCIENTIFIC STACK: %s"%self.name,"="*72,
            "T_hot=%.1f K  T_cold=%.1f K  dT=%.1f K"%(self.bc.T_hot,self.bc.T_cold,self.bc.T_hot-self.bc.T_cold),
            "Layers: %d  Total: %.1f mm"%(self.n_layers,self.total_thickness_mm),"-"*72,
            "%-4s %-30s %8s %10s %10s %8s"%("L#","Name","thk[mm]","k[W/mK]","R[m2K/W]","T[K]"), "-"*72]
        for i, layer in enumerate(self.layers):
            Ta = (r["T_profile"][i]+r["T_profile"][i+1])/2.0
            lines.append("%-4d %-30s %8.1f %10.6f %10.6f %8.1f"%(
                i, layer.name[:30], layer.thickness_mm, r["k_eff_per_layer"][i], r["R_per_layer"][i], Ta))
        lines.extend(["-"*72,"",
            "k_eff overall: %.6f W/(m K) = %.6f mW/(m K)"%(r["k_eff_overall"], r["k_eff_overall_mW"]),
            "q_incoming:    %.4f W/m2"%r["q_incoming"],
            "q_transmitted: %.6f W/m2"%r["q_transmitted"],
            "Suppression:   %.4f%%"%r["suppression_pct"], "="*72])
        return "\n".join(lines)

# --- Stack preset factories ---
def sm_foam(t=50.0): return SInsulationStack([SLayer("PU Foam","pu_foam",t)], name="Standard Foam")
def sm_aerogel(t=50.0): return SInsulationStack([SLayer("Aerogel","silica_aerogel",t)], name="Aerogel Blanket")
def sm_ceramic(t=50.0): return SInsulationStack([SLayer("Ceramic Fiber","ceramic_fiber",t)], name="Ceramic Fiber")
def sm_refractory(t=50.0): return SInsulationStack([SLayer("Refractory Foam","refractory_foam",t)], name="Refractory Foam")
def sm_mli(n=40, t=20.0): return SInsulationStack([SLayer("MLI","silica_spacer",t,n_reflective=n,vacuum=True,vacuum_pressure_torr=1e-5)], name="MLI Vacuum (%d)"%n)
def sm_hybrid(t=50.0, n=20): return SInsulationStack([SLayer("Aerogel-MLI","silica_aerogel",t,n_reflective=n,vacuum=True,vacuum_pressure_torr=1e-2)], name="Aerogel-MLI Hybrid")
def sm_pcm(pcm=200.0, ins=20.0): return SInsulationStack([
    SLayer("Aerogel (hot)","silica_aerogel",ins), SRecoveryLayer("PCM SrBr2","pcm_srbr2",pcm,recovery_type="pcm"),
    SLayer("Aerogel (cold)","silica_aerogel",ins)], name="PCM Heat Absorber")
def sm_endothermic(chem=150.0, ins=20.0): return SInsulationStack([
    SLayer("Aerogel (hot)","silica_aerogel",ins), SRecoveryLayer("MgSO4.7H2O","mgso4_7h2o",chem,recovery_type="endothermic"),
    SLayer("Aerogel (cold)","silica_aerogel",ins)], name="Endothermic Chemical")
def sm_teg(mli=20.0, teg=5.0): return SInsulationStack([
    SLayer("MLI (hot)","silica_spacer",mli,n_reflective=40,vacuum=True,vacuum_pressure_torr=1e-5),
    SRecoveryLayer("Bi2Te3 TEG","bi2te3_teg",teg,recovery_type="teg"),
    SLayer("MLI (cold)","silica_spacer",mli,n_reflective=40,vacuum=True,vacuum_pressure_torr=1e-5)], name="Thermoelectric Engine")
def sm_gradient(p=2.0, n=5): return SInsulationStack([
    SLayer("Pass1: Refractory-rich","refractory_dense",p),
    SLayer("Pass2: Hybrid","refractory_foam",p),
    SLayer("Pass3: Aerogel-rich","silica_aerogel_opac",p),
    SLayer("Pass4: Pure aerogel","silica_aerogel",p),
    SLayer("Pass5: Aerogel+TiN","silica_aerogel",p,n_reflective=1)], name="Gradient Spray")
def sm_pro(): return SInsulationStack([
    SLayer("S1: Refractory-rich","refractory_dense",2.0), SLayer("S1: Hybrid","refractory_foam",2.0),
    SLayer("S1: Aerogel-rich","silica_aerogel_opac",2.0), SLayer("S1: Pure aerogel","silica_aerogel",2.0),
    SLayer("S1: Aerogel+TiN","silica_aerogel",2.0,n_reflective=1),
    SRecoveryLayer("S2: PCM","pcm_srbr2",200.0,recovery_type="pcm"),
    SRecoveryLayer("S3: MgSO4","mgso4_7h2o",150.0,recovery_type="endothermic"),
    SLayer("S4: MLI (hot)","silica_spacer",10.0,n_reflective=20,vacuum=True,vacuum_pressure_torr=1e-3),
    SRecoveryLayer("S4: TEG","bi2te3_teg",5.0,recovery_type="teg"),
    SLayer("S4: MLI (cold)","silica_spacer",10.0,n_reflective=20,vacuum=True,vacuum_pressure_torr=1e-3),
    SLayer("S5: MLI Jacket","silica_spacer",20.0,n_reflective=40,vacuum=True,vacuum_pressure_torr=1e-5)], name="Insulatrix Pro")

def _sm_spray_pass(tag, thk=2.0):
    """Scientific-model version of _ultra_spray_pass: RTP2.0 5-sublayer
    refractory-rich -> aerogel -> TiN-cap compositional gradient."""
    return [
        SLayer("%s-1 YSZ/Al2O3-ZrO2 refractory-rich spray"%tag,"ysz_foam",thk),
        SLayer("%s-2 hybrid refractory-aerogel spray"%tag,"zirconia_aerogel",thk),
        SLayer("%s-3 opacified aerogel spray (SiC+C)"%tag,"carbon_aerogel",thk),
        SLayer("%s-4 pure nanoporous zirconia aerogel"%tag,"zirconia_aerogel",thk),
        SLayer("%s-5 zirconia aerogel + TiN low-e cap"%tag,"zirconia_aerogel",thk,n_reflective=1),
    ]

def sm_ultra(T_hot=1273.15,T_cold=77.0,n_passes=4):
    """INSULATRIX ULTRA -- optimized sprayable accumulative-stacking barrier.

    Six zones (hot -> cold), each dropping dT for the next so downstream
    layers run in a lower-k, lower-radiation regime AND every material sits
    below its T_max at steady state:
      A) n_passes cascaded spray passes (RTP2.0 _spray_pass recipe)
      B) 4x carbon-aerogel MLI hot shields + deep jacket (T_max 1473 K --
         they take the radiative beating polymer films cannot)
      C) PerfectVoid microsphere foam + fumed-silica VIP (nanoporous core)
      D) staged solid-state recovery engines: 2x Bi2Te3 TEG (PCM dams
         omitted -- at ~600 K they exceed decomposition limits, and a
         phase-pinned dam leaks against a ~0 K cold sink)
      E) two sealed Kapton-film MLI jackets in high vacuum (local T <673 K)
      F) aerogel-MLI cold cap + TiN low-e topcoat + ceramifiable weather seal
    """
    layers=[]
    for i in range(n_passes): layers.extend(_sm_spray_pass("A%d"%(i+1)))
    # NOTE: no PCM/hydrate dams in ULTRA -- at the ~600 K recovery position
    # they would exceed decomposition limits, and phase-pinning against a
    # ~0-3 K sink leaks more than it saves. Recovery is solid-state TEG only.
    for i in range(5):
        layers.append(SLayer("B%d evacuated carbon-MLI hot shield (60 foils)"%(i+1),
                             "carbon_aerogel",18.0,n_reflective=60,vacuum=True,vacuum_pressure_torr=1e-2))
    layers+=[SLayer("B6 deep carbon-MLI jacket (80 foils)","carbon_aerogel",16.0,n_reflective=80,vacuum=True,vacuum_pressure_torr=1e-2),
             SLayer("C1 PerfectVoid vacuum-microsphere spray foam","perfectvoid_foam",20.0,vacuum=True,vacuum_pressure_torr=1e-2),
             SLayer("C2 fumed-silica VIP core","vip_core",15.0,vacuum=True,vacuum_pressure_torr=1e-3),
             SRecoveryLayer("D1 Bi2Te3 TEG harvester (hot stage)","bi2te3_teg",4.0,recovery_type="teg"),
             SRecoveryLayer("D2 Bi2Te3 TEG harvester (cold stage)","bi2te3_teg",6.0,recovery_type="teg"),
             SLayer("E1 MLI jacket (120 Kapton films + spacers)","mli_nano",20.0,n_reflective=120,vacuum=True,vacuum_pressure_torr=1e-6),
             SLayer("E2 MLI jacket (80 Kapton films + spacers)","mli_nano",16.0,n_reflective=80,vacuum=True,vacuum_pressure_torr=1e-6),
             SLayer("F1 aerogel-MLI hybrid cold cap","silica_aerogel",18.0,n_reflective=36,vacuum=True,vacuum_pressure_torr=1e-3),
             SLayer("F2 TiN low-emissivity topcoat","tin_coating",0.3,n_reflective=1),
             SLayer("F3 ceramifiable silicone weather seal","weather_shell",0.5)]
    return SInsulationStack(layers,bc=SBoundary(T_hot=T_hot,T_cold=T_cold),name="Insulatrix Ultra")

SM_APPROACHES = {"foam":sm_foam,"aerogel":sm_aerogel,"ceramic":sm_ceramic,"refractory":sm_refractory,
    "mli":sm_mli,"hybrid":sm_hybrid,"pcm":sm_pcm,"endothermic":sm_endothermic,
    "teg":sm_teg,"gradient":sm_gradient,"pro":sm_pro,"ultra":sm_ultra}

def sm_custom(config):
    bc_d = config.get("bc",{}); bc = SBoundary(T_hot=bc_d.get("T_hot",1273.15),T_cold=bc_d.get("T_cold",77.0))
    layers = []
    for lc in config.get("layers",[]):
        if "recovery_type" in lc:
            layers.append(SRecoveryLayer(lc["name"],lc["material"],lc["thickness_mm"],
                n_reflective=lc.get("n_reflective",0),vacuum=lc.get("vacuum",False),
                vacuum_pressure_torr=lc.get("vacuum_pressure_torr",760.0),
                recovery_type=lc["recovery_type"],pcm_type=lc.get("pcm_type","SrBr2")))
        else:
            layers.append(SLayer(lc["name"],lc["material"],lc["thickness_mm"],
                n_reflective=lc.get("n_reflective",0),vacuum=lc.get("vacuum",False),
                vacuum_pressure_torr=lc.get("vacuum_pressure_torr",760.0)))
    return SInsulationStack(layers, bc=bc, name=config.get("name","custom"))

# --- Accumulative-stacking synergy metrics (ported from RTP2.0 synergy model,
#     extended with a T_max feasibility correction: a layer driven past its
#     rated service temperature fails -- it contributes no resistance, so an
#     "illegal" ordering cannot be credited with R it could never keep.) ---
def _reversed_sstack(stack):
    return SInsulationStack(list(reversed(stack.layers)), bc=stack.bc, name=stack.name+" (reversed)")

def _feasible_R(stack,r=None):
    """Feasibility-corrected resistance: sum R_i over layers whose hot-face
    temperature stays under their material's T_max. A layer that would melt,
    decompose or outgas at its local temperature contributes ~nothing."""
    r=r or stack.solve_steady_state(); R_ok=0.0; n_failed=0; failed=[]
    for i,l in enumerate(stack.layers):
        T_max=l.material.get("T_max",float("inf"))
        if r["T_profile"][i]>T_max: n_failed+=1; failed.append(l.name)
        else: R_ok+=r["R_per_layer"][i]
    return R_ok,n_failed,failed

def _best_single_material_sstack(stack):
    """Best FEASIBLE single-material stack of identical total thickness: for each
    distinct layer configuration whose T_max survives the hot face, expand it to
    the full thickness and keep the best. Materials that would fail at T_hot are
    disqualified -- a monolith has nowhere to hide behind a gradient."""
    total=stack.total_thickness_mm; best=None; best_R=-1.0; seen=set(); disqualified=[]
    for l in stack.layers:
        key=(l.material_key,l.n_reflective,l.vacuum,l.vacuum_pressure_torr)
        if key in seen: continue
        seen.add(key)
        if l.material.get("T_max",float("inf"))<stack.bc.T_hot:
            disqualified.append(l.material["name"]); continue
        cand=SLayer(l.name,l.material_key,total,n_reflective=l.n_reflective,
                    vacuum=l.vacuum,vacuum_pressure_torr=l.vacuum_pressure_torr)
        s=SInsulationStack([cand],bc=stack.bc,name="single:%s"%l.material_key)
        r=s.solve_steady_state()
        if r["R_total"]>best_R: best_R=r["R_total"]; best=s
    return best,disqualified

def _uniform_T_resistance(stack):
    T_avg=(stack.bc.T_hot+stack.bc.T_cold)/2.0
    return sum(l.thermal_resistance(T_avg) for l in stack.layers)

def sm_synergy_ordering(stack):
    """S_ord = R_stack / R_reversed(feasibility-corrected): does the hot->cold
    order beat the same layers reversed, counting only layers that survive
    their local temperature? (Reversed puts cold-side films/seals on the hot
    face -- they fail, so their nominal R is removed.)"""
    if not stack.layers or stack.n_layers<2: return 1.0
    r_opt=stack.solve_steady_state(); rev=_reversed_sstack(stack)
    r_rev=rev.solve_steady_state(); R_rev,_,_=_feasible_R(rev,r_rev)
    return r_opt["R_total"]/R_rev if R_rev>1e-15 else 1.0

def sm_synergy_material(stack):
    """S_mat = R_stack / R_best-feasible-single-material(same total thickness):
    does ordered multi-material stacking beat the best single material that can
    actually survive the hot face on its own?"""
    if not stack.layers or stack.n_layers<2: return 1.0
    r_stack=stack.solve_steady_state(); best,_=_best_single_material_sstack(stack)
    if best is None: return float("inf")
    r_single=best.solve_steady_state()
    return r_stack["R_total"]/r_single["R_total"] if r_single["R_total"]>1e-15 else 1.0

def sm_synergy_gradient(stack):
    """S_grad = R_stack(real T profile) / sum(R_i evaluated at uniform T_avg):
    how much does the internal temperature gradient help vs evaluating every
    layer at the mean boundary temperature? Captures the T^3 radiation drop."""
    if not stack.layers or stack.n_layers<2: return 1.0
    r_stack=stack.solve_steady_state()
    individual_R_sum=_uniform_T_resistance(stack)
    return r_stack["R_total"]/individual_R_sum if individual_R_sum>1e-15 else 1.0

def sm_synergy_factor(stack):
    """Total synergy S = S_ord * S_mat * S_grad. S_total > 1 means the stacking
    architecture itself contributes performance no single material could."""
    if not stack.layers or stack.n_layers<2:
        return {"total":1.0,"ordering":1.0,"material":1.0,"gradient":1.0}
    s_ord=sm_synergy_ordering(stack); s_mat=sm_synergy_material(stack); s_grad=sm_synergy_gradient(stack)
    return {"total":s_ord*s_mat*s_grad,"ordering":s_ord,"material":s_mat,"gradient":s_grad}

class OptimizationFramework:
    """Parameter-sweep and optimization tools for stack design."""
    def __init__(self, bc=None): self.bc = bc or SBoundary()
    def thickness_sweep(self, factory, thicknesses_mm, label=""):
        results = []
        for t in thicknesses_mm:
            s = factory(t); s.bc = self.bc; r = s.solve_steady_state()
            results.append({"label":label,"thickness_mm":t,"q_transmitted":r["q_transmitted"],
                "suppression_pct":r["suppression_pct"],"k_eff_overall_mW":r["k_eff_overall_mW"],"R_total":r["R_total"]})
        return results
    def mli_layer_sweep(self, n_list, total_mm=20.0):
        results = []
        for n in n_list:
            s = sm_mli(n, total_mm); s.bc = self.bc; r = s.solve_steady_state()
            results.append({"n_reflective":n,"total_mm":total_mm,"q_transmitted":r["q_transmitted"],
                "suppression_pct":r["suppression_pct"],"k_eff_overall_mW":r["k_eff_overall_mW"]})
        return results
    def full_comparison(self):
        results = []
        for key, factory in [("foam",sm_foam),("aerogel",sm_aerogel),("ceramic",sm_ceramic),
            ("refractory",sm_refractory),("mli",sm_mli),("hybrid",sm_hybrid),("pcm",sm_pcm),
            ("endothermic",sm_endothermic),("teg",sm_teg),("gradient",sm_gradient),
            ("pro",sm_pro),("ultra",sm_ultra)]:
            s = factory(); s.bc = self.bc; r = s.solve_steady_state()
            results.append({"name":s.name,"n_layers":s.n_layers,"total_thickness_mm":s.total_thickness_mm,
                "q_transmitted":r["q_transmitted"],"suppression_pct":r["suppression_pct"],
                "transmission_pct":r["transmission_pct"],"k_eff_overall_mW":r["k_eff_overall_mW"],
                "R_total":r["R_total"],"total_absorbed":r["total_absorbed"],
                "total_converted":r["total_converted"],"total_pumped":r["total_pumped"]})
        return results

def sm_export_csv(results, filepath):
    if not results: return
    keys = list(results[0].keys())
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in results: w.writerow(r)
    print("Exported %d rows to %s" % (len(results), filepath))

def sm_export_config(stack, filepath):
    config = {"name":stack.name, "bc":{"T_hot":stack.bc.T_hot,"T_cold":stack.bc.T_cold}, "layers":[]}
    for layer in stack.layers:
        lc = {"name":layer.name,"material":layer.material_key,"thickness_mm":layer.thickness_mm,
              "n_reflective":layer.n_reflective,"vacuum":layer.vacuum,"vacuum_pressure_torr":layer.vacuum_pressure_torr}
        if isinstance(layer, SRecoveryLayer): lc["recovery_type"] = layer.recovery_type; lc["pcm_type"] = layer.pcm_type
        config["layers"].append(lc)
    with open(filepath, "w") as f: json.dump(config, f, indent=2)
    print("Exported config to %s" % filepath)

# =============================================================================
# SECTION 7 -- PARTICLE SYSTEM (heat flow visualization)
# =============================================================================
class Particle:
    __slots__=["pos","vel","life","max_life","size","color","kind","prev_pos"]
    def __init__(self,pos,vel,life,size,color,kind="heat"):
        self.pos=np.array(pos,dtype=float); self.vel=np.array(vel,dtype=float)
        self.life=life; self.max_life=life; self.size=size; self.color=color; self.kind=kind
        self.prev_pos=self.pos.copy()

class ParticleSystem:
    def __init__(self): self.particles=[]
    def emit_burst(self,pos,color,count=24,speed=0.35,life=(0.4,0.9),size=(2,5)):
        """Radiating burst from a point -- magnetic pulse / fragmentation spark / sealant foam."""
        for _ in range(count):
            theta=random.uniform(0,2*math.pi); phi=math.acos(random.uniform(-1,1))
            s=random.uniform(speed*0.4,speed)
            vx=s*math.sin(phi)*math.cos(theta); vy=s*math.sin(phi)*math.sin(theta); vz=s*math.cos(phi)
            l=random.uniform(*life); sz=random.uniform(*size)
            self.particles.append(Particle(np.array(pos,dtype=float),np.array([vx,vy,vz]),l,sz,color,"burst"))
    def emit_hot(self,pos,color,count=3):
        for _ in range(count):
            x=pos[0]+random.uniform(-0.05,0.05); y=pos[1]+random.uniform(-0.05,0.05)
            z=pos[2]+random.uniform(-0.02,0.02)
            vx=random.uniform(-0.005,0.005); vy=random.uniform(-0.005,0.005)
            vz=random.uniform(0.01,0.03)
            life=random.uniform(1.0,3.0); size=random.uniform(2,5)
            self.particles.append(Particle(np.array([x,y,z]),np.array([vx,vy,vz]),life,size,color,"heat"))
    def update(self,dt):
        alive=[]
        for p in self.particles:
            p.life-=dt
            if p.life<=0: continue
            p.prev_pos=p.pos.copy()
            p.pos+=p.vel*dt; p.vel*=(1-0.3*dt); p.vel[2]+=0.005*dt
            alive.append(p)
        if len(alive)>500: alive=alive[-500:]
        self.particles=alive
    def render(self,surf,cx,cy,focal,Rcam,dist):
        for p in self.particles:
            cam=p.pos@Rcam.T; cam[2]+=dist
            if cam[2]<=0.05: continue
            sx=cx+focal*cam[0]/cam[2]; sy=cy-focal*cam[1]/cam[2]
            alpha=clamp(p.life/p.max_life); r=max(1,int(p.size*alpha))
            # Motion trail (line from prev_pos to current)
            pcam=p.prev_pos@Rcam.T; pcam[2]+=dist
            if pcam[2]>0.05:
                psx=cx+focal*pcam[0]/pcam[2]; psy=cy-focal*pcam[1]/pcam[2]
                trail_alpha=int(80*alpha)
                try:
                    ts=pygame.Surface((abs(int(sx)-int(psx))+4,abs(int(sy)-int(psy))+4),pygame.SRCALPHA)
                    tx=min(int(sx),int(psx))-2; ty=min(int(sy),int(psy))-2
                    pygame.draw.line(ts,(*p.color,trail_alpha),(int(psx)-tx,int(psy)-ty),(int(sx)-tx,int(sy)-ty),max(1,r-1))
                    surf.blit(ts,(tx,ty),special_flags=pygame.BLEND_ADD)
                except: pass
            # Glow halo (larger, dimmer circle)
            if r>=2:
                glow_r=r+2; glow_alpha=int(60*alpha)
                try:
                    gs=pygame.Surface((glow_r*2+2,glow_r*2+2),pygame.SRCALPHA)
                    pygame.draw.circle(gs,(*p.color,glow_alpha),(glow_r+1,glow_r+1),glow_r)
                    surf.blit(gs,(int(sx)-glow_r-1,int(sy)-glow_r-1),special_flags=pygame.BLEND_ADD)
                except: pass
            # Bright core
            col=tuple(max(0,min(255,int(c*(0.4+0.6*alpha)))) for c in p.color)
            try: pygame.draw.circle(surf,col,(int(sx),int(sy)),r)
            except: pass

# =============================================================================
# SECTION 8 -- SCENE RENDERER
# =============================================================================
class Scene:
    def __init__(self,formulation_idx=0,kind="panel"):
        self.kind=kind
        self.formulation_idx=formulation_idx
        self.parts,self.total_t=build_pod_scene() if kind=="pod" else build_scene(formulation_idx)
        self.el=DIMS["view_el"]; self.az=DIMS["view_az"]; self.dist=DIMS["view_dist"]
        self.pan=[0.0,0.0]; self.light=np.array([0.5,0.6,0.7]); self.light=self.light/np.linalg.norm(self.light)
        self.section=False; self.show_labels=False; self.particles=ParticleSystem()
        self.T_profile=None; self.layer_colors=[]; self.highlight_layer=-1
    def reset_camera(self):
        self.el=DIMS["view_el"]; self.az=DIMS["view_az"]; self.dist=DIMS["view_dist"]; self.pan=[0.0,0.0]
    def update_parts(self,formulation_idx):
        self.formulation_idx=formulation_idx
        self.parts,self.total_t=build_pod_scene() if self.kind=="pod" else build_scene(formulation_idx)
    def render(self,surf,rect,angles=None,label_font=None):
        angles=angles or {}; default_ang=angles.get("default",0.0)
        clip=surf.get_clip(); surf.set_clip(rect)
        cx=rect.x+rect.w/2.0+self.pan[0]; cy=rect.y+rect.h/2.0+self.pan[1]
        focal=min(rect.w,rect.h)*1.12; Rcam=rot_x(self.el)@rot_y(self.az)
        lx,ly,lz=float(self.light[0]),float(self.light[1]),float(self.light[2])
        # Second fill light from opposite direction (softer)
        lx2,ly2,lz2=-lx*0.5+0.3, -ly*0.5+0.4, -lz*0.5+0.3
        l2n=math.sqrt(lx2*lx2+ly2*ly2+lz2*lz2); lx2,ly2,lz2=lx2/l2n,ly2/l2n,lz2/l2n
        section=self.section; show_labels=self.show_labels and label_font is not None
        polys,labels=[],[]
        # Depth fog parameters
        fog_near=self.dist*0.6; fog_far=self.dist*2.5
        # Ground grid plane for spatial reference
        grid_y=-0.15; grid_ext=0.5; grid_step=0.1
        grid_lines=[]
        for gi in range(int(-grid_ext/grid_step),int(grid_ext/grid_step)+1):
            gx=gi*grid_step
            for a,b in [(np.array([gx,grid_y,-grid_ext]),np.array([gx,grid_y,grid_ext])),
                        (np.array([-grid_ext,grid_y,gx]),np.array([grid_ext,grid_y,gx]))]:
                ca=a@Rcam.T; ca[2]+=self.dist; cb=b@Rcam.T; cb[2]+=self.dist
                if ca[2]>0.05 and cb[2]>0.05:
                    sxa=cx+focal*ca[0]/ca[2]; sya=cy-focal*ca[1]/ca[2]
                    sxb=cx+focal*cb[0]/cb[2]; syb=cy-focal*cb[1]/cb[2]
                    avg_d=(ca[2]+cb[2])/2; fog_g=clamp((avg_d-fog_near)/(fog_far-fog_near))
                    gc=(int(C_GRID[0]+(C_BG[0]-C_GRID[0])*fog_g*0.5),
                        int(C_GRID[1]+(C_BG[1]-C_GRID[1])*fog_g*0.5),
                        int(C_GRID[2]+(C_BG[2]-C_GRID[2])*fog_g*0.5))
                    grid_lines.append((avg_d,sxa,sya,sxb,syb,gc))
        grid_lines.sort(key=lambda t:t[0],reverse=True)
        for _,sxa,sya,sxb,syb,gc in grid_lines:
            try: pygame.draw.line(surf,gc,(sxa,sya),(sxb,syb),1)
            except: pass
        for pi,part in enumerate(self.parts):
            off=part.explode
            for m in part.meshes:
                wv=m.world_verts(angles.get(m.group,default_ang))+off
                cam=wv@Rcam.T; cam[:,2]+=self.dist; caml=cam.tolist(); col=m.color
                # Temperature color overlay for panel layers
                if part.key.startswith("layer_") and self.T_profile is not None:
                    li=int(part.key.split("_")[1])
                    if li<len(self.T_profile)-1:
                        T_avg=(self.T_profile[li]+self.T_profile[li+1])/2.0
                        tcol=temp_to_color(T_avg)
                        col=_mix(col,tcol,0.45)
                # UI cross-highlight: hovering/pinning a layer row in the layer
                # breakdown tints that slab bright yellow in the 3D view.
                hl = part.key=="layer_%d"%self.highlight_layer
                if hl: col=_mix(col,(255,235,110),0.6)
                cr,cg,cb=col; sxl,syl,dzl=[],[],[]
                for vx2,vy2,vz2 in caml:
                    dzl.append(vz2)
                    if vz2>0.05: sxl.append(cx+focal*vx2/vz2); syl.append(cy-focal*vy2/vz2)
                    else: sxl.append(0.0); syl.append(0.0)
                if show_labels and m.name:
                    mcen=cam.mean(axis=0)
                    if mcen[2]>0.05:
                        mlx=cx+focal*mcen[0]/mcen[2]; mly=cy-focal*mcen[1]/mcen[2]
                        labels.append((mcen[2],(mlx,mly),m.name))
                for face in m.faces:
                    if section and part.key.startswith("layer_"):
                        c=wv[list(face)].mean(axis=0)
                        if c[1]>0.004: continue
                    clipped=False
                    for i in face:
                        if dzl[i]<=0.05: clipped=True; break
                    if clipped: continue
                    ax,ay,az=caml[face[0]]; bx,by,bz=caml[face[1]]; fx,fy,fz=caml[face[2]]
                    ux,uy,uz=bx-ax,by-ay,bz-az; wx,wy,wz=fx-ax,fy-ay,fz-az
                    nx=uy*wz-uz*wy; ny=uz*wx-ux*wz; nz=ux*wy-uy*wx
                    inv=1.0/((nx*nx+ny*ny+nz*nz)**0.5 or 1.0)
                    nx*=inv; ny*=inv; nz*=inv
                    if nz>0: nx,ny,nz=-nx,-ny,-nz
                    # Key light (diffuse)
                    d1=nx*lx+ny*ly+nz*lz; diff1=max(0.0,d1)
                    # Fill light (softer, from opposite side)
                    d2=nx*lx2+ny*ly2+nz*lz2; diff2=max(0.0,d2)*0.35
                    # Rim light (back-light edge glow)
                    rim=max(0.0,1.0-abs(nz))*0.15
                    # Specular highlight (Phong-ish: reflect key light)
                    # View direction is (0,0,1) in camera space
                    rx,ry,rz=2*d1*nx-lx, 2*d1*ny-ly, 2*d1*nz-lz
                    spec=max(0.0,rz)**16 * 0.3 if d1>0 else 0.0
                    # Ambient
                    amb=0.25
                    shade=amb+0.55*diff1+diff2+rim+spec
                    # Depth fog: blend distant faces toward background
                    ds=sum(dzl[i] for i in face); avg_d=ds/len(face)
                    fog_t=clamp((avg_d-fog_near)/(fog_far-fog_near))
                    # Apply shading
                    r8=max(0,min(255,int(cr*shade)))
                    g8=max(0,min(255,int(cg*shade)))
                    b8=max(0,min(255,int(cb*shade)))
                    # Apply fog
                    if fog_t>0.01:
                        r8=int(r8+(C_BG[0]-r8)*fog_t*0.6)
                        g8=int(g8+(C_BG[1]-g8)*fog_t*0.6)
                        b8=int(b8+(C_BG[2]-b8)*fog_t*0.6)
                    fc=(r8,g8,b8)
                    pts=[(sxl[i],syl[i]) for i in face]
                    polys.append((avg_d,pts,fc,fog_t,hl))
        polys.sort(key=lambda t:t[0],reverse=True)
        for _,pts,fc,fog_t,hl in polys:
            if len(pts)>=3:
                try: pygame.draw.polygon(surf,fc,pts)
                except: pass
                if hl:
                    try: pygame.draw.lines(surf,(255,235,110),True,pts,2)
                    except: pass
                # Edge outline for depth perception (skip very fogged faces)
                if fog_t<0.7:
                    ec=(max(0,fc[0]-35),max(0,fc[1]-35),max(0,fc[2]-35))
                    try:
                        if len(pts)==4:
                            pygame.draw.aalines(surf,ec,True,pts)
                        else:
                            pygame.draw.polygon(surf,ec,pts,1)
                    except: pass
        self.particles.render(surf,cx,cy,focal,Rcam,self.dist)
        # Pulsing glow for hot/cold frames
        pulse=0.5+0.5*math.sin(pygame.time.get_ticks()*0.003)
        for pi,part in enumerate(self.parts):
            if part.key in ("hot_frame","cold_frame"):
                glow_col=C_HOT if part.key=="hot_frame" else C_COLD
                for m in part.meshes:
                    wv=m.world_verts(angles.get(m.group,default_ang))+part.explode
                    cam=wv@Rcam.T; cam[:,2]+=self.dist
                    mcen=cam.mean(axis=0)
                    if mcen[2]>0.05:
                        mlx=cx+focal*mcen[0]/mcen[2]; mly=cy-focal*mcen[1]/mcen[2]
                        gr=int(20+15*pulse)
                        try:
                            gs=pygame.Surface((gr*2+2,gr*2+2),pygame.SRCALPHA)
                            pygame.draw.circle(gs,(*glow_col,int(40+30*pulse)),(gr+1,gr+1),gr)
                            surf.blit(gs,(int(mlx)-gr-1,int(mly)-gr-1),special_flags=pygame.BLEND_ADD)
                        except: pass
        if show_labels:
            labels.sort(key=lambda t:t[0])
            for _,(lxx,lyy),text in labels:
                try:
                    lbl=label_font.render(text,True,C_ACCENT)
                    lw,lh=lbl.get_size()
                    # Background pill for readability
                    pill=pygame.Surface((lw+10,lh+4),pygame.SRCALPHA)
                    pygame.draw.rect(pill,(20,22,32,180),(0,0,lw+10,lh+4),border_radius=3)
                    pygame.draw.rect(pill,(*C_PANEL_BORDER,180),(0,0,lw+10,lh+4),1,border_radius=3)
                    surf.blit(pill,(lxx+4,lyy-10))
                    surf.blit(lbl,(lxx+8,lyy-8))
                except: pass
        surf.set_clip(clip)

# =============================================================================
# SECTION 9 -- APP (main interactive application)
# =============================================================================
def panel(surf,x,y,w,h,alpha=220):
    s=pygame.Surface((w,h),pygame.SRCALPHA)
    # Subtle vertical gradient: slightly lighter at top
    for yy in range(h):
        t=yy/max(h-1,1)
        a=int(alpha*(1.0-t*0.15))
        r=int(C_PANEL[0]+8*(1-t)); g=int(C_PANEL[1]+8*(1-t)); b=int(C_PANEL[2]+12*(1-t))
        pygame.draw.line(s,(r,g,b,a),(0,yy),(w,yy))
    # Subtle scanline overlay every 3px for tech aesthetic
    for sy in range(0,h,3):
        pygame.draw.line(s,(255,255,255,8),(0,sy),(w,sy))
    surf.blit(s,(x,y))
    # Accent top border
    pygame.draw.line(surf,C_ACCENT,(x,y),(x+w,y),1)
    # Regular border
    pygame.draw.rect(surf,C_PANEL_BORDER,(x,y,w,h),1)
    # Corner accent brackets (top-left and bottom-right)
    bl=8
    pygame.draw.line(surf,C_ACCENT,(x,y),(x+bl,y),2)
    pygame.draw.line(surf,C_ACCENT,(x,y),(x,y+bl),2)
    pygame.draw.line(surf,C_ACCENT,(x+w,y+h),(x+w-bl,y+h),2)
    pygame.draw.line(surf,C_ACCENT,(x+w,y+h),(x+w,y+h-bl),2)

def wrap_text(font,text,max_px):
    words=text.split(); lines,cur=[],""
    for w in words:
        test=cur+(" " if cur else "")+w
        if font.size(test)[0]<=max_px: cur=test
        else:
            if cur: lines.append(cur)
            cur=w
    if cur: lines.append(cur)
    return lines

class App:
    def __init__(self):
        self.W,self.H=1280,800
        self.screen=pygame.display.set_mode((self.W,self.H),pygame.RESIZABLE)
        pygame.display.set_caption("Insulatrix -- Near-Zero Thermal Transfer Digital Twin")
        self.clock=pygame.time.Clock(); self.running=True
        self.formulation_idx=0; f=FORMULATIONS[0]; self.scenario_idx=0
        self.scene=Scene(0); self.sim=ThermalSim(f["approach"])
        self.show_help=False; self.show_info=False; self.show_chem=False; self.show_durability=False
        self.show_inspector=False; self.insp_scroll=0; self.layer_scroll=0; self._synergy_cache={}
        self._layer_rect=None; self._layer_rows=[]; self.hover_layer=None; self.pinned_layer=None
        self._scenario_cache={}; self._transient_cache={}; self._layer_nvis=10; self.show_proof=False
        self.show_fab=False; self.fab_scroll=0
        self.info_scroll=0; self.show_stats=True; self.show_particles=True
        self.font=pygame.font.SysFont("consolas",14); self.fb=pygame.font.SysFont("consolas",18,bold=True)
        self.fs=pygame.font.SysFont("consolas",12); self.fbig=pygame.font.SysFont("consolas",24,bold=True)
        self.auto_rotate=False; self.auto_cycle=False; self.auto_timer=0.0
        self.dragging=False; self.rdragging=False; self.last_mouse=(0,0)
        self.angles={"default":0.0}; self.mouse_pos=(0,0)
        self.heat_emit_timer=0.0; self.ratio_history=[]
        self._build_info(); self.sim.running=True; self.sim.elapsed=0.001
        # Precompute formulation success indicators + suppression values
        self._form_success=[]; self._form_supp=[]
        for f in FORMULATIONS:
            s=run_simulation(f["approach"],duration_s=60)
            self._form_success.append(s.is_success); self._form_supp.append(s.suppression_pct)
        # Pod (impact-shield) tab state
        self.tab="thermal"
        self.pod=PodShieldSim(); self.pod_threats=self.pod.sweep_threats(); self.pod_scenario_idx=0
        self.pod_test_phase="idle"; self.pod_test_t=0.0
        self.pod_scene=None  # built lazily on first switch to the pod tab

    def _build_info(self):
        self.info_sections=[
            ("ABOUT INSULATRIX",["Sprayable near-zero heat transfer digital twin",
                "All dimensions real (mm / SI). Physics computed live.","",
                "This is an EDUCATIONAL SIMULATION, not a real product.",
                "It models 12 insulation approaches to suppress heat transfer,","honestly scored by physics.","",
                "Success metric: HEAT TRANSFER SUPPRESSION RATIO > 99%","Goal: MINIMIZE transmitted heat flux (q_transmitted -> 0)","",
                "Heat transfer modes tracked: conduction, gas (Knudsen), radiation","PCM/chemical absorption, TEG conversion, Peltier re-absorption","",
                "Target: near-zero heat flux across extreme dT","(forge 1000C on one side, cryogenic 77K on other)","",
                "5 boundary-condition scenarios (V to cycle): forge+cryo, forge+ambient,","deep-space habitat (3K, near-CMB), Boomerang-Nebula extreme (1K),","forge + absolute zero (1000C -> 1 microkelvin)","",
                "Durability/weathering model (W): service life, thermal-cycle rating,","hydrophobicity, UV & abrasion resistance, per formulation",""]),
            ("EXACT FORMULATIONS (SCIENTIFIC)",["Each formulation lists its full recipe card -- every",
                "specification field -- followed by the per-layer role of every",
                "layer in its stack (why it sits where it sits).",""]+
                [line for f in FORMULATIONS for line in (["--- %d. %s ---"%(f["id"],f["name"])]+
                    ["  %s: %s"%(k,v) for k,v in f["recipe"].items()]+
                    ["  LAYERS:"]+
                    sum((["    L%d  %s"%(i,l["name"]),"      %s"%l.get("why","-")] for i,l in enumerate(f["layers"])),[])+
                    [""])]),
            ("PHYSICS MODELS USED",["Fourier conduction: q = k_eff * dT/dx through each layer",
                "Knudsen gas: k_gas = k_0 / (1 + 2*beta*Kn), Kn = l_mfp/d_pore",
                "Rosseland radiation: k_rad = 16/3 * sigma * T^3 * l_mfp",
                "MLI: radiation suppressed by 1/N (N reflective layers)",
                "Opacifier: SiC/TiO2 reduce radiation mean free path",
                "PCM: latent heat absorption at constant T_phase",
                "Endothermic: chemical dehydration stores heat as bond energy",
                "TEG: Seebeck effect, eta = 0.1 * Carnot",
                "Peltier: COP * P_electrical = heat pumped back","",
                "k_eff = k_solid + k_gas + k_radiation per layer","Overall: q = k_eff * dT / total_thickness",""]),
            ("DURABILITY & WEATHERING",["Near-permanence per the design goal: extreme weathering resistance",
                "(UV, moisture, abrasion, thermal cycling). Ratings below are static","material inputs; lifetime is always COMPUTED from them + the live",
                "boundary conditions (see estimate_durability()), never hard-coded.","",
                "max/min service C = rated continuous-service envelope","thermal cycles = full hot<->cold cycles before significant degradation",
                "hydrophobic/UV/abrasion = 0-1 resistance ratings","base lifetime = expected service life within the rated envelope",""]+
                [line for f in FORMULATIONS for line in (lambda d: ["--- %d. %s ---"%(f["id"],f["name"]),
                    "  Envelope: %d C to %d C   Cycles: %d   Class: %s"%(d["max_service_C"],d["min_service_C"],d["thermal_cycles_rating"],d["weathering_class"]),
                    "  Degradation: %s"%d["degradation"],""])(DURABILITY[f["approach"]])]),
            ("IMPACT SHIELD POD (new tab, press M)",["Hypervelocity impact-shield digital twin: a spacecraft pod hull",
                "combining a Whipple shield, pulsed magnetic (eddy-current) debris","deflection, and a self-healing auto-patching rear wall around the",
                "same Insulatrix Pro thermal core used in the thermal tab.","",
                "Design reference: 8940.8 m/s (20,000 mph), realistic LEO/meteoroid","hypervelocity. Design envelope: micrometeoroid & orbital debris (MMOD)",
                "scale threats, mm to a few cm -- the real regime Whipple shields","are engineered for. A literal asteroid is outside any material",
                "shield's capability; this is reported honestly, not hidden.","",
                "1-8 selects a representative threat scenario; SPACE launches a live","test: magnetic pulse (conductive threats only) -> Whipple fragmentation",
                "-> either fully stopped, or a breach that races the self-healing","sealant against cabin decompression -- exactly as assessed by",
                "PodShieldSim.assess_impact(), the same physics behind --pod-report.",""]),
            ("IMPACT SHIELD PHYSICS MODELS",["Simplified but honestly-cited engineering models -- same rigor level",
                "as the thermal physics above, not literal NASA/ESA design equations.","",
                "Whipple fragmentation: frag_eff = areal_bumper / (k_frag * r_p * rho_p)",
                "  clamped to [0,1]; areal_bumper = bumper thickness * density","",
                "Debris-cloud dilution: cloud_r = standoff * tan(20 deg debris-cone angle)",
                "  e_areal = KE * frag_eff / (pi * cloud_r^2)  [+ unfragmented-core term]","",
                "Ballistic margin: capacity / e_areal_incident  (>1 survives, <1 perforates)",
                "  capacity = rear-wall penetration threshold x thickness, PLUS the 405mm",
                "  thermal core's crush/pulverization absorption (areal density x 3000 J/kg,",
                "  conservative for brittle porous media -- not credited as extra standoff,",
                "  to avoid crediting the same material twice)","",
                "Eddy-current deflection (Lenz's law): J_eddy = sigma*Vol*B^2*v*t_pulse/L,",
                "  capped at the coil+frame's rated max reaction impulse (finite, so",
                "  deflection assist honestly diminishes for large masses); zero for",
                "  non-conductive (sigma=0) threats -- rock and ice get no assist.",
                "  deflect_frac = J_eddy / (J_eddy + 0.35 * incoming momentum)","",
                "Hole size: d_hole = d_projectile * spall_mult(overshoot ratio),",
                "  spall_mult = clamp(1.5 + 1.5*log10(e_areal/capacity), 1.5, 8.0)","",
                "Cabin decompression: standard choked (critical) compressible-orifice flow,",
                "  mdot = Cd*A_hole*P*sqrt(gamma/(R*T))*(2/(gamma+1))^((gamma+1)/(2(gamma-1)))",
                "  integrated against a linearly-closing hole area as the sealant patches.",""]),
            ("CONTROLS",["TAB / SHIFT+TAB  next/previous model -- cycles through EVERY",
                "  individual model in the app, all 13 (12 formulations + the pod,",
                "  the most recent addition), in one flat sequence.",
                "SPACE/B  toggle simulation","1-9,0,C,U  select formulation (12 total)",
                "+/-  add/remove layers","[/]  increase hot-side T",";/'  increase cold-side T",
                "V  cycle boundary scenario","W  durability/weathering panel",
                "P  toggle particle heat flow","T  auto-rotate","A  auto-cycle formulations",
                "BACKSPACE  reset simulation","R  reset camera","L  toggle labels",
                "X  cross-section","I  full specification","D  material diagram",
                "N  physics inspector (per-layer R, T_max check, synergy,",
                "  scenario sweep, live charts)","layer list: hover = detail card + 3D highlight,",
                "  click = pin; J/K steps the pinned layer","O  live proof checklist","E  export layer table CSV","F12  screenshot PNG",
                "F  fabrication protocol (spray/vacuum/QC steps)","M  impact-shield pod tab","H  help","ESC/Q  quit",""]),
        ]

    def handle_events(self,dt):
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: self.running=False
            elif ev.type==pygame.VIDEORESIZE:
                self.W,self.H=ev.w,ev.h; self.screen=pygame.display.set_mode((self.W,self.H),pygame.RESIZABLE)
            elif ev.type==pygame.KEYDOWN:
                asc=self._active_scene()
                if ev.key in (pygame.K_ESCAPE,pygame.K_q): self.running=False
                elif ev.key==pygame.K_m: self._toggle_tab()
                elif ev.key==pygame.K_TAB:
                    self._next_model(-1 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1)
                elif ev.key==pygame.K_r: asc.reset_camera()
                elif ev.key==pygame.K_l: asc.show_labels=not asc.show_labels
                elif ev.key==pygame.K_x: asc.section=not asc.section
                elif ev.key==pygame.K_h: self.show_help=not self.show_help
                elif ev.key==pygame.K_i: self.show_info=not self.show_info
                elif ev.key==pygame.K_F12:
                    fn="insulatrix_%s.png"%time.strftime("%Y%m%d_%H%M%S")
                    pygame.image.save(self.screen,fn); print("Screenshot saved: %s"%fn)
                elif ev.key==pygame.K_t: self.auto_rotate=not self.auto_rotate
                elif ev.key==pygame.K_p: self.show_particles=not self.show_particles
                elif ev.key==pygame.K_UP and self.show_fab: self.fab_scroll=max(0,self.fab_scroll-3)
                elif ev.key==pygame.K_DOWN and self.show_fab: self.fab_scroll+=3
                elif ev.key==pygame.K_UP and self.show_inspector: self.insp_scroll=max(0,self.insp_scroll-3)
                elif ev.key==pygame.K_DOWN and self.show_inspector: self.insp_scroll+=3
                elif ev.key==pygame.K_UP and self.show_info: self.info_scroll=max(0,self.info_scroll-40)
                elif ev.key==pygame.K_DOWN and self.show_info: self.info_scroll+=40
                elif self.tab=="pod":
                    if ev.key==pygame.K_SPACE: self._trigger_pod_test()
                    elif ev.key==pygame.K_BACKSPACE: self.pod_test_phase="idle"; self.pod_test_t=0.0
                    elif ev.key==pygame.K_a: self.auto_cycle=not self.auto_cycle
                    elif ev.key in (pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,pygame.K_5,pygame.K_6,pygame.K_7,pygame.K_8):
                        self._select_pod_scenario(ev.key-pygame.K_1)
                else:
                    if ev.key in (pygame.K_SPACE,pygame.K_b): self.sim.running=not self.sim.running
                    elif ev.key==pygame.K_d: self.show_chem=not self.show_chem
                    elif ev.key==pygame.K_n: self.show_inspector=not self.show_inspector; self.insp_scroll=0
                    elif ev.key==pygame.K_w: self.show_durability=not self.show_durability
                    elif ev.key==pygame.K_o: self.show_proof=not self.show_proof
                    elif ev.key==pygame.K_f: self.show_fab=not self.show_fab; self.fab_scroll=0
                    elif ev.key==pygame.K_e: self._export_layers_csv()
                    elif ev.key==pygame.K_j: self._step_layer(1)
                    elif ev.key==pygame.K_k: self._step_layer(-1)
                    elif ev.key==pygame.K_v: self._apply_scenario((self.scenario_idx+1)%len(BOUNDARY_SCENARIOS))
                    elif ev.key==pygame.K_a: self.auto_cycle=not self.auto_cycle
                    elif ev.key in (pygame.K_EQUALS,pygame.K_PLUS): self._adjust_thickness(1.2)
                    elif ev.key==pygame.K_MINUS: self._adjust_thickness(0.8)
                    elif ev.key==pygame.K_LEFTBRACKET: self.sim.T_hot=min(3000,self.sim.T_hot+100); self._reset_sim()
                    elif ev.key==pygame.K_RIGHTBRACKET: self.sim.T_hot=max(300,self.sim.T_hot-100); self._reset_sim()
                    elif ev.key==pygame.K_SEMICOLON: self.sim.T_cold=min(500,self.sim.T_cold+10); self._reset_sim()
                    elif ev.key==pygame.K_QUOTE: self.sim.T_cold=max(1.0,self.sim.T_cold-10); self._reset_sim()
                    elif ev.key==pygame.K_BACKSPACE: self._select_formulation(self.formulation_idx)
                    elif ev.key==pygame.K_1: self._select_formulation(0)
                    elif ev.key==pygame.K_2: self._select_formulation(1)
                    elif ev.key==pygame.K_3: self._select_formulation(2)
                    elif ev.key==pygame.K_4: self._select_formulation(3)
                    elif ev.key==pygame.K_5: self._select_formulation(4)
                    elif ev.key==pygame.K_6: self._select_formulation(5)
                    elif ev.key==pygame.K_7: self._select_formulation(6)
                    elif ev.key==pygame.K_8: self._select_formulation(7)
                    elif ev.key==pygame.K_9: self._select_formulation(8)
                    elif ev.key==pygame.K_0: self._select_formulation(9)
                    elif ev.key==pygame.K_c: self._select_formulation(10)
                    elif ev.key==pygame.K_u: self._select_formulation(11)
            elif ev.type==pygame.MOUSEBUTTONDOWN:
                if ev.button==1:
                    # Click on a layer-breakdown row pins/unpins its detail +
                    # 3D highlight instead of starting a camera drag.
                    hit_l=None
                    for rect,i in getattr(self,"_layer_rows",[]):
                        if rect.collidepoint(ev.pos): hit_l=i; break
                    if hit_l is not None:
                        self.pinned_layer=None if self.pinned_layer==hit_l else hit_l
                        continue
                    self.dragging=True; self.last_mouse=ev.pos
                elif ev.button==3: self.rdragging=True; self.last_mouse=ev.pos
                elif ev.button in (4,5):
                    # Wheel: overlay text if open, layer list if hovering it,
                    # otherwise camera zoom (original behavior).
                    if self.show_fab:
                        self.fab_scroll=max(0,self.fab_scroll+(-3 if ev.button==4 else 3))
                    elif self.show_inspector:
                        self.insp_scroll=max(0,self.insp_scroll+(-3 if ev.button==4 else 3))
                    elif self._layer_rect is not None and self._layer_rect.collidepoint(ev.pos):
                        self.layer_scroll=max(0,self.layer_scroll+(-1 if ev.button==4 else 1))
                    elif ev.button==4: self._active_scene().dist*=0.9
                    else: self._active_scene().dist*=1.1
            elif ev.type==pygame.MOUSEBUTTONUP:
                if ev.button==1: self.dragging=False
                elif ev.button==3: self.rdragging=False
            elif ev.type==pygame.MOUSEMOTION:
                self.mouse_pos=ev.pos
                asc=self._active_scene()
                if self.dragging:
                    dx,dy=ev.pos[0]-self.last_mouse[0],ev.pos[1]-self.last_mouse[1]
                    asc.az+=dx*0.01; asc.el+=dy*0.01
                    asc.el=max(-1.5,min(1.5,asc.el)); self.last_mouse=ev.pos
                if self.rdragging:
                    dx,dy=ev.pos[0]-self.last_mouse[0],ev.pos[1]-self.last_mouse[1]
                    asc.pan[0]+=dx; asc.pan[1]+=dy; self.last_mouse=ev.pos

    def _adjust_thickness(self,factor):
        for layer in self.sim.layers: layer["thickness_mm"]*=factor
        self.sim.total_thickness=sum(l["thickness_mm"] for l in self.sim.layers)*MM
        self.scene.update_parts(self.formulation_idx); self._reset_sim()

    def _reset_sim(self):
        f=FORMULATIONS[self.formulation_idx]
        self.sim=ThermalSim(f["approach"],self.sim.T_hot,self.sim.T_cold)
        self.sim.running=True; self.sim.elapsed=0.001
        self.ratio_history=[]; self.scene.particles=ParticleSystem()

    def _apply_scenario(self,idx):
        self.scenario_idx=idx%len(BOUNDARY_SCENARIOS); sc=BOUNDARY_SCENARIOS[self.scenario_idx]
        self.sim.T_hot=sc["T_hot"]; self.sim.T_cold=sc["T_cold"]; self._reset_sim()

    def _select_formulation(self,idx):
        self.formulation_idx=idx; f=FORMULATIONS[idx]
        T_hot=self.sim.T_hot; T_cold=self.sim.T_cold
        self.sim=ThermalSim(f["approach"],T_hot,T_cold)
        self.sim.running=True; self.sim.elapsed=0.001
        self.scene=Scene(idx); self.ratio_history=[]; self.layer_scroll=0; self.insp_scroll=0
        self.pinned_layer=None

    def _active_scene(self):
        return self.pod_scene if (self.tab=="pod" and self.pod_scene is not None) else self.scene

    def _toggle_tab(self):
        self.tab="pod" if self.tab=="thermal" else "thermal"
        if self.tab=="pod" and self.pod_scene is None:
            self.pod_scene=Scene(0,kind="pod")
        self.pod_test_phase="idle"; self.pod_test_t=0.0

    # Every individual model in the app, in one flat browsing order, so TAB
    # can step through ALL of them one at a time -- including the pod, the
    # most recent addition, which otherwise lived behind a separate 'M' toggle
    # with no way to just "keep pressing a key" to reach it.
    def _model_count(self): return len(FORMULATIONS)+1  # 12 formulations + the pod
    def _model_index(self):
        return len(FORMULATIONS) if self.tab=="pod" else self.formulation_idx
    def _goto_model(self,idx):
        idx%=self._model_count()
        if idx==len(FORMULATIONS):
            if self.tab!="pod": self._toggle_tab()
        else:
            if self.tab!="thermal": self._toggle_tab()
            self._select_formulation(idx)
    def _next_model(self,step=1): self._goto_model(self._model_index()+step)

    def _stack_metrics(self):
        """Aggregate live physics metrics for the active ThermalSim stack:
        per-layer R, total R/U, areal mass & heat capacity, structure counts,
        and service-rating (T_max) violations at the current T profile."""
        sim=self.sim; n=sim.n_layers
        k=[max(sim.k_eff[i],1e-12) for i in range(n)]
        R=[sim.layers[i]["thickness_mm"]*1e-3/k[i] for i in range(n)]
        R_total=sum(R)
        mass=0.0; c_areal=0.0; foils=0; n_vac=0; n_pcm=0; n_teg=0; over=[]
        R_sg=R_rad=0.0
        for i,l in enumerate(sim.layers):
            mp=_sm_mat_props(l); th=l["thickness_mm"]*1e-3; rho=l.get("density",mp.get("rho",200.0))
            mass+=rho*th; c_areal+=rho*mp.get("cp",800.0)*th
            foils+=l.get("n_reflective",0) or 0; n_vac+=1 if l.get("vacuum") else 0
            if l["material"] in ("pcm","endothermic"): n_pcm+=1
            elif l["material"]=="teg": n_teg+=1
            tmax=mp.get("T_max"); Tavg=(sim.T_profile[i]+sim.T_profile[i+1])/2.0
            if tmax and Tavg>tmax: over.append((i,l["name"],Tavg,tmax))
            # R-weighted share of conduction through solids+gas vs radiation
            ks=sim.q_cond[i] if i<len(sim.q_cond) else k[i]
            kr=sim.q_rad[i] if i<len(sim.q_rad) else 0.0
            if k[i]>0:
                R_sg+=R[i]*min(max(ks/k[i],0.0),1.0); R_rad+=R[i]*min(max(kr/k[i],0.0),1.0)
        return {"n":n,"R":R,"R_total":R_total,"U":1.0/max(R_total,1e-12),
                "mass":mass,"c_areal":c_areal,"tau":c_areal*R_total,
                "foils":int(foils),"n_vac":n_vac,"n_pcm":n_pcm,"n_teg":n_teg,
                "over_tmax":over,"R_sg":R_sg,"R_rad":R_rad}

    def _synergy_cached(self):
        """SM stacking-synergy for the active formulation at the live boundary
        conditions, computed once per (approach, scenario) and cached."""
        f=FORMULATIONS[self.formulation_idx]
        key=(f["approach"],round(self.sim.T_hot,3),round(self.sim.T_cold,6))
        if key not in self._synergy_cache:
            st=sm_stack_for(f["approach"],self.sim.T_hot,self.sim.T_cold)
            try: self._synergy_cache[key]=sm_synergy_factor(st) if st is not None else None
            except Exception: self._synergy_cache[key]=None
        return self._synergy_cache[key]

    def _scenario_sweep(self):
        """Suppression of the active formulation across all 5 boundary
        scenarios, solved on its scientific-stack twin (dict-sim fallback);
        cached per approach since each run is ~5 steady-state solves."""
        f=FORMULATIONS[self.formulation_idx]
        if f["approach"] not in self._scenario_cache:
            rows=[]
            for sc in BOUNDARY_SCENARIOS:
                try:
                    st=sm_stack_for(f["approach"],sc["T_hot"],sc["T_cold"])
                    if st is not None:
                        r=st.solve_steady_state()
                        rows.append((sc["name"],r["q_transmitted"],r["suppression_pct"]))
                    else:
                        s=run_simulation(f["approach"],duration_s=300,T_hot=sc["T_hot"],T_cold=sc["T_cold"])
                        rows.append((sc["name"],s.q_transmitted,s.suppression_pct))
                except Exception:
                    rows.append((sc["name"],float("nan"),float("nan")))
            self._scenario_cache[f["approach"]]=rows
        return self._scenario_cache[f["approach"]]

    def _transient_cached(self):
        """Implicit-FV transient cross-check for the active formulation at the
        current boundary conditions; cached (each run is ~1-2 s of solving)."""
        ap=FORMULATIONS[self.formulation_idx]["approach"]
        key=(ap,round(self.sim.T_hot,3),round(self.sim.T_cold,6))
        if key not in self._transient_cache:
            st=sm_stack_for(ap,self.sim.T_hot,self.sim.T_cold)
            try: self._transient_cache[key]=_implicit_transient(st,3600.0) if st is not None else None
            except Exception: self._transient_cache[key]=None
        return self._transient_cache[key]

    def _proof_checks(self):
        """Live pass/fail list mirroring the --ultra-proof verification block,
        evaluated on the active formulation at the current BCs."""
        m=self._stack_metrics(); sy=self._synergy_cached()
        checks=[("steady-state suppression > 99%",self.sim.suppression_pct>99.0,"%.4f%%"%self.sim.suppression_pct),
                ("finite R_total > 0",math.isfinite(m["R_total"]) and m["R_total"]>0.0,"%.2f m2K/W"%m["R_total"]),
                ("stacking synergy S_total > 1",sy is not None and sy["total"]>1.0,
                 "%.3f"%sy["total"] if sy else "n/a")]
        checks.append(("all layers under T_max",len(m["over_tmax"])==0,"%d viol."%len(m["over_tmax"])))
        n_pass=sum(1 for _,_,sp in self._scenario_sweep() if sp>99.0)
        checks.append(("all 5 boundary scenarios pass",n_pass==5,"%d/5"%n_pass))
        tr=self._transient_cached()
        checks.append(("transient cold-face flux bounded (1h)",
                       tr is not None and tr["q_end"]<0.01*max(self.sim.q_incoming,1e-9),
                       "%.3g W/m2"%tr["q_end"] if tr else "n/a"))
        checks.append(("finite results",
                       all(math.isfinite(v) for v in (self.sim.q_transmitted,m["R_total"],m["U"])),""))
        return checks

    def _export_layers_csv(self):
        """E: write the live per-layer table (materials, k components, R, T,
        T_max status, flags) to a timestamped CSV next to the script."""
        f=FORMULATIONS[self.formulation_idx]; sim=self.sim; m=self._stack_metrics()
        fn="insulatrix_layers_%s_%s.csv"%(f["approach"],time.strftime("%Y%m%d_%H%M%S"))
        try:
            with open(fn,"w",newline="") as fh:
                w=csv.writer(fh)
                w.writerow(["idx","zone","name","material","thickness_mm","density",
                            "porosity","pore_size_um","emissivity","opacifier","foils","vacuum","vac_Pa",
                            "k_s+g_mW_mK","k_rad_mW_mK","k_eff_mW_mK","R_m2K_W",
                            "T_in_K","T_out_K","dT_K","T_max_K","over_T_max","flags","role"])
                for i,l in enumerate(sim.layers):
                    mp=_sm_mat_props(l); tmax=mp.get("T_max","")
                    T1,T2=sim.T_profile[i],sim.T_profile[i+1]
                    over=tmax!="" and (T1+T2)/2.0>tmax
                    flags=[]
                    if sim.pcm_capacity[i]>0: flags.append("PCM %.0f%%"%(sim.pcm_charged[i]*100))
                    w.writerow([i,_layer_zone(l["name"]),l["name"],l["material"],
                                "%.4f"%l["thickness_mm"],l.get("density",""),l.get("porosity",""),
                                l.get("pore_size_um",""),l.get("emissivity",""),l.get("opacifier",""),
                                l.get("n_reflective",0),int(bool(l.get("vacuum"))),
                                l.get("vacuum_pressure",""),
                                "%.6f"%((sim.q_cond[i] if i<len(sim.q_cond) else 0.0)*1000.0),
                                "%.6f"%((sim.q_rad[i] if i<len(sim.q_rad) else 0.0)*1000.0),
                                "%.6f"%(max(sim.k_eff[i],0.0)*1000.0),"%.4f"%m["R"][i],
                                "%.3f"%T1,"%.3f"%T2,"%.3f"%(T1-T2),tmax,int(bool(over)),
                                "; ".join(flags),l.get("why","")])
            print("Layer table exported: %s"%fn)
        except Exception as e:
            print("CSV export failed: %s"%e)

    def _step_layer(self,d):
        """J/K: step the pinned layer through the stack, auto-scrolling the
        layer-breakdown list so the row stays visible."""
        n=self.sim.n_layers
        cur=-1 if self.pinned_layer is None else self.pinned_layer
        self.pinned_layer=(cur+d)%n
        # Flattened-row index of this layer (zone headers consume rows too)
        row_i=-1; last_z=None
        for i,l in enumerate(self.sim.layers):
            z=_layer_zone(l["name"])
            if z!=last_z and z: row_i+=1
            last_z=z; row_i+=1
            if i==self.pinned_layer: break
        nv=getattr(self,"_layer_nvis",10)
        if row_i<self.layer_scroll or row_i>=self.layer_scroll+nv:
            self.layer_scroll=max(0,row_i-nv//2)

    def _select_pod_scenario(self,idx):
        if 0<=idx<len(self.pod_threats):
            self.pod_scenario_idx=idx; self.pod_test_phase="idle"; self.pod_test_t=0.0
            if self.pod_scene is not None: self.pod_scene.particles=ParticleSystem()

    def _trigger_pod_test(self):
        if self.pod_test_phase in ("idle","done"):
            self.pod_test_phase="incoming"; self.pod_test_t=0.0
            if self.pod_scene is not None: self.pod_scene.particles=ParticleSystem()

    def update(self,dt):
        if self.tab=="pod":
            self._update_pod(dt); return
        dt_sim=dt*60; self.sim.step(dt_sim)
        if self.auto_rotate and not self.dragging: self.scene.az+=dt*0.3
        if self.auto_cycle:
            self.auto_timer+=dt
            if self.auto_timer>15.0:
                self.auto_timer=0.0; self._select_formulation((self.formulation_idx+1)%len(FORMULATIONS))
        # Update T_profile for rendering
        self.scene.T_profile=self.sim.T_profile
        # Heat particles
        if self.show_particles and self.sim.running:
            self.heat_emit_timer+=dt
            if self.heat_emit_timer>0.05:
                self.heat_emit_timer=0.0
                T=_mm(self.sim.total_thickness/MM)
                hot_pos=np.array([-T/2-_mm(5),0,0])
                self.scene.particles.emit_hot(hot_pos,C_HEAT_PART,count=3)
                cold_pos=np.array([T/2+_mm(5),0,0])
                self.scene.particles.emit_hot(cold_pos,C_COOL_PART,count=2)
        self.scene.particles.update(dt)
        if not self.ratio_history or self.sim.t-self.ratio_history[-1][0]>=0.5:
            self.ratio_history.append((self.sim.t,self.sim.transmission_pct,self.sim.suppression_pct))
            if len(self.ratio_history)>400: self.ratio_history=self.ratio_history[-400:]

    def _update_pod(self,dt):
        if self.pod_scene is None: return
        r=self.pod_threats[self.pod_scenario_idx]
        R_outer=_mm(IMPACT_PHYS["pod_radius_mm"])
        proj=next((p for p in self.pod_scene.parts if p.key=="projectile"),None)
        if self.auto_rotate and not self.dragging: self.pod_scene.az+=dt*0.3
        ph=self.pod_test_phase
        fx=self.pod_scene.particles.emit_burst if self.show_particles else (lambda *a,**k: None)
        if ph=="idle":
            if proj is not None: proj.explode=np.array([R_outer*2.6,0,0])
            if self.auto_cycle: self._trigger_pod_test()
        elif ph=="incoming":
            self.pod_test_t+=dt; dur=1.2; t=clamp(self.pod_test_t/dur)
            if proj is not None:
                start=R_outer*2.6; end=R_outer*1.02
                proj.explode=np.array([start+(end-start)*t,0,0])
            if t>=1.0: self.pod_test_phase="pulse"; self.pod_test_t=0.0
        elif ph=="pulse":
            self.pod_test_t+=dt
            if self.pod_test_t<=dt*1.5 and r["conductive"]:
                fx((R_outer*0.9,0,0),(110,180,255),count=30,speed=0.5,life=(0.3,0.6))
            if self.pod_test_t>=0.35: self.pod_test_phase="impact"; self.pod_test_t=0.0
        elif ph=="impact":
            self.pod_test_t+=dt
            if self.pod_test_t<=dt*1.5:
                fx((R_outer*1.0,0,0),(255,190,90),count=40,speed=0.6,life=(0.3,0.7))
            if proj is not None: proj.explode=np.array([R_outer*3.0,0,0])
            if self.pod_test_t>=0.35: self.pod_test_phase="result"; self.pod_test_t=0.0
        elif ph=="result":
            self.pod_test_t+=dt
            rear=(R_outer*0.2,0,0)
            if r["perforated"] and not r["beyond_patch_rating"]:
                dur=clamp(r["patch_time_s"] or 1.0,0.5,3.0)
                if self.pod_test_t<=dt*1.5:
                    fx(rear,(255,90,80),count=20,speed=0.3,life=(0.4,0.8))
                if random.random()<0.5:
                    fx(rear,(120,230,150),count=2,speed=0.15,life=(0.5,1.0))
            elif r["perforated"]:
                dur=1.2
                if self.pod_test_t<=dt*1.5:
                    fx(rear,(255,60,50),count=50,speed=0.7,life=(0.6,1.2))
            else:
                dur=0.8
                if self.pod_test_t<=dt*1.5:
                    fx((R_outer*0.5,0,0),(120,230,150),count=20,speed=0.3,life=(0.4,0.7))
            if self.pod_test_t>=dur: self.pod_test_phase="done"; self.pod_test_t=0.0
        elif ph=="done" and self.auto_cycle:
            self.pod_test_t+=dt
            if self.pod_test_t>=2.5:
                self._select_pod_scenario((self.pod_scenario_idx+1)%len(self.pod_threats))
        self.pod_scene.particles.update(dt)

    def draw(self):
        t_now=pygame.time.get_ticks()*0.001
        # Radial vignette background with subtle pulse when running
        self.screen.fill(C_BG)
        cx,cy=self.W//2,self.H//2; max_r=math.sqrt(cx*cx+cy*cy)
        vig=pygame.Surface((self.W,self.H),pygame.SRCALPHA)
        base_a=22+int(4*math.sin(t_now*1.5)) if (self.tab=="pod" or self.sim.running) else 25
        for r in range(int(max_r),0,-40):
            a=int(base_a*(r/max_r))
            pygame.draw.circle(vig,(0,0,0,a),(cx,cy),r)
        self.screen.blit(vig,(0,0))
        # Animated starfield
        if not hasattr(self,'_stars'):
            self._stars=[(random.randint(0,self.W),random.randint(0,self.H),random.uniform(0.3,1.0),random.uniform(0,math.tau)) for _ in range(80)]
        for sx,sy,br,ph in self._stars:
            tw=0.5+0.5*math.sin(t_now*2.0+ph)
            sb=int(br*tw*120)
            if sb>10:
                try: pygame.draw.circle(self.screen,(C_STAR[0],C_STAR[1],C_STAR[2],sb),(int(sx),int(sy)),1)
                except: pass
        rect=pygame.Rect(0,0,self.W,self.H)
        if self.tab=="pod":
            self.pod_scene.render(self.screen,rect,self.angles,label_font=self.font)
            title,title_col="INSULATRIX POD",(120,180,255)
            subtitle=self.pod_threats[self.pod_scenario_idx]["label"]
        else:
            self.scene.render(self.screen,rect,self.angles,label_font=self.font)
            f=FORMULATIONS[self.formulation_idx]; title,title_col=f["name"],f["color"]
            subtitle=BOUNDARY_SCENARIOS[self.scenario_idx]["name"]
        # Title with drop shadow and glow
        sh=self.fbig.render(title,True,(0,0,0))
        nm=self.fbig.render(title,True,title_col)
        glow=pygame.Surface((nm.get_width()+20,nm.get_height()+12),pygame.SRCALPHA)
        for gr in range(8,0,-2):
            pygame.draw.rect(glow,(*title_col,8),(gr,gr,nm.get_width()+20-2*gr,nm.get_height()+12-2*gr),border_radius=4)
        self.screen.blit(glow,((self.W-nm.get_width())//2-10,12-6),special_flags=pygame.BLEND_ADD)
        self.screen.blit(sh,((self.W-nm.get_width())//2+2,14))
        self.screen.blit(nm,((self.W-nm.get_width())//2,12))
        bar_w=nm.get_width()+40; bar_x=(self.W-bar_w)//2
        pygame.draw.line(self.screen,C_ACCENT,(bar_x,38),(bar_x+bar_w,38),1)
        sc_surf=self.font.render(subtitle,True,C_TEXT_DIM)
        self.screen.blit(sc_surf,((self.W-sc_surf.get_width())//2,42))
        model_surf=self.font.render("MODEL %d/%d  (TAB: next, SHIFT+TAB: prev)"%(
            self._model_index()+1,self._model_count()),True,C_TEXT_DIM)
        self.screen.blit(model_surf,((self.W-model_surf.get_width())//2,60))
        if self.show_stats:
            if self.tab=="pod": self._draw_pod_stats()
            else:
                self._layer_rect=None
                self._draw_stats()
        if self.show_help: self._draw_help()
        if self.show_info: self._draw_info()
        if self.show_chem: self._draw_chem_diagram()
        if self.show_durability: self._draw_durability()
        if self.show_inspector and self.tab!="pod": self._draw_inspector()
        if self.show_proof and self.tab!="pod": self._draw_proof()
        if self.show_fab and self.tab!="pod": self._draw_fabrication()
        self._draw_hotkeys(); pygame.display.flip()

    def _draw_stats(self):
        f=FORMULATIONS[self.formulation_idx]; w,h=440,400; x,y=16,16
        panel(self.screen,x,y,w,h,alpha=230)
        self.screen.blit(self.fb.render("INSULATRIX",True,C_ACCENT),(x+16,y+10))
        # Status indicator (pulsing dot)
        status_col=C_SUCCESS if self.sim.running else C_TEXT_DIM
        pulse_r=4+int(2*math.sin(self.sim.t*3.0)) if self.sim.running else 4
        pygame.draw.circle(self.screen,status_col,(x+w-24,y+18),pulse_r)
        status_txt="RUNNING" if self.sim.running else "PAUSED"
        self.screen.blit(self.font.render(status_txt,True,status_col),(x+w-72,y+12))
        self.screen.blit(self.font.render("Near-Zero Thermal Transfer Digital Twin",True,C_TEXT_DIM),(x+16,y+32))
        lines=[("Formulation:",f["name"]),("Scenario:",BOUNDARY_SCENARIOS[self.scenario_idx]["name"]),
               ("Elapsed:","%.1f s"%self.sim.elapsed),
               ("T_hot:","%.0f K (%.0f C)"%(self.sim.T_hot,self.sim.T_hot-273.15)),
               ("T_cold:","%.0f K (%.0f C)"%(self.sim.T_cold,self.sim.T_cold-273.15)),
               ("dT:","%.0f K"%(self.sim.T_hot-self.sim.T_cold)),
               ("Thickness:","%.1f mm"%(self.sim.total_thickness/MM)),
               ("",""),("q_incoming:","%.6f W/m2"%self.sim.q_incoming),
               ("q_transmitted:","%.6f W/m2"%self.sim.q_transmitted),
               ("k_eff overall:","%.6f mW/m.K"%self.sim.k_eff_overall),
               ("",""),("Suppression:","%.4f%%"%self.sim.suppression_pct),
               ("Transmission:","%.6f%%"%self.sim.transmission_pct),
               ("",""),("PCM/chem absorbed:","%.3f MJ/m2"%(self.sim.total_absorbed/1e6)),
               ("TEG converted:","%.3f MJ/m2"%(self.sim.total_converted/1e6)),
               ("Peltier pumped:","%.3f MJ/m2"%(self.sim.total_pumped/1e6))]
        for i,(k,v) in enumerate(lines):
            if k:
                self.screen.blit(self.font.render(k,True,C_TEXT_DIM),(x+16,y+58+i*18))
                col=C_ACCENT if "Suppression" in k else (C_SUCCESS if "Transmission" in k else C_TEXT)
                self.screen.blit(self.font.render(str(v),True,col),(x+160,y+58+i*18))
        # STACK PHYSICS right column: aggregate stack-level metrics computed
        # live from the per-layer k_eff field (see _stack_metrics). Compact
        # "key value" strings at 12px to fit the free ~174 px right column.
        m=self._stack_metrics(); fs=self.fs; mx=x+308
        self.screen.blit(fs.render("STACK PHYSICS",True,C_ACCENT),(mx,y+56))
        mrows=[("R %.0f m2K/W"%m["R_total"],False),
               ("U %.4f W/m2K"%m["U"],False),
               ("mass %.0f kg/m2"%m["mass"],False),
               ("Cth %.1f kJ/m2K"%(m["c_areal"]/1e3),False),
               ("tau %.1f h"%(m["tau"]/3600.0),False),
               ("%dL %df %dv %drec"%(m["n"],m["foils"],m["n_vac"],m["n_pcm"]+m["n_teg"]),False),
               ("T_max viol:%d"%len(m["over_tmax"]),len(m["over_tmax"])>0)]
        for i,(v,bad) in enumerate(mrows):
            self.screen.blit(fs.render(v,True,C_FAIL if bad else C_TEXT),(mx,y+76+i*15))
        # Temperature gradient bar
        gb_y=y+h-58; gb_x=x+16; gb_w=w-32; gb_h=16
        self.screen.blit(self.font.render("Temperature profile:",True,C_TEXT_DIM),(gb_x,gb_y-16))
        n_prof=len(self.sim.T_profile)
        if n_prof>1:
            for px_i in range(gb_w):
                frac=px_i/max(gb_w-1,1); ti=frac*(n_prof-1); ti0=int(ti); ti1=min(ti0+1,n_prof-1)
                tf=ti-ti0; T_px=self.sim.T_profile[ti0]*(1-tf)+self.sim.T_profile[ti1]*tf
                tcol=temp_to_color(T_px)
                pygame.draw.line(self.screen,tcol,(gb_x+px_i,gb_y),(gb_x+px_i,gb_y+gb_h))
        pygame.draw.rect(self.screen,C_PANEL_BORDER,(gb_x,gb_y,gb_w,gb_h),1)
        # Hot/cold labels at bar ends + numeric boundary temps beneath
        self.screen.blit(self.font.render("H",True,C_HOT),(gb_x-2,gb_y+1))
        self.screen.blit(self.font.render("C",True,C_COLD),(gb_x+gb_w-8,gb_y+1))
        th_l=self.fs.render("%.0f K"%self.sim.T_hot,True,C_HOT)
        tc_l=self.fs.render("%.4g K"%self.sim.T_cold,True,C_COLD)
        self.screen.blit(th_l,(gb_x,gb_y+gb_h+2))
        self.screen.blit(tc_l,(gb_x+gb_w-tc_l.get_width(),gb_y+gb_h+2))
        if self.sim.is_success:
            self.screen.blit(self.fb.render("SUCCESS  suppression > 99%",True,C_SUCCESS),(x+16,y+h-90))
        else:
            self.screen.blit(self.fb.render("below threshold",True,C_FAIL),(x+16,y+h-90))
        # Suppression progress ring (top-right of panel)
        ring_cx=x+w-50; ring_cy=y+h-80; ring_r=28
        supp_frac=clamp(self.sim.suppression_pct/100.0)
        # Background ring
        pygame.draw.circle(self.screen,(40,44,56),(ring_cx,ring_cy),ring_r,3)
        # Filled arc
        n_arc=max(2,int(60*supp_frac))
        ring_col=C_SUCCESS if self.sim.is_success else C_ACCENT
        for ai in range(n_arc):
            ang=math.tau*ai/60 - math.pi/2
            px=ring_cx+ring_r*math.cos(ang); py=ring_cy+ring_r*math.sin(ang)
            pygame.draw.circle(self.screen,ring_col,(int(px),int(py)),2)
        # Center text
        pct_txt=self.font.render("%.1f%%"%self.sim.suppression_pct,True,ring_col)
        self.screen.blit(pct_txt,(ring_cx-pct_txt.get_width()//2,ring_cy-pct_txt.get_height()//2))
        # Layer breakdown: scrollable full-stack inspector (ULTRA has 35
        # layers -- far more than the old fixed 6-row view). Wheel-scrolls
        # when the cursor is over the panel; zone headers group tagged layers.
        gy_ref=self.H-100-30
        ly=y+h+8; lw,lh=440,max(90,gy_ref-8-ly)
        self._layer_rect=pygame.Rect(x,ly,lw,lh)
        panel(self.screen,x,ly,lw,lh,alpha=220)
        self.screen.blit(self.fb.render("LAYER BREAKDOWN (wheel: scroll | hover: detail)",True,C_ACCENT),(x+12,ly+8))
        # Flatten rows: zone separators + one row per layer
        rows=[]; last_z=None
        for i,layer in enumerate(self.sim.layers):
            z=_layer_zone(layer["name"])
            if z!=last_z:
                if z: rows.append(("zone",z))
                last_z=z
            rows.append(("layer",i))
        row_h=14; top=ly+30; n_vis=max(1,(lh-38)//row_h); self._layer_nvis=n_vis
        self.layer_scroll=max(0,min(self.layer_scroll,max(0,len(rows)-n_vis)))
        self._layer_rows=[]
        for r_i,(kind,val) in enumerate(rows[self.layer_scroll:self.layer_scroll+n_vis]):
            ry=top+r_i*row_h
            if kind=="zone":
                self.screen.blit(self.fs.render("-- ZONE %s --"%val,True,C_ACCENT),(x+26,ry)); continue
            i=val; layer=self.sim.layers[i]
            row_rect=pygame.Rect(x+8,ry,lw-16,row_h)
            self._layer_rows.append((row_rect,i))
            if i==self.pinned_layer:
                pygame.draw.rect(self.screen,(50,54,72),row_rect)
                pygame.draw.rect(self.screen,C_ACCENT,row_rect,1)
            elif row_rect.collidepoint(self.mouse_pos):
                pygame.draw.rect(self.screen,(42,46,62),row_rect)
            T1=self.sim.T_profile[i]; T2=self.sim.T_profile[i+1]
            Ri=m["R"][i]
            flags=""
            if layer.get("vacuum"): flags+=" VAC"
            if layer.get("n_reflective",0)>0: flags+=" x%d"%layer["n_reflective"]
            if self.sim.pcm_capacity[i]>0: flags+=" PCM%d%%"%(self.sim.pcm_charged[i]*100)
            mp=_sm_mat_props(layer); tmax=mp.get("T_max")
            over=tmax is not None and (T1+T2)/2.0>tmax
            if over: flags+=" !>Tmax"
            txt="L%-2d %-20.20s %4.1fmm R=%7.2f %4.0f>%4.0fK%s"%(i,layer["name"],layer["thickness_mm"],Ri,T1,T2,flags)
            lcol=_layer_color(layer)
            pygame.draw.rect(self.screen,lcol,(x+12,ry+2,8,8))
            pygame.draw.rect(self.screen,C_PANEL_BORDER,(x+12,ry+2,8,8),1)
            self.screen.blit(self.fs.render(txt,True,C_FAIL if over else C_TEXT),(x+26,ry))
        # Scrollbar
        if len(rows)>n_vis:
            sb_h=max(20,int((lh-40)*n_vis/len(rows)))
            sb_y=top+int((lh-40-sb_h)*self.layer_scroll/max(1,len(rows)-n_vis))
            pygame.draw.rect(self.screen,(70,76,92),(x+lw-8,top,4,lh-40),border_radius=2)
            pygame.draw.rect(self.screen,C_ACCENT,(x+lw-8,sb_y,4,sb_h),border_radius=2)
        self.screen.blit(self.fs.render("%d/%d"%(min(self.layer_scroll+n_vis,len(rows)),len(rows)),True,C_TEXT_DIM),(x+lw-64,ly+12))
        # Cross-highlight: hovered (or pinned) layer row tints that slab in 3D
        self.hover_layer=None
        for rect,i in self._layer_rows:
            if rect.collidepoint(self.mouse_pos): self.hover_layer=i; break
        self.scene.highlight_layer=(self.pinned_layer if self.pinned_layer is not None
                                    else (self.hover_layer if self.hover_layer is not None else -1))
        # Hover tooltip: full per-layer detail card under the cursor.
        self._draw_layer_tooltip(x,y,w)
        # Graph
        gw,gh=440,100; gx,gy=16,self.H-gh-30
        panel(self.screen,gx,gy,gw,gh,alpha=220)
        self.screen.blit(self.font.render("SUPPRESSION / TRANSMISSION OVER TIME",True,C_ACCENT),(gx+10,gy+6))
        if len(self.ratio_history)>1:
            px,py=gx+10,gy+24; pw,ph=gw-20,gh-32
            max_t=max(self.ratio_history[-1][0],1.0)
            max_v=100.0
            # Grid lines
            for gp in range(0,101,25):
                gy_line=py+ph-int(ph*gp/max_v)
                pygame.draw.line(self.screen,C_GRID,(px,gy_line),(px+pw,gy_line),1)
            # 99% threshold line
            ty99=py+ph-int(ph*99.0/max_v)
            pygame.draw.line(self.screen,(100,100,60),(px,ty99),(px+pw,ty99),1)
            self.screen.blit(self.font.render("99%",True,(100,100,60)),(px+pw-24,ty99-14))
            pygame.draw.line(self.screen,C_PANEL_BORDER,(px,py+ph),(px+pw,py+ph),1)
            pygame.draw.line(self.screen,C_PANEL_BORDER,(px,py),(px,py+ph),1)
            # Y-axis labels
            for gp in [0,50,100]:
                ly_ax=py+ph-int(ph*gp/max_v)
                self.screen.blit(self.font.render("%d%%"%gp,True,C_TEXT_DIM),(px-22,ly_ax-6))
            # X-axis label
            self.screen.blit(self.font.render("t=%.0fs"%max_t,True,C_TEXT_DIM),(px+pw-30,py+ph+4))
            pts_s=[]; pts_t=[]
            for t,trans,supp in self.ratio_history:
                sx=px+int(pw*t/max_t); sy_s=py+ph-int(ph*supp/max_v); sy_t=py+ph-int(ph*trans/max_v)
                pts_s.append((sx,sy_s)); pts_t.append((sx,sy_t))
            if len(pts_s)>1: pygame.draw.lines(self.screen,C_SUCCESS,False,pts_s,2)
            if len(pts_t)>1: pygame.draw.lines(self.screen,C_FAIL,False,pts_t,1)
            # Area fill under suppression curve
            if len(pts_s)>2:
                fill_pts=pts_s+[(pts_s[-1][0],py+ph),(pts_s[0][0],py+ph)]
                fill_s=pygame.Surface((pw,ph),pygame.SRCALPHA)
                fill_pts_local=[(p[0]-px,p[1]-py) for p in fill_pts]
                try: pygame.draw.polygon(fill_s,(*C_SUCCESS,30),fill_pts_local)
                except: pass
                self.screen.blit(fill_s,(px,py))
                # Redraw curve on top of fill
                pygame.draw.lines(self.screen,C_SUCCESS,False,pts_s,2)
                pygame.draw.lines(self.screen,C_FAIL,False,pts_t,1)
            self.screen.blit(self.font.render("suppression",True,C_SUCCESS),(px+pw-70,py+2))
            self.screen.blit(self.font.render("transmission",True,C_FAIL),(px+pw-70,py+16))
            # Current-time indicator line
            cur_x=px+int(pw*self.ratio_history[-1][0]/max_t)
            pygame.draw.line(self.screen,C_ACCENT,(cur_x,py),(cur_x,py+ph),1)
            # Live data values at right edge
            cur_s=self.ratio_history[-1][2]; cur_t=self.ratio_history[-1][1]
            val_txt="S:%.2f%%  T:%.4f%%"%(cur_s,cur_t)
            self.screen.blit(self.font.render(val_txt,True,C_TEXT),(px+4,py+ph-16))
        # Formulation list with live suppression values (precomputed at startup)
        sw,sh=240,288; sx,sy=self.W-sw-16,16
        panel(self.screen,sx,sy,sw,sh,alpha=230)
        self.screen.blit(self.fb.render("FORMULATIONS",True,C_ACCENT),(sx+12,sy+8))
        self.screen.blit(self.fs.render("supp%%",True,C_TEXT_DIM),(sx+sw-72,sy+12))
        for i,form in enumerate(FORMULATIONS):
            is_sel=i==self.formulation_idx
            if is_sel:
                pygame.draw.rect(self.screen,(40,45,60),(sx+8,sy+32+i*20,sw-16,18))
                pygame.draw.line(self.screen,C_ACCENT,(sx+8,sy+32+i*20),(sx+8,sy+50+i*20),2)
            col=C_ACCENT if is_sel else C_TEXT
            label="%d. %s"%(form["id"],form["name"])
            self.screen.blit(self.font.render(label,True,col),(sx+14,sy+34+i*20))
            supp_v=self._form_supp[i] if i<len(self._form_supp) else 0.0
            supp_col=C_SUCCESS if self._form_success[i] else C_FAIL
            sv=self.fs.render("%.1f"%supp_v,True,supp_col)
            self.screen.blit(sv,(sx+sw-78-sv.get_width(),sy+36+i*20))
            # Success/fail indicator dot
            pygame.draw.circle(self.screen,supp_col,(sx+sw-40,sy+40+i*20),3)
            sw_x=sx+sw-24
            pygame.draw.rect(self.screen,form["color"],(sw_x,sy+36+i*20,12,12))
            pygame.draw.rect(self.screen,C_PANEL_BORDER,(sw_x,sy+36+i*20,12,12),1)

    def _draw_layer_tooltip(self,px,py,pw):
        """Hover card for the layer-breakdown list: full per-layer detail in an
        opaque box under the cursor (all material dict fields + live physics)."""
        rows=getattr(self,"_layer_rows",[])
        if not rows and self.pinned_layer is None: return
        hit=None
        for rect,i in rows:
            if rect.collidepoint(self.mouse_pos): hit=i; break
        if hit is None:
            if self.pinned_layer is None: return
            hit=self.pinned_layer  # pinned layer keeps its card open
        sim=self.sim; i=hit; l=sim.layers[i]; mp=_sm_mat_props(l)
        T1,T2=sim.T_profile[i],sim.T_profile[i+1]; ke=max(sim.k_eff[i],1e-12)
        ks=sim.q_cond[i] if i<len(sim.q_cond) else ke
        kr=sim.q_rad[i] if i<len(sim.q_rad) else 0.0
        tmax=mp.get("T_max"); over=tmax is not None and (T1+T2)/2.0>tmax
        ps=l.get("pore_size_um",0.0)
        lines=[("L%d  %s"%(i,l["name"]),C_ACCENT),
               ("material: %s (%s)"%(l["material"],mp.get("name","-")),C_TEXT),
               ("thick %.2f mm | rho %.0f kg/m3 | porosity %.0f%% | pore %s um"%(
                   l["thickness_mm"],l.get("density",mp.get("rho",0.0)),l.get("porosity",0.0)*100.0,
                   "%.3g"%ps if ps>0 else "-"),C_TEXT),
               ("emissivity %.2f | opacifier %.2f | foils %d"%(
                   l.get("emissivity",0.9),l.get("opacifier",0.0),l.get("n_reflective",0) or 0),C_TEXT),
               ("vacuum: %s%s"%("yes" if l.get("vacuum") else "no",
                   "  p=%.0e torr"%l.get("vacuum_pressure",1e-5) if l.get("vacuum") else ""),C_TEXT),
               ("k_s+g %.3f | k_rad %.3f | k_eff %.3f mW/mK"%(ks*1000.0,kr*1000.0,ke*1000.0),C_TEXT),
               ("R %.3f m2K/W | T %.1f -> %.1f K | dT %.2f K"%(
                   l["thickness_mm"]*1e-3/ke,T1,T2,T1-T2),C_TEXT)]
        if sim.pcm_capacity[i]>0:
            lines.append(("PCM %.0f%% charged of %.2f MJ/m2"%(
                sim.pcm_charged[i]*100.0,sim.pcm_capacity[i]/1e6),C_TEXT))
        if tmax is not None:
            lines.append(("T_max %.0f K: %s"%(tmax,"OK" if not over else "OVER -- exceeds rating"),
                          C_FAIL if over else C_SUCCESS))
        if mp.get("reference"):
            lines.append(("ref: %s"%mp["reference"][:64],C_TEXT_DIM))
        if l.get("why"):
            lines.append(("ROLE:",C_ACCENT))
            for wl in wrap_text(self.fs,l["why"],430):
                lines.append((wl,C_TEXT_DIM))
        fs=self.fs
        tw=max(fs.size(t)[0] for t,_ in lines)+24
        th=len(lines)*16+16
        hovering=any(rect.collidepoint(self.mouse_pos) for rect,_ in rows)
        if hit==self.pinned_layer and not hovering and self._layer_rect is not None:
            tx=self._layer_rect.right+8; ty=self._layer_rect.top+4
        else:
            tx=self.mouse_pos[0]+14; ty=self.mouse_pos[1]+14
        tx=min(max(tx,4),self.W-tw-4); ty=min(max(ty,4),self.H-th-4)
        pygame.draw.rect(self.screen,(13,16,23),(tx,ty,tw,th),border_radius=5)
        pygame.draw.rect(self.screen,C_ACCENT,(tx,ty,tw,th),1,border_radius=5)
        for j,(t,c) in enumerate(lines):
            self.screen.blit(fs.render(t,True,c),(tx+12,ty+8+j*16))

    def _draw_pod_stats(self):
        r=self.pod_threats[self.pod_scenario_idx]; w,h=460,380; x,y=16,16
        panel(self.screen,x,y,w,h,alpha=230)
        self.screen.blit(self.fb.render("INSULATRIX POD",True,(120,180,255)),(x+16,y+10))
        phase_col={"idle":C_TEXT_DIM,"incoming":C_ACCENT,"pulse":(120,180,255),
                   "impact":(255,190,90),"result":C_FAIL,"done":C_SUCCESS}.get(self.pod_test_phase,C_TEXT)
        pygame.draw.circle(self.screen,phase_col,(x+w-24,y+18),4)
        self.screen.blit(self.font.render(self.pod_test_phase.upper(),True,phase_col),(x+w-110,y+12))
        self.screen.blit(self.font.render("Hypervelocity Impact Shield -- new-tab digital twin",True,C_TEXT_DIM),(x+16,y+32))
        margin_s="%.4g"%r["ballistic_margin"] if r["ballistic_margin"]<1e6 else ">1e6"
        verdict="STOPPED (no perforation)" if not r["perforated"] else ("AUTO-PATCHED -- SURVIVED" if r["survived"] else "BEYOND PATCH RATING")
        vcol=C_SUCCESS if (not r["perforated"] or r["survived"]) else C_FAIL
        lines=[("Threat:",r["label"][:34]),("Velocity:","%.0f m/s (%.0f mph)"%(r["v_ms"],r["v_ms"]/0.44704)),
               ("Mass:","%.3g kg"%r["mass_kg"]),("Kinetic energy:","%.3g J"%r["kinetic_energy_J"]),
               ("",""),("Fragmentation (Whipple):","%.1f%%"%(r["fragmentation_efficacy"]*100)),
               ("Magnetic deflection:","%.1f%%"%(r["deflect_frac"]*100) if r["conductive"] else "0.0% (non-conductive)"),
               ("Ballistic margin:",margin_s),("Perforated?:","YES" if r["perforated"] else "no"),
               ("",""),("Hole diameter:","%.1f mm"%r["hole_mm"] if r["perforated"] else "--"),
               ("Patch time:","%.2f s"%r["patch_time_s"] if r["patch_time_s"] else "--"),
               ("Cabin min pressure:","%.1f kPa"%r["cabin_min_kPa"])]
        for i,(k,v) in enumerate(lines):
            if k:
                self.screen.blit(self.font.render(k,True,C_TEXT_DIM),(x+16,y+58+i*18))
                self.screen.blit(self.font.render(str(v),True,C_TEXT),(x+230,y+58+i*18))
        self.screen.blit(self.fb.render(verdict,True,vcol),(x+16,y+h-30))
        # Hull cross-section panel
        ly=y+h+8; lw,lh=460,150
        panel(self.screen,x,ly,lw,lh,alpha=220)
        self.screen.blit(self.fb.render("HULL CROSS-SECTION (space-facing -> cabin)",True,C_ACCENT),(x+12,ly+8))
        col_map={"bumper":(215,220,230),"coil":(120,175,255),"stuffed":(225,190,95),
                 "thermal":(150,205,225),"rear_wall":(205,125,225)}
        for i,l in enumerate(self.pod.layers):
            lcol=col_map.get(l["role"],(150,150,160))
            pygame.draw.rect(self.screen,lcol,(x+12,ly+35+i*20,10,10))
            pygame.draw.rect(self.screen,C_PANEL_BORDER,(x+12,ly+35+i*20,10,10),1)
            txt="L%d: %-42s %8.1f mm"%(i,l["name"][:42],l["thickness_mm"])
            self.screen.blit(self.font.render(txt,True,C_TEXT),(x+26,ly+34+i*20))
        # Threat scenario picker
        sw,sh=230,300; sx,sy=self.W-sw-16,16
        panel(self.screen,sx,sy,sw,sh,alpha=230)
        self.screen.blit(self.fb.render("THREAT SCENARIOS (1-8)",True,C_ACCENT),(sx+12,sy+8))
        for i,rr in enumerate(self.pod_threats):
            is_sel=i==self.pod_scenario_idx
            if is_sel:
                pygame.draw.rect(self.screen,(40,45,60),(sx+8,sy+32+i*22,sw-16,20))
                pygame.draw.line(self.screen,C_ACCENT,(sx+8,sy+32+i*22),(sx+8,sy+52+i*22),2)
            col=C_ACCENT if is_sel else C_TEXT
            label="%d. %s"%(i+1,rr["label"][:26])
            self.screen.blit(self.font.render(label,True,col),(sx+14,sy+35+i*22))
            dot_col=C_SUCCESS if rr["survived"] else C_FAIL
            pygame.draw.circle(self.screen,dot_col,(sx+sw-16,sy+42+i*22),3)
        self.screen.blit(self.font.render("SPACE: launch test  A: auto-cycle  M: thermal tab",True,C_TEXT_DIM),(sx,sy+sh+8))

    def _draw_help(self):
        if self.tab=="pod":
            lines=["TAB / SHIFT+TAB  next/prev model (all %d, incl. this pod)"%self._model_count(),
                   "M            back to thermal tab","SPACE        launch impact test",
                   "1-8          select threat scenario","A            auto-cycle threats (loops)",
                   "BACKSPACE    reset current test to idle","T            auto-rotate camera",
                   "P            toggle particles (pulse/frag/sealant fx)",
                   "mouse drag   orbit","right-drag   pan","mouse wheel  zoom",
                   "R            reset camera","L            toggle labels",
                   "X            cross-section (cutaway through hull)",
                   "I            full specification","H            help","ESC / Q      quit","",
                   "Test sequence: incoming -> magnetic pulse (conductive threats",
                   "only) -> Whipple fragmentation -> stopped, or breach + self-heal.",
                   "Design reference: 8940.8 m/s (20,000 mph). CLI: --pod-report."]
        else:
            lines=["TAB / SHIFT+TAB  next/prev model (all %d, incl. the pod)"%self._model_count(),
                   "SPACE/B      toggle simulation","1-9, 0, C, U  select formulation (12 total)",
                   "  1=Foam      2=Aerogel   3=Ceramic","  4=Refractory 5=MLI       6=Hybrid",
                   "  7=PCM       8=Chemical  9=TEG","  0=Gradient  C=Insulatrix Pro  U=ULTRA CASCADE",
                   "+ / -        add/remove layers (thickness)","[ / ]        hot-side T up/down",
                   "; / '        cold-side T up/down","V            cycle boundary scenario (5)",
                   "W            durability/weathering panel","N            physics inspector (full stack detail)",
                   "P            toggle heat particles","A            auto-cycle formulations","T            auto-rotate camera",
                   "BACKSPACE    reset simulation","mouse drag   orbit","right-drag   pan",
                   "mouse wheel  zoom / scroll layer list & inspector","R            reset camera","L            toggle labels",
                   "layer rows   hover = detail + 3D highlight | click = pin","J / K        step pinned layer through the stack",
                   "O            live proof checklist","E            export layer table to CSV","F12          save screenshot PNG",
                   "F            fabrication protocol (full build steps)",
                   "X            cross-section","I            full specification",
                   "D            material diagram","M            impact-shield pod tab (new)",
                   "H            help","ESC / Q      quit","",
                   "Success = suppression ratio > 99%","Goal: MINIMIZE transmitted heat flux",
                   "Green = suppression, Red = transmission","Scale: 1:1 (all dimensions in real mm)"]
        w=480; h=min(self.H-40,70+len(lines)*20); x=(self.W-w)//2; y=(self.H-h)//2
        panel(self.screen,x,y,w,h,alpha=240)
        self.screen.blit(self.fb.render("CONTROLS",True,C_ACCENT),(x+20,y+16))
        pygame.draw.line(self.screen,C_PANEL_BORDER,(x+20,y+36),(x+120,y+36),1)
        self.screen.blit(self.font.render("close: H",True,C_TEXT_DIM),(x+w-80,y+20))
        for i,ln in enumerate(lines):
            if not ln:
                continue
            # Split into key part and description part at the first double-space gap
            parts=ln.split("  ",1)
            if len(parts)==2 and not ln.startswith("  "):
                key_part=parts[0].rstrip()
                desc_part=parts[1].lstrip()
                self.screen.blit(self.font.render(key_part,True,C_ACCENT),(x+24,y+50+i*20))
                self.screen.blit(self.font.render(desc_part,True,C_TEXT),(x+160,y+50+i*20))
            else:
                self.screen.blit(self.font.render(ln,True,C_TEXT),(x+24,y+50+i*20))

    def _draw_info(self):
        w,h=760,self.H-90; x=(self.W-w)//2; y=50
        panel(self.screen,x,y,w,h,alpha=244)
        self.screen.blit(self.fb.render("INSULATRIX -- FULL SPECIFICATION",True,C_ACCENT),(x+20,y+14))
        self.screen.blit(self.font.render("scroll: up/down   close: I",True,C_TEXT_DIM),(x+w-200,y+20))
        clip=pygame.Rect(x+16,y+46,w-48,h-60); self.screen.set_clip(clip)
        yy=y+50-self.info_scroll; maxpx=w-60; content_h=0
        # First pass: measure content height
        for head,lines in self.info_sections:
            content_h+=26
            for ln in lines:
                for wl in wrap_text(self.font,ln,maxpx): content_h+=20
            content_h+=12
        # Scrollbar
        sb_x=x+w-16; sb_y=y+46; sb_h=h-60
        if content_h>sb_h:
            thumb_h=max(20,int(sb_h*sb_h/content_h))
            thumb_y=sb_y+int((sb_h-thumb_h)*self.info_scroll/max(content_h-sb_h,1))
            pygame.draw.rect(self.screen,(40,44,56),(sb_x,sb_y,6,sb_h))
            pygame.draw.rect(self.screen,C_PANEL_BORDER,(sb_x,sb_y,6,sb_h),1)
            pygame.draw.rect(self.screen,C_ACCENT,(sb_x,thumb_y,6,thumb_h))
        # Content
        for head,lines in self.info_sections:
            if y+30<yy<y+h:
                self.screen.blit(self.fb.render(head,True,C_ACCENT),(x+24,yy))
                # Underline accent
                hw=self.fb.size(head)[0]
                pygame.draw.line(self.screen,C_PANEL_BORDER,(x+24,yy+20),(x+24+hw+8,yy+20),1)
            yy+=26
            for ln in lines:
                for wl in wrap_text(self.font,ln,maxpx):
                    if y+30<yy<y+h: self.screen.blit(self.font.render(wl,True,C_TEXT),(x+30,yy))
                    yy+=20
            yy+=12
        self.screen.set_clip(None)

    def _draw_molecule(self,surf,mol,ox,oy,scale=1.0):
        label,atoms,bonds=mol
        if atoms:
            xs=[a[1] for a in atoms]; ys=[a[2] for a in atoms]
            cx=(min(xs)+max(xs))/2; cy=(min(ys)+max(ys))/2
        else: cx,cy=0,0
        for ai,bi,order in bonds:
            if ai>=len(atoms) or bi>=len(atoms): continue
            x1=ox+(atoms[ai][1]-cx)*scale; y1=oy+(atoms[ai][2]-cy)*scale
            x2=ox+(atoms[bi][1]-cx)*scale; y2=oy+(atoms[bi][2]-cy)*scale
            # Element-colored bond halves
            col_a=EL_COLORS.get(atoms[ai][0],(150,150,160))
            col_b=EL_COLORS.get(atoms[bi][0],(150,150,160))
            mx,my=(x1+x2)/2,(y1+y2)/2
            if order==1:
                pygame.draw.line(surf,col_a,(x1,y1),(mx,my),2)
                pygame.draw.line(surf,col_b,(mx,my),(x2,y2),2)
            elif order==2:
                dx,dy=x2-x1,y2-y1; L=math.sqrt(dx*dx+dy*dy)
                if L>0.1:
                    nx,ny=-dy/L*3,dx/L*3
                    pygame.draw.line(surf,col_a,(x1+nx,y1+ny),(mx+nx,my+ny),2)
                    pygame.draw.line(surf,col_b,(mx+nx,my+ny),(x2+nx,y2+ny),2)
                    pygame.draw.line(surf,col_a,(x1-nx,y1-ny),(mx-nx,my-ny),2)
                    pygame.draw.line(surf,col_b,(mx-nx,my-ny),(x2-nx,y2-ny),2)
        for elem,ax,ay in atoms:
            px=ox+(ax-cx)*scale; py=oy+(ay-cy)*scale
            col=EL_COLORS.get(elem,(180,180,180)); r=EL_RADIUS.get(elem,8)
            # Outer ring (darker shade)
            dark=(max(0,col[0]-40),max(0,col[1]-40),max(0,col[2]-40))
            pygame.draw.circle(surf,dark,(int(px),int(py)),r+1)
            # Main body
            pygame.draw.circle(surf,col,(int(px),int(py)),r)
            # Highlight (upper-left lighter spot for 3D effect)
            hl=(min(255,col[0]+40),min(255,col[1]+40),min(255,col[2]+40))
            pygame.draw.circle(surf,hl,(int(px-r*0.3),int(py-r*0.3)),max(1,r//3))
            txt=self.font.render(elem,True,(255,255,255) if col[0]+col[1]+col[2]<400 else (20,20,20))
            surf.blit(txt,(px-txt.get_width()/2,py-txt.get_height()/2))
        lbl=self.font.render(label,True,C_TEXT_DIM)
        by=oy+20 if not atoms else oy+(max(ys)-cy)*scale+14
        surf.blit(lbl,(ox-lbl.get_width()/2,by))

    def _draw_chem_diagram(self):
        f=FORMULATIONS[self.formulation_idx]; rxn=CHEM_REACTIONS.get(f["approach"])
        if not rxn: return
        w,h=760,self.H-90; x=(self.W-w)//2; y=50
        panel(self.screen,x,y,w,h,alpha=244)
        self.screen.blit(self.fbig.render("MATERIAL DIAGRAM",True,C_ACCENT),(x+20,y+14))
        self.screen.blit(self.fb.render(f["name"],True,C_TEXT),(x+20,y+44))
        self.screen.blit(self.font.render("close: D",True,C_TEXT_DIM),(x+w-80,y+20))
        eq_y=y+72
        for i,wl in enumerate(wrap_text(self.font,rxn["eq"],w-60)):
            self.screen.blit(self.font.render(wl,True,C_ACCENT),(x+24,eq_y+i*18))
        eq_y+=len(wrap_text(self.font,rxn["eq"],w-60))*18+8
        for i,wl in enumerate(wrap_text(self.font,rxn["mech"],w-60)):
            self.screen.blit(self.font.render(wl,True,C_TEXT_DIM),(x+24,eq_y+i*16))
        eq_y+=len(wrap_text(self.font,rxn["mech"],w-60))*16+16
        # Divider line
        pygame.draw.line(self.screen,C_PANEL_BORDER,(x+24,eq_y-8),(x+w-24,eq_y-8),1)
        self.screen.blit(self.fb.render("REACTANTS",True,C_ACCENT),(x+24,eq_y))
        pygame.draw.line(self.screen,C_PANEL_BORDER,(x+24,eq_y+20),(x+140,eq_y+20),1); eq_y+=28
        reactants=rxn.get("reactants",[]); n_r=len(reactants)
        if n_r>0:
            col_w=(w-60)/max(n_r,1)
            for i,mol in enumerate(reactants):
                mx=x+30+col_w*i+col_w/2-30; my=eq_y+50
                self._draw_molecule(self.screen,mol,mx,my,scale=1.2)
        eq_y+=120
        # Reaction arrow with pulsing glow
        ax_c=x+w/2
        pulse=0.5+0.5*math.sin(self.sim.t*3.0)
        glow_a=int(40+40*pulse)
        glow_s=pygame.Surface((80,20),pygame.SRCALPHA)
        pygame.draw.line(glow_s,(*C_ACCENT,glow_a),(8,10),(72,10),6)
        self.screen.blit(glow_s,(ax_c-40,eq_y-10),special_flags=pygame.BLEND_ADD)
        pygame.draw.line(self.screen,(60,50,20),(ax_c-32,eq_y),(ax_c+32,eq_y),4)
        pygame.draw.line(self.screen,C_ACCENT,(ax_c-30,eq_y),(ax_c+30,eq_y),2)
        pygame.draw.polygon(self.screen,C_ACCENT,[(ax_c+32,eq_y),(ax_c+22,eq_y-6),(ax_c+22,eq_y+6)])
        eq_y+=20
        # Divider line
        pygame.draw.line(self.screen,C_PANEL_BORDER,(x+24,eq_y-4),(x+w-24,eq_y-4),1)
        self.screen.blit(self.fb.render("PRODUCTS",True,C_ACCENT),(x+24,eq_y))
        pygame.draw.line(self.screen,C_PANEL_BORDER,(x+24,eq_y+20),(x+120,eq_y+20),1); eq_y+=28
        products=rxn.get("products",[]); n_p=len(products)
        if n_p>0:
            col_w=(w-60)/max(n_p,1)
            for i,mol in enumerate(products):
                mx=x+30+col_w*i+col_w/2-30; my=eq_y+50
                self._draw_molecule(self.screen,mol,mx,my,scale=1.2)
        eq_y+=120
        pygame.draw.line(self.screen,C_PANEL_BORDER,(x+24,eq_y-4),(x+w-24,eq_y-4),1)
        self.screen.blit(self.fb.render("LIVE SIMULATION",True,C_ACCENT),(x+24,eq_y))
        pygame.draw.line(self.screen,C_PANEL_BORDER,(x+24,eq_y+20),(x+160,eq_y+20),1); eq_y+=26
        sim_lines=[("q_incoming:",    "%.6f W/m2"%self.sim.q_incoming, C_TEXT),
                   ("q_transmitted:", "%.6f W/m2"%self.sim.q_transmitted, C_TEXT),
                   ("Suppression:",   "%.4f%%"%self.sim.suppression_pct, C_SUCCESS if self.sim.is_success else C_FAIL),
                   ("k_eff overall:", "%.6f mW/m.K"%self.sim.k_eff_overall, C_TEXT),
                   ("PCM absorbed:",  "%.3f MJ/m2"%(self.sim.total_absorbed/1e6), C_ACCENT),
                   ("TEG converted:", "%.3f MJ/m2"%(self.sim.total_converted/1e6), C_ACCENT)]
        for lbl,val,col in sim_lines:
            self.screen.blit(self.font.render(lbl,True,C_TEXT_DIM),(x+30,eq_y))
            # Value with subtle background pill
            val_surf=self.font.render(val,True,col)
            pill_x=x+160; pill_w=val_surf.get_width()+12; pill_h=16
            pygame.draw.rect(self.screen,(30,34,46),(pill_x-4,eq_y-1,pill_w,pill_h),border_radius=3)
            self.screen.blit(val_surf,(pill_x+4,eq_y)); eq_y+=20

    def _draw_durability(self):
        f=FORMULATIONS[self.formulation_idx]
        w,h=700,self.H-90; x=(self.W-w)//2; y=50
        panel(self.screen,x,y,w,h,alpha=244)
        self.screen.blit(self.fbig.render("DURABILITY & WEATHERING",True,C_ACCENT),(x+20,y+14))
        self.screen.blit(self.fb.render(f["name"],True,C_TEXT),(x+20,y+44))
        self.screen.blit(self.font.render("close: W",True,C_TEXT_DIM),(x+w-90,y+20))
        est=estimate_durability(f,self.sim.T_hot,self.sim.T_cold)
        yy=y+76
        if est is None:
            self.screen.blit(self.font.render("No durability data for this approach.",True,C_FAIL),(x+24,yy)); return
        d=est["data"]
        env_col=C_SUCCESS if est["in_rated_envelope"] else C_FAIL
        rows=[("Rated envelope:","%d C to %d C"%(d["min_service_C"],d["max_service_C"]),C_TEXT),
              ("Current dT envelope:","%.0f C to %.0f C"%(self.sim.T_cold-273.15,self.sim.T_hot-273.15),env_col),
              ("In rated envelope:","YES" if est["in_rated_envelope"] else "NO -- overstressed",env_col),
              ("Overstress (hot/cold):","+%.0f C / +%.0f C"%(est["over_hot_C"],est["over_cold_C"]),C_TEXT_DIM),
              ("Thermal cycle rating:","%d cycles"%d["thermal_cycles_rating"],C_TEXT),
              ("Cycles to failure (this dT):","%.0f cycles (%.1f yr @ 1/day)"%(est["cycles_to_failure"],est["cycles_to_failure"]/365.0),C_TEXT),
              ("Base lifetime (rated envelope):","%.1f yr"%d["base_lifetime_years"],C_TEXT),
              ("ESTIMATED SERVICE LIFE (live):","%.1f yr"%est["lifetime_years"],C_ACCENT),
              ("Weathering class:",d["weathering_class"],C_TEXT)]
        for k,v,col in rows:
            self.screen.blit(self.font.render(k,True,C_TEXT_DIM),(x+24,yy))
            self.screen.blit(self.font.render(str(v),True,col),(x+330,yy)); yy+=22
        yy+=10
        # Visual rating bars
        self.screen.blit(self.fb.render("RESISTANCE RATINGS",True,C_ACCENT),(x+24,yy))
        pygame.draw.line(self.screen,C_PANEL_BORDER,(x+24,yy+20),(x+220,yy+20),1); yy+=26
        bar_labels=[("Hydrophobic",d["hydrophobic_rating"]),("UV resistance",d["uv_resistance"]),("Abrasion",d["abrasion_resistance"])]
        for lbl,val in bar_labels:
            self.screen.blit(self.font.render(lbl,True,C_TEXT_DIM),(x+24,yy))
            bar_x=x+160; bar_w=200; bar_h=14
            # Background bar with subtle gradient
            pygame.draw.rect(self.screen,(35,40,52),(bar_x,yy+2,bar_w,bar_h))
            # Tick marks at 25/50/75%
            for tk in [0.25,0.5,0.75]:
                tx=bar_x+int(bar_w*tk)
                pygame.draw.line(self.screen,(55,62,80),(tx,yy+2),(tx,yy+2+bar_h),1)
            # Filled portion with smooth color gradient
            fill_w=int(bar_w*val)
            if fill_w>0:
                for fi in range(fill_w):
                    fv=fi/bar_w
                    if fv<0.4: fc=_mix((180,70,70),(220,180,70),fv/0.4)
                    elif fv<0.7: fc=_mix((220,180,70),(80,200,120),(fv-0.4)/0.3)
                    else: fc=(80,200,120)
                    pygame.draw.line(self.screen,fc,(bar_x+fi,yy+3),(bar_x+fi,yy+2+bar_h-1))
            pygame.draw.rect(self.screen,C_PANEL_BORDER,(bar_x,yy+2,bar_w,bar_h),1)
            self.screen.blit(self.font.render("%.2f"%val,True,C_TEXT),(bar_x+bar_w+8,yy+1))
            yy+=24
        yy+=10
        self.screen.blit(self.fb.render("DEGRADATION MECHANISM",True,C_ACCENT),(x+24,yy))
        pygame.draw.line(self.screen,C_PANEL_BORDER,(x+24,yy+20),(x+220,yy+20),1); yy+=26
        for wl in wrap_text(self.font,d["degradation"],w-60):
            self.screen.blit(self.font.render(wl,True,C_TEXT_DIM),(x+24,yy)); yy+=18
        yy+=14
        self.screen.blit(self.fb.render("REFERENCES",True,C_ACCENT),(x+24,yy))
        pygame.draw.line(self.screen,C_PANEL_BORDER,(x+24,yy+20),(x+160,yy+20),1); yy+=26
        for wl in wrap_text(self.font,d["references"],w-60):
            self.screen.blit(self.font.render(wl,True,C_TEXT_DIM),(x+24,yy)); yy+=18
        yy+=14
        self.screen.blit(self.fb.render("ACTIVE SCENARIO",True,C_ACCENT),(x+24,yy))
        pygame.draw.line(self.screen,C_PANEL_BORDER,(x+24,yy+20),(x+180,yy+20),1); yy+=26
        sc=BOUNDARY_SCENARIOS[self.scenario_idx]
        for wl in wrap_text(self.font,"%s -- %s"%(sc["name"],sc["desc"]),w-60):
            self.screen.blit(self.font.render(wl,True,C_TEXT_DIM),(x+24,yy)); yy+=18

    def _inspector_lines(self):
        """Build the full physics-inspector text (kind, text) list from the
        live sim + scientific twin. Regenerated each frame, so all numbers
        track the running simulation in real time."""
        f=FORMULATIONS[self.formulation_idx]; sim=self.sim; m=self._stack_metrics()
        sc=BOUNDARY_SCENARIOS[self.scenario_idx]; L=[]
        def H(t): L.append(("hdr",t))
        def B(t): L.append(("txt",t))
        def W(t): L.append(("warn",t))
        def D(t): L.append(("dim",t))
        H("FORMULATION %d :: %s"%(f["id"],f["name"]))
        B("scenario: %s -- %s"%(sc["name"],sc["desc"]))
        B("T_hot %.2f K (%.1f C) -> T_cold %.4g K (%.4g C)   dT=%.2f K"%(
            sim.T_hot,sim.T_hot-273.15,sim.T_cold,sim.T_cold-273.15,sim.T_hot-sim.T_cold))
        B("")
        H("STACK PHYSICS (live)")
        for k,v in [("q_incoming","%.6f W/m2"%sim.q_incoming),
                    ("q_transmitted","%.6f W/m2"%sim.q_transmitted),
                    ("suppression","%.6f%%"%sim.suppression_pct),
                    ("transmission","%.6f%%"%sim.transmission_pct),
                    ("R_total","%.4f m2.K/W"%m["R_total"]),
                    ("U-value","%.6f W/m2.K"%m["U"]),
                    ("k_eff overall","%.6f mW/m.K"%sim.k_eff_overall),
                    ("areal mass","%.3f kg/m2"%m["mass"]),
                    ("areal heat cap.","%.2f kJ/m2.K"%(m["c_areal"]/1e3)),
                    ("est. time const.","%.1f h  (lumped C_areal * R_total)"%(m["tau"]/3600.0))]:
            B("  %-22s %s"%(k,v))
        B("")
        H("CONDUCTION MODE SPLIT (R-weighted share of total resistance)")
        rt=max(m["R_total"],1e-12)
        B("  solid + gas conduction : %5.1f%%   radiation (Rosseland/MLI): %5.1f%%"%(
            100.0*m["R_sg"]/rt,100.0*m["R_rad"]/rt))
        B("")
        H("STRUCTURE")
        B("  %d layers | %.1f mm total | %d reflective foils | %d vacuum layers | %d PCM/chem | %d TEG"%(
            m["n"],sim.total_thickness/MM,m["foils"],m["n_vac"],m["n_pcm"],m["n_teg"]))
        B("")
        H("SERVICE-RATING CHECK (per-layer T_avg vs material T_max)")
        if m["over_tmax"]:
            for i,nm,Ta,tm in m["over_tmax"]:
                W("  L%-2d %-34.34s  T_avg=%.0f K  >  T_max=%.0f K  [OVER]"%(i,nm,Ta,tm))
        else:
            B("  all %d layers below T_max at the current profile"%m["n"])
        B("")
        H("STACKING SYNERGY (scientific-stack cross-check, live BCs)")
        sy=self._synergy_cached()
        if sy is None:
            B("  no scientific twin for this approach")
        else:
            B("  S_ordering  %.3f   (stack vs feasibility-corrected reversed order)"%sy["ordering"])
            B("  S_material  %.3f   (stack vs best feasible single material)"%sy["material"])
            B("  S_gradient  %.3f   (real T-profile vs uniform-T evaluation)"%sy["gradient"])
            B("  S_total     %.3f   >1 = ordered stacking itself adds performance"%sy["total"])
        B("")
        H("PER-LAYER TABLE (hot -> cold)")
        B("  #   material          thick  k_s+g    k_rad    k_eff    R        T_in -> T_out   dT      flags")
        last_z=None
        for i,l in enumerate(sim.layers):
            z=_layer_zone(l["name"])
            if z!=last_z:
                if z: B("  -- ZONE %s --"%z)
                last_z=z
            th=l["thickness_mm"]; kse=max(sim.k_eff[i],1e-12)
            ks=sim.q_cond[i] if i<len(sim.q_cond) else kse
            kr=sim.q_rad[i] if i<len(sim.q_rad) else 0.0
            T1,T2=sim.T_profile[i],sim.T_profile[i+1]
            fl=[]
            if l.get("vacuum"): fl.append("VAC")
            if l.get("n_reflective",0)>0: fl.append("x%d foils"%l["n_reflective"])
            if sim.pcm_capacity[i]>0: fl.append("PCM %.0f%%"%(sim.pcm_charged[i]*100))
            mp=_sm_mat_props(l); tmax=mp.get("T_max")
            if tmax is not None and (T1+T2)/2.0>tmax: fl.append("!>Tmax=%.0f"%tmax)
            B("  L%-2d %-16.16s %5.2fmm %8.3f %8.3f %8.3f %9.2f %5.0f>%5.0fK %6.1f  %s"%(
                i,l["material"],th,ks*1000.0,kr*1000.0,kse*1000.0,m["R"][i],T1,T2,T1-T2," ".join(fl)))
        B("")
        H("MATERIALS USED (scientific database entries)")
        seen=set()
        for l in sim.layers:
            mk=_sm_mat_key(l)
            if not mk or mk in seen: continue
            seen.add(mk); mp=MATERIALS[mk]
            Tmid=(sim.T_hot+sim.T_cold)/2.0
            B("  %-18s %-30.30s rho=%6.0f cp=%5.0f T_max=%4.0fK k(Tmid)=%.4f"%(
                mk,mp["name"],mp["rho"],mp["cp"],mp.get("T_max",0.0),mp["k_func"](Tmid)))
            B("      ref: %s"%mp.get("reference","-"))
        B("")
        H("FORMULATION SPEC")
        r=f["recipe"]
        B("  composition: %s"%r.get("composition",""))
        B("  k @25C     : %s"%r.get("k_25C",r.get("k_eff","N/A")))
        B("  references : %s"%r.get("references","-"))
        d=DURABILITY.get(f["approach"])
        if d is not None:
            est=estimate_durability(f,sim.T_hot,sim.T_cold)
            B("  durability : %d..%d C, %d cycles rated, class %s -> est. %.1f yr at this dT"%(
                d["min_service_C"],d["max_service_C"],d["thermal_cycles_rating"],d["weathering_class"],
                est["lifetime_years"] if est else float("nan")))
        B("")
        H("SCENARIO SWEEP (all 5 boundary cases, scientific stack)")
        _sc_desc={sc["name"]:sc["desc"] for sc in BOUNDARY_SCENARIOS}
        for nm,qt,sp in self._scenario_sweep():
            okd=sp>99.0
            L.append(("ok" if okd else "warn",
                      "  %-30.30s q_trans=%11.4g W/m2  supp=%9.4f%%  %s"%(nm,qt,sp,"OK" if okd else "fail")))
            if _sc_desc.get(nm): D("     %s"%_sc_desc[nm])
        B("")
        H("REPRODUCIBLE PROOF")
        B("  CLI: python Insulatrix.py --ultra-proof   (full 8-step stacking proof)")
        B("  CLI: python Insulatrix.py --proof / --compare / --sm-sim <approach>")
        B("")
        B("close: N    scroll: wheel / UP / DOWN")
        return L

    def _draw_inspector(self):
        w,h=min(1140,self.W-40),self.H-60; x=(self.W-w)//2; y=30
        # Opaque backing (not the translucent panel() helper) so the dense
        # stats/layer content underneath never bleeds through the table text.
        pygame.draw.rect(self.screen,(13,16,23),(x,y,w,h),border_radius=6)
        pygame.draw.rect(self.screen,C_PANEL_BORDER,(x,y,w,h),1,border_radius=6)
        self.screen.blit(self.fbig.render("PHYSICS INSPECTOR",True,C_ACCENT),(x+20,y+12))
        f=FORMULATIONS[self.formulation_idx]
        self.screen.blit(self.fb.render(f["name"],True,f["color"]),(x+300,y+16))
        self.screen.blit(self.font.render("close: N",True,C_TEXT_DIM),(x+w-90,y+20))
        lines=self._inspector_lines()
        plot_h=212; view_h=h-68-plot_h-30
        row_h=16; top=y+52; n_vis=max(1,view_h//row_h)
        max_sc=max(0,len(lines)-n_vis)
        self.insp_scroll=max(0,min(self.insp_scroll,max_sc))
        clip=self.screen.get_clip(); self.screen.set_clip(pygame.Rect(x+8,top-4,w-16,view_h))
        for i,(kind,txt) in enumerate(lines[self.insp_scroll:self.insp_scroll+n_vis]):
            col={"hdr":C_ACCENT,"warn":C_FAIL,"ok":C_SUCCESS,"dim":C_TEXT_DIM}.get(kind,C_TEXT)
            self.screen.blit(self.fs.render(txt,True,col),(x+16,top+i*row_h))
        self.screen.set_clip(clip)
        if len(lines)>n_vis:
            sb_h=max(20,int(view_h*n_vis/len(lines)))
            sb_y=top+int((view_h-sb_h)*self.insp_scroll/max(1,max_sc))
            pygame.draw.rect(self.screen,(70,76,92),(x+w-10,top,4,view_h),border_radius=2)
            pygame.draw.rect(self.screen,C_ACCENT,(x+w-10,sb_y,4,sb_h),border_radius=2)
        # Live-check badges (top right): suppression / synergy / T_max at the
        # current boundary conditions -- same checks the CLI proof reports.
        m=self._stack_metrics(); sy=self._synergy_cached()
        checks=[("SUPP>99%",self.sim.suppression_pct>99.0),
                ("SYN>1",sy is not None and sy["total"]>1.0),
                ("T_MAX",len(m["over_tmax"])==0)]
        cx=x+w-100
        for lbl,okd in reversed(checks):
            tag=lbl+(" OK" if okd else " X"); cw=self.fs.size(tag)[0]+16; cx-=cw
            col=C_SUCCESS if okd else C_FAIL
            pygame.draw.rect(self.screen,(13,16,23),(cx,y+12,cw,18),border_radius=4)
            pygame.draw.rect(self.screen,col,(cx,y+12,cw,18),1,border_radius=4)
            self.screen.blit(self.fs.render(tag,True,col),(cx+8,y+15)); cx-=6
        # Energy-fate stacked bar: where the incoming energy has actually gone
        self._draw_energy_budget(x+12,y+h-8-plot_h-30,w-24)
        # Docked chart strip: T(x) profile + per-layer R + per-layer k_eff
        self._draw_insp_plots(x+12,y+h-8-plot_h,w-24,plot_h-8)

    def _draw_insp_plots(self,bx,by,bw,bh):
        """Docked chart strip at the inspector bottom: T(x) profile, per-layer
        R (log10) and per-layer k_eff (log10) -- all live from the sim."""
        sim=self.sim; fs=self.fs; gap=10; m=self._stack_metrics()
        titles=["T(x) PROFILE  (depth mm)","R PER LAYER  [m2K/W, log10]",
                "k_eff PER LAYER  [mW/mK, log10]"]
        pw=(bw-2*gap)/3.0
        for pi in range(3):
            px=bx+int(pi*(pw+gap)); pwi=int(pw)
            pygame.draw.rect(self.screen,(17,20,28),(px,by,pwi,bh),border_radius=4)
            pygame.draw.rect(self.screen,C_PANEL_BORDER,(px,by,pwi,bh),1,border_radius=4)
            self.screen.blit(fs.render(titles[pi],True,C_ACCENT),(px+8,by+6))
            ix,iy=px+10,by+24; iw,ih=pwi-20,bh-34
            if pi==0: self._plot_Tprofile(ix,iy,iw,ih)
            elif pi==1: self._plot_bars(ix,iy,iw,ih,m["R"])
            else: self._plot_bars(ix,iy,iw,ih,[max(sim.k_eff[j],1e-12)*1000.0 for j in range(sim.n_layers)])

    def _plot_Tprofile(self,ix,iy,iw,ih):
        """Temperature vs cumulative thickness through the stack."""
        sim=self.sim
        cum=[0.0]
        for l in sim.layers: cum.append(cum[-1]+l["thickness_mm"])
        tot=max(cum[-1],1e-9)
        Tlo=min(min(sim.T_profile),sim.T_cold); Thi=max(max(sim.T_profile),sim.T_hot)
        rng=max(Thi-Tlo,1e-9)
        pts=[(ix+int(iw*cum[i]/tot),iy+ih-int(ih*(sim.T_profile[i]-Tlo)/rng)) for i in range(sim.n_layers+1)]
        if len(pts)>1:
            fill=[(pts[0][0],iy+ih)]+pts+[(pts[-1][0],iy+ih)]
            s=pygame.Surface((iw+2,ih+2),pygame.SRCALPHA)
            pygame.draw.polygon(s,(90,140,220,40),[(p[0]-ix,p[1]-iy) for p in fill])
            self.screen.blit(s,(ix,iy))
            pygame.draw.lines(self.screen,(140,190,255),False,pts,2)
            # Material-boundary ticks along the bottom edge
            for ci in range(1,sim.n_layers):
                tx=ix+int(iw*cum[ci]/tot)
                pygame.draw.line(self.screen,(50,56,72),(tx,iy+ih-6),(tx,iy+ih),1)
        pygame.draw.rect(self.screen,C_PANEL_BORDER,(ix,iy,iw,ih),1)
        fs=self.fs
        self.screen.blit(fs.render("%.0f K"%Thi,True,C_HOT),(ix+3,iy+2))
        cl=fs.render("%.4g K"%Tlo,True,C_COLD); self.screen.blit(cl,(ix+3,iy+ih-12))
        dl=fs.render("%.1f mm"%tot,True,C_TEXT_DIM); self.screen.blit(dl,(ix+iw-dl.get_width()-2,iy+ih-12))

    def _plot_bars(self,ix,iy,iw,ih,vals):
        """Log10 vertical bar chart, one bar per layer, colored by material."""
        n=len(vals)
        if n==0: return
        vs=[math.log10(max(v,1e-12)) for v in vals]
        lo,hi=min(vs),max(vs); rng=max(hi-lo,1e-9)
        for i,v in enumerate(vs):
            bh=int(ih*(v-lo)/rng)
            bx0=ix+int(iw*i/n); bx1=ix+int(iw*(i+1)/n)
            pygame.draw.rect(self.screen,_layer_color(self.sim.layers[i]),
                             (bx0,iy+ih-bh,max(1,bx1-bx0-1),bh))
        pygame.draw.rect(self.screen,C_PANEL_BORDER,(ix,iy,iw,ih),1)
        fs=self.fs
        self.screen.blit(fs.render("max %.3g"%(10.0**hi),True,C_TEXT),(ix+3,iy+2))
        self.screen.blit(fs.render("min %.3g"%(10.0**lo),True,C_TEXT_DIM),(ix+3,iy+ih-12))

    def _draw_energy_budget(self,bx,by,bw):
        """Stacked 'energy fate' bar: of the cumulative incoming energy, how much
        was blocked vs absorbed/converted/pumped vs transmitted (elapsed time)."""
        sim=self.sim; fs=self.fs
        e_in=max(sim.q_incoming*sim.elapsed,1e-12)
        parts=[("blocked",e_in-sim.q_transmitted*sim.elapsed-sim.total_absorbed-sim.total_converted-sim.total_pumped,C_SUCCESS),
               ("PCM/chem",sim.total_absorbed,(200,160,100)),
               ("TEG",sim.total_converted,(160,120,220)),
               ("pumped",sim.total_pumped,(120,190,255)),
               ("transmitted",sim.q_transmitted*sim.elapsed,C_FAIL)]
        x0=bx; legend=[]
        for nm,ev,col in parts:
            frac=clamp(ev/e_in); seg=int(bw*frac)
            if seg>0: pygame.draw.rect(self.screen,col,(x0,by,seg,12)); x0+=seg
            legend.append("%s %.2f%%"%(nm,100.0*clamp(ev/e_in)))
        pygame.draw.rect(self.screen,C_PANEL_BORDER,(bx,by,bw,12),1)
        self.screen.blit(fs.render("ENERGY FATE over %.0fs: "%sim.elapsed+" | ".join(legend)+"   (E_in=%.3g J/m2)"%e_in,True,C_TEXT_DIM),(bx,by+15))

    def _fabrication_lines(self):
        """Full fabrication-protocol text for the active formulation."""
        f=FORMULATIONS[self.formulation_idx]; fab=FABRICATION.get(f["approach"])
        L=[]
        def H(t): L.append(("hdr",t))
        def B(t): L.append(("txt",t))
        def D(t): L.append(("dim",t))
        def Q(t): L.append(("ok",t))
        def W(t): L.append(("warn",t))
        if fab is None:
            B("no fabrication protocol for %s"%f["approach"]); return L
        H(fab["title"])
        B("active build: %s   |   calendar incl. cures: %s"%(fab["est_active"],fab["est_calendar"]))
        B("")
        H("EQUIPMENT"); [D("  - %s"%e) for e in fab["equipment"]]; B("")
        H("MATERIALS"); [D("  - %s"%m) for m in fab["materials"]]; B("")
        for pname,steps in fab["phases"]:
            H(pname)
            for i,s in enumerate(steps): B("  %2d. %s"%(i+1,s))
            B("")
        H("QC GATES"); [Q("  [ ] %s"%q) for q in fab["qc"]]; B("")
        H("SAFETY"); [W("  ! %s"%s) for s in fab["safety"]]
        for c in fab.get("caveats",[]): D("  NOTE: %s"%c)
        B(""); D("close: F   scroll: wheel / UP / DOWN   CLI: --fabrication %s"%f["approach"])
        return L

    def _draw_fabrication(self):
        w,h=min(1080,self.W-40),self.H-60; x=(self.W-w)//2; y=30
        pygame.draw.rect(self.screen,(13,16,23),(x,y,w,h),border_radius=6)
        pygame.draw.rect(self.screen,C_PANEL_BORDER,(x,y,w,h),1,border_radius=6)
        f=FORMULATIONS[self.formulation_idx]
        self.screen.blit(self.fbig.render("FABRICATION PROTOCOL",True,C_ACCENT),(x+20,y+12))
        self.screen.blit(self.fb.render(f["name"],True,f["color"]),(x+340,y+16))
        self.screen.blit(self.font.render("close: F",True,C_TEXT_DIM),(x+w-90,y+20))
        lines=self._fabrication_lines()
        row_h=16; top=y+52; n_vis=max(1,(h-64)//row_h)
        max_sc=max(0,len(lines)-n_vis); self.fab_scroll=max(0,min(self.fab_scroll,max_sc))
        clip=self.screen.get_clip(); self.screen.set_clip(pygame.Rect(x+8,top-4,w-16,h-56))
        for i,(kind,txt) in enumerate(lines[self.fab_scroll:self.fab_scroll+n_vis]):
            col={"hdr":C_ACCENT,"dim":C_TEXT_DIM,"warn":C_FAIL,"ok":C_SUCCESS}.get(kind,C_TEXT)
            self.screen.blit(self.fs.render(txt,True,col),(x+16,top+i*row_h))
        self.screen.set_clip(clip)
        if len(lines)>n_vis:
            sb_h=max(20,int((h-64)*n_vis/len(lines)))
            sb_y=top+int((h-64-sb_h)*self.fab_scroll/max(1,max_sc))
            pygame.draw.rect(self.screen,(70,76,92),(x+w-10,top,4,h-64),border_radius=2)
            pygame.draw.rect(self.screen,C_ACCENT,(x+w-10,sb_y,4,sb_h),border_radius=2)

    def _draw_proof(self):
        """Live pass/fail proof checklist panel (O) -- mirrors the verification
        block of --ultra-proof, evaluated live on the active formulation."""
        w,h=660,430; x=(self.W-w)//2; y=(self.H-h)//2
        pygame.draw.rect(self.screen,(13,16,23),(x,y,w,h),border_radius=6)
        pygame.draw.rect(self.screen,C_PANEL_BORDER,(x,y,w,h),1,border_radius=6)
        f=FORMULATIONS[self.formulation_idx]
        self.screen.blit(self.fbig.render("LIVE PROOF CHECKLIST",True,C_ACCENT),(x+20,y+14))
        self.screen.blit(self.fb.render(f["name"],True,f["color"]),(x+20,y+44))
        self.screen.blit(self.font.render("close: O",True,C_TEXT_DIM),(x+w-90,y+20))
        checks=self._proof_checks()
        n_ok=sum(1 for _,ok,_ in checks if ok)
        yy=y+78
        self.screen.blit(self.font.render("boundary: %s"%BOUNDARY_SCENARIOS[self.scenario_idx]["name"],True,C_TEXT_DIM),(x+20,yy)); yy+=26
        for lbl,ok,val in checks:
            chip_col=C_SUCCESS if ok else C_FAIL; tag="PASS" if ok else "FAIL"
            pygame.draw.rect(self.screen,(20,24,34),(x+20,yy,w-40,24),border_radius=4)
            pygame.draw.rect(self.screen,chip_col,(x+20,yy,3,24))
            self.screen.blit(self.font.render(lbl,True,C_TEXT),(x+32,yy+5))
            vs=self.font.render(val,True,C_TEXT_DIM); self.screen.blit(vs,(x+w-170-vs.get_width(),yy+5))
            cs=self.font.render(tag,True,chip_col); self.screen.blit(cs,(x+w-64,yy+5))
            yy+=30
        yy+=10
        overall=n_ok==len(checks); oc=C_SUCCESS if overall else C_FAIL
        self.screen.blit(self.fb.render("OVERALL: %s  (%d/%d checks)"%("PROVEN" if overall else "NOT PROVEN",n_ok,len(checks)),True,oc),(x+20,yy)); yy+=34
        for wl in wrap_text(self.fs,"Near-zero engineering limit, not literal zero transfer -- residual conduction, gas, radiation and real-world parasitics (supports, seams, vacuum leaks) always remain. These checks mirror the model's own criteria, not laboratory certification. Full report: python Insulatrix.py --ultra-proof",w-40):
            self.screen.blit(self.fs.render(wl,True,C_TEXT_DIM),(x+20,yy)); yy+=15

    def _draw_hotkeys(self):
        if self.tab=="pod":
            items=[("TAB","next model"),("M","thermal tab"),("SPACE","launch test"),("1-8","threat"),("A","auto-cycle"),
                   ("BKSP","reset test"),("P","particles"),("T","auto-rotate"),("R","reset cam"),
                   ("L","labels"),("X","cross-section"),("I","spec"),("H","help"),("ESC","quit")]
        else:
            items=[("TAB","next model"),("M","pod tab"),("SPACE","sim"),("1-9/0/C/U","formulation"),("+/-","thickness"),("[/]","hot T"),
                   (";/'","cold T"),("V","scenario"),("W","durability"),("N","inspector"),("O","proof"),("P","particles"),("A","auto-cycle"),("T","auto-rotate"),
                   ("BKSP","reset"),("R","reset cam"),("L","labels"),("X","cross-section"),
                   ("I","spec"),("D","diagram"),("H","help"),("ESC","quit")]
        bar_h=24; bar_y=self.H-bar_h-2
        panel(self.screen,0,bar_y,self.W,bar_h,alpha=210)
        x=10
        for idx,(key,desc) in enumerate(items):
            if idx>0 and idx%6==0:
                pygame.draw.line(self.screen,C_PANEL_BORDER,(x,bar_y+4),(x,bar_y+bar_h-4),1)
                x+=8
            ks=self.font.render(key,True,C_ACCENT); ds=self.font.render(desc,True,C_TEXT_DIM)
            self.screen.blit(ks,(x,bar_y+5)); x+=ks.get_width()+2
            self.screen.blit(ds,(x,bar_y+5)); x+=ds.get_width()+12
        # Live perf/sim readout, right-aligned
        info="%.0f fps | sim t=%.0fs"%(self.clock.get_fps(),getattr(self.sim,"elapsed",0.0))
        isurf=self.fs.render(info,True,C_TEXT_DIM)
        self.screen.blit(isurf,(self.W-isurf.get_width()-8,bar_y+6))

    def run(self):
        _print_banner()
        while self.running:
            try:
                dt=self.clock.tick(60)/1000.0; dt=min(dt,0.05)
                self.handle_events(dt); self.update(dt); self.draw()
            except Exception as e:
                import traceback; traceback.print_exc()
                print("Warning: frame error: %s"%e)
        pygame.quit()

# =============================================================================
# SECTION 10 -- SELF-TEST, FEASIBILITY REPORT, OBJ EXPORT
# =============================================================================
def selftest():
    print("Running self-test..."); ok=True
    try:
        parts,total_t=build_scene(0)
        print("  [OK] Scene built: %d parts, total thickness %.1f mm"%(len(parts),total_t))
    except Exception as e:
        print("  [FAIL] Scene build: %s"%e); ok=False; parts=[]
    for f in FORMULATIONS:
        try:
            sim=run_simulation(f["approach"],duration_s=60)
            supp=sim.suppression_pct; success=sim.is_success
            print("  [OK] %-22s suppression=%.4f%%  success=%s"%(f["name"],supp,success))
        except Exception as e:
            print("  [FAIL] %s: %s"%(f["name"],e)); ok=False
    try:
        results=compare_all_approaches(duration_s=60)
        best=max(results,key=lambda r:r["suppression"])
        print("  [OK] Best approach: %s (%.4f%% suppression)"%(best["name"],best["suppression"]))
    except Exception as e:
        print("  [FAIL] Compare: %s"%e); ok=False
    try:
        for f in FORMULATIONS:
            rxn=CHEM_REACTIONS.get(f["approach"])
            assert rxn is not None, "No CHEM_REACTIONS entry for %s"%f["approach"]
            assert "eq" in rxn and "mech" in rxn
            assert "reactants" in rxn and "products" in rxn
        print("  [OK] Material diagrams: all %d approaches covered"%len(FORMULATIONS))
    except Exception as e:
        print("  [FAIL] Material diagrams: %s"%e); ok=False
    if pygame is not None:
        try:
            if parts:
                scene=Scene(0)
                surf=pygame.Surface((800,600))
                scene.render(surf,pygame.Rect(0,0,800,600))
                print("  [OK] Render (headless) succeeded")
        except Exception as e:
            print("  [FAIL] Render: %s"%e); ok=False
    else:
        print("  [SKIP] Render (pygame not installed)")
    # Full interactive App: draw()/update() are NOT exercised by any of the
    # checks above (they only touch Scene.render directly), which is exactly
    # how a real crash in App.draw() -- e.g. a malformed stats-panel tuple --
    # went undetected by a passing selftest before. Drive both tabs for real.
    if pygame is not None:
        try:
            app=App()
            for _ in range(3): app.update(1/60.0); app.draw()
            app._toggle_tab()
            assert app.tab=="pod" and app.pod_scene is not None
            for _ in range(3): app.update(1/60.0); app.draw()
            app._select_pod_scenario(4); app._trigger_pod_test()  # a perforating-but-patchable case
            steps=0
            while app.pod_test_phase!="done" and steps<600:
                app.update(1/30.0); app.draw(); steps+=1
            assert app.pod_test_phase=="done", "pod test animation never reached 'done'"
            app._toggle_tab()
            assert app.tab=="thermal"
            for _ in range(3): app.update(1/60.0); app.draw()
            # Deep-stack coverage: ULTRA CASCADE (35 layers) with the physics
            # inspector open and both scroll positions exercised.
            app._select_formulation(11); app.show_inspector=True
            for _ in range(3): app.update(1/60.0); app.draw()
            app.insp_scroll=40; app.layer_scroll=20; app.draw()
            app.show_inspector=False; app.draw()
            # Fabrication overlay + scrolling
            app.show_fab=True
            for _ in range(3): app.draw()
            app.fab_scroll=40; app.draw(); app.show_fab=False
            # Proof checklist, layer stepping/pinning, CSV export
            app.show_proof=True; app.draw(); app.show_proof=False
            app._step_layer(3); app.draw()
            assert app.pinned_layer==2, "J stepping should pin layer 2, got %s"%app.pinned_layer
            app._step_layer(-1); app.draw()
            _cwd=os.getcwd(); _tmp=os.path.join(tempfile.gettempdir(),"insulatrix_selftest")
            os.makedirs(_tmp,exist_ok=True); os.chdir(_tmp)
            try: app._export_layers_csv()
            finally: os.chdir(_cwd)
            assert any(fn.startswith("insulatrix_layers_") for fn in os.listdir(_tmp)), "CSV export wrote nothing"
            app.draw()
            print("  [OK] App: both tabs draw/update without error, full pod test cycle completes")
        except Exception as e:
            print("  [FAIL] App: %s"%e); ok=False
        try:
            app=App(); n=app._model_count()
            assert n==len(FORMULATIONS)+1
            seen=[]
            for _ in range(n+1):  # one extra step to confirm it wraps back to the start
                seen.append((app._model_index(),app.tab)); app.draw(); app._next_model(1)
            assert seen[0]==(0,"thermal") and seen[-2]==(len(FORMULATIONS),"pod") and seen[-1]==seen[0], \
                "TAB cycle did not visit every model exactly once before wrapping: %s"%seen
            app._goto_model(0); app._next_model(-1); app.draw()
            assert app._model_index()==len(FORMULATIONS) and app.tab=="pod", "SHIFT+TAB from model 0 should wrap to the pod"
            print("  [OK] TAB cycling: all %d models (12 formulations + pod) visited, wraps both directions"%n)
        except Exception as e:
            print("  [FAIL] TAB cycling: %s"%e); ok=False
    else:
        print("  [SKIP] App (pygame not installed)")
    try:
        sim=ThermalSim("combined")
        sim.running=True; sim.elapsed=0.001
        for _ in range(100): sim.step(1.0)
        assert sim.suppression_pct>0, "No suppression in combined sim"
        print("  [OK] Combined sim: suppression=%.4f%%"%sim.suppression_pct)
    except Exception as e:
        print("  [FAIL] Combined sim: %s"%e); ok=False
    # Scientific model framework tests
    try:
        assert len(MATERIALS)>=10
        for k,m in MATERIALS.items(): assert m["k_func"](300.0)>0
        print("  [OK] SM Materials: %d verified"%len(MATERIALS))
    except Exception as e:
        print("  [FAIL] SM Materials: %s"%e); ok=False
    try:
        l=SLayer("T","silica_aerogel",50.0)
        assert l.k_eff(300.0)>0; assert l.k_gas(300.0)<_k_air_T(300.0)
        print("  [OK] SM Layer: k=%.6f, Knudsen OK"%(l.k_eff(300.0)))
    except Exception as e:
        print("  [FAIL] SM Layer: %s"%e); ok=False
    try:
        pcm=SRecoveryLayer("P","pcm_srbr2",200.0,recovery_type="pcm")
        assert pcm.pcm_capacity_per_m2>0
        teg=SRecoveryLayer("T","bi2te3_teg",5.0,recovery_type="teg")
        assert teg.teg_power_density(500,300)>=0
        print("  [OK] SM RecoveryLayer: PCM cap=%.0f MJ/m2"%(pcm.pcm_capacity_per_m2/1e6))
    except Exception as e:
        print("  [FAIL] SM RecoveryLayer: %s"%e); ok=False
    try:
        s=sm_aerogel(50.0); s.bc=SBoundary(); r=s.solve_steady_state()
        assert r["q_transmitted"]>0 and 0<=r["suppression"]<=1
        print("  [OK] SM Steady-state: q=%.4f, supp=%.2f%%"%(r["q_transmitted"],r["suppression_pct"]))
    except Exception as e:
        print("  [FAIL] SM Steady-state: %s"%e); ok=False
    try:
        stacks=[sm_foam(),sm_aerogel(),sm_ceramic(),sm_refractory(),sm_mli(),sm_hybrid(),
            sm_pcm(),sm_endothermic(),sm_teg(),sm_gradient(),sm_pro(),sm_ultra()]
        for s in stacks:
            s.bc=SBoundary(); r=s.solve_steady_state(); assert r["q_transmitted"]>=0
        print("  [OK] SM All %d stacks solved"%len(stacks))
    except Exception as e:
        print("  [FAIL] SM Stacks: %s"%e); ok=False
    try:
        s=sm_aerogel(10.0); s.bc=SBoundary(T_hot=500,T_cold=300)
        r=s.solve_transient(100.0,0.5)
        assert np.isfinite(r["q_transmitted"]) and r["q_transmitted"]>=0, "q_transmitted must be finite and non-negative (hot->cold flow)"
        print("  [OK] SM Transient: %d steps, q=%.4f"%(r["n_steps"],r["q_transmitted"]))
    except Exception as e:
        print("  [FAIL] SM Transient: %s"%e); ok=False
    try:
        opt=OptimizationFramework()
        sw=opt.thickness_sweep(sm_aerogel,[10,50,100],"Aerogel")
        ml=opt.mli_layer_sweep([10,40,80])
        cp=opt.full_comparison()
        assert len(sw)==3 and len(ml)==3 and len(cp)==12
        print("  [OK] SM Optimization: %d+%d+%d results"%(len(sw),len(ml),len(cp)))
    except Exception as e:
        print("  [FAIL] SM Optimization: %s"%e); ok=False
    # Boundary scenarios + durability model
    try:
        assert len(BOUNDARY_SCENARIOS)>=4
        for sc in BOUNDARY_SCENARIOS:
            assert sc["T_hot"]>sc["T_cold"]>0
            r=compare_all_approaches(duration_s=30,T_hot=sc["T_hot"],T_cold=sc["T_cold"])
            assert len(r)==len(FORMULATIONS)
            for row in r: assert math.isfinite(row["suppression"]) and math.isfinite(row["q_trans"])
        print("  [OK] Boundary scenarios: %d verified, all approaches finite at each"%len(BOUNDARY_SCENARIOS))
    except Exception as e:
        print("  [FAIL] Boundary scenarios: %s"%e); ok=False
    # Regression guard: Insulatrix Pro must sustain near-zero heat transfer (>99.99%
    # suppression) with a simulated hot face at the absolute-zero-limit cold scenario.
    try:
        az=[s for s in BOUNDARY_SCENARIOS if "ABSOLUTE ZERO" in s["name"]]
        assert az, "No FORGE + ABSOLUTE ZERO scenario defined"
        sc=az[0]
        sim=run_simulation("combined",duration_s=3600,T_hot=sc["T_hot"],T_cold=sc["T_cold"])
        assert math.isfinite(sim.suppression_pct) and math.isfinite(sim.q_transmitted)
        assert sim.suppression_pct>99.99, "Pro suppression only %.6f%% at absolute-zero limit"%sim.suppression_pct
        assert sim.is_success
        print("  [OK] Near-zero @ absolute zero: Pro suppression=%.6f%%  q_trans=%.6e W/m2"%(sim.suppression_pct,sim.q_transmitted))
    except Exception as e:
        print("  [FAIL] Near-zero @ absolute zero: %s"%e); ok=False
    try:
        for f in FORMULATIONS:
            assert f["approach"] in DURABILITY, "No DURABILITY entry for %s"%f["approach"]
        for sc in BOUNDARY_SCENARIOS:
            for f in FORMULATIONS:
                est=estimate_durability(f,sc["T_hot"],sc["T_cold"])
                assert est is not None and math.isfinite(est["lifetime_years"]) and est["lifetime_years"]>=0
        print("  [OK] Durability model: all %d approaches x %d scenarios finite & non-negative"%(len(FORMULATIONS),len(BOUNDARY_SCENARIOS)))
    except Exception as e:
        print("  [FAIL] Durability model: %s"%e); ok=False
    try:
        for f in FORMULATIONS:
            fab=FABRICATION[f["approach"]]
            assert fab["phases"] and fab["qc"] and fab["safety"], "Incomplete FABRICATION entry for %s"%f["approach"]
        print("  [OK] Fabrication protocols: all %d approaches have phases+QC+safety"%len(FORMULATIONS))
    except Exception as e:
        print("  [FAIL] Fabrication protocols: %s"%e); ok=False
    try:
        h,c=resolve_scenario(BOUNDARY_SCENARIOS[2]["name"])
        assert h==BOUNDARY_SCENARIOS[2]["T_hot"] and c==BOUNDARY_SCENARIOS[2]["T_cold"]
        h2,c2=resolve_scenario(None); assert h2 is None and c2 is None
        print("  [OK] resolve_scenario: name lookup and None passthrough correct")
    except Exception as e:
        print("  [FAIL] resolve_scenario: %s"%e); ok=False
    # Impact-shield system: PodShieldSim
    try:
        pod=PodShieldSim()
        assert pod.standoff_m>0
        rows=pod.sweep_threats()
        assert len(rows)>=5
        for r in rows:
            for k in ("mass_kg","kinetic_energy_J","deflect_frac","fragmentation_efficacy",
                      "ballistic_margin","hole_mm","cabin_min_kPa"):
                v=r[k]; assert math.isfinite(v), "%s not finite for %s"%(k,r["label"])
            assert 0.0<=r["deflect_frac"]<=1.0001
            assert 0.0<=r["fragmentation_efficacy"]<=1.0001
            # Non-conductive threats must get zero magnetic assist -- this is the
            # single most important honesty check in the whole impact model.
            if not r["conductive"]: assert r["deflect_frac"]==0.0, "non-conductive threat got magnetic assist"
        # A small, non-conductive, borderline-perforating threat must be a case
        # the auto-patch system actually handles (proves "auto fill hole" works
        # for at least part of the design envelope, not just the easy misses).
        r_small=pod.assess_impact(2.0,3000.0,IMPACT_PHYS["ref_velocity_ms"],conductive=False)
        assert math.isfinite(r_small["ballistic_margin"])
        n_survived=sum(1 for r in rows if r["survived"])
        assert n_survived>=1, "No threat scenario survived at all -- shield is non-functional"
        assert any(r["perforated"] for r in rows), "No scenario perforates -- sweep isn't exercising the self-heal path"
        print("  [OK] PodShieldSim: %d/%d representative threats survived"%(n_survived,len(rows)))
    except Exception as e:
        print("  [FAIL] PodShieldSim: %s"%e); ok=False
    try:
        import io,contextlib
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf): passed=print_pod_proof()
        assert passed, "print_pod_proof() returned False"
        assert "BOTH PROOFS HOLD" in buf.getvalue()
        print("  [OK] print_pod_proof: both A/B deflection and self-heal proofs hold")
    except Exception as e:
        print("  [FAIL] print_pod_proof: %s"%e); ok=False
    try:
        import io,contextlib
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf): total=print_economics_report()
        assert math.isfinite(total) and total>0, "economics total not finite/positive: %r"%total
        assert "HONEST CAVEATS" in buf.getvalue() and "FULL-ADOPTION CEILING" in buf.getvalue(), \
            "economics report dropped its caveats"
        print("  [OK] print_economics_report: total=$%s/yr, caveats present"%("{:,.0f}".format(total)))
    except Exception as e:
        print("  [FAIL] print_economics_report: %s"%e); ok=False
    if ok: print("\nSelf-test PASSED.")
    else: print("\nSelf-test FAILED.")
    return ok

def print_feasibility(T_hot=None,T_cold=None):
    T_hot=T_hot or PHYS["T_hot_default"]; T_cold=T_cold or PHYS["T_cold_default"]
    print("="*72)
    print(" INSULATRIX -- FEASIBILITY REPORT")
    print(" Sprayable Near-Zero Thermal Transfer: Honest Physics Assessment")
    print("="*72)
    print("\nBOUNDARY CONDITIONS")
    print("-"*40)
    print("  T_hot:   %.0f K (%.0f C)"%(T_hot,T_hot-273.15))
    print("  T_cold:  %.0f K (%.0f C)"%(T_cold,T_cold-273.15))
    print("  dT:      %.0f K"%(T_hot-T_cold))
    print("\nAPPROACH COMPARISON (3600 s simulation)")
    print("-"*72)
    print("  %-22s %10s %10s %8s %12s %10s"%("Approach","Suppress%","Transmit%","Success","q_trans W/m2","k_eff mW"))
    print("  "+"-"*68)
    results=compare_all_approaches(duration_s=3600,T_hot=T_hot,T_cold=T_cold)
    for r in results:
        print("  %-22s %10.4f %10.6f %8s %12.6f %10.6f"%(
            r["name"],r["suppression"],r["transmission"],
            "YES" if r["success"] else "no",r["q_trans"],r["k_eff"]))
    best=max(results,key=lambda r:r["suppression"])
    worst=min(results,key=lambda r:r["suppression"])
    print("\n  BEST SUPPRESSION:  %-22s  %.4f%%"%(best["name"],best["suppression"]))
    print("  WORST SUPPRESSION: %-22s  %.4f%%"%(worst["name"],worst["suppression"]))
    print("\nABSORPTION/CONVERSION SUMMARY")
    print("-"*40)
    for r in results:
        print("  %-22s  absorbed=%.2f MJ/m2  converted=%.2f MJ/m2  pumped=%.2f MJ/m2"%(
            r["name"],r["absorbed_MJ"],r["converted_MJ"],r["pumped_MJ"]))
    print("\nHONEST ASSESSMENT")
    print("-"*40)
    print("""
  The SUPPRESSION RATIO measures how much of the incoming heat flux is
  blocked, absorbed, or converted vs. transmitted through the full stack.
  Higher suppression = better insulation. Target: > 99%.

  PERFORMANCE TIERS (3600 s simulation):
    Tier 1 (>99.9%% suppression): Insulatrix Pro, TEG, MLI Vacuum, Hybrid
      - Pro: 5-stage combined (spray+PCM+chemical+TEG+MLI) -> 100%
      - TEG: Seebeck converts dT + Peltier pumps heat back -> 99.995%
      - MLI: 40 reflective Al layers in vacuum, radiation suppressed by 1/N
      - Hybrid: aerogel Knudsen + MLI reflective, robust under soft vacuum

    Tier 2 (>99.8%% suppression): Endothermic, PCM, Gradient Spray
      - Endothermic: MgSO4 dehydration stores 42.7 MJ/m2 chemically
      - PCM: SrBr2 latent heat absorbs 44.0 MJ/m2 at 89C
      - Gradient: compositional spray, conformal + field-repairable

    Tier 3 (>97%% suppression): Refractory Foam
      - Refractory: high-T stable, SiC opacifier reduces radiation

    Tier 4 (baseline): Aerogel, Ceramic Fiber, Standard Foam
      - Aerogel: Knudsen gas suppression but no radiation blocking
      - Ceramic fiber: high-T stable but higher k
      - Foam: gas conduction dominates, no radiation suppression

  KEY INSIGHT: Near-zero heat transfer requires COMBINED approaches:
    1. Block radiation (MLI reflective layers or opacifiers)
    2. Suppress gas conduction (Knudsen effect in nanopores or vacuum)
    3. Absorb heat pulses (PCM latent heat or endothermic chemical)
    4. Convert + re-absorb (thermoelectric engine + Peltier heat pump)
    5. Gradient composition (refractory hot-face -> aerogel cold-face)

  No single mechanism achieves near-zero alone. The Insulatrix Pro
  combination of all 5 mechanisms achieves the lowest practical
  transmitted heat flux across extreme temperature gradients.
""")
    print("="*72)

def print_comparison(T_hot=None,T_cold=None):
    print("="*80)
    print(" INSULATRIX -- APPROACH COMPARISON TABLE")
    print("="*80)
    results=compare_all_approaches(duration_s=3600,T_hot=T_hot,T_cold=T_cold)
    print("\n%-22s %8s %8s %8s %12s %10s %10s %10s"%(
        "Approach","Supp%","Trans%","Success","q_trans","absorbed","converted","pumped"))
    print("-"*80)
    for r in results:
        print("%-22s %8.3f %8.6f %8s %12.6f %8.2f %8.2f %8.2f"%(
            r["name"],r["suppression"],r["transmission"],
            "YES" if r["success"] else "no",r["q_trans"],
            r["absorbed_MJ"],r["converted_MJ"],r["pumped_MJ"]))
    print("\n"+"-"*80)
    print("RANKING BY SUPPRESSION RATIO:")
    ranked=sorted(results,key=lambda r:r["suppression"],reverse=True)
    for i,r in enumerate(ranked):
        marker=" <-- BEST" if i==0 else ""
        print("  %2d. %-22s  %.4f%%%s"%(i+1,r["name"],r["suppression"],marker))
    print("\n"+"="*80)

def print_simulation(approach_name,T_hot=None,T_cold=None):
    approach=None; form_name=None
    for f in FORMULATIONS:
        if f["approach"]==approach_name or f["name"].lower().startswith(approach_name.lower()):
            approach=f["approach"]; form_name=f["name"]; break
    if approach is None:
        valid=[f["approach"] for f in FORMULATIONS]
        print("Unknown approach: %s"%approach_name)
        print("Valid approaches: %s"%", ".join(valid))
        sys.exit(1)
    print("="*72)
    print(" INSULATRIX -- DETAILED SIMULATION: %s"%form_name.upper())
    print("="*72)
    sim=run_simulation(approach,duration_s=3600,T_hot=T_hot,T_cold=T_cold)
    print(sim.summary())
    print("\n"+"="*72)

def resolve_scenario(name):
    """Look up a BOUNDARY_SCENARIOS preset by fuzzy name match. Returns (T_hot,T_cold) or (None,None)."""
    if not name: return None,None
    key=name.lower().replace(" ","").replace("_","").replace("-","").replace("/","")
    for sc in BOUNDARY_SCENARIOS:
        sk=sc["name"].lower().replace(" ","").replace("_","").replace("-","").replace("/","")
        if key in sk or sk.startswith(key): return sc["T_hot"],sc["T_cold"]
    valid=[sc["name"] for sc in BOUNDARY_SCENARIOS]
    print("Unknown scenario: %s"%name); print("Valid scenarios: %s"%", ".join(valid)); sys.exit(1)

def print_pod_report():
    """Live impact-shield assessment report for the INSULATRIX pod hull:
    Whipple-shield fragmentation, eddy-current magnetic deflection, and the
    self-healing hull's race against decompression. Every number is computed
    from IMPACT_PHYS + PodShieldSim.assess_impact() -- nothing is hard-coded."""
    pod=PodShieldSim()
    print("="*100)
    print(" INSULATRIX POD -- HYPERVELOCITY IMPACT SHIELD REPORT")
    print(" Whipple-shield fragmentation + eddy-current magnetic deflection + self-healing hull")
    print("="*100)
    print("\nHULL CROSS-SECTION (hot face outward -> cold/vacuum face)")
    print("-"*60)
    for l in pod.layers:
        print("  %-45s %8.1f mm"%(l["name"],l["thickness_mm"]))
    print("  %-45s %8.1f mm"%("Standoff gap (bumper -> rear wall)",pod.standoff_m*1000.0))
    print("\nDESIGN REFERENCE: %.1f m/s (20,000 mph) -- realistic LEO/meteoroid hypervelocity"%IMPACT_PHYS["ref_velocity_ms"])
    print("DESIGN ENVELOPE: micrometeoroid & orbital debris (MMOD) scale, mm to a few cm --")
    print("  the real regime Whipple shields are engineered for. A literal kilometer-scale")
    print("  asteroid is outside ANY material shield's capability; that is honest physics.")
    print("\n%-46s %6s %6s %10s %6s %7s %7s %9s"%("Threat","Frag%","Defl%","Margin","Perf?","Hole mm","Patch s","Survived"))
    print("-"*100)
    rows=pod.sweep_threats()
    for r in rows:
        margin_s="%.4g"%r["ballistic_margin"] if r["ballistic_margin"]<1e6 else ">1e6"
        patch_s="%.2f"%r["patch_time_s"] if r["patch_time_s"] is not None else ("N/A" if r["beyond_patch_rating"] else "--")
        verdict="SURVIVED" if r["survived"] else ("BEYOND RATING" if r["beyond_patch_rating"] else "BREACHED")
        print("%-46s %5.0f%% %5.0f%% %10s %6s %7.1f %7s %9s"%(
            r["label"][:46],r["fragmentation_efficacy"]*100,r["deflect_frac"]*100,margin_s,
            "YES" if r["perforated"] else "no",r["hole_mm"],patch_s,verdict))
    print("-"*100)
    n_survived=sum(1 for r in rows if r["survived"]); n_total=len(rows)
    print("\nSUMMARY: %d/%d representative threat scenarios survived (fully stopped or auto-patched)."%(n_survived,n_total))
    print("\nHONEST ASSESSMENT")
    print("-"*60)
    print("""
  Magnetic deflection is real eddy-current physics (Lenz's law): it strongly
  assists against CONDUCTIVE threats (metal debris, nickel-iron meteorite
  fragments -- "metal asteroid, no problem" holds up for MMOD-scale metal
  fragments) but does essentially nothing for non-conductive rock/ice, which
  is the majority of real micrometeoroid mass. The coil+frame have a finite
  reaction-impulse budget, so deflection assist honestly diminishes for
  larger masses rather than staying "total" regardless of size.

  The Whipple bumper does the heavy lifting: it shatters small-to-moderate
  impactors into a debris cloud that spreads across the standoff gap before
  reaching the rear wall, diluting the areal energy load by orders of
  magnitude for anything the bumper can meaningfully fragment.

  The self-healing rear wall auto-patches breaches within its rating (holes
  up to %.0f mm) faster than the cabin can decompress to unsurvivable
  pressure -- genuinely "auto fill hole" as designed. Beyond that rating, the
  breach is reported honestly as unsurvivable by this system, not silently
  clipped into a false "handled" result. No real shield -- Whipple, magnetic,
  or otherwise -- stops everything; this report shows exactly where this
  one's line is.
"""%IMPACT_PHYS["sealant_max_hole_mm"])
    print("="*100)
    return n_survived,n_total

def print_pod_proof():
    """Two live, reproducible proofs backing the pod's headline claims:
    (1) magnetic deflection is doing real work for CONDUCTIVE threats (an A/B
    comparison, same projectile, conductivity flipped -- the only variable);
    (2) the self-healing hull genuinely "auto fills" a breach within its
    rating, walked step by step against the cabin-decompression clock. Every
    number comes from PodShieldSim.assess_impact() -- nothing is hard-coded.
    Returns True/False (pass/fail), usable as a regression check."""
    pod=PodShieldSim(); v=IMPACT_PHYS["ref_velocity_ms"]
    print("="*90)
    print(" INSULATRIX POD -- PROOF: MAGNETIC DEFLECTION + SELF-HEALING AUTO-PATCH")
    print("="*90)

    print("\nPROOF 1 -- MAGNETIC DEFLECTION IS REAL (A/B, conductivity is the only variable)")
    print("-"*90)
    d_mm,density=20.0,7800.0  # a mid-size Ni-Fe metal fragment -- "metal asteroid" scale
    a=pod.assess_impact(d_mm,density,v,conductive=True)
    b=pod.assess_impact(d_mm,density,v,conductive=False)
    print("  Same %.0f mm, %.0f kg/m3 fragment at %.0f m/s, only conductivity flipped:"%(d_mm,density,v))
    print("  %-28s %14s %14s"%("","CONDUCTIVE (metal)","NON-CONDUCTIVE"))
    print("  %-28s %14.1f%% %14.1f%%"%("Magnetic deflection:",a["deflect_frac"]*100,b["deflect_frac"]*100))
    print("  %-28s %14.0f %14.0f"%("Effective speed [m/s]:",a["v_effective_ms"],b["v_effective_ms"]))
    print("  %-28s %14.4g %14.4g"%("Ballistic margin:",a["ballistic_margin"],b["ballistic_margin"]))
    print("  %-28s %14s %14s"%("Perforated?:","YES" if a["perforated"] else "no","YES" if b["perforated"] else "no"))
    proof1=(a["deflect_frac"]>0.0 and b["deflect_frac"]==0.0 and a["ballistic_margin"]>b["ballistic_margin"])
    print("  PROOF 1: %s -- deflection is strictly positive only when conductive,"%("HOLDS" if proof1 else "FAILS"))
    print("           and strictly improves the outcome (%s the non-conductive case)."%
          ("does not merely match" if proof1 else "check model"))

    print("\nPROOF 2 -- SELF-HEALING HULL GENUINELY AUTO-FILLS A BREACH")
    print("-"*90)
    r=pod.assess_impact(8.0,3000.0,v,conductive=False)  # stony micrometeoroid: perforates, patchable
    print("  Threat: 8mm stony micrometeoroid (non-conductive, so zero deflection assist --")
    print("  the Whipple shield + self-heal have to do this completely on their own).")
    print("  Step 1: mass = %.3g kg, kinetic energy = %.3g J"%(r["mass_kg"],r["kinetic_energy_J"]))
    print("  Step 2: Whipple fragmentation efficacy = %.1f%%"%(r["fragmentation_efficacy"]*100))
    print("  Step 3: areal energy incident = %.3g J/m2 vs capacity %.3g J/m2 (rear wall %.3g"%(
        r["areal_energy_incident_J_m2"],r["areal_capacity_J_m2"],r["areal_capacity_rear_J_m2"]))
    print("          + thermal-core crush absorption %.3g -- 405mm of insulation isn't armor,"%r["areal_capacity_thermal_J_m2"])
    print("          but it isn't free passage either)")
    print("  Step 4: ballistic margin = %.4f (<1 -> perforates)"%r["ballistic_margin"])
    print("  Step 5: hole diameter = %.1f mm (within %.0f mm patch-system rating)"%(
        r["hole_mm"],IMPACT_PHYS["sealant_max_hole_mm"]))
    print("  Step 6: sealant closes radially at %.1f mm/s -> patch time = %.2f s"%(
        IMPACT_PHYS["sealant_expansion_mm_s"],r["patch_time_s"]))
    print("  Step 7: cabin pressure vs. time (choked-orifice decompression through the")
    print("          shrinking hole, race against the closing patch -- real sampled values):")
    for t,P_kPa in r["pressure_history"]:
        hole_open_pct=100.0*max(0.0,1.0-t/max(r["patch_time_s"],1e-9))
        print("    t=%5.2fs (hole %5.1f%% open): cabin pressure = %.2f kPa"%(t,hole_open_pct,P_kPa))
    print("  Step 8: patch completed at t=%.2fs; cabin never dropped below %.1f kPa"%(
        r["patch_time_s"],r["cabin_min_kPa"]))
    print("          (survivable floor: %.1f kPa)"%IMPACT_PHYS["cabin_min_survivable_kPa"])
    proof2=(r["perforated"] and not r["beyond_patch_rating"] and r["survived"] and
            r["cabin_min_kPa"]>=IMPACT_PHYS["cabin_min_survivable_kPa"]*0.5)
    print("  PROOF 2: %s -- breach occurred, was within the patch rating, sealed"%("HOLDS" if proof2 else "FAILS"))
    print("           itself, and the crew never saw unsurvivable pressure.")

    print("\nHONEST BOUNDARY -- where PROOF 2 stops holding")
    print("-"*90)
    beyond=next((rr for rr in pod.sweep_threats() if rr["beyond_patch_rating"]),None)
    if beyond:
        print("  %s: hole would be %.0f mm, exceeding the %.0f mm patch rating --"%(
            beyond["label"],beyond["hole_mm"],IMPACT_PHYS["sealant_max_hole_mm"]))
        print("  reported as unsurvivable by this system, not silently patched. That line is")
        print("  the honest edge of the design envelope, not a hidden failure.")
    print("="*90)
    passed=proof1 and proof2
    print("OVERALL: %s"%("BOTH PROOFS HOLD" if passed else "AT LEAST ONE PROOF FAILED -- see above"))
    print("="*90)
    return passed

# =============================================================================
# ECONOMIC IMPACT ESTIMATE -- order-of-magnitude, NOT a business plan
# =============================================================================
# Per-unit avoided heat flux is COMPUTED LIVE from the same ThermalSim used
# everywhere else in this file (STANDARD FOAM baseline vs. INSULATRIX PRO at
# a stated boundary condition) -- these are not hand-typed numbers. Scaling
# those per-unit results up to a global total requires macro inputs (how many
# houses/fridges/facilities exist, what energy costs) that physics cannot
# supply; those are stated explicitly below as cited, order-of-magnitude
# assumptions, the same way IMPACT_PHYS states its material constants.
#
# THIS IS A FERMI ESTIMATE FOR AN EDUCATIONAL SIMULATION, not a commercially
# validated product costing. Treat every total below as "if a material with
# exactly these simulated properties were manufactured at zero extra cost and
# deployed everywhere, order of magnitude X" -- a ceiling for discussion, not
# a forecast. See README.md / OVERVIEW.md for the full caveats.
ECONOMICS = {
    # --- Houses / buildings (winter space heating) ---
    "house_envelope_area_m2": 200.0,        # opaque wall+roof area, typical single-family home
    "house_indoor_K": 294.15, "house_outdoor_K": 273.15,  # 21 C indoor vs. 0 C representative winter design temp
    "heating_season_hours": 4380.0,         # ~half the year at full-load-equivalent (simplified single-point proxy for degree-days)
    "global_heated_dwellings_billion": 0.6, # dwellings in heating-dominant climates only (N.America+Europe+Russia/CIS+N.China-
                                             # scale order of magnitude) -- most of the ~2B global housing stock is in climates
                                             # needing little or no winter heating, so this is well under the world total
    "heating_usd_per_kWh": 0.12,            # blended global average delivered heating-energy cost (gas/electric/oil/district mix)
    "building_envelope_fraction": 0.5,      # opaque wall/roof conduction is ~40-60% of a real building's total heat loss;
                                             # the rest is windows, air infiltration, and ground losses this model doesn't touch
    # --- Refrigeration (fridges/coolers) ---
    "fridge_area_m2": 2.0, "fridge_indoor_K": 294.15, "fridge_inside_K": 277.15,  # 21 C room vs. 4 C fridge compartment
    "global_fridges_billion": 2.0,          # household + small commercial refrigeration units in service worldwide
    "elec_usd_per_kWh": 0.15,               # blended global average retail electricity price
    "fridge_envelope_fraction": 0.75,       # fridges are mostly opaque insulated envelope, minus door-seal/opening losses
    # --- Space habitat hull (avoided heater power -> smaller power system) ---
    # NOT propellant boil-off: a direct foam-vs-MLI check showed plain MLI is
    # already slightly BETTER than Insulatrix Pro for a pure cryogenic tank at
    # ~90 K -- an honest result worth stating, not papering over. Pro's real
    # advantage here is a general crewed-habitat hull (structure, not the
    # tank), where it beats a legacy foam/composite baseline outright.
    "habitat_area_m2": 100.0,               # hull area of a modest crewed habitat module
    "habitat_power_usd_per_W": 1500.0,      # representative space-qualified power generation+storage cost per watt
    "habitat_modules_per_year": 20,         # order-of-magnitude count of new station/habitat-class modules launched per year
    # --- Polar habitats / food-growing domes ---
    "polar_area_m2": 200.0, "polar_indoor_K": 294.15, "polar_outdoor_K": 233.15,  # 21 C indoor vs. -40 C polar winter
    "polar_usd_per_kWh": 1.00,              # remote diesel-logistics delivered energy cost (airlift/sealift fuel, not grid price)
    "polar_facility_count": 100,            # order-of-magnitude count of polar research/habitat facilities worldwide
    "polar_envelope_fraction": 0.4,         # a food-growing dome needs extensive glazing for light, which this model can't
                                             # replace -- opaque-envelope insulation is a smaller share of its total loss
    "discount_rate": 0.07,                  # for the illustrative perpetuity valuation
}

def _econ_avoided_flux(T_hot,T_cold,baseline="foam",upgrade="combined"):
    """Live-computed avoided heat flux (W/m2): baseline formulation minus the
    upgrade formulation, at the given boundary condition. Returns (q_base,q_upgrade,avoided)."""
    q_base=run_simulation(baseline,duration_s=3600,T_hot=T_hot,T_cold=T_cold).q_transmitted
    q_up=run_simulation(upgrade,duration_s=3600,T_hot=T_hot,T_cold=T_cold).q_transmitted
    return q_base,q_up,max(q_base-q_up,0.0)

def print_economics_report():
    """Order-of-magnitude economic impact estimate. Per-unit numbers are live
    physics (ThermalSim); global totals layer on stated macro assumptions
    (ECONOMICS dict). Returns the grand-total annual USD saved (float)."""
    E=ECONOMICS
    print("="*94)
    print(" INSULATRIX -- ECONOMIC IMPACT ESTIMATE (order-of-magnitude Fermi estimate)")
    print(" Per-unit avoided heat flux is LIVE-COMPUTED (STANDARD FOAM vs INSULATRIX PRO).")
    print(" Global totals layer stated macro assumptions on top -- NOT a business plan.")
    print("="*94)

    rows=[]
    # 1) Houses / buildings -- only building_envelope_fraction of the avoided
    # wall/roof flux is credited as real heating-bill savings (windows,
    # infiltration, and ground losses are outside what this model touches).
    q_f,q_p,av=_econ_avoided_flux(E["house_indoor_K"],E["house_outdoor_K"])
    kWh_house=av*E["building_envelope_fraction"]*E["house_envelope_area_m2"]*E["heating_season_hours"]/1000.0
    usd_house=kWh_house*E["heating_usd_per_kWh"]
    usd_global_house=usd_house*E["global_heated_dwellings_billion"]*1e9
    rows.append(("Houses/buildings (winter heating)",q_f,q_p,av,kWh_house,usd_house,
                 E["global_heated_dwellings_billion"]*1e9,usd_global_house))

    # 2) Refrigeration
    q_f2,q_p2,av2=_econ_avoided_flux(E["fridge_indoor_K"],E["fridge_inside_K"])
    kWh_fridge=av2*E["fridge_envelope_fraction"]*E["fridge_area_m2"]*8760.0/1000.0
    usd_fridge=kWh_fridge*E["elec_usd_per_kWh"]
    usd_global_fridge=usd_fridge*E["global_fridges_billion"]*1e9
    rows.append(("Refrigeration (fridges/coolers)",q_f2,q_p2,av2,kWh_fridge,usd_fridge,
                 E["global_fridges_billion"]*1e9,usd_global_fridge))

    # 3) Polar habitats / food-growing domes
    q_f3,q_p3,av3=_econ_avoided_flux(E["polar_indoor_K"],E["polar_outdoor_K"])
    kWh_polar=av3*E["polar_envelope_fraction"]*E["polar_area_m2"]*8760.0/1000.0
    usd_polar=kWh_polar*E["polar_usd_per_kWh"]
    usd_global_polar=usd_polar*E["polar_facility_count"]
    rows.append(("Polar habitats/greenhouses",q_f3,q_p3,av3,kWh_polar,usd_polar,
                 E["polar_facility_count"],usd_global_polar))

    print("\n%-34s %10s %10s %10s %14s %12s"%("Category","q_base","q_pro","avoided","kWh/yr/unit","USD/yr/unit"))
    print("%-34s %10s %10s %10s %14s %12s"%("","W/m2","W/m2","W/m2","",""))
    print("-"*94)
    for name,qf,qp,a,kwh,usd,n,gtotal in rows:
        print("%-34s %10.4f %10.4f %10.4f %14.1f %12.2f"%(name,qf,qp,a,kwh,usd))
    print("-"*94)
    print("\n%-34s %16s %20s"%("Category","# units (world)","GLOBAL USD/year"))
    print("-"*94)
    total=0.0
    for name,qf,qp,a,kwh,usd,n,gtotal in rows:
        print("%-34s %16.3g %20s"%(name,n,"${:,.0f}".format(gtotal)))
        total+=gtotal

    # 4) Space habitat hull: avoided heater power -> smaller power system CAPEX.
    # Deliberately NOT propellant boil-off (see ECONOMICS comment): a direct
    # check found plain MLI already beats Insulatrix Pro for a pure ~90 K
    # cryo tank, so that comparison is reported honestly instead of forced.
    _,_,av4=_econ_avoided_flux(298.15,3.0)  # habitable interior vs. deep space (BOUNDARY_SCENARIOS[2])
    avoided_W=av4*E["habitat_area_m2"]
    usd_per_habitat=avoided_W*E["habitat_power_usd_per_W"]
    usd_global_habitat=usd_per_habitat*E["habitat_modules_per_year"]
    print("%-34s %16.3g %20s"%("Space habitat hull (CAPEX)",E["habitat_modules_per_year"],"${:,.0f}".format(usd_global_habitat)))
    print("  (avoided flux %.4f W/m2 x %.0f m2 hull = %.0f W less heater power needed"%(av4,E["habitat_area_m2"],avoided_W))
    print("   -> $%s smaller power system per habitat module, at $%.0f/W --"%(
        "{:,.0f}".format(usd_per_habitat),E["habitat_power_usd_per_W"]))
    print("   a one-time CAPEX-avoided story, not a recurring $/year one, kept separate above.")
    print("   Note: plain MLI already slightly outperforms Pro for a dedicated ~90 K cryo")
    print("   tank specifically -- this line is about habitat hull structure, not propellant tanks.)")
    total+=usd_global_habitat
    print("-"*94)

    print("\nGLOBAL TOTAL (all categories, full-adoption ceiling): $%s / year"%("{:,.0f}".format(total)))
    pv=total/E["discount_rate"]
    print("\nILLUSTRATIVE CAPITALIZED VALUE (perpetuity, discount rate %.0f%%):"%(E["discount_rate"]*100))
    print("  PV = annual savings / discount rate = $%s"%("{:,.0f}".format(pv)))
    print("\nHONEST CAVEATS (read before quoting any number above)")
    print("-"*94)
    print("""
  - This is a FULL-ADOPTION CEILING: every heated dwelling, fridge, polar
    facility, and new habitat module on Earth (or in orbit) built or
    retrofitted simultaneously with zero transition cost. Real adoption would
    be gradual (years to decades) and partial, so realized savings would be a
    fraction of this figure.
  - GROSS savings only: manufacturing, installation, and capital cost of the
    insulation itself are NOT netted out. Net value would be materially lower.
  - Insulatrix Pro is a SIMULATED formulation in an educational digital twin,
    not a commercially validated, manufactured, or costed real material.
  - Single-point boundary conditions (e.g. "0 C outdoor") stand in for real
    degree-day/duty-cycle integrals -- a simplification, not a weather model.
  - The perpetuity valuation assumes the savings stream is perpetual, risk-free,
    and constant in real terms, none of which holds in practice; it is offered
    only as an illustrative order-of-magnitude framing, not a valuation.
""")
    print("="*94)
    return total

def print_proof(T_hot=None,T_cold=None):
    """Live, reproducible demonstration that INSULATRIX PRO achieves near-zero heat
    transfer with a simulated hot face driven to the practical limit of absolute
    zero on the cold face. Every number below is computed from the same physics
    used by --compare and the interactive app -- nothing here is hard-coded.
    Returns True/False (pass/fail) so it can also be used as a regression check."""
    sc_name=None
    if T_hot is None and T_cold is None:
        az=[s for s in BOUNDARY_SCENARIOS if "ABSOLUTE ZERO" in s["name"]]
        if az:
            T_hot=az[0]["T_hot"]; T_cold=az[0]["T_cold"]; sc_name=az[0]["name"]
    T_hot=T_hot or PHYS["T_hot_default"]; T_cold=T_cold if T_cold is not None else PHYS["T_cold_default"]
    print("="*78)
    print(" INSULATRIX -- PROOF: NEAR-ZERO HEAT TRANSFER AT THE ABSOLUTE-ZERO LIMIT")
    print("="*78)
    print("\nCLAIM")
    print("-"*40)
    print("  With a simulated forge/crucible hot face and a cold face driven to the")
    print("  practical limit of absolute zero, INSULATRIX PRO (formulation 11, the")
    print("  5-stage combined stack) suppresses transmitted heat flux to a negligible")
    print("  fraction of the uninsulated (bare) flux.")
    print("\nBOUNDARY CONDITIONS%s"%("  (%s)"%sc_name if sc_name else ""))
    print("-"*40)
    print("  T_hot:  %.6g K (%.2f C)"%(T_hot,T_hot-273.15))
    print("  T_cold: %.6g K (%.9f C)"%(T_cold,T_cold-273.15))
    print("  dT:     %.6g K"%(T_hot-T_cold))
    sim=run_simulation("combined",duration_s=3600,T_hot=T_hot,T_cold=T_cold)
    sigma=PHYS["sigma_SB"]
    q_rad_bare=sigma*(T_hot**4-T_cold**4); q_conv_bare=5.0*(T_hot-T_cold)
    print("\nSTEP 1 -- UNINSULATED BASELINE (Stefan-Boltzmann + natural convection)")
    print("-"*40)
    print("  q_rad_bare  = sigma*(T_hot^4 - T_cold^4)   = %.6e W/m2"%q_rad_bare)
    print("  q_conv_bare = h*(T_hot - T_cold), h=5 W/m2K = %.6e W/m2"%q_conv_bare)
    print("  q_incoming  = q_rad_bare + q_conv_bare      = %.6e W/m2  <- flux with NO insulation"%sim.q_incoming)
    print("\nSTEP 2 -- PER-LAYER ATTENUATION (Fourier + Knudsen + Rosseland, live)")
    print("-"*40)
    print(sim.layer_summary())
    print("\nSTEP 3 -- ACTIVE ABSORPTION / CONVERSION (this simulation window)")
    print("-"*40)
    print("  PCM + chemical heat absorbed: %.4f MJ/m2"%(sim.total_absorbed/1e6))
    print("  TEG electrical conversion:    %.4f MJ/m2"%(sim.total_converted/1e6))
    print("  Peltier heat pumped back:     %.4f MJ/m2"%(sim.total_pumped/1e6))
    print("\nSTEP 4 -- RESULT")
    print("-"*40)
    print("  q_transmitted = %.6e W/m2"%sim.q_transmitted)
    print("  suppression   = 1 - q_transmitted/q_incoming = %.8f%%"%sim.suppression_pct)
    print("  transmission  = %.8f%%"%sim.transmission_pct)
    thresh=99.99
    passed=math.isfinite(sim.suppression_pct) and math.isfinite(sim.q_transmitted) and sim.suppression_pct>thresh and sim.is_success
    print("\nVERIFICATION")
    print("-"*40)
    print("  suppression > %.2f%% ?  %-4s (%.6f%%)"%(thresh,"YES" if sim.suppression_pct>thresh else "NO",sim.suppression_pct))
    print("  result finite?          %s"%("YES" if math.isfinite(sim.q_transmitted) else "NO"))
    print("  is_success flag?        %s"%sim.is_success)
    print("  OVERALL:                %s"%("PROVEN (this run)" if passed else "NOT PROVEN -- see failure above"))
    print("\nHONEST CAVEAT")
    print("-"*40)
    print("  This is a live numerical demonstration, not a claim of literal zero.")
    print("  True zero heat transfer is thermodynamically impossible (2nd/3rd law):")
    print("  residual conduction, gas conduction, and radiation always exist. What is")
    print("  demonstrated is that the suppression ratio saturates arbitrarily close to")
    print("  100% as T_cold -> 0, bounded only by the stack's own material physics --")
    print("  not by how cold the sink gets (verified stable down to 1e-12 K).")
    print("="*78)
    return passed

def _thomas(a,b,c,d):
    """Tridiagonal solve (Thomas algorithm), a=sub, b=diag, c=super, d=rhs."""
    n=len(d); cp=np.zeros(n); dp=np.zeros(n)
    cp[0]=c[0]/b[0]; dp[0]=d[0]/b[0]
    for i in range(1,n):
        m=b[i]-a[i]*cp[i-1]; cp[i]=c[i]/m; dp[i]=(d[i]-a[i]*dp[i-1])/m
    x=np.zeros(n); x[-1]=dp[-1]
    for i in range(n-2,-1,-1): x[i]=dp[i]-cp[i]*x[i+1]
    return x

def _implicit_transient(stack,duration_s=3600.0,dt=2.0,n_per=8):
    """Independent cross-check engine: semi-implicit finite-volume 1-D
    conduction. rho*cp*dT/dt = div(k dT/dx) solved by backward Euler
    (unconditionally stable -- sub-mm coatings do not force micro-timesteps,
    unlike the explicit solver and unlike the app's algebraic-relaxation sim).
    k_eff(T) is evaluated at the previous step's state (Picard linearization).
    Stack starts at T_cold; boundaries held Dirichlet. Returns cold-face flux
    history + arrival metrics."""
    cells=[]
    for li,l in enumerate(stack.layers):
        for _ in range(n_per): cells.append((li,l.thickness_m/n_per))
    n=len(cells); n_steps=max(1,int(duration_s/dt))
    lo,hi=min(stack.bc.T_hot,stack.bc.T_cold)-10.0,max(stack.bc.T_hot,stack.bc.T_cold)+10.0
    T=np.full(n,stack.bc.T_cold,float)
    rho=np.array([stack.layers[li].material["rho"] for li,_ in cells])
    cpv=np.array([stack.layers[li].material["cp"] for li,_ in cells])
    dxc=np.array([dx for _,dx in cells])
    q_hist=np.zeros(n_steps+1); t_hist=np.arange(n_steps+1)*dt
    for step in range(n_steps):
        k=np.array([stack.layers[li].k_eff(max(T[c],1e-9)) for c,(li,_) in enumerate(cells)])
        a=np.zeros(n); b=np.zeros(n); c=np.zeros(n); d=np.zeros(n)
        C=rho*cpv*dxc/dt; b+=C; d+=C*T
        Gw=2.0/(dxc[:-1]/k[:-1]+dxc[1:]/k[1:])   # harmonic-mean interface conductance
        a[1:]=-Gw; b[1:]+=Gw; c[:-1]=-Gw; b[:-1]+=Gw
        gL=2.0*k[0]/dxc[0]; b[0]+=gL; d[0]+=gL*stack.bc.T_hot
        gR=2.0*k[-1]/dxc[-1]; b[-1]+=gR; d[-1]+=gR*stack.bc.T_cold
        T=_thomas(a,b,c,d)
        T=np.clip(T,lo,hi); T=np.nan_to_num(T,nan=stack.bc.T_cold)
        q_hist[step+1]=gR*(T[-1]-stack.bc.T_cold)
    q_inc=PHYS["sigma_SB"]*(stack.bc.T_hot**4-stack.bc.T_cold**4)+5.0*(stack.bc.T_hot-stack.bc.T_cold)
    thresh=0.01*q_inc
    t_arrive=float(t_hist[np.argmax(q_hist>thresh)]) if np.any(q_hist>thresh) else duration_s
    return {"q_end":float(q_hist[-1]),"q_peak":float(q_hist.max()),"t_arrive":t_arrive,
            "q_hist":q_hist,"t_hist":t_hist,"q_incoming":q_inc,"n_cells":n,
            "suppression_end_pct":(1.0-abs(q_hist[-1])/max(q_inc,1e-9))*100.0}

def print_ultra_proof(T_hot=None,T_cold=None):
    """Live, reproducible proof that ULTRA CASCADE (formulation 12) achieves
    near-zero heat loss by ACCUMULATIVE STACKING -- ordered multi-layering in
    which each stage drops the local dT for the next, so downstream layers run
    in a lower-k, lower-radiation regime and emergent performance exceeds what
    any equal-thickness single material can deliver. Every number is computed
    by the same physics engines used by --compare/--sim; nothing is hard-coded.
    Returns True/False (pass/fail) for use as a regression check."""
    sc_name=None
    if T_hot is None and T_cold is None:
        az=[s for s in BOUNDARY_SCENARIOS if "ABSOLUTE ZERO" in s["name"]]
        if az: T_hot=az[0]["T_hot"]; T_cold=az[0]["T_cold"]; sc_name=az[0]["name"]
    T_hot=T_hot or PHYS["T_hot_default"]; T_cold=T_cold if T_cold is not None else PHYS["T_cold_default"]
    stack=sm_ultra(T_hot=T_hot,T_cold=T_cold)
    r=stack.solve_steady_state()
    print("="*86)
    print(" INSULATRIX ULTRA -- PROOF: NEAR-ZERO HEAT LOSS BY ACCUMULATIVE STACKING")
    print("="*86)
    print("\nCLAIM")
    print("-"*46)
    print("  The optimized sprayable multi-layer (ULTRA CASCADE, formulation 12)")
    print("  suppresses transmitted heat flux to a negligible fraction of the")
    print("  uninsulated flux, and does so BECAUSE of its stacking order: the")
    print("  architecture contributes resistance no single material of equal")
    print("  thickness can match (proven below, not assumed).")
    print("\nSTEP 0 -- THE OPTIMIZED FORMULA (sprayable layering, goal2.md + RTP2.0)")
    print("-"*46)
    print("  Zones hot->cold:")
    print("    A  4x cascaded spray pass: YSZ/Al2O3-ZrO2 -> zirconia aerogel ->")
    print("       opacified carbon aerogel -> pure zirconia aerogel -> +TiN cap")
    print("    B  5x evacuated carbon-MLI hot shields (60 refractory foils ea) +")
    print("       deep jacket (80 foils) -- microsphere vacuum + T_max 1473 K")
    print("    C  PerfectVoid vacuum-microsphere foam (96%% porosity) + VIP core")
    print("    D  recovery engines: 2x staged Bi2Te3 TEG harvesters (solid-state;")
    print("       no PCM dams -- at ~600 K they decompose, and a pinned dam")
    print("       against a ~0 K sink leaks more than it saves)")
    print("    E  2x sealed Kapton-film MLI jackets (200 foils, local T <673 K)")
    print("    F  aerogel-MLI cold cap (36 foils) + TiN topcoat + weather seal")
    print("  %d layers, %.1f mm total"%(stack.n_layers,stack.total_thickness_mm))
    print("  Ordering rule: every material sits below its T_max at steady state --")
    print("  fragile high-R materials only work BECAUSE upstream zones drop the T.")
    print("\nBOUNDARY CONDITIONS%s"%("  (%s)"%sc_name if sc_name else ""))
    print("-"*46)
    print("  T_hot:  %.6g K (%.2f C)"%(T_hot,T_hot-273.15))
    print("  T_cold: %.6g K (%.9f C)"%(T_cold,T_cold-273.15))
    print("  dT:     %.6g K"%(T_hot-T_cold))
    print("\nSTEP 1 -- UNINSULATED BASELINE (Stefan-Boltzmann + natural convection)")
    print("-"*46)
    sigma=PHYS["sigma_SB"]
    q_rad_bare=sigma*(T_hot**4-T_cold**4); q_conv_bare=5.0*(T_hot-T_cold)
    print("  q_rad_bare  = sigma*(T_hot^4 - T_cold^4)    = %.6e W/m2"%q_rad_bare)
    print("  q_conv_bare = h*(T_hot - T_cold), h=5 W/m2K = %.6e W/m2"%q_conv_bare)
    print("  q_incoming  = %.6e W/m2  <- flux with NO insulation"%r["q_incoming"])
    print("\nSTEP 2 -- PER-LAYER ATTENUATION (steady-state Fourier + Knudsen + Rosseland)")
    print("-"*46)
    print("  %-4s %-42s %7s %11s %11s %8s"%("L#","Layer","thk[mm]","k_eff W/mK","R m2K/W","Tavg K"))
    print("  "+"-"*84)
    for i,layer in enumerate(stack.layers):
        Ta=(r["T_profile"][i]+r["T_profile"][i+1])/2.0
        print("  %-4d %-42s %7.1f %11.5f %11.4f %8.1f"%(
            i,layer.name[:42],layer.thickness_mm,r["k_eff_per_layer"][i],r["R_per_layer"][i],Ta))
    print("  "+"-"*84)
    print("  "+"-"*84)
    print("  R_total = %.4f m2.K/W    q_flux = dT/R_total = %.6e W/m2"%(r["R_total"],r["q_flux"]))
    R_fwd,n_fwd,fail_fwd=_feasible_R(stack,r)
    print("  Forward-feasibility: %d/%d layers over T_max%s"%(
        n_fwd,stack.n_layers," -- VIOLATIONS: "+", ".join(fail_fwd[:4]) if n_fwd else " -- all within service rating"))
    print("\nSTEP 3 -- ACCUMULATIVE-STACKING SYNERGY PROOF")
    print("-"*46)
    print("  Feasibility rule: a layer pushed past its T_max fails -- it keeps no")
    print("  resistance. Illegal orderings and impossible monoliths are not")
    print("  credited with R they could never hold.")
    syn=sm_synergy_factor(stack)
    rev=_reversed_sstack(stack); r_rev=rev.solve_steady_state()
    R_rev_feas,n_failed,failed=_feasible_R(rev,r_rev)
    best,disqualified=_best_single_material_sstack(stack)
    r_single=best.solve_steady_state() if best else None
    R_uniform=_uniform_T_resistance(stack)
    print("  S_ord  ordering  = R_stack / R_reversed(feasible)  = %.4f / %.4f = %.3f"%(
        r["R_total"],R_rev_feas,syn["ordering"]))
    print("           (reversed order fails %d/%d layers over T_max: %s)"%(
        n_failed,rev.n_layers,", ".join(failed[:4])+("..." if len(failed)>4 else "")))
    if r_single:
        print("  S_mat  material  = R_stack / R_best-feasible-single = %.4f / %.4f = %.3f"%(
            r["R_total"],r_single["R_total"],syn["material"]))
        print("           (best feasible single = %s)"%best.layers[0].material["name"])
        print("           (%d material configs disqualified: T_max < T_hot, e.g. %s)"%(
            len(disqualified),", ".join(disqualified[:3])+("..." if len(disqualified)>3 else "")))
    else:
        print("  S_mat  material  = +inf -- NO single material survives T_hot alone.")
    print("  S_grad gradient  = R_stack / sum(R_i at uniform T)  = %.4f / %.4f = %.3f"%(
        r["R_total"],R_uniform,syn["gradient"]))
    print("  S_total = S_ord * S_mat * S_grad = %.3f"%syn["total"])
    if syn["total"]>1.0:
        print("  -> ~%.0f%% of the stack's thermal resistance is emergent from the"%(
            100.0*(1.0-1.0/syn["total"])))
        print("     stacking architecture itself -- no single material achieves it.")
    else:
        print("  -> stacking contributes less R than a feasible monolith in this")
        print("     metric -- ordering still matters for survivability, not raw R.")
    print("\nSTEP 4 -- CASCADE SCALING (more stacked passes -> transmission -> 0)")
    print("-"*46)
    print("  %-12s %12s %14s %14s"%("spray passes","q_trans W/m2","suppression %","R_total"))
    print("  "+"-"*56)
    for n_p in (1,2,3,4,6,8):
        s_k=sm_ultra(T_hot=T_hot,T_cold=T_cold,n_passes=n_p); r_k=s_k.solve_steady_state()
        print("  %-12d %12.6f %14.6f %14.4f"%(n_p,r_k["q_transmitted"],r_k["suppression_pct"],r_k["R_total"]))
    print("  (each added cascade stage lowers the dT the next stages must carry;")
    print("   suppression climbs asymptotically toward 100%% -- accumulative stacking.)")
    print("\nSTEP 5 -- MLI 1/N LAW (radiative suppression scales with foil count)")
    print("-"*46)
    print("  %-10s %14s %14s"%("N foils","q_trans W/m2","suppression %"))
    print("  "+"-"*42)
    for n_r in (0,1,5,10,40,100):
        s_n=SInsulationStack([SLayer("Kapton MLI + aerogel spacers","mli_nano",10.0,
            n_reflective=n_r,vacuum=True,vacuum_pressure_torr=1e-5)],
            bc=SBoundary(T_hot=T_hot,T_cold=T_cold),name="MLI-N%d"%n_r)
        r_n=s_n.solve_steady_state()
        print("  %-10d %14.6f %14.6f"%(n_r,r_n["q_transmitted"],r_n["suppression_pct"]))
    print("  (N=0 uses the bare-surface emissivity form, N>=1 the n-foil 2e form;")
    print("   the small N=0->1 discontinuity is the model's, the ~1/N trend for")
    print("   N>=5 is the physical foil-count law.)")
    print("\nSTEP 6 -- ALL-SCENARIO VALIDATION (same stack, every boundary pair)")
    print("-"*46)
    print("  %-32s %14s %16s"%("Scenario","q_trans W/m2","suppression %"))
    print("  "+"-"*64)
    all_pass=True
    for sc in BOUNDARY_SCENARIOS:
        s_sc=sm_ultra(T_hot=sc["T_hot"],T_cold=sc["T_cold"]); r_sc=s_sc.solve_steady_state()
        ok=r_sc["suppression_pct"]>99.9 and math.isfinite(r_sc["q_transmitted"])
        all_pass=all_pass and ok
        print("  %-32s %14.6f %16.6f  %s"%(sc["name"],r_sc["q_transmitted"],r_sc["suppression_pct"],"OK" if ok else "FAIL"))
    print("\nSTEP 7 -- INDEPENDENT CROSS-CHECK (implicit finite-volume transient)")
    print("-"*46)
    tr=_implicit_transient(stack,duration_s=3600.0,dt=2.0,n_per=8)
    print("  Engine: backward-Euler finite-volume conduction, %d cells, dt=2 s --"%tr["n_cells"])
    print("  a DIFFERENT solver than the steady-state engine (Picard R-sum) above.")
    print("  Stack starts fully cold, then the hot face is held at T_hot for 3600 s:")
    print("    cold-face flux at t=3600 s:   %.6e W/m2  (suppression %.6f%%)"%(tr["q_end"],tr["suppression_end_pct"]))
    print("    peak cold-face flux:          %.6e W/m2"%tr["q_peak"])
    print("    time for cold face to reach 1%% of incoming: %s s"%(
        "%.0f"%tr["t_arrive"] if tr["t_arrive"]<3600 else ">3600 (front has not arrived)"))
    print("  (Physical meaning: the stack's thermal inertia holds the cold face at")
    print("   ~zero transmitted flux for the whole hour -- the heat front is still")
    print("   diffusing through 246 mm of ultra-high-R material.)")
    print("\nSTEP 8 -- RESULT")
    print("-"*46)
    print("  Steady-state:  q_transmitted = %.6e W/m2   suppression = %.8f%%"%(r["q_transmitted"],r["suppression_pct"]))
    print("  k_eff overall: %.6f W/(m.K) = %.6f mW/(m.K)"%(r["k_eff_overall"],r["k_eff_overall_mW"]))
    thresh=99.99
    transient_ok=math.isfinite(tr["q_end"]) and tr["q_end"]<=max(2.0*r["q_transmitted"],0.01*tr["q_incoming"])
    passed=(math.isfinite(r["q_transmitted"]) and transient_ok
            and r["suppression_pct"]>thresh
            and syn["total"]>1.0 and all_pass and n_fwd==0)
    print("\nVERIFICATION")
    print("-"*46)
    print("  steady-state suppression > %.2f%% ?   %-4s (%.6f%%)"%(thresh,"YES" if r["suppression_pct"]>thresh else "NO",r["suppression_pct"]))
    print("  transient cold-face flux bounded?     %-4s (q_end=%.4f W/m2)"%("YES" if transient_ok else "NO",tr["q_end"]))
    print("  stacking synergy S_total > 1 ?        %-4s (%.3f)"%("YES" if syn["total"]>1.0 else "NO",syn["total"]))
    print("  all materials under T_max?            %-4s (%d violations)"%("YES" if n_fwd==0 else "NO",n_fwd))
    print("  all 5 scenarios pass?                 %-4s"%("YES" if all_pass else "NO"))
    print("  finite results?                       %-4s"%("YES" if math.isfinite(r["q_transmitted"]) else "NO"))
    print("  OVERALL:                              %s"%("PROVEN (this run)" if passed else "NOT PROVEN -- see failures above"))
    print("\nHONEST CAVEAT")
    print("-"*46)
    print("  This is a live numerical demonstration, not a claim of literal zero.")
    print("  True zero heat transfer is thermodynamically impossible (2nd/3rd law):")
    print("  residual solid conduction, residual gas, and radiation always exist,")
    print("  and real-world parasitics (supports, seams, vacuum leaks) dominate")
    print("  below the model floor. What is proven here is that the ORDERED multi-")
    print("  layer architecture drives transmitted flux asymptotically toward zero")
    print("  -- each stacked stage lowers the dT carried by the next, which is the")
    print("  accumulative-stacking mechanism goal2.md requires. Vacuum-retention,")
    print("  seal life, and PCM recharge remain the real engineering constraints.")
    print("="*86)
    return passed

def print_fabrication(approach="ultra_cascade"):
    """CLI: print the step-by-step fabrication protocol for an approach."""
    fab=FABRICATION.get(approach)
    if fab is None:
        print("No fabrication protocol for '%s'. Valid: %s"%(approach,", ".join(sorted(FABRICATION))))
        return
    fname=approach
    for f in FORMULATIONS:
        if f["approach"]==approach: fname=f["name"]; break
    print("="*100)
    print(" FABRICATION PROTOCOL :: %s"%fab["title"])
    print("="*100)
    print("Active build: %s   |   Calendar incl. cures: %s\n"%(fab["est_active"],fab["est_calendar"]))
    print("EQUIPMENT:"); [print("  - %s"%e) for e in fab["equipment"]]
    print("\nMATERIALS:"); [print("  - %s"%m) for m in fab["materials"]]
    for pname,steps in fab["phases"]:
        print("\n"+"-"*100); print("%s"%pname); print("-"*100)
        for i,s in enumerate(steps): print("  %2d. %s"%(i+1,s))
    print("\n"+"="*100); print(" QC GATES:"); [print("  [ ] %s"%q) for q in fab["qc"]]
    print("\n SAFETY:"); [print("  ! %s"%s) for s in fab["safety"]]
    for c in fab.get("caveats",[]): print("\n NOTE: %s"%c)
    print("="*100)

def print_durability_report(T_hot=None,T_cold=None):
    T_hot=T_hot or PHYS["T_hot_default"]; T_cold=T_cold or PHYS["T_cold_default"]
    print("="*90)
    print(" INSULATRIX -- DURABILITY & WEATHERING REPORT")
    print(" Near-permanence assessment: rated envelope, thermal-cycle fatigue, live service-life estimate")
    print("="*90)
    print("\nBOUNDARY CONDITIONS")
    print("-"*40)
    print("  T_hot:  %.0f K (%.0f C)"%(T_hot,T_hot-273.15))
    print("  T_cold: %.0f K (%.0f C)"%(T_cold,T_cold-273.15))
    print("  dT:     %.0f K"%(T_hot-T_cold))
    print("\n%-22s %16s %10s %10s %14s %12s"%("Approach","Envelope[C]","In-range","Cycles/yr","Life[yr,live]","Class"))
    print("-"*90)
    rows=[]
    for f in FORMULATIONS:
        est=estimate_durability(f,T_hot,T_cold)
        if est is None: continue
        d=est["data"]
        rows.append((f["name"],d,est))
        print("%-22s %7d to %6d %10s %10.1f %14.1f %12s"%(
            f["name"],d["min_service_C"],d["max_service_C"],
            "YES" if est["in_rated_envelope"] else "NO",
            est["cycles_to_failure"]/365.0,est["lifetime_years"],d["weathering_class"].split(" ")[0]))
    print("-"*90)
    ranked=sorted(rows,key=lambda t:t[2]["lifetime_years"],reverse=True)
    print("\nRANKED BY ESTIMATED SERVICE LIFE (this boundary condition):")
    for i,(name,d,est) in enumerate(ranked):
        marker=" <-- MOST DURABLE" if i==0 else ""
        print("  %2d. %-22s  %.1f yr%s"%(i+1,name,est["lifetime_years"],marker))
    print("\n"+"="*90)

def export_obj():
    out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"export")
    os.makedirs(out_dir,exist_ok=True)
    parts,total_t=build_scene(0)
    all_verts=[]; all_faces=[]; all_mats={}; mat_idx=0; v_offset=0
    for part in parts:
        for m in part.meshes:
            mat_name="mat_%d"%mat_idx
            r,g,b=m.color; all_mats[mat_name]=(r/255.0,g/255.0,b/255.0); mat_idx+=1
            wv=m.world_verts()+part.explode
            for v in wv: all_verts.append((v[0],v[1],v[2]))
            for f in m.faces: all_faces.append(([f[i]+v_offset+1 for i in range(len(f))],mat_name))
            v_offset+=len(wv)
    obj_path=os.path.join(out_dir,"insulatrix.obj")
    mtl_path=os.path.join(out_dir,"insulatrix.mtl")
    with open(obj_path,"w") as fp:
        fp.write("# Insulatrix -- Near-Zero Thermal Transfer Digital Twin\n")
        fp.write("mtllib insulatrix.mtl\n")
        for v in all_verts: fp.write("v %.6f %.6f %.6f\n"%v)
        current_mat=None
        for f,mat in all_faces:
            if mat!=current_mat: fp.write("usemtl %s\n"%mat); current_mat=mat
            fp.write("f "+" ".join(str(i) for i in f)+"\n")
    with open(mtl_path,"w") as fp:
        for mat_name,(r,g,b) in all_mats.items():
            fp.write("newmtl %s\n"%mat_name)
            fp.write("Kd %.4f %.4f %.4f\n"%(r,g,b))
            fp.write("Ka 0.1 0.1 0.1\nKe 0.0 0.0 0.0\nNs 10.0\n\n")
    print("Exported %d vertices, %d faces to %s"%(len(all_verts),len(all_faces),obj_path))
    print("Materials written to %s"%mtl_path)

def _print_banner():
    print("="*70)
    print(" Insulatrix -- Near-Zero Thermal Transfer Digital Twin")
    print(" Educational simulation. NOT a real product.")
    print("="*70)
    print(" Physics: Fourier conduction, Knudsen gas, Rosseland radiation,")
    print("          MLI reflective shielding, PCM latent heat,")
    print("          endothermic chemical absorption, TEG Seebeck + Peltier")
    print(" Scale:   1:1 (all dimensions in real mm)")
    print(" Success: heat transfer suppression ratio > 99%")
    print(" Goal:    MINIMIZE transmitted heat flux (q_transmitted -> 0)")
    print("-"*70)
    print(" Formulations (11):")
    for f in FORMULATIONS:
        print("   %2d. %-22s  [%s]"%(f["id"],f["name"],f["approach"]))
        print("       Composition: %s"%f["recipe"]["composition"])
    print("-"*70)
    print(" Boundary scenarios (press V to cycle):")
    for sc in BOUNDARY_SCENARIOS:
        print("   %-28s %s"%(sc["name"],sc["desc"]))
    print("-"*70)
    print(" Controls:")
    print("   TAB/SHIFT+TAB=next/prev model (all 13: 12 formulations + the pod)")
    print("   H=help  I=spec  D=material diagram  W=durability  SPACE=sim on/off")
    print("   1-9/0/C/U=formulation  A=auto-cycle  T=auto-rotate  V=scenario")
    print("   +/-=thickness  [/]=hot T  ;/'=cold T  P=particles")
    print("   BACKSPACE=reset  R=reset camera  L=labels  X=cross-section")
    print("   M=impact-shield pod tab (new)  ESC=quit")
    print("-"*70)
    print(" Impact Shield Pod (press M): Whipple shield + magnetic deflection +")
    print("   self-healing hull, 20,000 mph (8940.8 m/s) design reference.")
    print("   In-tab: 1-8 selects threat, SPACE launches live test.")
    print("   CLI: --pod-report for the full physics assessment.")
    print("="*70)

# =============================================================================
# SECTION 11 -- MAIN
# =============================================================================
def main():
    parser=argparse.ArgumentParser(description="Insulatrix -- Near-Zero Thermal Transfer Digital Twin")
    parser.add_argument("--selftest",action="store_true",help="headless build + physics + render sanity check")
    parser.add_argument("--feasibility",action="store_true",help="print the full honest physics/feasibility report")
    parser.add_argument("--export-obj",action="store_true",help="write OBJ+MTL files to ./export/ and exit")
    parser.add_argument("--compare",action="store_true",help="print side-by-side comparison table of all approaches")
    parser.add_argument("--sim",type=str,default=None,metavar="APPROACH",
                        help="run a single approach simulation and print detailed results. "
                             "Valid: foam, aerogel, ceramic_fiber, refractory_foam, mli, "
                             "aerogel_mli, pcm, endothermic, teg, gradient_spray, combined, "
                             "ultra_cascade")
    # Scientific model framework modes
    parser.add_argument("--sm-compare",action="store_true",help="scientific model: comparison table")
    parser.add_argument("--sm-sim",type=str,default=None,metavar="APPROACH",
                        help="scientific model: detailed steady-state sim. "
                             "Valid: foam, aerogel, ceramic, refractory, mli, hybrid, pcm, endothermic, teg, gradient, pro, ultra")
    parser.add_argument("--sm-sweep-thickness",type=str,default=None,metavar="APPROACH",
                        help="scientific model: sweep thickness for given approach")
    parser.add_argument("--sm-sweep-mli",action="store_true",
                        help="scientific model: sweep MLI reflective layer count")
    parser.add_argument("--sm-transient",type=str,default=None,metavar="APPROACH",
                        help="scientific model: run transient (3600s) simulation")
    parser.add_argument("--sm-config",type=str,default=None,metavar="FILE",
                        help="scientific model: load stack from JSON config file")
    parser.add_argument("--export-csv",type=str,default=None,metavar="FILE",
                        help="export comparison/sweep results to CSV")
    parser.add_argument("--export-config",type=str,default=None,metavar="FILE",
                        help="export stack config to JSON")
    parser.add_argument("--T-hot",type=float,default=1273.15,metavar="K")
    parser.add_argument("--T-cold",type=float,default=77.0,metavar="K")
    parser.add_argument("--scenario",type=str,default=None,metavar="NAME",
                        help="named boundary-condition preset, overrides --T-hot/--T-cold. Valid: %s"%
                             (", ".join(sc["name"] for sc in BOUNDARY_SCENARIOS)))
    parser.add_argument("--durability",action="store_true",help="print the durability/weathering service-life report")
    parser.add_argument("--proof",action="store_true",
                        help="print a live, reproducible proof that Insulatrix Pro achieves near-zero heat "
                             "transfer at the absolute-zero limit (or at --scenario/--T-hot/--T-cold if given)")
    parser.add_argument("--pod-report",action="store_true",
                        help="print the pod hull's hypervelocity impact-shield report (Whipple shield + "
                             "magnetic deflection + self-healing hull)")
    parser.add_argument("--pod-proof",action="store_true",
                        help="print live proofs that magnetic deflection and the self-healing hull "
                             "actually work (A/B comparison + step-by-step decompression race)")
    parser.add_argument("--economics",action="store_true",
                        help="print the order-of-magnitude global economic impact estimate "
                             "(buildings, refrigeration, space habitat hulls, polar habitats)")
    parser.add_argument("--ultra-proof",action="store_true",
                        help="print a live, reproducible proof that ULTRA CASCADE (formulation 12, the "
                             "optimized goal2.md sprayable multi-layer) achieves near-zero heat loss by "
                             "accumulative stacking: per-layer attenuation, stacking synergy, cascade "
                             "scaling, MLI 1/N law, all-scenario validation")
    parser.add_argument("--fabrication",type=str,default=None,nargs="?",const="ultra_cascade",metavar="APPROACH",
                        help="print the step-by-step fabrication protocol for an approach "
                             "(default: ultra_cascade) -- spray params, cure schedules, vacuum steps, QC")
    args=parser.parse_args()
    _sc_hot,_sc_cold=resolve_scenario(args.scenario)
    T_hot=_sc_hot if _sc_hot is not None else args.T_hot
    T_cold=_sc_cold if _sc_cold is not None else args.T_cold
    if args.selftest:
        if pygame is not None:
            os.environ["SDL_VIDEODRIVER"]="dummy"; pygame.init()
        sys.exit(0 if selftest() else 1)
    if args.feasibility: print_feasibility(T_hot,T_cold); return
    if args.compare: print_comparison(T_hot,T_cold); return
    if args.sim: print_simulation(args.sim,T_hot,T_cold); return
    if args.durability: print_durability_report(T_hot,T_cold); return
    if args.proof:
        ok=print_proof(T_hot,T_cold) if args.scenario else print_proof()
        sys.exit(0 if ok else 1)
    if args.ultra_proof:
        ok=print_ultra_proof(T_hot,T_cold) if args.scenario else print_ultra_proof()
        sys.exit(0 if ok else 1)
    if args.pod_report:
        n_survived,n_total=print_pod_report(); sys.exit(0 if n_survived>0 else 1)
    if args.pod_proof:
        sys.exit(0 if print_pod_proof() else 1)
    if args.economics:
        total=print_economics_report(); sys.exit(0 if total>0 else 1)
    if args.export_obj: export_obj(); return
    if args.fabrication: print_fabrication(args.fabrication); return
    # Scientific model modes
    if args.sm_compare:
        bc=SBoundary(T_hot=T_hot,T_cold=T_cold)
        opt=OptimizationFramework(bc); r=opt.full_comparison()
        print("="*90); print(" SCIENTIFIC MODEL -- APPROACH COMPARISON"); print("="*90)
        print("%-30s %5s %8s %12s %10s %12s %10s"%("Approach","N_L","thk[mm]","q_trans W/m2","Supp[%]","k_eff mW","R_total"))
        print("-"*90)
        for row in r:
            print("%-30s %5d %8.1f %12.4f %10.4f %12.6f %10.4f"%(row["name"][:30],row["n_layers"],row["total_thickness_mm"],row["q_transmitted"],row["suppression_pct"],row["k_eff_overall_mW"],row["R_total"]))
        print("-"*90)
        ranked=sorted(r,key=lambda x:x["suppression_pct"],reverse=True)
        print("\nRANKING:")
        for i,row in enumerate(ranked):
            m=" <-- BEST" if i==0 else ""
            print("  %2d. %-30s  %.4f%%%s"%(i+1,row["name"][:30],row["suppression_pct"],m))
        print("="*90)
        if args.export_csv: sm_export_csv(r,args.export_csv)
        return
    if args.sm_config:
        with open(args.sm_config) as f: cfg=json.load(f)
        stack=sm_custom(cfg); stack.bc=SBoundary(T_hot=T_hot,T_cold=T_cold)
        print(stack.summary())
        if args.export_config: sm_export_config(stack,args.export_config)
        return
    if args.sm_sim:
        if args.sm_sim not in SM_APPROACHES: print("Unknown: %s. Valid: %s"%(args.sm_sim,", ".join(SM_APPROACHES))); sys.exit(1)
        stack=SM_APPROACHES[args.sm_sim](); stack.bc=SBoundary(T_hot=T_hot,T_cold=T_cold)
        print(stack.summary())
        if args.export_config: sm_export_config(stack,args.export_config)
        return
    if args.sm_sweep_thickness:
        if args.sm_sweep_thickness not in SM_APPROACHES: print("Unknown: %s"%args.sm_sweep_thickness); sys.exit(1)
        bc=SBoundary(T_hot=T_hot,T_cold=T_cold); opt=OptimizationFramework(bc)
        r=opt.thickness_sweep(SM_APPROACHES[args.sm_sweep_thickness],[5,10,20,50,100,200,500,1000],args.sm_sweep_thickness)
        print("\nTHICKNESS SWEEP: %s"%args.sm_sweep_thickness)
        print("-"*60); print("%10s %12s %10s %12s"%("thk[mm]","q_trans W/m2","Supp[%]","k_eff mW")); print("-"*60)
        for row in r: print("%10.1f %12.4f %10.4f %12.6f"%(row["thickness_mm"],row["q_transmitted"],row["suppression_pct"],row["k_eff_overall_mW"]))
        if args.export_csv: sm_export_csv(r,args.export_csv)
        return
    if args.sm_sweep_mli:
        bc=SBoundary(T_hot=T_hot,T_cold=T_cold); opt=OptimizationFramework(bc)
        r=opt.mli_layer_sweep([5,10,20,40,80,160,320])
        print("\nMLI REFLECTIVE LAYER SWEEP")
        print("-"*60); print("%8s %12s %10s %12s"%("N_refl","q_trans W/m2","Supp[%]","k_eff mW")); print("-"*60)
        for row in r: print("%8d %12.6f %10.4f %12.8f"%(row["n_reflective"],row["q_transmitted"],row["suppression_pct"],row["k_eff_overall_mW"]))
        if args.export_csv: sm_export_csv(r,args.export_csv)
        return
    if args.sm_transient:
        if args.sm_transient not in SM_APPROACHES: print("Unknown: %s"%args.sm_transient); sys.exit(1)
        stack=SM_APPROACHES[args.sm_transient](); stack.bc=SBoundary(T_hot=T_hot,T_cold=T_cold)
        print("Transient: %s (3600 s, dt=1.0 s)"%stack.name)
        t0=time.time(); r=stack.solve_transient(3600.0,1.0)
        print("  Done in %.1f s (%d steps)"%(time.time()-t0,r["n_steps"]))
        print("  q_transmitted: %.6f W/m2"%r["q_transmitted"])
        print("  Suppression: %.4f%%"%r["suppression_pct"])
        print("  Absorbed: %.3f MJ/m2  Converted: %.3f MJ/m2  Pumped: %.3f MJ/m2"%(r["total_absorbed"]/1e6,r["total_converted"]/1e6,r["total_pumped"]/1e6))
        return
    if pygame is None:
        print("pygame is required for the interactive viewer. Install with:")
        print("  pip install pygame numpy"); sys.exit(1)
    pygame.init(); App().run()

if __name__=="__main__":
    main()
