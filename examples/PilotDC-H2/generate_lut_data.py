"""Generate H2 plasma LUT data files for plasmaArc solver.

Computes LTE thermodynamic, transport and radiation properties for a
pure H2 atmosphere from 1500 K to 25000 K at 100 K intervals using minplascalc.
"""

import json
import sys
import numpy as np
import minplascalc as mpc
from pathlib import Path

# ── Physical constants ────────────────────────────────────────────────────────
eV   = 1.602176634e-19   # J per eV
h_   = 6.62607015e-34    # J·s
c_   = 2.99792458e8      # m/s
hc_  = h_ * c_           # J·m
hcm  = h_ * 2.99792458e10  # J·cm  (h × c in cm/s units, for cm-1 conversion)

# ── Temperature grid ─────────────────────────────────────────────────────────
T_START = 1500.0
T_DELTA = 100.0
N_PTS   = 236            # 1500 K to 25000 K inclusive
TEMPS   = [T_START + i * T_DELTA for i in range(N_PTS)]

P_REF   = 101325.0       # Pa

# ── Species data ─────────────────────────────────────────────────────────────
# H2 — Diatomic molecule
H2_JSON = {
    "name": "H2",
    "stoichiometry": {"H": 2},
    "molar_mass": 2.01565006414e-3,        # kg/mol  (2 × m_H)
    "charge_number": 0,
    "ionisation_energy": 15.42593 * eV,    # J
    "dissociation_energy": 4.47803 * eV,   # J  (D0, NIST)
    "sigma_s": 2,                          # homonuclear
    "g0": 1,
    "w_e": 4401.21316 * hcm,              # J  (4401.21 cm-1 → J)
    "b_e": 60.853 * hcm,                  # J  (60.853 cm-1 → J)
    "polarisability": 8.02e-31,            # m^3  (0.802 Å^3, NIST)
    "multiplicity": 1,
    "effective_electrons": 2.0,
    # constant momentum-transfer cross section for e-H2 scattering (~1 eV)
    "electron_cross_section": 1.0e-19,
    "emission_lines": [],                  # H2 molecular bands not included
    "sources": [
        "NIST Chemistry WebBook, https://webbook.nist.gov/chemistry/",
        "NIST CCCBDB, https://cccbdb.nist.gov/"
    ]
}

# H — Monatomic atom
# Energy levels: n=1..10, each lump as J_eff = n^2 - 0.5 → g = 2n^2
# E_n = 13.59844*(1 - 1/n^2) eV  (measured from ground state)
IE_H = 13.59844 * eV
H_LEVELS = [[float(n**2 - 0.5), 13.59844 * (1.0 - 1.0/n**2) * eV]
             for n in range(1, 11)]

# Key emission lines: [wavelength (m), g×A (s-1), E_upper (J)]
# Lyman series (n→1) and Balmer series (n→2)
H_LINES = [
    [121.567e-9,  3.759e9, 10.199 * eV],   # Ly-α  2→1
    [102.572e-9,  3.345e8, 12.088 * eV],   # Ly-β  3→1
    [ 97.254e-9,  7.668e7, 12.749 * eV],   # Ly-γ  4→1
    [ 95.005e-9,  2.036e7, 13.055 * eV],   # Ly-δ  5→1
    [ 93.782e-9,  6.492e6, 13.221 * eV],   # Ly-ε  6→1
    [656.279e-9,  1.025e9, 12.088 * eV],   # Hα    3→2
    [486.133e-9,  4.075e8, 12.749 * eV],   # Hβ    4→2
    [434.047e-9,  1.978e8, 13.055 * eV],   # Hγ    5→2
    [410.174e-9,  9.732e7, 13.221 * eV],   # Hδ    6→2
    [397.007e-9,  5.334e7, 13.321 * eV],   # Hε    7→2
   [1875.10e-9,   3.334e8, 12.749 * eV],   # Pa-α  4→3
   [1281.81e-9,   1.978e8, 13.055 * eV],   # Pa-β  5→3
]

H_JSON = {
    "name": "H",
    "stoichiometry": {"H": 1},
    "molar_mass": 1.00782503207e-3,        # kg/mol
    "charge_number": 0,
    "ionisation_energy": IE_H,
    "energy_levels": H_LEVELS,
    "polarisability": 6.67e-31,            # m^3  (0.667 Å^3, NIST)
    "multiplicity": 2,                     # doublet 2S1/2
    "effective_electrons": 1.0,
    "electron_cross_section": 6.5e-20,     # m^2  (e-H momentum transfer CS)
    "emission_lines": H_LINES,
    "sources": [
        "NIST Chemistry WebBook, https://webbook.nist.gov/chemistry/",
        "NIST ASD, https://physics.nist.gov/asd"
    ]
}

# H+ — Bare proton, no electronic structure
HP_JSON = {
    "name": "H+",
    "stoichiometry": {"H": 1},
    "molar_mass": 1.00727646677e-3,        # kg/mol  (proton mass)
    "charge_number": 1,
    "ionisation_energy": 54.418 * eV,      # J  (H^2+ threshold, irrelevant here)
    "energy_levels": [[0.0, 0.0]],         # single ground state
    "polarisability": 1.0e-41,             # negligible
    "multiplicity": 1,
    "effective_electrons": None,
    "electron_cross_section": None,
    "emission_lines": [],
    "sources": ["NIST Chemistry WebBook, https://webbook.nist.gov/chemistry/"]
}


def build_mixture(T: float) -> mpc.mixture.LTE:
    sp_dir = Path(__file__).parent / "species_data"
    sp_h2 = mpc.species.from_file(sp_dir / "H2.json")
    sp_h  = mpc.species.from_file(sp_dir / "H.json")
    sp_hp = mpc.species.from_file(sp_dir / "H+.json")
    return mpc.mixture.LTE(
        species=[sp_h2, sp_h, sp_hp],
        x0=[1.0, 0.0, 0.0],
        T=T, P=P_REF,
        gfe_initial_particles=1e20,
        gfe_rtol=1e-8,
        gfe_max_iter=2000,
    )


def write_foam_table(path: Path, keyword: str, start_t: float, delta_t: float,
                     data: list):
    lines = [
        f"{keyword}",
        "{",
        f"    startT {start_t:.0f};",
        f"    deltaT {delta_t:.0f};",
        "    dataTable",
        "    (",
    ]
    for v in data:
        lines.append(f"        {float(v)!r}")
    lines += ["    );", "}"]
    path.write_text("\n".join(lines) + "\n")


def write_species_foam(path: Path, species_names: list, start_t: float,
                        delta_t: float, xdata: dict):
    """Write dataSpecies.foam with one block per species."""
    lines = [
        "species",
        "(",
    ]
    for sp in species_names:
        lines.append(f"    {sp}")
    lines.append(");")

    for sp in species_names:
        lines += [
            f"{sp}",
            "{",
            f"    startT {start_t:.0f};",
            f"    deltaT {delta_t:.0f};",
            "    dataTable",
            "    (",
        ]
        for v in xdata[sp]:
            lines.append(f"        {float(v)!r}")
        lines += ["    );", "}"]

    path.write_text("\n".join(lines) + "\n")


def main():
    out_dir = Path(__file__).parent / "constant" / "dataH2"
    sp_dir  = Path(__file__).parent / "species_data"
    sp_dir.mkdir(exist_ok=True)

    # Write species JSON files
    print("Writing species data files...")
    (sp_dir / "H2.json").write_text(json.dumps(H2_JSON, indent=2))
    (sp_dir / "H.json" ).write_text(json.dumps(H_JSON,  indent=2))
    (sp_dir / "H+.json").write_text(json.dumps(HP_JSON, indent=2))

    # Accumulate LUT data
    rho_data  = []
    cp_data   = []
    mu_data   = []
    tk_data   = []
    ek_data   = []
    em_data   = []   # emission = absorption (net emission coefficient, W/m³/sr)
    sp_x      = {"H2": [], "H": [], "H+": [], "e": []}

    print(f"Computing LTE properties for {N_PTS} temperature points...")
    print(f"  T range: {T_START:.0f} – {T_START + (N_PTS-1)*T_DELTA:.0f} K")

    prev_mix = None
    for i, T in enumerate(TEMPS):
        if i % 20 == 0:
            print(f"  T = {T:.0f} K  ({i+1}/{N_PTS})", flush=True)

        try:
            mix = build_mixture(T)
            mix.calculate_composition()   # trigger LTE solve
        except Exception as exc:
            print(f"  WARNING: LTE solve failed at T={T:.0f} K: {exc}", file=sys.stderr)
            # carry forward last good values
            if i == 0:
                raise
            rho_data.append(rho_data[-1])
            cp_data.append(cp_data[-1])
            mu_data.append(mu_data[-1])
            tk_data.append(tk_data[-1])
            ek_data.append(ek_data[-1])
            em_data.append(em_data[-1])
            for sp in sp_x:
                sp_x[sp].append(sp_x[sp][-1])
            continue

        rho_data.append(mix.calculate_density())
        cp_data.append(mix.calculate_heat_capacity())
        mu_data.append(mix.calculate_viscosity())
        tk_data.append(mix.calculate_thermal_conductivity())
        ek_data.append(mix.calculate_electrical_conductivity())
        em_val = mix.calculate_total_emission_coefficient()
        em_data.append(em_val)

        # Species mole fractions
        nd = mix.calculate_composition()   # number density [m-3] for [H2, H, H+, e]
        nd_total = sum(nd)
        names = ["H2", "H", "H+", "e"]
        for j, sp in enumerate(names):
            sp_x[sp].append(nd[j] / nd_total if nd_total > 0 else 0.0)

    # Write .foam files
    print("Writing .foam data files...")

    write_foam_table(out_dir / "dataRho.foam",
                     "rhoLookupTable", T_START, T_DELTA, rho_data)
    write_foam_table(out_dir / "dataCp.foam",
                     "CpLookupTable",  T_START, T_DELTA, cp_data)
    write_foam_table(out_dir / "dataMu.foam",
                     "muLookupTable",  T_START, T_DELTA, mu_data)
    write_foam_table(out_dir / "dataTk.foam",
                     "tkLookupTable",  T_START, T_DELTA, tk_data)
    write_foam_table(out_dir / "dataEk.foam",
                     "ekLookupTable",  T_START, T_DELTA, ek_data)
    write_foam_table(out_dir / "dataAbsorptionCoeffs.foam",
                     "absorptionCoeffs", T_START, T_DELTA, em_data)
    write_foam_table(out_dir / "dataEmissionCoeffs.foam",
                     "emissionCoeffs",   T_START, T_DELTA, em_data)

    write_species_foam(out_dir / "dataSpecies.foam",
                       ["H2", "H", "H+", "e"], T_START, T_DELTA, sp_x)

    print(f"Done. Files written to: {out_dir}")
    print("\nSample values at 10000 K:")
    idx = int((10000 - T_START) / T_DELTA)
    print(f"  rho  = {rho_data[idx]:.6e} kg/m3")
    print(f"  Cp   = {cp_data[idx]:.6e} J/(kg·K)")
    print(f"  mu   = {mu_data[idx]:.6e} Pa·s")
    print(f"  k    = {tk_data[idx]:.6e} W/(m·K)")
    print(f"  sigma= {ek_data[idx]:.6e} S/m")
    print(f"  emit = {em_data[idx]:.6e} W/(m3·sr)")


if __name__ == "__main__":
    main()
