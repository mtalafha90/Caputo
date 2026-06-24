# Caputo Fractional Surface Flux Transport (SFT)

A 1-D, axisymmetric **Surface Flux Transport** model of the solar photospheric
magnetic field in which the time derivative is replaced by a **Caputo
fractional derivative** of order `q ∈ (0, 1]`. The fractional order introduces
*memory*: the field at time `t` depends on its entire history, weighted so that
the recent past matters most. For `q = 1` the model reduces exactly to the
classical (integer-order) SFT equation.

The model is solved in the **annular-flux `W`-formulation**

```
W(θ, t) = R⊙ · sin θ · B(θ, t)
```

which avoids the coordinate singularity of `B` at the poles and matches the
structure of the classical `transp.py` SFT integrator.

## Governing equation

```
  C D_t^q W  =  F_W[W]  +  S(θ, t) · R⊙ · sin θ
```

where the transport operator `F_W` contains

- **advection** by the meridional flow `u(θ)` (van Leer TVD or upwind),
- **supergranular diffusion** with diffusivity `η`,
- an optional **radial-decay** term `−W/τ`,

and `S` is an active-region emergence source (Hathaway-type cycle envelope,
Jiang-type latitude drift, cycle-parity reversal, tilt quenching, and a
flux-balance correction).

The Caputo derivative is discretised with the **L1 scheme**

```
  C D_t^q W(t_n) ≈ c_q · Σ_{k=0}^{n-1} a_k (W_{n-k} − W_{n-k-1}),
  a_k = (k+1)^{1-q} − k^{1-q},   c_q = 1 / (Δt^q · Γ(2−q)),
```

with optional **short-memory truncation** to cap the per-step cost.

## Repository contents

| File | Description |
|------|-------------|
| `1D_W_caputo_fixed.py` | The solver, source model, and a command-line example that writes a butterfly diagram. |
| `verify_advection.py`  | Verifies the van Leer advection scheme (TVD test + grid-convergence of the dipole and field). Writes `fig4_advection_verification.png`. |
| `memory_diagnostics.py`| Visualises the fractional memory (butterflies vs `q`, polar-field lag, memory-term magnitude, Mittag-Leffler decay tails, L1 weights). Writes `fig1`–`fig3`. |
| `*.png` | Pre-generated example figures. |

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. `scipy` is only needed for the `flowtype=3` flow profile.

## Usage

Run the default example (33 years, `q = 0.99`, 181 latitudes):

```bash
python3 1D_W_caputo_fixed.py
```

The CLI exposes the main parameters (defaults reproduce the original example):

```bash
# Strongly fractional, shorter run, coarser grid
python3 1D_W_caputo_fixed.py --q 0.7 --years 20 --n-theta 91 --out butterfly_q07.png

# Classical SFT (q = 1) with first-order upwind advection
python3 1D_W_caputo_fixed.py --q 1.0 --advection-scheme upwind

# Cap the Caputo memory at 2000 steps to speed up long fractional runs
python3 1D_W_caputo_fixed.py --q 0.8 --short-memory 2000

python3 1D_W_caputo_fixed.py --help   # full option list
```

Reproduce the diagnostic and verification figures:

```bash
python3 verify_advection.py      # -> fig4_advection_verification.png
python3 memory_diagnostics.py    # -> fig1..fig3  (heavy: full-memory L1 runs)
```

The solver functions are also importable for use in your own scripts (the
module file name starts with a digit, so load it via `importlib`):

```python
import importlib.util
spec = importlib.util.spec_from_file_location("sft", "1D_W_caputo_fixed.py")
sft = importlib.util.module_from_spec(spec); spec.loader.exec_module(sft)

theta, t, B = sft.run_fractional_sft_1d_W(q=0.8, n_theta=181, dt=86400.0,
                                          n_steps=1000, R=6.96e8, eta=600e3,
                                          u0=10.0, tau=None, short_memory_M=None)
```

## Key parameters

| Parameter | CLI flag | Default | Meaning |
|-----------|----------|---------|---------|
| `q`               | `--q`                | 0.99   | Caputo fractional order in `(0, 1]` |
| `n_theta`         | `--n-theta`          | 181    | colatitude grid points (≥ 5) |
| `dt`              | `--dt-days`          | 0.5 d  | time step |
| run length        | `--years`            | 33     | simulated time |
| `η`               | `--eta`              | 600 km²/s | supergranular diffusivity |
| `u0`              | `--u0`               | 10 m/s | peak meridional flow speed |
| `τ`               | `--tau-years`        | 10     | radial-decay timescale (`≤0` disables) |
| source amplitude  | `--source-strength`  | 0.02   | emergence-source scaling |
| flow profile      | `--flowtype`         | 2      | meridional flow selector (1–5) |
| cycle period      | `--cycle-years`      | 6      | activity cycle length |
| memory truncation | `--short-memory`     | none   | L1 short-memory length (steps) |
| advection         | `--advection-scheme` | vanleer| `vanleer` (TVD) or `upwind` |

## Numerical notes

- **Advection** uses a second-order MUSCL reconstruction with van Leer
  (harmonic-mean) limited slopes; it is TVD and retains peaks far better than
  first-order upwind, which carries numerical diffusion `~|u|·R·Δθ/2`.
  `verify_advection.py` checks translation speed, the TVD property, and
  grid convergence.
- **Cost**: with full memory the L1 scheme is `O(N²)` in the number of time
  steps. Use `--short-memory` for long fractional runs.
- The global monopole is removed from `B` every step to prevent spurious net
  flux accumulation.

## Units

SI-like throughout: `R` [m], `u` [m/s], `η` [m²/s], `τ`, `t`, `dt` [s],
`B` [G, arbitrary scaling], `W = R sin θ · B` [m·G].
