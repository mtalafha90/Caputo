#!/usr/bin/env python3
"""
Verification of the van Leer advection scheme in 1D_W_caputo_fixed.py.

Test A (TVD, uniform flow): advect a Gaussian with constant u; the exact
    solution is pure translation. Checks: no new extrema (TVD), correct
    propagation speed, peak retention vs first-order upwind.
    (Note: a non-uniform decelerating flow piles flux up, so amplitude
    growth there is physical compression, NOT instability - do not use
    the solar flow profile for a TVD check.)

Test B (integrated-metric convergence): axial dipole moment D(t) of the
    full q=1 model on 181/361/721 grids, both schemes.

Test C (field-level convergence): relative L2 error of the whole
    butterfly B(theta, t) against the vanleer@721 reference.

Findings encoded below (also printed):
  - both schemes are stable; vanleer is TVD and retains peaks far better;
  - integrated metrics (dipole, polar field) converge to <1.5% even with
    coarse upwind, because numerical diffusion ~|u| vanishes at the equator
    and poles where those metrics are decided;
  - field morphology (surge widths/shapes) is where the scheme matters:
    upwind@181 carries ~30% field-level error, vanleer@361 ~7%;
  - the diffusivity is the physical eta = 600 km2/s (6e8 m2/s); an earlier
    1000x-too-small value (600e3 m2/s = 0.6 km2/s) let the poleward flux
    pile up into a thin ring near 75 deg instead of diffusing into a smooth
    polar cap (the "polar field" pile-up artifact).
"""
import os
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load the solver module that lives next to this script. Its filename starts
# with a digit, so it cannot be imported with a normal `import` statement.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SFT_PATH = os.path.join(_HERE, "1D_W_caputo_fixed.py")
spec = importlib.util.spec_from_file_location("sft", _SFT_PATH)
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)

DAY = 86400.0
YEAR = 365.25 * DAY
R = 6.96e8


# ----------------------------------------------------------------------
# Test A: TVD with uniform velocity
# ----------------------------------------------------------------------
def test_A_uniform_advection(n=181, n_steps=1500, dt=43200.0, u=5.0):
    """Advect a Gaussian with constant velocity; exact solution = translation."""
    print(f"=== Test A: uniform advection (u = {u} m/s, {n_steps} steps) ===")
    theta = np.linspace(0, np.pi, n)
    dth = theta[1] - theta[0]
    lat = 90 - np.degrees(theta)
    u_const = np.full(n, u)
    W0 = np.exp(-((lat + 30) / 4) ** 2)
    S = np.zeros(n)
    profiles = {"initial": W0.copy()}
    for scheme in ("upwind", "vanleer"):
        W = W0.copy(); W[0] = W[-1] = 0
        for _ in range(n_steps):
            rhs, _ = sft.w_transport_rhs(W=W, theta=theta, dtheta=dth, R=R, eta=0.0,
                                         u_theta=u_const, tau=None, source_g_per_s=S,
                                         advection_scheme=scheme, dt_eff=dt)
            W = W + dt * rhs; W[0] = W[-1] = 0
        profiles[scheme] = W
        tvd_ok = W.max() <= W0.max() + 1e-12 and W.min() >= -1e-12
        print(f"{scheme:8s}: peak retained = {W.max():.3f}, "
              f"peak lat = {lat[np.argmax(W)]:+.1f} deg, TVD = {tvd_ok}")
    dlat_exact = np.degrees(u * n_steps * dt / R)
    print(f"exact: peak = 1.000, peak lat = {-30 + dlat_exact:+.1f} deg")
    return lat, profiles, dlat_exact


# ----------------------------------------------------------------------
# Full-model runs for Tests B and C
# ----------------------------------------------------------------------
def stable_dt(n_theta, eta, safety=0.4, dt_cap=DAY):
    """Explicit-diffusion CFL: dt < (R*dtheta)^2 / (2*eta) (binds at the
    equator where sin(theta)=1). With the physical eta this constrains fine
    grids, so the convergence runs must shrink dt with resolution."""
    dtheta = np.pi / (n_theta - 1)
    return min(dt_cap, safety * (R * dtheta) ** 2 / (2.0 * eta))


def full_run(n_theta, scheme, eta=600e6, years=18.0):  # eta = 600 km^2/s
    th0 = np.linspace(0.0, np.pi, n_theta)
    src = sft.TranspSource1D(
        latitude_deg=90.0 - np.rad2deg(th0),
        cycleper_days=11.0 * 365.25, flowtype=2,
        tau_seconds=10.0 * YEAR, blat=0.0, bjoy=0.0,
        seed=1, source_strength=0.02)
    dt = stable_dt(n_theta, eta)
    # Store on a fixed ~4-day cadence so runs with different dt remain
    # comparable; the field error below resamples onto the reference times.
    store_every = max(1, int(round(4.0 * DAY / dt)))
    return sft.run_fractional_sft_1d_W(
        q=1.0, n_theta=n_theta, dt=dt, n_steps=int(years * YEAR / dt),
        R=R, eta=eta, u0=10.0, tau=10.0 * YEAR, short_memory_M=None,
        source_model=src, store_every=store_every, flowtype=2,
        advection_scheme=scheme)


def tests_BC_convergence(cases, years=18.0):
    """Full-model dipole and field-level convergence vs the vanleer@721 reference."""
    print("\n=== Tests B & C: full-model convergence, q=1, %.0f yr ===" % years)
    runs = {}
    for label, (nn, sc) in cases.items():
        print("running", label, "...", flush=True)
        runs[label] = full_run(nn, sc, years=years)

    thR, tR, BR = runs["vanleer 721"]
    DR = np.array([sft.axial_dipole_moment(thR, b) for b in BR])

    def resample(th, t, Bs):
        """Interpolate a run's butterfly onto the reference (tR, thR) grid,
        in space then time, so runs with different dt/snapshot times compare."""
        Bsp = np.array([np.interp(thR, th, b) for b in Bs])      # (nt, nthR)
        return np.array([np.interp(tR, t, Bsp[:, j])
                         for j in range(thR.size)]).T            # (ntR, nthR)

    dip_err, fld_err, dipoles = {}, {}, {}
    for label in cases:
        th, t, Bs = runs[label]
        D = np.array([sft.axial_dipole_moment(th, b) for b in Bs])
        dipoles[label] = (t / YEAR, D)
        Bg = resample(th, t, Bs)
        fld_err[label] = np.linalg.norm(Bg - BR) / np.linalg.norm(BR)
        dip_err[label] = np.max(np.abs(np.interp(tR, t, D) - DR)) / np.max(np.abs(DR))

    print("\nerrors vs vanleer@721 reference:")
    for label in cases:
        print(f"  {label:12s}: dipole {dip_err[label]:6.3f}   field L2 {fld_err[label]:6.3f}")
    return dipoles, fld_err


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def make_figure(lat, profiles, dlat_exact, dipoles, fld_err,
                outfile="fig4_advection_verification.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    ax = axes[0]
    ax.plot(lat, profiles["initial"], "k:", label="initial")
    ax.axvline(-30 + dlat_exact, color="gray", lw=0.8, label="exact peak position")
    ax.plot(lat, profiles["upwind"], color="tab:red", label="upwind, 1500 steps")
    ax.plot(lat, profiles["vanleer"], color="tab:blue", label="van Leer, 1500 steps")
    ax.set_xlim(-45, 30)
    ax.set_xlabel("Latitude (deg)"); ax.set_ylabel("W (normalized)")
    ax.set_title("A: uniform advection (exact = translation)")
    ax.legend(fontsize=8)

    ax = axes[1]
    styles = {"upwind 181": dict(color="tab:red", ls=":"),
              "upwind 721": dict(color="tab:red", ls="-"),
              "vanleer 181": dict(color="tab:blue", ls=":"),
              "vanleer 721": dict(color="tab:blue", ls="-", lw=2)}
    for label, st in styles.items():
        if label not in dipoles:
            continue
        t, D = dipoles[label]
        ax.plot(t, D, label=label, **st)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Axial dipole D(t)")
    ax.set_title("B: integrated metric converges on all variants")
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = [l for l in ["upwind 181", "vanleer 181", "vanleer 361", "upwind 721"]
              if l in fld_err]
    vals = [fld_err[l] for l in labels]
    colors = ["tab:red", "tab:blue", "tab:blue", "tab:red"][:len(labels)]
    ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel("relative L2 error of B(lat, t)")
    ax.set_title("C: field morphology vs vanleer@721")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)
    print(f"\nsaved {outfile}")


def main():
    lat, profiles, dlat_exact = test_A_uniform_advection()
    cases = {"upwind 181": (181, "upwind"), "upwind 721": (721, "upwind"),
             "vanleer 181": (181, "vanleer"), "vanleer 361": (361, "vanleer"),
             "vanleer 721": (721, "vanleer")}
    dipoles, fld_err = tests_BC_convergence(cases)
    make_figure(lat, profiles, dlat_exact, dipoles, fld_err)


if __name__ == "__main__":
    main()
