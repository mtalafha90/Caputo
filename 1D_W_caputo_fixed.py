#!/usr/bin/env python3
"""
Time-fractional (Caputo) 1D axisymmetric Surface Flux Transport (SFT)
in the original annular-flux W formulation used by transp.py.

Key change compared with the direct-B version:
    W(theta,t) = R_sun sin(theta) B(theta,t)

The Caputo derivative is applied to W:
    C D_t^q W = F_W[W] + S(theta,t) R_sun sin(theta)

For q=1, this reduces to the original explicit W update structure:
    W^{n+1} = W^n + dt * F_W[W^n]

Numerics:
- Caputo derivative: L1 scheme with optional short-memory truncation
- Transport operator: follows the structure of transp.py
- Source: adapted from transp.py
- Butterfly diagram: latitude vs time

Units here are SI-like:
- R [m]
- u [m/s]
- eta [m^2/s]
- tau [s]
- t, dt [s]
- B [G, arbitrary scaling]
- W = R sin(theta) B [m G]
"""

from __future__ import annotations
import math
import argparse
import numpy as np

# NumPy 1.x/2.x compatibility: np.trapz was renamed to np.trapezoid
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Meridional flow profiles, adapted from transp.py
# -----------------------------------------------------------------------------
def meridional_flow_profile(
    theta: np.ndarray,
    u0: float,
    flowtype: int = 2,
    smooth_taper: bool = True,
    latitude0: float = 65.0,
    taper_width: float = 20.0,
) -> np.ndarray:

    latitude = 90.0 - np.degrees(theta)
    abs_lat = np.abs(latitude)

    if flowtype == 1:
        uc = u0 * np.sin(np.pi * latitude / 90.0)

    elif flowtype == 2:
        uc = u0 * np.sin(2.4 * latitude * np.pi / 180.0)

    elif flowtype == 3:
        from scipy import special
        latitude_ref = 89.0
        V = 7.0
        W = 1.0
        uc = u0 * special.erf(
            V * np.cos(np.pi / 2.0 * latitude / latitude_ref)
        ) * special.erf(
            W * np.sin(np.pi / 2.0 * latitude / latitude_ref)
        )

    elif flowtype == 4:
        p = 3.24
        uc = u0 * (np.sin(theta) ** p) * np.cos(theta)

    elif flowtype == 5:
        uc = u0 * np.tanh(np.pi / 2.0 * latitude / 6.0) * (
            np.cos(np.pi / 2.0 * latitude / 90.0)
        ) ** 2

    else:
        raise ValueError("flowtype must be 1, 2, 3, 4, or 5")

    if smooth_taper:
        taper = 0.5 * (1.0 - np.tanh((abs_lat - latitude0) / taper_width))
        uc *= taper
    else:
        uc[abs_lat > latitude0] = 0.0

    uc[0] = 0.0
    uc[-1] = 0.0

    return uc

def remove_monopole_from_W(W: np.ndarray, theta: np.ndarray, R: float) -> np.ndarray:
    """
    Remove the global magnetic monopole from B and rebuild W.
    This prevents artificial accumulation of net flux.
    """
    B = recover_B_from_W(W, theta, R)
    s = np.sin(theta)

    mean_B = _trapz(B * s, theta) / _trapz(s, theta)

    Bcorr = B - mean_B
    Wcorr = R * s * Bcorr

    Wcorr[0] = 0.0
    Wcorr[-1] = 0.0

    return Wcorr
# -----------------------------------------------------------------------------
# Source adapted from transp.py
# -----------------------------------------------------------------------------
class TranspSource1D:
    """
    Axisymmetric source adapted from transp.py.

    Main ingredients retained:
    - Hathaway et al.-type cycle temporal profile
    - Jiang et al.-type emergence latitude drift
    - cycle parity reversal
    - Joy/tilt quenching through bjoy
    - flux correction so one polarity ring is corrected for near-zero net flux

    This returns S in G/s, so the W-equation source term is:
        S(theta,t) * R_sun * sin(theta)
    """

    def __init__(
        self,
        latitude_deg: np.ndarray,
        *,
        cycleper_days: float = 11.0 * 365.25,
        flowtype: int = 2,
        tau_seconds: float | None = None,
        blat: float = 0.0,
        bjoy: float = 0.0,
        seed: int = 1,
        source_strength: float = 1.0,
    ):
        self.latitude = np.array(latitude_deg, dtype=float)
        self.cycleper_days = float(cycleper_days)
        self.flowtype = int(flowtype)
        self.tau_days = (tau_seconds / 86400.0) if tau_seconds not in (None, 0.0) else 1000.0 * 365.25
        self.blat = float(blat)
        self.bjoy = float(bjoy)
        self.source_strength = float(source_strength)
        self.rng = np.random.default_rng(seed)

        self.ahat = 0.00185
        self.bhat = 48.7
        self.chat = 0.71

        self.current_cycle = None
        self.sourcescale = None
        self.sourcescale1 = None

    def _new_cycle_sourcescale(self) -> None:
        self.sourcescale1 = 0.0015 * np.exp(7.0 / self.tau_days * 365.25)
        sigma = 0.13
        gaussian = self.rng.normal(0.0, sigma)
        self.sourcescale = self.sourcescale1 * 10.0**gaussian

    def __call__(self, t_seconds: float) -> np.ndarray:
        t_days = t_seconds / 86400.0
        phase = (t_days / self.cycleper_days) % 1.0
        tc = 12.0 * (phase * self.cycleper_days / 365.25)

        cycleno = int(t_days // self.cycleper_days) + 1
        if self.current_cycle != cycleno or self.sourcescale is None:
            self.current_cycle = cycleno
            self._new_cycle_sourcescale()

        denom = np.exp(tc**2 / self.bhat**2) - self.chat
        if denom <= 0.0 or tc <= 0.0:
            ampli = 0.0
        else:
            ampli = self.sourcescale * (self.ahat * tc**3 / denom)

        evenodd = 1 - 2 * (cycleno % 2)  # 1 for even, -1 for odd

        lambdan = 14.6 + self.blat * (self.sourcescale - self.sourcescale1) / self.sourcescale1
        lambdai = 26.4 - 34.2 * phase + 16.1 * phase**2
        lambda0 = lambdai * (lambdan / 14.6)

        fwhm = (0.14 + 1.05 * phase - 0.78 * phase**2) * lambdai
        if fwhm <= 1.0e-8:
            return np.zeros_like(self.latitude)

        joynorm0 = 1.5
        joynorm = joynorm0
        if tc > 0.0 and denom > 0.0:
            ampli0 = self.sourcescale1 * (self.ahat * tc**3 / denom)
            if abs(ampli0) > 1.0e-30:
                joynorm = joynorm0 * (1.0 - self.bjoy * ((ampli - ampli0) / ampli0))

        tilt_sep = joynorm * np.sin(np.deg2rad(lambda0))
        latitude = self.latitude

        bandn1 = evenodd * ampli * np.exp(-(latitude - lambda0 - tilt_sep)**2 / (2.0 * fwhm**2))
        bandn2a = -evenodd * ampli * np.exp(-(latitude - lambda0 + tilt_sep)**2 / (2.0 * fwhm**2))
        bands2a = evenodd * ampli * np.exp(-(latitude + lambda0 - tilt_sep)**2 / (2.0 * fwhm**2))
        bands1 = -evenodd * ampli * np.exp(-(latitude + lambda0 + tilt_sep)**2 / (2.0 * fwhm**2))

        # Flux correction following transp.py logic.
        thetaf = np.linspace(0.0, np.pi, 181)
        latf = 90.0 - np.rad2deg(thetaf)
        bandn1f = evenodd * ampli * np.exp(-(latf - lambda0 - tilt_sep)**2 / (2.0 * fwhm**2))
        bandn2af = -evenodd * ampli * np.exp(-(latf - lambda0 + tilt_sep)**2 / (2.0 * fwhm**2))

        fluxband1 = _trapz(-np.sin(thetaf) * bandn1f, thetaf)
        fluxband2 = _trapz(-np.sin(thetaf) * bandn2af, thetaf)

        fluxcorrection = 1.0
        if ampli != 0.0 and abs(fluxband2) > 1.0e-30:
            fluxcorrection = -fluxband1 / fluxband2

        bandn2 = fluxcorrection * bandn2a
        bands2 = fluxcorrection * bands2a

        # Original transp.py source is used in a day-based update. Convert G/day -> G/s.
        source_g_per_day = bandn1 + bandn2 + bands1 + bands2
        return self.source_strength * source_g_per_day / 86400.0


# -----------------------------------------------------------------------------
# Fractional weights
# -----------------------------------------------------------------------------
def l1_weights(q: float, nmax: int) -> np.ndarray:
    """L1 weights a_k = (k+1)^(1-q) - k^(1-q), k=0..nmax."""
    p = 1.0 - q
    k = np.arange(nmax + 1, dtype=float)
    a = (k + 1.0) ** p - np.where(k > 0, k ** p, 0.0)
    return a


# -----------------------------------------------------------------------------
# W-form transport operator
# -----------------------------------------------------------------------------
def recover_B_from_W(W: np.ndarray, theta: np.ndarray, R: float) -> np.ndarray:
    """
    Recover B from W=R sin(theta) B and use the same style of pole extrapolation
    as transp.py.
    """
    n = theta.size
    s = np.sin(theta)
    B = np.zeros_like(W)
    B[1:n-1] = W[1:n-1] / (R * s[1:n-1])

    if n >= 5:
        # Original transp.py pole treatment: assuming third derivative = 0.
        B[0] = B[2] + 0.5 * (B[1] - B[3])
        B[n-1] = B[n-3] + 0.5 * (B[n-2] - B[n-4])
    else:
        B[0] = B[1]
        B[n-1] = B[n-2]
    return B


def axial_dipole_moment(theta: np.ndarray, B: np.ndarray) -> float:
    """Axial dipole moment D = (3/2) int B cos(theta) sin(theta) dtheta."""
    return 1.5 * _trapz(B * np.cos(theta) * np.sin(theta), theta)


def w_transport_rhs(
    W: np.ndarray,
    theta: np.ndarray,
    dtheta: float,
    R: float,
    eta: float,
    u_theta: np.ndarray,
    tau: float | None,
    source_g_per_s: np.ndarray,
    advection_scheme: str = "vanleer",
    dt_eff: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Non-periodic W-form transport RHS for 1D SFT.

    Evolves:
        W = R sin(theta) B

    This avoids np.roll in latitude, because theta is not periodic.
    Boundary fluxes at the poles are set to zero.
    """

    n = theta.size
    B = recover_B_from_W(W, theta, R)

    # --------------------------------------------------
    # 1) Advective flux at cell faces
    # --------------------------------------------------
    # faces between i and i+1, length n-1
    theta_face = theta[:-1] + 0.5 * dtheta

    # face velocity
    u_face = 0.5 * (u_theta[:-1] + u_theta[1:])

    # W at faces using upwind
    # NOTE: with rhs = +d(F)/dtheta below, u>0 (poleward in the north)
    # corresponds to flow toward *decreasing* theta, so the upwind cell
    # is the one at larger theta (W_right).
    W_left = W[:-1]
    W_right = W[1:]

    if advection_scheme == "upwind":
        # First-order upwind. Stable but carries numerical diffusion
        # ~ |u| R dtheta / 2 ~ 6e7 m^2/s at 1 deg resolution, ~10% of the
        # physical eta = 6e8; van Leer below keeps the correction O(dtheta^2).
        W_up = np.where(u_face >= 0.0, W_right, W_left)

    elif advection_scheme == "vanleer":
        # Second-order MUSCL reconstruction with van Leer (harmonic-mean)
        # limited slopes. TVD; numerical diffusion O(dtheta^2) in smooth
        # regions, reverting to first-order upwind at extrema.
        sigma = np.zeros_like(W)
        dWm = W[1:-1] - W[:-2]   # backward differences
        dWp = W[2:] - W[1:-1]    # forward differences
        prod = dWm * dWp
        denom = np.where(np.abs(dWm + dWp) > 0.0, dWm + dWp, 1.0)
        sigma[1:-1] = np.where(prod > 0.0, 2.0 * prod / denom, 0.0)

        # Face i+1/2 sits between cells i and i+1.
        # u_face >= 0: flow toward decreasing theta -> upwind cell is i+1,
        #              the face is its lower edge: W = W[i+1] - sigma[i+1]/2.
        # u_face <  0: flow toward increasing theta -> upwind cell is i,
        #              the face is its upper edge: W = W[i] + sigma[i]/2.
        # TVD requires scaling the reconstruction by (1 - CFL) when the
        # scheme is advanced with forward Euler (Sweby/LeVeque form).
        # CFL << 1 here, but the factor costs nothing and guarantees no
        # spurious growth from the second-order correction.
        if dt_eff is not None:
            nu = np.abs(u_face) * dt_eff / (R * dtheta)
        else:
            nu = 0.0
        W_face_from_right = W_right - 0.5 * (1.0 - nu) * sigma[1:]
        W_face_from_left = W_left + 0.5 * (1.0 - nu) * sigma[:-1]
        W_up = np.where(u_face >= 0.0, W_face_from_right, W_face_from_left)

    else:
        raise ValueError("advection_scheme must be 'upwind' or 'vanleer'.")

    # advective W flux
    F_adv = (u_face / R) * W_up

    # --------------------------------------------------
    # 2) Diffusive flux at cell faces
    # --------------------------------------------------
    dB_dtheta_face = (B[1:] - B[:-1]) / dtheta
    F_diff = (eta / R) * np.sin(theta_face) * dB_dtheta_face

    # Total face flux
    F_face = F_adv + F_diff

    # --------------------------------------------------
    # 3) Divergence of flux with zero pole flux
    # --------------------------------------------------
    rhs = np.zeros_like(W)

    # interior cells
    rhs[1:-1] = (F_face[1:] - F_face[:-1]) / dtheta

    # pole boundary fluxes: F_{-1/2}=0 and F_{N-1/2}=0
    rhs[0] = (F_face[0] - 0.0) / dtheta
    rhs[-1] = (0.0 - F_face[-1]) / dtheta

    # --------------------------------------------------
    # 4) Decay term
    # --------------------------------------------------
    if tau is not None and tau > 0.0:
        rhs -= W / tau

    # --------------------------------------------------
    # 5) Source term in W-form
    # --------------------------------------------------
    rhs += source_g_per_s * R * np.sin(theta)

    # Keep exact pole values stable
    rhs[0] = 0.0
    rhs[-1] = 0.0

    return rhs, B


# -----------------------------------------------------------------------------
# Caputo-W runner
# -----------------------------------------------------------------------------
def run_fractional_sft_1d_W(
    *,
    q: float,
    n_theta: int,
    dt: float,
    n_steps: int,
    R: float,
    eta: float,
    u0: float,
    tau: float | None,
    short_memory_M: int | None,
    source_model=None,
    B0: np.ndarray | None = None,
    store_every: int = 1,
    flowtype: int = 2,
    fractional_time_scale: float = 86400.0,
    advection_scheme: str = "vanleer",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the Caputo time-fractional 1D SFT in W-form.

    For q=1:
        W_new = W_old + dt * RHS(W_old)

    For 0<q<1, L1 explicit form:
        c_q (W_n - W_{n-1} + H_n) = T0^(1-q) RHS(W_{n-1})
        W_n = W_{n-1} - H_n + T0^(1-q) RHS/c_q

    T0 is `fractional_time_scale`; it is included for dimensional consistency
    because RHS from the classical SFT operator has units W/s.
    """
    if not (0.0 < q <= 1.0):
        raise ValueError("q must be in (0,1].")
    if n_theta < 5:
        raise ValueError("n_theta must be at least 5 for pole extrapolation.")

    theta = np.linspace(0.0, np.pi, n_theta)
    dtheta = theta[1] - theta[0]
    latitude = 90.0 - np.rad2deg(theta)
    sin_theta = np.sin(theta)

    u_theta = meridional_flow_profile(
    theta,
    u0=u0,
    flowtype=flowtype,
    smooth_taper=True,
    latitude0=65.0,
    taper_width=20.0,
)

    if B0 is None:
        B = np.zeros(n_theta, dtype=float)
    else:
        B = np.array(B0, dtype=float).copy()
        if B.size != n_theta:
            raise ValueError("B0 must have length n_theta.")

    W = R * sin_theta * B
    W[0] = 0.0
    W[-1] = 0.0

    if q == 1.0:
        cq = 1.0 / dt
        a = None
        rhs_scale = 1.0
    else:
        cq = 1.0 / (dt ** q * math.gamma(2.0 - q))
        a = l1_weights(q, n_steps + 1)
        rhs_scale = fractional_time_scale ** (1.0 - q)

    dW_hist = np.zeros((n_steps + 1, n_theta), dtype=float)

    t_store = [0.0]
    B_store = [B.copy()]

    for n in range(1, n_steps + 1):
        t = n * dt
        S = np.zeros(n_theta, dtype=float) if source_model is None else source_model(t)

        rhs, B_old = w_transport_rhs(
            W=W,
            theta=theta,
            dtheta=dtheta,
            R=R,
            eta=eta,
            u_theta=u_theta,
            tau=tau,
            source_g_per_s=S,
            advection_scheme=advection_scheme,
            dt_eff=rhs_scale / cq,
        )

        if q == 1.0 or n <= 1:
            H = 0.0
        else:
            if short_memory_M is None:
                m_start = 1
            else:
                m_start = max(1, n - short_memory_M)
            weights = a[n - np.arange(m_start, n)]
            H = weights @ dW_hist[m_start:n]

        Wnew = W - (H if isinstance(H, np.ndarray) else 0.0) + rhs_scale * rhs / cq
        Wnew[0] = 0.0
        Wnew[-1] = 0.0
        Wnew = remove_monopole_from_W(Wnew, theta, R)

        if not np.isfinite(Wnew).all():
            raise FloatingPointError(
                f"Non-finite W at step {n}, t={t / 86400.0:.2f} days. "
                "Try q closer to 1, smaller dt, weaker source_strength, or shorter memory."
            )

        dW_hist[n] = Wnew - W
        W = Wnew
        B = recover_B_from_W(W, theta, R)

        if (n % store_every) == 0:
            t_store.append(t)
            B_store.append(B.copy())

    return theta, np.array(t_store), np.array(B_store)


# -----------------------------------------------------------------------------
# Butterfly plotting
# -----------------------------------------------------------------------------
def plot_butterfly(theta: np.ndarray, t: np.ndarray, B_store: np.ndarray, outfile: str = "butterfly_W_caputo.png") -> None:
    lat_deg = 90.0 - np.degrees(theta)
    t_years = t / (365.25 * 24.0 * 3600.0)
    data = B_store.T

    vmax = np.nanpercentile(np.abs(data), 99.0)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[t_years.min(), t_years.max(), lat_deg.min(), lat_deg.max()],
        interpolation="nearest",
        vmin=-vmax,
        vmax=vmax,
        cmap="RdBu_r",
    )
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("SFT Butterfly Diagram: Caputo fractional W-form")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("B [G / arb. scaling]")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main example
# -----------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI for the example run. Defaults reproduce the original script exactly."""
    p = argparse.ArgumentParser(
        description="Time-fractional (Caputo) 1D axisymmetric SFT in W-form.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Start close to the classical case first (q=0.99). Then reduce q gradually.
    p.add_argument("--q", type=float, default=0.99,
                   help="Caputo fractional order in (0, 1]; q=1 is classical SFT.")
    p.add_argument("--n-theta", type=int, default=181,
                   help="Number of colatitude grid points (>=5).")
    p.add_argument("--dt-days", type=float, default=0.5, help="Time step in days.")
    p.add_argument("--years", type=float, default=33.0,
                   help="Total simulated time in years.")
    p.add_argument("--R", type=float, default=6.96e8, help="Solar radius [m].")
    p.add_argument("--eta", type=float, default=600e6,
                   help="Supergranular diffusivity [m^2/s] (600 km^2/s = 6.0e8).")
    p.add_argument("--u0", type=float, default=10.0,
                   help="Peak meridional flow speed [m/s].")
    p.add_argument("--tau-years", type=float, default=10.0,
                   help="Radial-decay timescale in years; <=0 disables decay.")
    p.add_argument("--source-strength", type=float, default=0.02,
                   help="Source amplitude scaling.")
    p.add_argument("--flowtype", type=int, default=2, choices=[1, 2, 3, 4, 5],
                   help="Meridional flow profile selector.")
    p.add_argument("--cycle-years", type=float, default=11.0,
                   help="Activity cycle period in years (matches the Hathaway "
                        "envelope, which is tuned for ~11 yr).")
    # For q<1, this truncates the Caputo memory. Omit for full memory.
    p.add_argument("--short-memory", type=int, default=None,
                   help="Truncate Caputo memory to this many steps (default: full).")
    p.add_argument("--store-every", type=int, default=5,
                   help="Store output every N steps.")
    p.add_argument("--advection-scheme", default="vanleer",
                   choices=["vanleer", "upwind"], help="Advection scheme.")
    p.add_argument("--seed", type=int, default=1, help="RNG seed for the source.")
    p.add_argument("--out", default="butterfly_W_caputo.png",
                   help="Output butterfly figure path.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    dt = args.dt_days * 86400.0
    n_steps = int(args.years * 365.25 * 86400.0 / dt)
    tau = args.tau_years * 365.25 * 86400.0 if args.tau_years > 0.0 else None

    theta0 = np.linspace(0.0, np.pi, args.n_theta)
    source_model = TranspSource1D(
        latitude_deg=90.0 - np.rad2deg(theta0),
        cycleper_days=args.cycle_years * 365.25,
        flowtype=args.flowtype,
        tau_seconds=tau,
        blat=0.0,
        bjoy=0.0,
        seed=args.seed,
        source_strength=args.source_strength,
    )

    theta, t_store, B_store = run_fractional_sft_1d_W(
        q=args.q,
        n_theta=args.n_theta,
        dt=dt,
        n_steps=n_steps,
        R=args.R,
        eta=args.eta,
        u0=args.u0,
        tau=tau,
        short_memory_M=args.short_memory,
        source_model=source_model,
        store_every=args.store_every,
        flowtype=args.flowtype,
        fractional_time_scale=86400.0,
        advection_scheme=args.advection_scheme,
    )

    plot_butterfly(theta, t_store, B_store, outfile=args.out)
    print(f"Saved butterfly diagram to {args.out}")


if __name__ == "__main__":
    main()
