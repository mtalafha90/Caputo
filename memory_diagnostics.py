#!/usr/bin/env python3
"""
Diagnostics to visualize the Caputo memory in the 1D W-form SFT model.

Produces:
  fig1_butterflies_q.png    : butterfly diagrams, q = 1.0 vs q = 0.7, same source
  fig2_polar_memory.png     : (a) N polar field vs time for several q
                              (b) size of the memory term ||H|| relative to the
                                  instantaneous update, vs time
  fig3_decay_weights.png    : (a) free-decay of unsigned flux, log-log, with
                                  t^-q guide lines (Mittag-Leffler tails)
                              (b) the L1 weights a_k for several q
"""
import math
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("sft", "/home/claude/sft/1D_W_caputo.py")
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)

R = 6.96e8
ETA = 600e3
U0 = 10.0
FLOWTYPE = 2
T0 = 86400.0
DAY = 86400.0
YEAR = 365.25 * DAY


def run_with_diagnostics(q, n_theta, dt, n_steps, tau, source_fn, B0=None,
                         store_every=2):
    """Same scheme as run_fractional_sft_1d_W, but also records polar fields,
    unsigned flux, and the norm of the memory term H vs the instantaneous update."""
    theta = np.linspace(0.0, np.pi, n_theta)
    dth = theta[1] - theta[0]
    lat = 90.0 - np.degrees(theta)
    s = np.sin(theta)
    u = sft.meridional_flow_profile(theta, u0=U0, flowtype=FLOWTYPE)

    B = np.zeros(n_theta) if B0 is None else np.array(B0, float).copy()
    W = R * s * B
    W[0] = W[-1] = 0.0

    if q == 1.0:
        cq = 1.0 / dt
        a = None
        rhs_scale = 1.0
    else:
        cq = 1.0 / (dt ** q * math.gamma(2.0 - q))
        a = sft.l1_weights(q, n_steps + 1)
        rhs_scale = T0 ** (1.0 - q)

    dW = np.zeros((n_steps + 1, n_theta))
    capN = lat >= 70.0
    capS = lat <= -70.0

    out = dict(t=[0.0], B=[B.copy()], pfN=[0.0], pfS=[0.0],
               uflux=[np.trapezoid(np.abs(B) * s, theta)],
               mem_ratio=[0.0])

    for n in range(1, n_steps + 1):
        t = n * dt
        S = source_fn(t) if source_fn is not None else np.zeros(n_theta)
        rhs, _ = sft.w_transport_rhs(W=W, theta=theta, dtheta=dth, R=R, eta=ETA,
                                     u_theta=u, tau=tau, source_g_per_s=S)
        if q == 1.0 or n <= 1:
            H = np.zeros(n_theta)
        else:
            w = a[n - np.arange(1, n)]
            H = w @ dW[1:n]

        upd = rhs_scale * rhs / cq
        Wn = W - H + upd
        Wn[0] = Wn[-1] = 0.0
        Wn = sft.remove_monopole_from_W(Wn, theta, R)
        if not np.isfinite(Wn).all():
            raise FloatingPointError(f"blow-up at step {n}")

        dW[n] = Wn - W
        W = Wn
        B = sft.recover_B_from_W(W, theta, R)

        if n % store_every == 0:
            out["t"].append(t)
            out["B"].append(B.copy())
            out["pfN"].append(np.trapezoid((B * s)[capN], theta[capN]) /
                              np.trapezoid(s[capN], theta[capN]))
            out["pfS"].append(np.trapezoid((B * s)[capS], theta[capS]) /
                              np.trapezoid(s[capS], theta[capS]))
            out["uflux"].append(np.trapezoid(np.abs(B) * s, theta))
            nH = np.linalg.norm(H)
            nU = np.linalg.norm(upd)
            out["mem_ratio"].append(nH / nU if nU > 0 else 0.0)

    out["theta"] = theta
    out["lat"] = lat
    for k in ("t", "B", "pfN", "pfS", "uflux", "mem_ratio"):
        out[k] = np.array(out[k])
    return out


def make_source(n_theta, pulse_years=None):
    theta0 = np.linspace(0.0, np.pi, n_theta)
    src = sft.TranspSource1D(
        latitude_deg=90.0 - np.rad2deg(theta0),
        cycleper_days=6.0 * 365.25, flowtype=FLOWTYPE,
        tau_seconds=10.0 * YEAR, blat=0.0, bjoy=0.0,
        seed=1, source_strength=0.02)
    if pulse_years is None:
        return src
    def pulsed(t):
        return src(t) if t < pulse_years * YEAR else np.zeros(n_theta)
    return pulsed


# ----------------------------------------------------------------------
# Experiment 1: same cyclic source, different q  (24 years, dt = 1 day)
# ----------------------------------------------------------------------
n_theta = 181
dt = 1.0 * DAY
years = 24.0
n_steps = int(years * YEAR / dt)
tau = 10.0 * YEAR

qs = [1.0, 0.9, 0.8, 0.7]
runs = {}
for q in qs:
    print(f"running cyclic source, q = {q} ...", flush=True)
    runs[q] = run_with_diagnostics(q, n_theta, dt, n_steps, tau,
                                   make_source(n_theta), store_every=4)

# --- fig 1: butterflies q=1 vs q=0.7
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for ax, q in zip(axes, (1.0, 0.7)):
    r = runs[q]
    data = r["B"].T
    vmax = np.nanpercentile(np.abs(data), 99.0)
    im = ax.imshow(data, origin="lower", aspect="auto",
                   extent=[r["t"].min() / YEAR, r["t"].max() / YEAR,
                           r["lat"].min(), r["lat"].max()],
                   vmin=-vmax, vmax=vmax, cmap="RdBu_r", interpolation="nearest")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"q = {q}")
    fig.colorbar(im, ax=ax, label="B [arb.]")
axes[1].set_xlabel("Time (years)")
fig.suptitle("Same source, classical vs fractional", y=0.99)
fig.tight_layout()
fig.savefig("fig1_butterflies_q.png", dpi=180)
plt.close(fig)

# --- fig 2: polar field + memory term size
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(qs)))
for q, c in zip(qs, colors):
    r = runs[q]
    axes[0].plot(r["t"] / YEAR, r["pfN"], color=c, label=f"q = {q}")
    axes[1].plot(r["t"] / YEAR, r["mem_ratio"], color=c, label=f"q = {q}")
axes[0].axhline(0, color="k", lw=0.5)
for k in range(1, 5):
    for ax in axes:
        ax.axvline(6 * k, color="gray", lw=0.5, ls=":")
axes[0].set_ylabel("North polar field [arb.]")
axes[0].set_title("Polar cap field (|lat| > 70°): reversals lag and smooth as q drops")
axes[0].legend()
axes[1].set_ylabel(r"$\|H_n\| / \|\Delta W_{inst}\|$")
axes[1].set_title("Memory term relative to instantaneous update")
axes[1].set_xlabel("Time (years)")
axes[1].set_yscale("log")
fig.tight_layout()
fig.savefig("fig2_polar_memory.png", dpi=180)
plt.close(fig)

# ----------------------------------------------------------------------
# Experiment 2: decay with finite tau, no source -> Mittag-Leffler tails
# ----------------------------------------------------------------------
lat0 = 90.0 - np.degrees(np.linspace(0.0, np.pi, n_theta))
B0 = np.exp(-((lat0 - 15.0) / 8.0) ** 2) - np.exp(-((lat0 + 15.0) / 8.0) ** 2)

qs_decay = [1.0, 0.8, 0.6]
decays = {}
tau_d = 0.5 * YEAR
n_steps_d = int(50.0 * YEAR / dt)
for q in qs_decay:
    print(f"running decay, q = {q} ...", flush=True)
    decays[q] = run_with_diagnostics(q, n_theta, dt, n_steps_d, tau_d,
                                     None, B0=B0, store_every=4)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
colors = plt.cm.plasma(np.linspace(0.0, 0.7, len(qs_decay)))
for q, c in zip(qs_decay, colors):
    r = decays[q]
    ty = r["t"] / YEAR
    f = r["uflux"] / r["uflux"][0]
    axes[0].loglog(ty[1:], np.maximum(f[1:], 1e-16), color=c, label=f"q = {q}")
    if q < 1.0:
        m = ty > 20.0
        ref = f[m][0] * (ty[m] / ty[m][0]) ** (-q)
        axes[0].loglog(ty[m], ref, color=c, ls="--", lw=1)
axes[0].set_ylim(1e-10, 2)
axes[0].set_xlabel("Time (years)")
axes[0].set_ylabel("Unsigned flux (normalized)")
axes[0].set_title("Decay with $\\tau$ = 0.5 yr, no source\n"
                  "q=1: exponential; q<1: Mittag-Leffler, dashed $t^{-q}$ guides")
axes[0].legend()

k = np.arange(1, 2001)
for q, c in zip([0.99, 0.9, 0.8, 0.7, 0.6], plt.cm.viridis(np.linspace(0, 0.85, 5))):
    axes[1].loglog(k, sft.l1_weights(q, 2000)[1:], color=c, label=f"q = {q}")
axes[1].set_xlabel("lag k (steps into the past)")
axes[1].set_ylabel(r"L1 weight $a_k$")
axes[1].set_title("How strongly the past is weighted")
axes[1].legend()
fig.tight_layout()
fig.savefig("fig3_decay_weights.png", dpi=180)
plt.close(fig)
print("done")
