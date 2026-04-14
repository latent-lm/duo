"""
Simulate and visualise Brownian bridge trajectories on the 2D Poincaré disk D².

Bridge SDE on the local chart (d=2, underlying drift f₀=0, σ=1):

    dx = [ (d-1)/2 · (1-‖x‖²)² / ‖y-x‖² · (y-x)          ← bridge drift
           - d/4 · (1-‖x‖²) · x ]  dt                       ← Itô correction
         + (1-‖x‖²)/2  · dW                                  ← hyperbolic noise

Forward Euler–Maruyama:
    x_{n+1} = x_n  +  f(x_n, y) Δt  +  g(x_n) √Δt · εₙ,    εₙ ~ N(0, I)

NOTE — The one-step *posterior* in the ELBO derivation uses −f Δt
(reverse-τ direction), but for *simulating* bridge states from q(·|y)
we step forward with +f Δt since f is the h-transform (forward) drift.

Along the radial direction toward y the drift simplifies to
    dr/dt = (1-r²)/2      ⟹     r(t) = tanh(t/2).
So the mean trajectory needs  t ≈ 2 artanh(0.99) ≈ 5.3  to reach r ≈ 0.99.
With T = 1000 steps and Δt = 1/T = 0.001 the total time is only 1.0,
which gives r ≈ tanh(0.5) ≈ 0.46 — far from the boundary.
We therefore set Δt = 0.01 (total time = 10) so the bridge has enough
time to approach the boundary target.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ── dimension ────────────────────────────────────────────────────────────
D = 2


def drift(x, y):
    """Bridge drift  f(x, y)  on the Poincaré disk (d = D)."""
    norm_x_sq = x @ x
    diff = y - x
    norm_diff_sq = diff @ diff
    conf = 1.0 - norm_x_sq
    bridge = (D - 1) / 2.0 * conf**2 / norm_diff_sq * diff
    ito = -D / 4.0 * conf * x
    return bridge + ito


def diffusion(x):
    """Scalar diffusion coefficient  g(x) = (1−‖x‖²)/2."""
    return (1.0 - x @ x) / 2.0


def simulate_bridge(y, T=1000, dt=0.01, seed=None):
    """
    Sample bridge states {x_n}_{n=0}^{T} from q(·|y) via forward EM.

    One step:  x_{n+1} = x_n + f(x_n, y) Δt + g(x_n) √Δt · ε
    starting from x_0 = 0 (origin), walking toward boundary target y.
    """
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)

    x = np.zeros(D)
    traj = np.empty((T + 1, D))
    traj[0] = x

    for n in range(T):
        f = drift(x, y)
        g = diffusion(x)
        eps = rng.standard_normal(D)

        x = x + f * dt + g * sqrt_dt * eps

        # numerical safety: stay strictly inside the disk
        r = np.linalg.norm(x)
        if r >= 0.9999:
            x = x / r * 0.9999

        traj[n + 1] = x

    return traj  # traj[0] = origin,  traj[T] near y


# ── Parameters ───────────────────────────────────────────────────────────
T = 1000
DT = 0.03           # Δt — total bridge time = T·Δt = 10.0
N_BRIDGES = 1

# target on the boundary at 40°
y = np.array([np.cos(np.pi * 2 / 9), np.sin(np.pi * 2 / 9)])  # 40°

# ── Simulate ─────────────────────────────────────────────────────────────
bridges = [simulate_bridge(y, T=T, dt=DT, seed=100 + j) for j in range(N_BRIDGES)]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))

# disk
theta = np.linspace(0, 2 * np.pi, 500)
ax.fill(np.cos(theta), np.sin(theta), color="#f7f7f7", zorder=0)
ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)

# geodesic-distance circles
for r in (0.2, 0.4, 0.6, 0.8):
    ax.plot(r * np.cos(theta), r * np.sin(theta),
            "k-", alpha=0.06, lw=0.5)

# background bridges (light)
for j, traj in enumerate(bridges):
    ax.plot(traj[:, 0], traj[:, 1], "-", color="steelblue",
            alpha=0.15, linewidth=0.6)

# colour-coded primary trajectory
traj0 = bridges[0]
t_arr = np.linspace(0, 1, len(traj0))
pts = traj0.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap="viridis", array=t_arr[:-1],
                    linewidths=2.0, zorder=4)
ax.add_collection(lc)
cbar = plt.colorbar(lc, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Normalised time  $n / T$", fontsize=11)

# markers
ax.plot(0, 0, "o", color="limegreen", markersize=11, zorder=6,
        markeredgecolor="k", markeredgewidth=1, label="Origin (prior)")
ax.plot(y[0], y[1], "*", color="red", markersize=20, zorder=6,
        markeredgecolor="k", markeredgewidth=0.5,
        label=rf"Target $y$=({y[0]:.2f}, {y[1]:.2f})")

# endpoint scatter for all bridges, labelled individually
ends = np.array([b[T] for b in bridges])
ax.scatter(ends[:, 0], ends[:, 1], c="orange", s=60, zorder=7,
           edgecolors="k", linewidths=0.5, label="Endpoints $x_T$")
for j, (ex, ey) in enumerate(ends):
    ax.annotate(f"{j+1}", (ex, ey), fontsize=8, fontweight="bold",
                ha="center", va="bottom", xytext=(0, 6),
                textcoords="offset points", zorder=8)

ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.15)
ax.set_aspect("equal")
ax.set_xlabel("$x_1$", fontsize=13)
ax.set_ylabel("$x_2$", fontsize=13)
ax.set_title(
    r"Brownian Bridge on Poincaré Disk $\mathbb{D}^2$"
    + f"\n$T={T}$,  $\\Delta t={DT}$,  total time $= {T*DT:.0f}$"
    + f",  {N_BRIDGES} realisations",
    fontsize=13,
)
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.15)

# ── Zoomed inset around the target ────────────────────────────────────────
inset = ax.inset_axes([0.02, 0.55, 0.38, 0.38])  # [x, y, w, h] in axes coords
pad = 0.08
inset.set_xlim(y[0] - pad - 0.35, y[0] + pad)
inset.set_ylim(y[1] - pad - 0.05, y[1] + pad + 0.05)
inset.set_aspect("equal")
inset.set_title("Zoom near target", fontsize=9)
inset.tick_params(labelsize=7)

# redraw boundary arc in inset
inset.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1.5)
# redraw trajectories
for j, traj in enumerate(bridges):
    inset.plot(traj[:, 0], traj[:, 1], "-", color="steelblue",
               alpha=0.25, linewidth=0.5)
# target
inset.plot(y[0], y[1], "*", color="red", markersize=14,
           markeredgecolor="k", markeredgewidth=0.5)
# endpoints with labels
inset.scatter(ends[:, 0], ends[:, 1], c="orange", s=50, zorder=7,
              edgecolors="k", linewidths=0.5)
for j, (ex, ey) in enumerate(ends):
    inset.annotate(f"{j+1}", (ex, ey), fontsize=7, fontweight="bold",
                   ha="center", va="bottom", xytext=(0, 5),
                   textcoords="offset points", zorder=8)
ax.indicate_inset_zoom(inset, edgecolor="gray", linewidth=1)

plt.tight_layout()
out = "scripts/hyper_bridge.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\nTarget y = [{y[0]:.4f}, {y[1]:.4f}]")
for j, traj in enumerate(bridges):
    end = traj[T]
    dist = np.linalg.norm(end - y)
    print(f"  Bridge {j+1}:  x_T=[{end[0]:+.4f}, {end[1]:+.4f}]  "
          f"||x_T||={np.linalg.norm(end):.4f}  "
          f"||x_T - y||={dist:.4f}")
