"""Contract tests for ``unigram/geo_bridge.py``.

Written against ``unigram/notes/geo_bridge_ARCH.md`` BEFORE the implementation
is fixed (TDD). The target module currently compiles but is riddled with
dangling references (``HyperBridge.sample_chi``, ``torch.pi()``, undefined
``ps``/``src``/``dest``, bare-name converter calls, a wrong dimension check and
a missing ``rho`` computation). These tests therefore FAIL today and PASS once
the implementer makes ``geo_bridge.py`` match the ARCH contract.

Run with::

    cd <repo root>
    conda run -n duo python -m pytest unigram/tests -x -q

Import resolves via ``unigram/tests/conftest.py`` (puts ``unigram/`` on path).
"""

from __future__ import annotations

import math

import pytest
import torch

from geo_bridge import (
    Coordinate,
    FreeBinaryHyperbolicHeatKernel as HK,
    GeoUtils,
    Geometry,
)

DTYPE = torch.float64
B = 8


# ---------------------------------------------------------------------------
# Tiny shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ts():
    """A batch of strictly-positive heat times, shape (B,), float64."""
    return torch.linspace(0.1, 2.0, B, dtype=DTYPE)


@pytest.fixture
def word_embedding():
    """A (vocab, 2) embedding table; only the angle atan2(e[1], e[0]) is used."""
    torch.manual_seed(0)
    return torch.randn(5, 2, dtype=DTYPE)


@pytest.fixture
def targets():
    """Long indices into the word_embedding table, shape (B,)."""
    return torch.arange(B, dtype=torch.long) % 5


def lorentz_inner(z):
    """Minkowski self-product -z0^2 + sum(z[1:]^2), shape (...)."""
    return -z[..., 0] ** 2 + (z[..., 1:] ** 2).sum(-1)


def lorentz_dist(x, y):
    """Hyperbolic geodesic distance arccosh(-<x, y>_L) between Lorentz points."""
    inner = -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(-1)
    return torch.acosh((-inner).clamp_min(1.0))


# ===========================================================================
# 1. Reference-integrity / smoke -- every public method runs without raising
#    (this is what catches the dangling-reference bugs).
# ===========================================================================


def test_binary_free_hyperbolic_heat_kernel_returns_pair(ts):
    torch.manual_seed(0)
    out = HK.binary_free_hyperbolic_heat_kernel(ts)
    assert isinstance(out, tuple) and len(out) == 2


def test_binary_free_poincare_heat_kernel_polar_returns_pair(ts):
    torch.manual_seed(0)
    out = HK.binary_free_poincare_heat_kernel(ts, Coordinate.HYPERBOLIC_POLAR)
    assert isinstance(out, tuple) and len(out) == 2


def test_binary_free_poincare_heat_kernel_cartesian_returns_tensor(ts):
    torch.manual_seed(0)
    out = HK.binary_free_poincare_heat_kernel(ts, Coordinate.CARTESIAN)
    assert torch.is_tensor(out)


def test_binary_free_lorentz_heat_kernel_polar_returns_pair(ts):
    torch.manual_seed(0)
    out = HK.binary_free_lorentz_heat_kernel(ts, Coordinate.HYPERBOLIC_POLAR)
    assert isinstance(out, tuple) and len(out) == 2


def test_binary_free_lorentz_heat_kernel_cartesian_returns_tensor(ts):
    torch.manual_seed(0)
    out = HK.binary_free_lorentz_heat_kernel(ts, Coordinate.CARTESIAN)
    assert torch.is_tensor(out)


def test_binary_free_lorentz_heat_kernel_default_returns_cartesian_tensor(ts):
    torch.manual_seed(0)
    out = HK.binary_free_lorentz_heat_kernel(ts)
    assert torch.is_tensor(out)


def test_binary_poincare_bridge_runs(ts, targets, word_embedding):
    torch.manual_seed(0)
    out = HK.binary_poincare_bridge(ts, targets, word_embedding)
    assert isinstance(out, tuple) and len(out) == 2


def test_binary_lorentz_bridge_runs(ts, targets, word_embedding):
    torch.manual_seed(0)
    out = HK.binary_lorentz_bridge(ts, targets, word_embedding)
    assert torch.is_tensor(out)


def test_binary_poincare_bridge_warps_then_rotates_angle(ts, targets, word_embedding):
    # Bridge keeps the free-kernel rho (same seed -> identical RNG stream), reshapes
    # the uniform free angle through the Poisson-kernel inverse-CDF
    # 2*atan(exp(-rho)*tan(theta/2)), then rotates by the target angle atan2(e1, e0).
    # It is NOT a pure rotation of the free angle.
    torch.manual_seed(0)
    rho_free, theta_free = HK.binary_free_hyperbolic_heat_kernel(ts)
    torch.manual_seed(0)
    rho_b, theta_b = HK.binary_poincare_bridge(
        ts, targets, word_embedding, Coordinate.HYPERBOLIC_POLAR
    )
    e = word_embedding[targets].to(DTYPE)
    target_angle = torch.atan2(e[..., 1], e[..., 0])
    warped = 2 * torch.atan((-rho_free).exp() * torch.tan(theta_free * 0.5))
    assert torch.equal(rho_b, rho_free)
    assert torch.allclose(theta_b, warped + target_angle, atol=1e-12)
    # the Poisson warp genuinely changes the angle -> not a bare rotation
    assert float((warped - theta_free).abs().max()) > 1e-6


def test_geodesic_cartesian_input_runs():
    torch.manual_seed(0)
    src = HK.binary_hyperbolic_polar_to_lorentz_cartesian(
        torch.full((B,), 0.3, dtype=DTYPE), torch.zeros(B, dtype=DTYPE)
    )
    dest = HK.binary_hyperbolic_polar_to_lorentz_cartesian(
        torch.full((B,), 0.5, dtype=DTYPE), torch.full((B,), 0.7, dtype=DTYPE)
    )
    out = HK.geodesic(
        0.5, src_cartesian=src, dest_cartesian=dest, cartesian_model=Geometry.LORENTZ
    )
    assert torch.is_tensor(out)


def test_geodesic_polar_input_runs():
    out = HK.geodesic(
        0.5,
        src_radial=torch.full((B,), 0.3, dtype=DTYPE),
        src_angular=torch.zeros(B, dtype=DTYPE),
        dest_radial=torch.full((B,), 0.5, dtype=DTYPE),
        dest_angular=torch.full((B,), 0.7, dtype=DTYPE),
    )
    assert isinstance(out, tuple) and len(out) == 2


def test_polar_to_poincare_cartesian_runs():
    rhos = torch.full((B,), 0.4, dtype=DTYPE)
    thetas = torch.zeros(B, dtype=DTYPE)
    out = GeoUtils.binary_hyperbolic_polar_to_poincare_cartesian(rhos, thetas)
    assert torch.is_tensor(out)


def test_polar_to_lorentz_cartesian_runs():
    rhos = torch.full((B,), 0.4, dtype=DTYPE)
    thetas = torch.zeros(B, dtype=DTYPE)
    out = HK.binary_hyperbolic_polar_to_lorentz_cartesian(rhos, thetas)
    assert torch.is_tensor(out)


def test_lorentz_cartesian_to_polar_runs():
    z = HK.binary_hyperbolic_polar_to_lorentz_cartesian(
        torch.full((B,), 0.4, dtype=DTYPE), torch.zeros(B, dtype=DTYPE)
    )
    out = GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar(z)
    assert isinstance(out, tuple) and len(out) == 2


def test_poincare_cartesian_to_lorentz_cartesian_runs():
    z = torch.full((B, 2), 0.3, dtype=DTYPE)
    out = GeoUtils.poincare_cartesian_to_lorentz_cartesian(z)
    assert torch.is_tensor(out)


# ===========================================================================
# 2. Shape & dtype
# ===========================================================================


def test_free_hyperbolic_polar_shapes(ts):
    torch.manual_seed(0)
    rhos, thetas = HK.binary_free_hyperbolic_heat_kernel(ts)
    assert rhos.shape == (B,) and thetas.shape == (B,)


def test_free_poincare_cartesian_shape(ts):
    torch.manual_seed(0)
    z = HK.binary_free_poincare_heat_kernel(ts, Coordinate.CARTESIAN)
    assert z.shape == (B, 2)


def test_free_lorentz_cartesian_shape(ts):
    torch.manual_seed(0)
    z = HK.binary_free_lorentz_heat_kernel(ts, Coordinate.CARTESIAN)
    assert z.shape == (B, 3)


def test_free_hyperbolic_polar_dtype_preserved(ts):
    torch.manual_seed(0)
    rhos, thetas = HK.binary_free_hyperbolic_heat_kernel(ts)
    assert rhos.dtype == DTYPE and thetas.dtype == DTYPE


def test_polar_to_lorentz_cartesian_dtype_preserved():
    rhos = torch.full((B,), 0.4, dtype=DTYPE)
    thetas = torch.zeros(B, dtype=DTYPE)
    z = HK.binary_hyperbolic_polar_to_lorentz_cartesian(rhos, thetas)
    assert z.dtype == DTYPE


# ===========================================================================
# 3. Manifold invariants
# ===========================================================================


def test_poincare_cartesian_norm_strictly_below_one(ts):
    torch.manual_seed(0)
    z = HK.binary_free_poincare_heat_kernel(ts, Coordinate.CARTESIAN)
    assert torch.all(z.norm(dim=-1) < 1.0)


def test_lorentz_cartesian_time_component_at_least_one(ts):
    torch.manual_seed(0)
    z = HK.binary_free_lorentz_heat_kernel(ts, Coordinate.CARTESIAN)
    assert torch.all(z[..., 0] >= 1.0)


def test_lorentz_cartesian_on_hyperboloid(ts):
    torch.manual_seed(0)
    z = HK.binary_free_lorentz_heat_kernel(ts, Coordinate.CARTESIAN)
    assert torch.allclose(
        lorentz_inner(z), torch.full((B,), -1.0, dtype=DTYPE), atol=1e-6
    )


def test_free_hyperbolic_rho_nonnegative(ts):
    torch.manual_seed(0)
    rhos, _ = HK.binary_free_hyperbolic_heat_kernel(ts)
    assert torch.all(rhos >= 0.0)


def test_free_hyperbolic_theta_in_open_interval(ts):
    torch.manual_seed(0)
    _, thetas = HK.binary_free_hyperbolic_heat_kernel(ts)
    assert torch.all(thetas >= -math.pi) and torch.all(thetas < math.pi)


# ===========================================================================
# 4. Round-trips / converter correctness
# ===========================================================================


def test_polar_lorentz_roundtrip_recovers_rho():
    rho = torch.tensor([0.2, 0.7, 1.3], dtype=DTYPE)
    theta = torch.tensor([0.0, 1.0, -2.0], dtype=DTYPE)
    z = HK.binary_hyperbolic_polar_to_lorentz_cartesian(rho, theta)
    rho_rt, _ = GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar(z)
    assert torch.allclose(rho_rt, rho, atol=1e-6)


def test_polar_lorentz_roundtrip_recovers_theta():
    rho = torch.tensor([0.2, 0.7, 1.3], dtype=DTYPE)
    theta = torch.tensor([0.0, 1.0, -2.0], dtype=DTYPE)
    z = HK.binary_hyperbolic_polar_to_lorentz_cartesian(rho, theta)
    _, theta_rt = GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar(z)
    assert torch.allclose(theta_rt, theta, atol=1e-6)


def test_lorentz_cartesian_to_polar_pins_components():
    # Explicit Lorentz point built from known (rho, theta); the converter must
    # return exactly those, pinning rho=arccosh(z0) and theta=atan2(z2, z1).
    rho = torch.tensor([0.9], dtype=DTYPE)
    theta = torch.tensor([0.6], dtype=DTYPE)
    z = torch.stack(
        [torch.cosh(rho), torch.sinh(rho) * torch.cos(theta), torch.sinh(rho) * torch.sin(theta)],
        dim=-1,
    )
    rho_out, theta_out = GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar(z)
    assert torch.allclose(rho_out, rho, atol=1e-6)
    assert torch.allclose(theta_out, theta, atol=1e-6)


def test_poincare_to_lorentz_lands_on_hyperboloid():
    z = torch.tensor([[0.3, 0.4], [-0.1, 0.5]], dtype=DTYPE)  # ||z|| < 1
    lifted = GeoUtils.poincare_cartesian_to_lorentz_cartesian(z)
    assert torch.allclose(
        lorentz_inner(lifted), torch.full((2,), -1.0, dtype=DTYPE), atol=1e-6
    )


def test_lorentz_to_poincare_roundtrip():
    # lift Poincare -> Lorentz, project back -> recover the original disk point.
    z = torch.tensor([[0.3, 0.4], [-0.1, 0.5]], dtype=DTYPE)  # ||z|| < 1
    lifted = GeoUtils.poincare_cartesian_to_lorentz_cartesian(z)
    back = GeoUtils.lorentz_cartesian_to_poincare_cartesian(lifted)
    assert back.shape == z.shape
    assert torch.all(back.norm(dim=-1) < 1.0)
    assert torch.allclose(back, z, atol=1e-6)


# ===========================================================================
# 5. Error paths
# ===========================================================================


def test_free_lorentz_heat_kernel_raises_on_large_rho():
    # rho ~ 25 > _LORENTZ_RHO_MAX (20): force it via huge ts so cartesian fails.
    torch.manual_seed(0)
    big_ts = torch.full((B,), 1e6, dtype=DTYPE)
    with pytest.raises(ValueError):
        HK.binary_free_lorentz_heat_kernel(big_ts, Coordinate.CARTESIAN)


def test_lorentz_bridge_raises_on_large_rho(targets, word_embedding):
    torch.manual_seed(0)
    big_ts = torch.full((B,), 1e6, dtype=DTYPE)
    with pytest.raises(ValueError):
        HK.binary_lorentz_bridge(big_ts, targets, word_embedding, Coordinate.CARTESIAN)


def test_geodesic_polar_raises_on_large_rho():
    big = torch.full((B,), 25.0, dtype=DTYPE)
    with pytest.raises(ValueError):
        HK.geodesic(
            0.5,
            src_radial=big,
            src_angular=torch.zeros(B, dtype=DTYPE),
            dest_radial=torch.full((B,), 0.5, dtype=DTYPE),
            dest_angular=torch.zeros(B, dtype=DTYPE),
        )


def test_converter_bound_path_raises_on_large_rho():
    big = torch.full((B,), 25.0, dtype=DTYPE)
    with pytest.raises(ValueError):
        GeoUtils._check_lorentz_rho_bound(big, d=2)


def test_lorentz_cartesian_to_polar_raises_on_wrong_last_dim():
    z = torch.zeros(B, 2, dtype=DTYPE)  # last dim must be 3
    with pytest.raises(ValueError):
        GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar(z)


def test_geodesic_raises_when_no_source_form_given():
    with pytest.raises(ValueError):
        HK.geodesic(
            0.5,
            dest_radial=torch.full((B,), 0.5, dtype=DTYPE),
            dest_angular=torch.zeros(B, dtype=DTYPE),
        )


def test_geodesic_raises_when_both_source_forms_given():
    src = HK.binary_hyperbolic_polar_to_lorentz_cartesian(
        torch.full((B,), 0.3, dtype=DTYPE), torch.zeros(B, dtype=DTYPE)
    )
    with pytest.raises(ValueError):
        HK.geodesic(
            0.5,
            src_cartesian=src,
            cartesian_model=Geometry.LORENTZ,
            src_radial=torch.full((B,), 0.3, dtype=DTYPE),
            src_angular=torch.zeros(B, dtype=DTYPE),
            dest_radial=torch.full((B,), 0.5, dtype=DTYPE),
            dest_angular=torch.zeros(B, dtype=DTYPE),
        )


def test_geodesic_raises_on_bad_cartesian_model():
    src = HK.binary_hyperbolic_polar_to_lorentz_cartesian(
        torch.full((B,), 0.3, dtype=DTYPE), torch.zeros(B, dtype=DTYPE)
    )
    dest = HK.binary_hyperbolic_polar_to_lorentz_cartesian(
        torch.full((B,), 0.5, dtype=DTYPE), torch.zeros(B, dtype=DTYPE)
    )
    with pytest.raises(ValueError):
        HK.geodesic(
            0.5, src_cartesian=src, dest_cartesian=dest, cartesian_model="banana"
        )


# ===========================================================================
# 6. geodesic endpoints
# ===========================================================================


def _lorentz_endpoints():
    src = HK.binary_hyperbolic_polar_to_lorentz_cartesian(
        torch.full((B,), 0.3, dtype=DTYPE), torch.zeros(B, dtype=DTYPE)
    )
    dest = HK.binary_hyperbolic_polar_to_lorentz_cartesian(
        torch.full((B,), 0.8, dtype=DTYPE), torch.full((B,), 0.5, dtype=DTYPE)
    )
    return src, dest


def test_geodesic_t0_returns_source():
    src, dest = _lorentz_endpoints()
    out = HK.geodesic(
        0.0, src_cartesian=src, dest_cartesian=dest, cartesian_model=Geometry.LORENTZ
    )
    assert torch.allclose(out, src, atol=1e-6)


def test_geodesic_t1_returns_dest():
    src, dest = _lorentz_endpoints()
    out = HK.geodesic(
        1.0, src_cartesian=src, dest_cartesian=dest, cartesian_model=Geometry.LORENTZ
    )
    assert torch.allclose(out, dest, atol=1e-6)


def test_geodesic_interpolant_on_hyperboloid():
    src, dest = _lorentz_endpoints()
    out = HK.geodesic(
        0.5, src_cartesian=src, dest_cartesian=dest, cartesian_model=Geometry.LORENTZ
    )
    assert torch.allclose(
        lorentz_inner(out), torch.full((B,), -1.0, dtype=DTYPE), atol=1e-6
    )


def test_geodesic_constant_speed():
    # gamma(t) moves at constant speed: d(src, gamma(t)) == t * d(src, dest).
    src, dest = _lorentz_endpoints()
    total = lorentz_dist(src, dest)
    for frac in (0.25, 0.5, 0.75):
        gamma = HK.geodesic(
            frac, src_cartesian=src, dest_cartesian=dest, cartesian_model=Geometry.LORENTZ
        )
        assert torch.allclose(lorentz_dist(src, gamma), frac * total, atol=1e-6)


def test_geodesic_poincare_lift_t0_returns_source():
    # cartesian_model=POINCARE: input AND output are Poincare-disk (B, 2). t=0
    # lifts src to Lorentz, interpolates, then projects back -> recovers src_p.
    src_p = torch.full((B, 2), 0.2, dtype=DTYPE)
    dest_p = torch.full((B, 2), -0.3, dtype=DTYPE)
    out = HK.geodesic(
        0.0, src_cartesian=src_p, dest_cartesian=dest_p, cartesian_model=Geometry.POINCARE
    )
    assert out.shape == (B, 2)
    assert torch.allclose(out, src_p, atol=1e-6)


def test_geodesic_poincare_lift_t1_returns_dest():
    # Pins the dest-symmetry decision: dest honors cartesian_model=POINCARE too,
    # and the chart-aware output returns Poincare coords -> recovers dest_p.
    src_p = torch.full((B, 2), 0.2, dtype=DTYPE)
    dest_p = torch.full((B, 2), -0.3, dtype=DTYPE)
    out = HK.geodesic(
        1.0, src_cartesian=src_p, dest_cartesian=dest_p, cartesian_model=Geometry.POINCARE
    )
    assert out.shape == (B, 2)
    assert torch.allclose(out, dest_p, atol=1e-6)


def test_geodesic_cartesian_output_without_model_raises():
    # New chart-aware contract: requesting CARTESIAN output from polar inputs
    # without naming the output chart (cartesian_model) is a ValueError.
    with pytest.raises(ValueError):
        HK.geodesic(
            0.5,
            src_radial=torch.full((B,), 0.3, dtype=DTYPE),
            src_angular=torch.zeros(B, dtype=DTYPE),
            dest_radial=torch.full((B,), 0.5, dtype=DTYPE),
            dest_angular=torch.zeros(B, dtype=DTYPE),
            output_coord=Coordinate.CARTESIAN,
        )


# ===========================================================================
# 7. Intentional divergence from the reference (no parity test).
# ===========================================================================
# geo_bridge is the *corrected* d==2 module: its free heat kernel returns a
# UNIFORM angle (H^2 from the origin is rotationally symmetric), while the
# reference hyper_bridge.py folds the target-conditioned Poisson-kernel angle
# into its "free" kernel. The two no longer match bit-for-bit, by design -- the
# Poisson angle lives in geo_bridge's bridge methods instead. The former
# bit-exact parity test was dropped when this divergence was adopted.


# ===========================================================================
# 8. Light distributional sanity
# ===========================================================================


def test_free_hyperbolic_outputs_finite(ts):
    torch.manual_seed(0)
    rhos, thetas = HK.binary_free_hyperbolic_heat_kernel(ts)
    assert torch.all(torch.isfinite(rhos)) and torch.all(torch.isfinite(thetas))


def test_free_hyperbolic_theta_mean_near_zero():
    # Symmetric angular law -> sample mean of theta ~ 0 with loose tolerance.
    torch.manual_seed(0)
    big_ts = torch.full((20000,), 1.0, dtype=DTYPE)
    _, thetas = HK.binary_free_hyperbolic_heat_kernel(big_ts)
    assert abs(float(thetas.mean())) < 0.05


def test_free_hyperbolic_angle_is_uniform():
    # Free kernel from the origin is rotationally symmetric -> theta is uniform on
    # [-pi, pi): all low-order circular Fourier coefficients vanish.
    torch.manual_seed(0)
    big_ts = torch.full((200000,), 1.0, dtype=DTYPE)
    _, thetas = HK.binary_free_hyperbolic_heat_kernel(big_ts)
    assert abs(float(torch.cos(thetas).mean())) < 0.02
    assert abs(float(torch.sin(thetas).mean())) < 0.02
    assert abs(float(torch.cos(2 * thetas).mean())) < 0.02


def test_binary_poincare_bridge_angle_follows_poisson_kernel():
    # Bridge angle relative to the target follows the H^2 Poisson kernel: its mean
    # resultant E[cos(theta - target)] equals E[tanh(rho/2)] (the closed-form first
    # circular moment of (cosh rho - sinh rho cos .)^{-1}).
    torch.manual_seed(0)
    n = 200000
    big_ts = torch.full((n,), 1.0, dtype=DTYPE)
    word_embedding = torch.tensor([[0.0, 1.0]], dtype=DTYPE)  # target angle = pi/2
    targets = torch.zeros(n, dtype=torch.long)
    rhos, thetas = HK.binary_poincare_bridge(
        big_ts, targets, word_embedding, Coordinate.HYPERBOLIC_POLAR
    )
    emp = float(torch.cos(thetas - math.pi / 2).mean())
    expected = float(torch.tanh(rhos / 2).mean())
    assert abs(emp - expected) < 0.02
