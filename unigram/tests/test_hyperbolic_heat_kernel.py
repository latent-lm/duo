"""Contract tests for the d-dimensional ``HyperbolicHeatKernel`` in ``geo_bridge.py``.

Covers, per the requested scope:
- coordinate-conversion round-trip consistency (polar <-> Lorentz <-> Poincare);
- consistency of the new general-d methods with the binary (d == 2) methods;
- radial-law correctness (vs the exact H^3 kernel + the Euclidean short-time limit);
- free-kernel / bridge / geodesic shapes, manifold invariants, and concentration.

Import resolves via ``unigram/tests/conftest.py`` (puts ``unigram/`` on path).
Run: ``cd <repo root> && conda run -n duo python -m pytest unigram/tests -q``
"""

from __future__ import annotations

import math

import pytest
import torch

from geo_bridge import (
    BinaryHyperbolicHeatKernel as BHK,
    Coordinate,
    GeoUtils,
    Geometry,
    HyperbolicHeatKernel as HK,
)

DTYPE = torch.float64


def lorentz_inner(z):
    return -z[..., 0] ** 2 + (z[..., 1:] ** 2).sum(-1)


def euclid_mean(d):  # E[chi_d]: the short-time radial scale E[rho]/sqrt(t) as t -> 0
    return math.sqrt(2.0) * math.gamma((d + 1) / 2.0) / math.gamma(d / 2.0)


def lorentz_dist(a, b):
    inner = -a[..., 0] * b[..., 0] + (a[..., 1:] * b[..., 1:]).sum(-1)
    return torch.acosh((-inner).clamp_min(1.0))


# ===========================================================================
# 1. Coordinate-conversion round-trips (general d)
# ===========================================================================


@pytest.mark.parametrize("d", [2, 3, 4, 5, 8])
def test_polar_lorentz_polar_roundtrip(d):
    torch.manual_seed(0)
    rho = torch.rand(200, dtype=DTYPE) * 1.5 + 0.1
    u = torch.randn(200, d, dtype=DTYPE)
    u = u / u.norm(dim=-1, keepdim=True)
    z = GeoUtils.hyperbolic_polar_to_lorentz_cartesian(rho, u)
    assert torch.allclose(lorentz_inner(z), torch.full((200,), -1.0, dtype=DTYPE), atol=1e-6)
    r2, u2 = GeoUtils.lorentz_cartesian_to_hyperbolic_polar(z)
    assert torch.allclose(r2, rho, atol=1e-6)
    assert torch.allclose(u2, u, atol=1e-6)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_polar_poincare_lorentz_polar_roundtrip(d):
    # polar -> Poincare -> Lorentz -> polar
    torch.manual_seed(0)
    rho = torch.rand(200, dtype=DTYPE) * 1.5 + 0.1
    u = torch.randn(200, d, dtype=DTYPE)
    u = u / u.norm(dim=-1, keepdim=True)
    zp = GeoUtils.hyperbolic_polar_to_poincare_cartesian(rho, u)
    assert torch.all(zp.norm(dim=-1) < 1.0)
    zl = GeoUtils.poincare_cartesian_to_lorentz_cartesian(zp)
    r2, u2 = GeoUtils.lorentz_cartesian_to_hyperbolic_polar(zl)
    assert torch.allclose(r2, rho, atol=1e-6)
    assert torch.allclose(u2, u, atol=1e-6)


@pytest.mark.parametrize("d", [2, 3, 4, 7])
def test_lorentz_poincare_lorentz_roundtrip(d):
    # Lorentz -> Poincare -> Lorentz
    torch.manual_seed(0)
    rho = torch.rand(200, dtype=DTYPE) * 1.5 + 0.1
    u = torch.randn(200, d, dtype=DTYPE)
    z = GeoUtils.hyperbolic_polar_to_lorentz_cartesian(rho, u)
    zp = GeoUtils.lorentz_cartesian_to_poincare_cartesian(z)
    assert torch.all(zp.norm(dim=-1) < 1.0)
    z2 = GeoUtils.poincare_cartesian_to_lorentz_cartesian(zp)
    assert torch.allclose(z2, z, atol=1e-6)


@pytest.mark.parametrize("d", [2, 3, 6])
def test_poincare_lorentz_poincare_roundtrip(d):
    # Poincare -> Lorentz -> Poincare
    torch.manual_seed(0)
    z = torch.randn(200, d, dtype=DTYPE)
    z = z * (0.6 / z.norm(dim=-1, keepdim=True))  # ||z|| = 0.6 < 1
    zl = GeoUtils.poincare_cartesian_to_lorentz_cartesian(z)
    assert torch.allclose(lorentz_inner(zl), torch.full((200,), -1.0, dtype=DTYPE), atol=1e-6)
    z2 = GeoUtils.lorentz_cartesian_to_poincare_cartesian(zl)
    assert torch.allclose(z2, z, atol=1e-6)


# ===========================================================================
# 2. Consistency of the general-d converters with the binary (d == 2) ones
# ===========================================================================


def test_polar_to_lorentz_matches_binary_at_d2():
    torch.manual_seed(0)
    rho = torch.rand(100, dtype=DTYPE) * 1.5 + 0.1
    theta = (torch.rand(100, dtype=DTYPE) - 0.5) * 2 * math.pi
    u = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    z_dd = GeoUtils.hyperbolic_polar_to_lorentz_cartesian(rho, u)
    z_bin = GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(rho, theta)
    assert torch.allclose(z_dd, z_bin, atol=1e-9)


def test_polar_to_poincare_matches_binary_at_d2():
    torch.manual_seed(0)
    rho = torch.rand(100, dtype=DTYPE) * 1.5 + 0.1
    theta = (torch.rand(100, dtype=DTYPE) - 0.5) * 2 * math.pi
    u = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    z_dd = GeoUtils.hyperbolic_polar_to_poincare_cartesian(rho, u)
    z_bin = GeoUtils.binary_hyperbolic_polar_to_poincare_cartesian(rho, theta)
    assert torch.allclose(z_dd, z_bin, atol=1e-9)


def test_lorentz_to_polar_matches_binary_at_d2():
    # general-d returns a unit vector u; binary returns the scalar angle atan2(z2, z1).
    torch.manual_seed(0)
    rho = torch.rand(100, dtype=DTYPE) * 1.5 + 0.1
    theta = (torch.rand(100, dtype=DTYPE) - 0.5) * 2 * math.pi
    z = GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(rho, theta)
    r_dd, u_dd = GeoUtils.lorentz_cartesian_to_hyperbolic_polar(z)
    r_bin, th_bin = GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar(z)
    assert torch.allclose(r_dd, r_bin, atol=1e-9)
    assert torch.allclose(torch.atan2(u_dd[..., 1], u_dd[..., 0]), th_bin, atol=1e-9)


# ===========================================================================
# 3. Consistency of the general-d kernel/bridge with the binary ones (d == 2,
#    distributional: same laws, independent RNG)
# ===========================================================================


def test_free_radial_matches_binary_at_d2():
    torch.manual_seed(0)
    rho_dd, _ = HK.free_hyperbolic_heat_kernel(torch.full((4,), 1.0, dtype=DTYPE), 20000, 2)
    torch.manual_seed(0)
    rho_bin, _ = BHK.binary_free_hyperbolic_heat_kernel(torch.full((80000,), 1.0, dtype=DTYPE))
    # same H^2 radial law -> matching mean / std (loose, Monte-Carlo + grid tolerance)
    assert abs(float(rho_dd.mean()) - float(rho_bin.mean())) < 0.03
    assert abs(float(rho_dd.std()) - float(rho_bin.std())) < 0.03


def test_bridge_concentration_matches_binary_at_d2():
    # Both bridges put the Poisson-kernel angle on the target; first circular moment
    # to the target is E[tanh(rho/2)] in either parametrization.
    torch.manual_seed(0)
    we = torch.tensor([[1.0, 0.0]], dtype=DTYPE)  # target direction = +x axis (angle 0)
    rho_dd, u_dd = HK.poincare_bridge(
        torch.full((4,), 0.7, dtype=DTYPE), torch.zeros(4, 20000, dtype=torch.long), we, Coordinate.HYPERBOLIC_POLAR
    )
    emp_dd = float((u_dd[..., 0]).mean())  # <u, e_x>
    torch.manual_seed(0)
    rho_b, th_b = BHK.binary_poincare_bridge(
        torch.full((80000,), 0.7, dtype=DTYPE), torch.zeros(80000, dtype=torch.long), we, Coordinate.HYPERBOLIC_POLAR
    )
    emp_b = float(torch.cos(th_b).mean())
    assert abs(emp_dd - emp_b) < 0.02


# ===========================================================================
# 4. Radial-law correctness (vs exact H^3 + Euclidean short-time limit)
# ===========================================================================


def test_free_radial_matches_H3_closed_form():
    g = torch.linspace(1e-4, 40, 400000, dtype=DTYPE)
    dn = g * torch.sinh(g) * torch.exp(-g * g / 2.0)
    dn = dn / torch.trapz(dn, g)
    exact = float(torch.trapz(g * dn, g))
    rho, _ = HK.free_hyperbolic_heat_kernel(torch.full((4,), 1.0, dtype=DTYPE), 20000, 3)
    assert abs(float(rho.mean()) - exact) < 0.03


@pytest.mark.parametrize("d", [2, 3, 4, 5, 8])
def test_free_radial_euclidean_short_time_limit(d):
    torch.manual_seed(0)
    rho, _ = HK.free_hyperbolic_heat_kernel(torch.full((4,), 1e-3, dtype=DTYPE), 20000, d)
    assert abs(float(rho.mean()) / math.sqrt(1e-3) - euclid_mean(d)) < 0.04


# ===========================================================================
# 5. Free-kernel shapes / invariants
# ===========================================================================


def test_free_kernel_shapes_and_invariants():
    torch.manual_seed(0)
    ts = torch.linspace(0.2, 2.0, 4, dtype=DTYPE)
    rho, u = HK.free_hyperbolic_heat_kernel(ts, 5, 4)
    assert rho.shape == (4, 5) and u.shape == (4, 5, 4)
    assert torch.all(rho >= 0)
    assert torch.allclose(u.norm(dim=-1), torch.ones(4, 5, dtype=DTYPE), atol=1e-10)
    zp = HK.free_poincare_heat_kernel(ts, 5, 4, Coordinate.CARTESIAN)
    assert zp.shape == (4, 5, 4) and torch.all(zp.norm(dim=-1) < 1.0)
    zl = HK.free_lorentz_heat_kernel(ts, 5, 4, Coordinate.CARTESIAN)
    assert zl.shape == (4, 5, 5)
    assert torch.allclose(lorentz_inner(zl), torch.full((4, 5), -1.0, dtype=DTYPE), atol=1e-6)


def test_free_angle_is_uniform_on_sphere():
    torch.manual_seed(0)
    _, u = HK.free_hyperbolic_heat_kernel(torch.full((4,), 1.0, dtype=DTYPE), 20000, 5)
    # uniform on S^{d-1} -> mean direction ~ 0
    assert float(u.reshape(-1, 5).mean(0).norm()) < 0.05


# ===========================================================================
# 6. Bridge
# ===========================================================================


def test_bridge_concentrates_toward_target():
    torch.manual_seed(0)
    we = torch.randn(7, 4, dtype=DTYPE)
    rho, u = HK.poincare_bridge(
        torch.full((4,), 0.7, dtype=DTYPE), torch.zeros(4, 20000, dtype=torch.long), we, Coordinate.HYPERBOLIC_POLAR
    )
    x = we[0] / we[0].norm()
    assert float((u * x).sum(-1).mean()) > 0.3
    assert torch.allclose(u.norm(dim=-1), torch.ones(4, 20000, dtype=DTYPE), atol=1e-10)


def test_lorentz_bridge_on_manifold_and_shape():
    torch.manual_seed(0)
    we = torch.randn(5, 6, dtype=DTYPE)
    z = HK.lorentz_bridge(
        torch.full((3,), 0.5, dtype=DTYPE), torch.zeros(3, 4, dtype=torch.long), we, Coordinate.CARTESIAN
    )
    assert z.shape == (3, 4, 7)
    assert torch.allclose(lorentz_inner(z), torch.full((3, 4), -1.0, dtype=DTYPE), atol=1e-6)


# ===========================================================================
# 7. Geodesic
# ===========================================================================


def _lorentz_endpoints(d=5):
    torch.manual_seed(0)
    B, L = 4, 3
    src = GeoUtils.hyperbolic_polar_to_lorentz_cartesian(
        torch.full((B, L), 0.3, dtype=DTYPE), torch.randn(B, L, d, dtype=DTYPE)
    )
    dst = GeoUtils.hyperbolic_polar_to_lorentz_cartesian(
        torch.full((B, L), 0.8, dtype=DTYPE), torch.randn(B, L, d, dtype=DTYPE)
    )
    return src, dst


def test_geodesic_endpoints_and_on_manifold():
    src, dst = _lorentz_endpoints()
    out0 = HK.geodesic(0.0, src_cartesian=src, dest_cartesian=dst, cartesian_model=Geometry.LORENTZ)
    out1 = HK.geodesic(1.0, src_cartesian=src, dest_cartesian=dst, cartesian_model=Geometry.LORENTZ)
    mid = HK.geodesic(0.5, src_cartesian=src, dest_cartesian=dst, cartesian_model=Geometry.LORENTZ)
    assert torch.allclose(out0, src, atol=1e-6)
    assert torch.allclose(out1, dst, atol=1e-6)
    assert torch.allclose(lorentz_inner(mid), torch.full((4, 3), -1.0, dtype=DTYPE), atol=1e-6)


def test_geodesic_constant_speed():
    src, dst = _lorentz_endpoints()
    total = lorentz_dist(src, dst)
    for frac in (0.25, 0.5, 0.75):
        g = HK.geodesic(frac, src_cartesian=src, dest_cartesian=dst, cartesian_model=Geometry.LORENTZ)
        assert torch.allclose(lorentz_dist(src, g), frac * total, atol=1e-5)


def test_geodesic_per_sample_t():
    # per-sample t column of shape (B, L, 1) must broadcast against the ambient points.
    src, dst = _lorentz_endpoints()  # (4, 3, 6)
    t0 = torch.zeros(4, 3, 1, dtype=DTYPE)
    t1 = torch.ones(4, 3, 1, dtype=DTYPE)
    assert torch.allclose(
        HK.geodesic(t0, src_cartesian=src, dest_cartesian=dst, cartesian_model=Geometry.LORENTZ), src, atol=1e-6)
    assert torch.allclose(
        HK.geodesic(t1, src_cartesian=src, dest_cartesian=dst, cartesian_model=Geometry.LORENTZ), dst, atol=1e-6)
    tr = torch.rand(4, 3, 1, dtype=DTYPE)
    out = HK.geodesic(tr, src_cartesian=src, dest_cartesian=dst, cartesian_model=Geometry.LORENTZ)
    assert out.shape == src.shape
    assert torch.allclose(lorentz_inner(out), torch.full((4, 3), -1.0, dtype=DTYPE), atol=1e-6)
    # a uniform per-sample column matches the scalar-t result exactly
    g_col = HK.geodesic(torch.full((4, 3, 1), 0.4, dtype=DTYPE), src_cartesian=src, dest_cartesian=dst, cartesian_model=Geometry.LORENTZ)
    g_scalar = HK.geodesic(0.4, src_cartesian=src, dest_cartesian=dst, cartesian_model=Geometry.LORENTZ)
    assert torch.allclose(g_col, g_scalar, atol=1e-9)


def test_bridge_antipodal_target():
    # target = -e_1: the Householder reflection must map e_1 -> -e_1 without degenerate collapse.
    torch.manual_seed(0)
    we = torch.zeros(1, 4, dtype=DTYPE); we[0, 0] = -1.0
    rho, u = HK.poincare_bridge(
        torch.full((2,), 0.7, dtype=DTYPE), torch.zeros(2, 5000, dtype=torch.long), we, Coordinate.HYPERBOLIC_POLAR
    )
    assert float((u * we[0]).sum(-1).mean()) > 0.3  # concentrates toward -e_1
    assert torch.allclose(u.norm(dim=-1), torch.ones(2, 5000, dtype=DTYPE), atol=1e-10)


def test_geodesic_poincare_lift_t0_returns_source():
    torch.manual_seed(0)
    sp = torch.randn(4, 3, 5, dtype=DTYPE); sp = sp * (0.3 / sp.norm(dim=-1, keepdim=True))
    dp = torch.randn(4, 3, 5, dtype=DTYPE); dp = dp * (0.5 / dp.norm(dim=-1, keepdim=True))
    out = HK.geodesic(0.0, src_cartesian=sp, dest_cartesian=dp, cartesian_model=Geometry.POINCARE)
    assert out.shape == (4, 3, 5)
    assert torch.allclose(out, sp, atol=1e-6)


def test_geodesic_polar_input_runs():
    torch.manual_seed(0)
    sr = torch.full((4, 3), 0.3, dtype=DTYPE); sa = torch.randn(4, 3, 5, dtype=DTYPE)
    dr = torch.full((4, 3), 0.6, dtype=DTYPE); da = torch.randn(4, 3, 5, dtype=DTYPE)
    rho, u = HK.geodesic(0.5, src_radial=sr, src_angular=sa, dest_radial=dr, dest_angular=da)
    assert rho.shape == (4, 3) and u.shape == (4, 3, 5)


# ===========================================================================
# 8. Error paths
# ===========================================================================


def test_sample_radial_rejects_d_lt_2():
    with pytest.raises(ValueError):
        HK.sample_radial(torch.full((2,), 1.0, dtype=DTYPE), 1, 3)


def test_free_lorentz_raises_on_large_rho():
    # Radial BM escapes ~linearly (E[rho] ~ (d-1)/2 * t for large t), so t=30, d=4
    # gives rho ~ 47 > _LORENTZ_RHO_MAX=20 -> the cartesian path must refuse.
    torch.manual_seed(0)
    with pytest.raises(ValueError):
        HK.free_lorentz_heat_kernel(torch.full((2,), 30.0, dtype=DTYPE), 2, 4, Coordinate.CARTESIAN)


def test_geodesic_raises_when_both_source_forms_given():
    src, dst = _lorentz_endpoints()
    with pytest.raises(ValueError):
        HK.geodesic(
            0.5, src_cartesian=src, cartesian_model=Geometry.LORENTZ,
            src_radial=torch.full((4, 3), 0.3, dtype=DTYPE), src_angular=torch.randn(4, 3, 5, dtype=DTYPE),
            dest_cartesian=dst,
        )
