# ARCH: `unigram/geo_bridge.py`

Fix-to-intent contract for the cleaned-up, self-contained binary (`d == 2`)
hyperbolic heat-kernel + bridge + geodesic module. This is a **reorganization**
of the `d == 2` code paths from `unigram/hyper_bridge.py` (the reference, ground
truth for behavior). The module compiles but is riddled with dangling references
from a sloppy extraction; this document pins the intended API so a test-writer
and implementer can fix it without re-deriving intent.

**Authority:** where this doc and the current `geo_bridge.py` source disagree,
this doc + the reference win. The implementer changes `geo_bridge.py` to match.

**Status (as-built, revised):** the module has been implemented and is covered by
`unigram/tests/test_geo_bridge.py`. One design point was *revised after* this doc
was first written and now intentionally diverges from the reference: the free
heat kernel returns a **uniform** angle (H^2 from the origin is rotationally
symmetric), and the target-conditioned **Poisson-kernel angle moved into the
bridge methods**. The sections below are updated to this as-built design; for the
*angle* specifically, the code + this Status note supersede any older
"reference says X" intent. The uniform-angle change is intentionally **not
propagated** to `hyper_bridge.py` (which keeps its Poisson-angle-in-free-kernel
convention), so geo_bridge is no longer bit-exact with it and the parity test was
dropped. (Any pre-existing edits to `hyper_bridge.py` in the working tree are
unrelated to this change.)

## Scope

In:
- Two string-tag dataclasses: `Geometry`, `Coordinate`.
- A `GeoUtils` class holding the boundary-check helpers and the pure coordinate
  converters as `@staticmethod`s.
- `FreeBinaryHyperbolicHeatKernel(GeoUtils)`: the closed-form `H^2` sampler,
  bridge, and geodesic, all `@staticmethod`.
- Two module-level numeric constants `_LORENTZ_RHO_MAX`, `_SPHERE_T_MAX`.

Out (explicitly NOT in this module — they live only in `hyper_bridge.py`):
- `FreeHyperbolicHeatKernel` (`H^d`, `d >= 3`) and its angular samplers
  (`_angular_boost`, `_angular_icdf`, `_angular_vmf`, `_reflect_to_target`).
- `FreeSphericalHeatKernel` and the whole `S^d` path.
- `_gruet_radial`, `_polar_direction` (the `d >= 3` dispatcher), and the
  `d >= 3` Poincare/sphere converters
  (`poincare_polar_to_lorentz_cartesian`, `poincare_cartesian_to_poincare_polar`,
  `sphere_*`).
- `_uniform_sphere`, `_check_sphere_t_bound`: present in `GeoUtils` (carried over
  from the extraction) but **unused** by any `FreeBinaryHyperbolicHeatKernel`
  method. They are kept verbatim for parity but are dead within this module. Do
  NOT delete (pre-existing, not orphaned by the fix); do NOT wire them in.

This module does **not** import from `hyper_bridge.py` (see Open Questions §1).
No existing files are modified. No callers exist yet (`grep` finds none), so
there is no downstream contract to preserve beyond this document.

## Module layout

Single file: `/share/thickstun/sychou/workspace/research/duo-dev/unigram/geo_bridge.py`

Top-to-bottom order (keep the existing order; fixes are surgical):
1. Module docstring, `import torch`, `from dataclasses import dataclass`,
   `from typing import Optional, Tuple`.
2. `@dataclass Geometry`, `@dataclass Coordinate`.
3. Constants `_LORENTZ_RHO_MAX = 20.0`, `_SPHERE_T_MAX = 0.5`.
4. `class GeoUtils:` — helpers + converters (all `@staticmethod`).
5. `class FreeBinaryHyperbolicHeatKernel(GeoUtils):` — sampler/bridge/geodesic.
6. A module-level `_geodesic_kernel` (NEW — inlined; see Open Questions §1).

## Calling-convention rule for GeoUtils (resolves bug #5)

**Every method of `GeoUtils` is decorated `@staticmethod`** (those that mutate no
state and take no `self`/`cls`). Cross-references between them use the fully
qualified form `GeoUtils.<method>(...)`, never a bare name and never `self.`.
The boundary-check helpers additionally take their args positionally as today.

`FreeBinaryHyperbolicHeatKernel` inherits `GeoUtils`, so inside its methods a
converter may be called as either `GeoUtils.<m>(...)` or
`FreeBinaryHyperbolicHeatKernel.<m>(...)`; **standardize on `GeoUtils.<m>(...)`**
for the converters/helpers and `FreeBinaryHyperbolicHeatKernel.<m>(...)` for the
kernel/bridge methods (matches how the reference qualifies its own static
methods, e.g. `FreeBinaryHyperbolicHeatKernel.sample_chi`).

Rationale: the current file is inconsistent — `binary_hyperbolic_polar_to_lorentz_cartesian`
has `@staticmethod` but the sibling converters do not, and several callers use
bare names (`binary_free_hyperbolic_heat_kernel(...)`,
`binary_hyperbolic_polar_to_poincare_cartesian(...)`,
`_check_lorentz_rho_bound(...)`, `_geodesic_kernel(...)`,
`poincare_cartesian_to_lorentz_cartesian(...)`) that resolve to nothing now that
the functions are class members. Uniform `@staticmethod` + qualified calls is the
minimal fix that makes every call site resolvable.

## Geometry / Coordinate tags

```python
@dataclass
class Geometry:
    POINCARE: str = "poincare"
    LORENTZ: str = "lorentz"

@dataclass
class Coordinate:
    HYPERBOLIC_POLAR: str = "polar"
    CARTESIAN: str = "cartesian"
```

- Keep exactly these field names and string values. `geo_bridge`'s `Geometry`
  intentionally differs from the reference's (`LORENTZ_POLAR`/`LORENTZ_CARTESIAN`)
  — do NOT import the reference's enum; `geo_bridge` only needs
  `POINCARE`/`LORENTZ` for the `cartesian_model` dispatch in `geodesic`.
- `Coordinate.HYPERBOLIC_POLAR == "polar"`, `Coordinate.CARTESIAN == "cartesian"`.
  These string values must match the reference's `Coordinate.POLAR`/`.CARTESIAN`
  so behavior is identical; only the attribute name differs
  (`HYPERBOLIC_POLAR` vs `POLAR`).
- Used purely as namespaced string constants; never instantiated.

## Constants

```python
_LORENTZ_RHO_MAX: float = 20.0   # Lorentz-Cartesian refused above this rho
_SPHERE_T_MAX: float = 0.5       # carried over, UNUSED in this module
```

## GeoUtils methods

All `@staticmethod`. Shapes: `B` is `batch_size`; `(...)` is an arbitrary
leading batch shape. Tensors are floating point (`float64` is the working dtype;
methods are dtype-agnostic and preserve the input dtype).

### `_check_lorentz_rho_bound`
```python
@staticmethod
def _check_lorentz_rho_bound(rhos: torch.Tensor, d: int, ts: Optional[torch.Tensor] = None) -> None
```
- `rhos`: `(...)` radial values. `d`: int. `ts`: optional `(...)` for the error msg.
- Returns `None`. Raises `ValueError` iff `rhos.numel() > 0` and
  `rhos.max() > _LORENTZ_RHO_MAX (= 20.0)`. Empty tensor → no-op return.
- Error string embeds `d`, optional `max(ts)`, and `max(rho)`; mentions
  `output_coord=HYPERBOLIC_POLAR` as the escape hatch (the reference says
  `POLAR`; in this module the doc/escape-hatch text should read
  `HYPERBOLIC_POLAR` to match the local enum — cosmetic, not load-bearing).
- Reference: `_check_lorentz_rho_bound` (hyper_bridge L74). Behavior identical.

### `_uniform_sphere` (UNUSED — keep verbatim)
```python
@staticmethod
def _uniform_sphere(B: int, d: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor
```
- Returns `(B, d)` unit vectors (rows on `S^{d-1}`). Raises `ValueError` if `d == 1`.
- Reference: `_uniform_sphere` (hyper_bridge L98). Dead in this module.

### `_check_sphere_t_bound` (UNUSED — keep verbatim)
```python
@staticmethod
def _check_sphere_t_bound(ts: torch.Tensor, d: int) -> None
```
- Returns `None`; raises `ValueError` iff `ts.max() > _SPHERE_T_MAX (= 0.5)`.
- Reference: `_check_sphere_t_bound` (hyper_bridge L108). Dead in this module.

### `_binary_polar_direction`
```python
@staticmethod
def _binary_polar_direction(thetas: torch.Tensor) -> torch.Tensor
```
- `thetas`: `(B,)` scalar angles.
- Returns `(B, 2)` = `stack([cos(thetas), sin(thetas)], dim=-1)`. Unit-norm rows
  by construction (`cos^2 + sin^2 = 1`).
- This is the `d == 2` specialization of the reference's `_polar_direction`
  (hyper_bridge L126), with the `d >= 3` unit-vector branch dropped. Reference's
  signature is `(rhos, thetas_or_u)`; here it takes only `thetas`. Keep the
  geo_bridge signature.
- Invariant: output rows have `||.|| == 1`.

### `binary_hyperbolic_polar_to_poincare_cartesian`
```python
@staticmethod
@torch.no_grad()
def binary_hyperbolic_polar_to_poincare_cartesian(rhos: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor
```
- `rhos`: `(B,)` `>= 0`. `thetas`: `(B,)`.
- Returns `(B, 2)` Poincare-disk Cartesian `z = tanh(rho/2) * direction`.
- Body: `direction = GeoUtils._binary_polar_direction(thetas=thetas)`; scale
  `tanh(rho/2)` clamped to `1 - eps`.
- **Invariant:** `||z|| < 1` strictly, for arbitrarily large `rho`.
- Reference: `poincare_polar_to_poincare_cartesian` (hyper_bridge L143, `d == 2`
  path). Note geo_bridge currently MISSING the `@staticmethod` decorator — add it.

### `binary_hyperbolic_polar_to_lorentz_cartesian`
```python
@staticmethod
@torch.no_grad()
def binary_hyperbolic_polar_to_lorentz_cartesian(rhos: torch.FloatTensor, thetas: torch.FloatTensor) -> torch.FloatTensor
```
- `rhos`: `(B,)` `>= 0`. `thetas`: `(B,)`.
- Returns `(B, 3)` = `stack([cosh(rho), sinh(rho)*cos(theta), sinh(rho)*sin(theta)], dim=-1)`.
- **Component order:** `z = (cosh ρ, sinh ρ cos θ, sinh ρ sin θ)`, i.e.
  `z[...,0]` is the time component, `z[...,1:]` spatial.
- **Invariant:** `<z,z>_L == -1` i.e. `-z0^2 + z1^2 + z2^2 == -1` (exact up to
  float cancellation; the caller gates large rho via `_check_lorentz_rho_bound`).
- This converter does NOT itself call `_check_lorentz_rho_bound`; callers do.
- Reference: `FreeBinaryHyperbolicHeatKernel.poincare_polar_to_lorentz_cartesian`
  (hyper_bridge L439). Body is identical and ALREADY correct in geo_bridge.
  (The simple `stack` form, NOT the rescaled module-level `d>=3` form at L160.)

### `binary_lorentz_cartesian_to_hyperbolic_polar`  (resolves bug #3)
```python
@staticmethod
@torch.no_grad()
def binary_lorentz_cartesian_to_hyperbolic_polar(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```
- `z`: `(..., 3)` Lorentz-Cartesian H^2 point, component order
  `(cosh ρ, sinh ρ cos θ, sinh ρ sin θ)`, with `z[...,0] >= 1`.
- Returns `Tuple[(...), (...)]` = `(rhos, thetas)`:
  - `rhos = arccosh(z[..., 0].clamp_min(1.0))`, shape `(...)`, `>= 0`.
  - `thetas = atan2(z[..., 2], z[..., 1])`, shape `(...)`, range `(-pi, pi]`.
- **Error:** raise `ValueError(f"Cartesian dimension should be 3, not {d}.")`
  when `z.shape[-1] != 3`. (NOTE: an `H^2` Lorentz point is 3-dimensional — time
  + 2 spatial. The current code checks `!= 2` and the message says "2"; both are
  wrong. Use `3`.)
- **Three bugs to fix (all in geo_bridge L190–211):**
  1. `rhos` is never computed → returns undefined `rhos`. Add
     `rhos = torch.acosh(z[..., 0].clamp_min(1.0))`.
  2. dimension check `d != 2` is wrong → must be `d != 3`.
  3. `atan2(z[...,1], z[...,0])` uses the **time** component `z[...,0]` as the
     `x` arg → must use the two **spatial** components:
     `atan2(z[..., 2], z[..., 1])`.
- Reference: `lorentz_cartesian_to_poincare_polar` (hyper_bridge L234, `d == 2`
  path; reference operates on `z[...,0]` for rho and `spatial = z[...,1:]` then
  `atan2(spatial[...,1], spatial[...,0])`). geo_bridge's name and `(rho, theta)`
  return match the reference's `d == 2` output. The reference also clamps
  `z[...,0]` with `.clamp_min(1.0)` before `acosh` — replicate that.

### `poincare_cartesian_to_lorentz_cartesian`
```python
@staticmethod
@torch.no_grad()
def poincare_cartesian_to_lorentz_cartesian(z: torch.Tensor) -> torch.Tensor
```
- `z`: `(..., d)` Poincare-disk Cartesian, `||z|| < 1`. (In this module always
  `d == 2`, but the op is dimension-generic; do not special-case.)
- Returns `(..., d + 1)` Lorentz-Cartesian via stereographic lift
  `z -> ((1 + ||z||^2)/(1 - ||z||^2), 2z/(1 - ||z||^2))`.
- **Invariant:** `-out[...,0]^2 + sum(out[...,1:]^2) == -1`.
- Reference: `poincare_cartesian_to_lorentz_cartesian` (hyper_bridge L212). Body
  identical and correct in geo_bridge. Add `@staticmethod` (currently missing).
- Used by `geodesic` when `cartesian_model == Geometry.POINCARE`.

### `lorentz_cartesian_to_poincare_cartesian`  (NEW — implement the stub)
```python
@staticmethod
@torch.no_grad()
def lorentz_cartesian_to_poincare_cartesian(z: torch.Tensor) -> torch.Tensor
```
- `z`: `(..., d + 1)` Lorentz-Cartesian on the hyperboloid, `z[..., 0] >= 1`
  (in this module always `(..., 3)`). Inverse of
  `poincare_cartesian_to_lorentz_cartesian`.
- Returns `(..., d)` Poincare-disk Cartesian via the stereographic projection
  `z -> z[..., 1:] / (1 + z[..., 0])`.
- **Invariant:** `||out|| < 1`; round-trips with
  `poincare_cartesian_to_lorentz_cartesian` to `atol=1e-6` for `rho <= _LORENTZ_RHO_MAX`.
- **Status:** geo_bridge currently has this as a `pass` stub (added by the user)
  with a copy-pasted-wrong docstring. Implement the body verbatim from the
  reference; fix the docstring to describe the projection `z[1:]/(1+z[0])` and the
  `(..., d+1) -> (..., d)` shape change (NOT the `-z0^2+...=-1` text it copied).
- Reference: `lorentz_cartesian_to_poincare_cartesian` (hyper_bridge L266):
  ```python
  spatial = z_lorentz[..., 1:]
  t = z_lorentz[..., 0]
  denom = (1.0 + t).unsqueeze(-1).clamp_min(torch.finfo(z_lorentz.dtype).tiny)
  return spatial / denom
  ```
  Use this body (parameter name `z` per the geo_bridge stub).
- Used by `geodesic` to return Poincare output when `cartesian_model == POINCARE`.

## FreeBinaryHyperbolicHeatKernel methods

`class FreeBinaryHyperbolicHeatKernel(GeoUtils):` — all methods `@staticmethod`,
all `@torch.no_grad()` (except `sample_chi`/`sample_chi_old`, which lack the
decorator in the reference; keep that as-is). `ts` dtype is the working dtype
(typically `torch.float64`); outputs preserve it.

### `sample_chi`
```python
@staticmethod
def sample_chi(ns: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor
```
- `ns`: `(B,)` int degrees of freedom. Returns `(B,)` chi samples via
  `chi^2(n) ~ Gamma(n/2, rate=0.5)`, `.sqrt()`. Invariant: output `>= 0`.
- Reference: `FreeBinaryHyperbolicHeatKernel.sample_chi` (hyper_bridge L467).
  Identical and correct.

### `sample_chi_old`
```python
@staticmethod
def sample_chi_old(ns: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor
```
- `ns`: any shape int dof. Returns same-shape chi samples via summed squared
  normals. Legacy/parity only; not on the hot path.
- Reference: hyper_bridge L489. Identical and correct.

### `binary_free_hyperbolic_heat_kernel`  (resolves bug #1)
```python
@staticmethod
@torch.no_grad()
def binary_free_hyperbolic_heat_kernel(ts: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]
```
- `ts`: `(B,)` heat times `> 0`.
- Returns `Tuple[(B,), (B,)]` = `(rhos, thetas)` in `HYPERBOLIC_POLAR`. No
  `output_coord` arg (this is the raw polar producer; the cartesian option lives
  in `binary_free_poincare_heat_kernel`). Remove the stray `output_coord`
  mention from the docstring.
- **As-built body (UNIFORM angle — diverges from the reference):**
  ```python
  ns = torch.poisson(ts / 8).to(torch.int64)
  ss = ts.sqrt() * FreeBinaryHyperbolicHeatKernel.sample_chi(2 * ns + 3, ts.dtype)
  vs = torch.rand_like(ts)
  rhos = torch.acosh(vs.square() + (1 - vs.square()) * torch.cosh(ss))
  us = torch.rand_like(ts)
  thetas = (us - 0.5) * (2 * torch.pi)   # uniform on [-pi, pi)
  return rhos, thetas
  ```
- **Angle law (revised):** the free heat kernel from the origin is rotationally
  symmetric, so `theta` is **uniform on `[-pi, pi)`** — NOT the Poisson-kernel
  conditional. The Poisson-kernel angle `(cosh ρ − sinh ρ cos θ)^{-1}` is applied
  later, inside the bridge methods (see `binary_poincare_bridge`). This is the one
  intentional divergence from `hyper_bridge.py`, whose
  `binary_free_poincare_heat_kernel` folds the Poisson angle into its "free" kernel.
  The **radial** marginal still matches the reference (Gruet); only the angle differs.
- **Common pitfall:** `torch.pi` is a float constant — `torch.pi()` raises
  `TypeError`. Use `torch.pi`.
- **Invariants:** `rhos >= 0`; `thetas` uniform on `[-pi, pi)`. Empty `ts` → empty
  `(rhos, thetas)`.

### `binary_free_poincare_heat_kernel`  (resolves bug #2)
```python
@staticmethod
@torch.no_grad()
def binary_free_poincare_heat_kernel(ts: torch.FloatTensor, output_coord: Optional[str] = None)
```
- `ts`: `(B,)`. `output_coord`: `None` / `"polar"` / `"cartesian"`.
- **Default behavior:** `output_coord is None` behaves as `HYPERBOLIC_POLAR`
  (the `== CARTESIAN` test fails, so the polar tuple is returned). Matches
  reference (default returns `(ps, thetas)`).
- Returns: `HYPERBOLIC_POLAR` → `Tuple[(B,), (B,)]` `(rhos, thetas)`;
  `CARTESIAN` → `(B, 2)` Poincare-disk `z` with `||z|| < 1`.
- **Intended body:**
  ```python
  rhos, thetas = FreeBinaryHyperbolicHeatKernel.binary_free_hyperbolic_heat_kernel(ts=ts)
  if output_coord == Coordinate.CARTESIAN:
      return GeoUtils.binary_hyperbolic_polar_to_poincare_cartesian(rhos, thetas)
  return rhos, thetas
  ```
- **Three bugs to fix (geo_bridge L335–338):**
  1. L335 bare `binary_free_hyperbolic_heat_kernel(ts=ts)` → qualify as
     `FreeBinaryHyperbolicHeatKernel.binary_free_hyperbolic_heat_kernel(ts=ts)`.
  2. L337 bare `binary_hyperbolic_polar_to_poincare_cartesian(ps, thetas)` →
     qualify as `GeoUtils.binary_hyperbolic_polar_to_poincare_cartesian(rhos, thetas)`.
  3. Undefined var `ps` (twice, L337 + L338) → must be `rhos`.
- Reference: `binary_free_poincare_heat_kernel` (hyper_bridge L511). Same
  contract; geo_bridge just delegates the polar production to the new method.

### `binary_free_lorentz_heat_kernel`  (resolves bug #4 part 1)
```python
@staticmethod
@torch.no_grad()
def binary_free_lorentz_heat_kernel(ts: torch.FloatTensor, output_coord: Optional[str] = None)
```
- `ts`: `(B,)`. `output_coord`: `None` / `"polar"` / `"cartesian"`.
- **Default behavior:** `None` → returns Lorentz **CARTESIAN** (the method
  returns polar ONLY when `output_coord == HYPERBOLIC_POLAR`; every other value
  including `None` falls through to cartesian). This matches the reference
  (`binary_free_lorentz_heat_kernel` default = cartesian, L538).
- Returns: `HYPERBOLIC_POLAR` → `Tuple[(B,), (B,)]`;
  `CARTESIAN`/`None` → `(B, 3)` Lorentz-Cartesian; raises `ValueError` if
  `max(rho) > _LORENTZ_RHO_MAX`.
- **Intended body:**
  ```python
  rhos, thetas = FreeBinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel(
      ts=ts, output_coord=Coordinate.HYPERBOLIC_POLAR
  )
  if output_coord == Coordinate.HYPERBOLIC_POLAR:
      return rhos, thetas
  GeoUtils._check_lorentz_rho_bound(rhos, d=2, ts=ts)
  return GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(rhos, thetas)
  ```
- **Bugs to fix (geo_bridge L365–366):** bare `_check_lorentz_rho_bound(...)` →
  `GeoUtils._check_lorentz_rho_bound(...)`; bare
  `binary_hyperbolic_polar_to_lorentz_cartesian(...)` →
  `GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(...)`. (The delegate
  call on L360 is already correctly qualified.)
- Reference: `binary_free_lorentz_heat_kernel` (hyper_bridge L538).

### `binary_poincare_bridge`
```python
@staticmethod
@torch.no_grad()
def binary_poincare_bridge(
    ts: torch.FloatTensor,
    targets: torch.LongTensor,
    word_embedding: torch.FloatTensor,
    output_coord: Optional[str] = None,
)
```
- `ts`: `(B,)`. `targets`: `(B,)` long, indices into `word_embedding`.
  `word_embedding`: `(vocab_size, 2)`; only the angular part is used via
  `atan2(e[...,1], e[...,0])`. `output_coord`: `None` / `"polar"` / `"cartesian"`.
- **Default behavior:** `None` → `HYPERBOLIC_POLAR` (returns polar unless
  explicitly `CARTESIAN`). Matches reference (default polar, L566).
- Returns: `HYPERBOLIC_POLAR` → `Tuple[(B,), (B,)]`; `CARTESIAN` → `(B, 2)`
  Poincare-disk, `||z|| < 1`.
- Body (as-built): draw `(rhos, thetas)` polar from
  `binary_free_poincare_heat_kernel` (uniform angle), reshape the uniform angle
  into the Poisson-kernel angle via the inverse-CDF
  `thetas = 2 * atan(exp(-rho) * tan(thetas * 0.5))` (concentration `exp(-rho)`,
  centered at 0), then add `target_angle = atan2(e[...,1], e[...,0])`, then convert
  if cartesian.
- **Invariant:** given `rho`, `theta - target_angle` follows the Poisson kernel
  `(cosh ρ − sinh ρ cos ·)^{-1}` (first circular moment `tanh(ρ/2)`), so the
  sample concentrates toward the target direction. NOT a pure rotation of the
  free angle.
- Reference: `binary_poincare_bridge` (hyper_bridge L566) — same target rotation,
  but the reference inherits the Poisson angle from its free kernel whereas
  geo_bridge applies it here. The output distribution is the same.

### `binary_lorentz_bridge`
```python
@staticmethod
@torch.no_grad()
def binary_lorentz_bridge(
    ts: torch.FloatTensor,
    targets: torch.LongTensor,
    word_embedding: torch.FloatTensor,
    output_coord: Optional[str] = None,
)
```
- Args as `binary_poincare_bridge`.
- **Default behavior:** `None` → Lorentz **CARTESIAN** (returns polar only when
  `== HYPERBOLIC_POLAR`). Matches reference (default cartesian, L604).
- Returns: `HYPERBOLIC_POLAR` → `Tuple[(B,), (B,)]`; `CARTESIAN`/`None` →
  `(B, 3)` Lorentz-Cartesian; raises `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
- Body: delegate to `binary_poincare_bridge(..., output_coord=POLAR)`, then
  bound-check + convert. ALREADY correctly qualified in geo_bridge (L433–442) —
  confirm resolution post-fix. Note the `_check_lorentz_rho_bound` /
  `binary_hyperbolic_polar_to_lorentz_cartesian` calls here are bare (L441–442)
  → qualify with `GeoUtils.` (same bug class as `binary_free_lorentz_heat_kernel`).
- Reference: `binary_lorentz_bridge` (hyper_bridge L604). Identical.

### `geodesic`  (resolves bug #4 part 2 + Open Questions §2, §3)
```python
@staticmethod
@torch.no_grad()
def geodesic(
    t,
    src_cartesian: Optional[torch.FloatTensor] = None,
    dest_cartesian: Optional[torch.FloatTensor] = None,
    cartesian_model: Optional[str] = None,
    src_radial: Optional[torch.FloatTensor] = None,
    src_angular: Optional[torch.FloatTensor] = None,
    dest_radial: Optional[torch.FloatTensor] = None,
    dest_angular: Optional[torch.FloatTensor] = None,
    output_coord: Optional[str] = None,
)
```
- `t`: float or broadcastable tensor, geodesic fraction (`0` → src, `1` → dest).
- Endpoints: exactly one form per endpoint:
  - cartesian: `src_cartesian` / `dest_cartesian`, shape `(B, 3)`
    (Lorentz) or `(B, 2)` (Poincare) per `cartesian_model`.
  - polar: `(src_radial (B,), src_angular (B,))` and
    `(dest_radial (B,), dest_angular (B,))`.
- `cartesian_model`: `Geometry.POINCARE` or `Geometry.LORENTZ`; required when a
  cartesian endpoint is given. Selects how cartesian inputs are lifted to the
  ambient Lorentz hyperboloid: POINCARE → `poincare_cartesian_to_lorentz_cartesian`,
  LORENTZ → identity.
- `output_coord`: `None` defaults to `CARTESIAN` if `src_cartesian is not None`
  else `HYPERBOLIC_POLAR` (mirror reference L700–701, with the renamed `src`).
- Returns (the CARTESIAN branch is chart-aware per the user's edit — adopted):
  - `CARTESIAN` + `cartesian_model == Geometry.LORENTZ` → `(B, 3)`
    Lorentz-Cartesian interpolant on the hyperboloid (raw `interpolate`).
  - `CARTESIAN` + `cartesian_model == Geometry.POINCARE` → `(B, 2)` Poincare-disk
    via `GeoUtils.lorentz_cartesian_to_poincare_cartesian(interpolate)` (`||z|| < 1`).
  - `CARTESIAN` + `cartesian_model` not in `{LORENTZ, POINCARE}` (e.g. `None`,
    which happens for polar inputs that explicitly request `CARTESIAN`) →
    `ValueError`. To get cartesian output you MUST name the output chart.
  - `HYPERBOLIC_POLAR` → `Tuple[(B,), (B,)]` `(rhos, thetas)` via
    `GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar(interpolate)`.
  - **Symmetry note:** the output chart equals the input chart — Lorentz input →
    Lorentz output, Poincare input → Poincare output. Polar input → polar output
    by default.
- **Errors:** `ValueError` if neither-or-both forms of an endpoint are given
  (the two guard blocks at L494–505 are correct, keep them). `ValueError` if a
  cartesian endpoint is given with `cartesian_model` not in
  `{POINCARE, LORENTZ}`. `ValueError` (via `_check_lorentz_rho_bound`) if a polar
  `rho > _LORENTZ_RHO_MAX`.

- **Bugs to fix (geo_bridge L507–530):**
  1. **Parameter naming (Open Q §2 — DECISION: keep `*_cartesian` names,
     minimal churn).** The guards (L494–505) already use `src_cartesian` /
     `dest_cartesian`, but the body (L508, L510, L520) references undefined `src`
     / `dest`. Rename the body's `src`/`dest` to `src_cartesian`/`dest_cartesian`.
     Do NOT rename the signature.
  2. L508 `src` → `src_cartesian`. L510 `if src is not None` →
     `if src_cartesian is not None`. L512 `poincare_cartesian_to_lorentz_cartesian(z=src)`
     → `GeoUtils.poincare_cartesian_to_lorentz_cartesian(z=src_cartesian)`.
     L514 `x_amb = src` → `x_amb = src_cartesian`.
  3. L519 bare `binary_hyperbolic_polar_to_lorentz_cartesian(...)` →
     `GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(rhos=src_radial, thetas=src_angular)`.
     L518 bare `_check_lorentz_rho_bound(src_radial, d=2)` →
     `GeoUtils._check_lorentz_rho_bound(src_radial, d=2)`.
  4. **dest branch (Open Q §3 — DECISION: dest honors `cartesian_model`
     symmetrically).** The current dest branch (L520–524) ignores
     `cartesian_model` and assumes Lorentz. Make it mirror the src branch:
     ```python
     if dest_cartesian is not None:
         if cartesian_model == Geometry.POINCARE:
             y_amb = GeoUtils.poincare_cartesian_to_lorentz_cartesian(z=dest_cartesian)
         elif cartesian_model == Geometry.LORENTZ:
             y_amb = dest_cartesian
         else:
             raise ValueError(...)  # same message as src
     else:
         GeoUtils._check_lorentz_rho_bound(dest_radial, d=2)
         y_amb = GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(rhos=dest_radial, thetas=dest_angular)
     ```
     Rationale: src and dest must be lifted into the same ambient chart for
     `_geodesic_kernel`'s Lorentz inner product to be meaningful; a single
     `cartesian_model` applying to both endpoints is the only coherent contract,
     and `geo_bridge` already pairs both endpoints with one `cartesian_model`
     param. (The reference's `geodesic` takes Lorentz-only `src`/`dest` and has
     no `cartesian_model`; geo_bridge intentionally generalizes to accept
     Poincare cartesian inputs — this is the documented added capability.)
  5. L526 `_geodesic_kernel(x_amb, y_amb, t, kappa=-1)` → call the module-level
     inlined `_geodesic_kernel` (see Open Questions §1); name `interpolate`
     (geo_bridge's local) is fine, keep it.
  6. **Return block (geo_bridge L546–553, the user's chart-aware edit — adopt
     and fix its bugs):**
     ```python
     if output_coord == Coordinate.CARTESIAN:
         if cartesian_model == Geometry.LORENTZ:
             return interpolate
         elif cartesian_model == Geometry.POINCARE:
             return GeoUtils.lorentz_cartesian_to_poincare_cartesian(z=interpolate)
         else:
             raise ValueError(
                 f"cartesian_model, {cartesian_model}, is not supported, only "
                 f"support ({Geometry.LORENTZ}, {Geometry.POINCARE})."
             )
     return GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar(interpolate)
     ```
     Bugs to fix in the user's version:
     - `Geometry.Lorentz` / `Geometry.Poincare` (L547, L549, L552) → wrong
       attribute casing; the fields are `Geometry.LORENTZ` / `Geometry.POINCARE`
       (`AttributeError` otherwise).
     - `lorentz_cartesian_to_poincare_cartesian(z=interpolate)` (L550) → bare name
       → qualify `GeoUtils.lorentz_cartesian_to_poincare_cartesian(...)`.
     - the stub returns `None` → implement its body (see the new GeoUtils entry).
     - `binary_lorentz_cartesian_to_hyperbolic_polar(interpolate)` (L553) → bare
       name → `GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar(interpolate)`.
       The interpolated ambient point is `(B, 3)`, satisfying the `(..., 3)` contract.
- **Invariants:** for Lorentz output, `<gamma,gamma>_L == -1`; `t == 0` →
  src (on-manifold), `t == 1` → dest; constant geodesic speed.
- Reference: `FreeBinaryHyperbolicHeatKernel.geodesic` (hyper_bridge L642) for
  the polar-input + Lorentz-cartesian-input + `_geodesic_kernel` structure;
  geo_bridge adds the `cartesian_model` Poincare-lift on top.

## Module-level private helper (NEW — inlined)

### `_geodesic_kernel`  (resolves Open Questions §1)
```python
@torch.no_grad()
def _geodesic_kernel(x: torch.Tensor, y: torch.Tensor, t, kappa: int) -> torch.Tensor
```
- `x`, `y`: `(B, 3)` ambient Lorentz-Cartesian on the hyperboloid. `t`: float or
  broadcastable tensor. `kappa`: `-1` (hyperbolic) is the only value `geodesic`
  passes; the body supports `-1` and `+1` (sphere) verbatim from the reference.
- Returns `(B, 3)` constant-speed geodesic interpolant. Uses the differential
  distance form `cosh d - 1 = <x-y, x-y>_L / 2` and a small-`d` Euclidean
  fallback (`d < 1e-6`).
- **DECISION: inline a verbatim copy** of `_geodesic_kernel` (hyper_bridge
  L350–390) as a module-level function in `geo_bridge.py`, NOT
  `from hyper_bridge import _geodesic_kernel`.
  Rationale: geo_bridge is deliberately self-contained — it re-implements every
  converter and bound-check internally rather than importing them. Importing one
  private helper from `hyper_bridge` would break that property and couple the two
  modules at a single arbitrary point; a ~40-line verbatim copy is the
  consistent, minimal-coupling choice. Mark it module-level (not a `GeoUtils`
  method) to match the reference's placement and because `geodesic` calls it as a
  bare module function.

## Coordinate / invariant cheat-sheet

| quantity | shape | invariant |
|---|---|---|
| polar `(rhos, thetas)` | `(B,), (B,)` | `rho >= 0`; free-kernel `theta` uniform on `[-pi, pi)` (bridge adds a target rotation) |
| Poincare-disk `z` | `(B, 2)` | `||z|| < 1` (strict) |
| Lorentz-Cartesian `z` | `(B, 3)` | `z[0] >= 1`; `-z0^2 + z1^2 + z2^2 = -1` |
| Lorentz component order | — | `(cosh ρ, sinh ρ cos θ, sinh ρ sin θ)` |
| heat times `ts` | `(B,)` | `> 0` |
| `word_embedding` | `(vocab_size, 2)` | only angle `atan2(e[1], e[0])` used |

## Symbol → reference mapping

| geo_bridge symbol | hyper_bridge counterpart (line) | behavior delta |
|---|---|---|
| `Geometry` | `Geometry` (L33) | fields differ: `POINCARE`,`LORENTZ` vs `POINCARE`,`LORENTZ_POLAR`,`LORENTZ_CARTESIAN` |
| `Coordinate` | `Coordinate` (L42) | attr renamed `HYPERBOLIC_POLAR` (was `POLAR`); string values identical |
| `GeoUtils._check_lorentz_rho_bound` | `_check_lorentz_rho_bound` (L74) | moved into class as staticmethod; behavior same |
| `GeoUtils._uniform_sphere` | `_uniform_sphere` (L98) | moved into class; UNUSED here |
| `GeoUtils._check_sphere_t_bound` | `_check_sphere_t_bound` (L108) | moved into class; UNUSED here |
| `GeoUtils._binary_polar_direction` | `_polar_direction` (L126) | `d == 2` specialization; `(thetas)` not `(rhos, thetas_or_u)` |
| `GeoUtils.binary_hyperbolic_polar_to_poincare_cartesian` | `poincare_polar_to_poincare_cartesian` (L143) | `d == 2` only; add missing `@staticmethod` |
| `GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian` | `FreeBinaryHyperbolicHeatKernel.poincare_polar_to_lorentz_cartesian` (L439) | identical; already correct |
| `GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar` | `lorentz_cartesian_to_poincare_polar` (L234) | `d == 2` path; 3 bugs to fix |
| `GeoUtils.poincare_cartesian_to_lorentz_cartesian` | `poincare_cartesian_to_lorentz_cartesian` (L212) | identical; add `@staticmethod` |
| `GeoUtils.lorentz_cartesian_to_poincare_cartesian` (NEW, user stub) | `lorentz_cartesian_to_poincare_cartesian` (L266) | implement `z[1:]/(1+z[0])`; fix docstring; add `@staticmethod` |
| `FreeBinaryHyperbolicHeatKernel.sample_chi` | `…sample_chi` (L467) | identical |
| `FreeBinaryHyperbolicHeatKernel.sample_chi_old` | `…sample_chi_old` (L489) | identical |
| `FreeBinaryHyperbolicHeatKernel.binary_free_hyperbolic_heat_kernel` | polar branch of `binary_free_poincare_heat_kernel` (L526–531) | split out as own method; **angle revised to UNIFORM** (reference uses Poisson) — intentional divergence |
| `FreeBinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel` | `…binary_free_poincare_heat_kernel` (L511) | delegates polar to new method; fix `ps`/bare names |
| `FreeBinaryHyperbolicHeatKernel.binary_free_lorentz_heat_kernel` | `…binary_free_lorentz_heat_kernel` (L538) | fix bare-name calls |
| `FreeBinaryHyperbolicHeatKernel.binary_poincare_bridge` | `…binary_poincare_bridge` (L566) | applies the Poisson-kernel warp internally (reference inherits it from its free kernel); same output distribution |
| `FreeBinaryHyperbolicHeatKernel.binary_lorentz_bridge` | `…binary_lorentz_bridge` (L604) | fix bare-name calls |
| `FreeBinaryHyperbolicHeatKernel.geodesic` | `…geodesic` (L642) | + `cartesian_model` Poincare lift on input AND chart-aware output; fix src/dest names, dest symmetry, `Geometry.Lorentz/Poincare` casing, bare calls |
| `_geodesic_kernel` (module-level, NEW) | `_geodesic_kernel` (L350) | inlined verbatim copy |

## Edge cases & invariants (module-wide)

- Empty `ts` (`numel == 0`): `binary_free_hyperbolic_heat_kernel` and downstream
  return empty tensors; `_check_lorentz_rho_bound` no-ops. (Reference relies on
  `torch.poisson`/`rand_like` handling empty input; preserve, do not add explicit
  empty-guards beyond what the reference has — the reference's `d == 2` path has
  no empty-guard, so neither should this.)
- `output_coord` accepts only `None`, `"polar"`, `"cartesian"`. Any other string
  is treated as "not cartesian"/"not polar" per each method's branch (no
  validation; matches reference — do not add a value check).
- `_LORENTZ_RHO_MAX = 20.0` gate applies ONLY on the cartesian-Lorentz path;
  polar outputs are unbounded.
- `geodesic` requires both endpoints lifted to the SAME ambient chart; a single
  `cartesian_model` governs both cartesian endpoints.
- NOT handled: `d >= 3`, spherical geometry, mixing a Poincare src with a
  Lorentz dest (one `cartesian_model` for both), validation of `output_coord`
  string values.

## Open questions (all resolved with recommendations)

1. **`_geodesic_kernel` placement** → RESOLVED: inline a verbatim module-level
   copy from hyper_bridge L350; do NOT import. (Keeps geo_bridge self-contained,
   consistent with how every other helper was re-implemented.)
2. **`geodesic` parameter naming** → RESOLVED: keep signature names
   `src_cartesian`/`dest_cartesian`/`src_radial`/… ; fix the body's stray
   `src`/`dest` to use them. Minimal churn; the guards already use the long names.
3. **`geodesic` dest `cartesian_model`** → RESOLVED: dest honors `cartesian_model`
   symmetrically with src (POINCARE→lift, LORENTZ→identity, else `ValueError`).
4. **`binary_lorentz_cartesian_to_hyperbolic_polar` input shape** → RESOLVED:
   `(..., 3)` for `H^2`, component order `(cosh ρ, sinh ρ cos θ, sinh ρ sin θ)`;
   `rhos = arccosh(z[...,0])`, `thetas = atan2(z[...,2], z[...,1])`; raise on
   `shape[-1] != 3`.

BLOCKING: none. Every gap in the broken source is resolvable from the reference +
math spec; all four open questions have a defensible default chosen above.
