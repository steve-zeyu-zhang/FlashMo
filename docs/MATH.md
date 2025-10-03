# MotionSiT geometric factorized interpolant and reproduction choices

MotionSiT uses the geometric factorized interpolant defined in FlashMo
Section 3.2 and the SO(3)-valued VAE posterior described in the OpenReview
discussion. Training uses original AMASS/SMPL rotations indexed by HumanML3D.
The VAE latent space is SO(3)-valued; using axis-angle inputs alone does not
constrain an ordinary Euclidean VAE's latent space to SO(3).

## Representation and conventions

A motion clip contains joint rotation matrices and root translation parameters,
derived from raw AMASS axis-angle poses and `trans`. Rotation
matrices multiply column vectors. Translations are Euclidean vectors, never
passed through SO(3) Exp/Log. Temporal compression produces rotations with shape
`[B, L, K, 3, 3]` and a separate Euclidean root latent `[B, L, 3]`; latent masks
have shape `[B, L]`, with `True` indicating valid time steps.

The transformer feature width is 512 in the paper configuration. The number
of product SO(3) factors is configured separately; the default of 22 follows
the selected SMPL body joints. The paper does not specify this product factor
count. Temporal compression and latent joint identities are learned.

For a vector `a`, `hat(a)` is its skew matrix, and `Exp(a)` abbreviates
`matrix_exp(hat(a))`. We use **body angular velocity** throughout:

```text
dR/dt = R hat(omega),     omega = vee(R^T dR/dt).
```

`so3_exp` implements Rodrigues' formula with a Taylor series at zero.
`so3_log` returns the principal rotation vector with angle in `[0, pi]`, using a
dominant quaternion component for stability near pi. Log has an unavoidable
axis-sign ambiguity at the pi cut locus; no global smooth principal logarithm
exists. Six-dimensional rotation inputs represent the first two matrix columns.

## SO(3) variational posterior

The encoder predicts a mean rotation `M` and diagonal tangent log-standard
deviation `log_s`. Reparameterized samples are

```text
epsilon ~ N(0, I),  v = exp(log_s) * epsilon,  Z = M Exp(v).
```

This is the left-centered SO(3) posterior specified in the OpenReview response.
The default SO(3) prior is `Exp(N(0,I))`, matching the flow
starting distribution, with its full wrapped density used in the KL. The
optional `prior_type: haar` instead uses a uniform Haar prior.
This prior selection is an explicit reconstruction choice;
the FlashMo rebuttal does not specify it. The root latent uses an ordinary
Gaussian posterior and unit Gaussian prior.

The exponential map is many-to-one. Consequently, a Gaussian log probability of
the tangent vector alone is not the SO(3) posterior log probability. For a
relative rotation with axis `u` and principal angle `theta`, the implementation
sums its preimages:

```text
q(R | M, s) = sum_{k in Z} Normal(u (theta + 2 pi k); 0, diag(s^2))
                         * (theta + 2 pi k)^2 / (4 sin(theta/2)^2).
```

The density convention is Riemannian volume on SO(3), with total volume
`8 pi^2`. With a Haar prior, the Monte Carlo KL estimate uses
`log q + log(8 pi^2)`. With the default wrapped prior it instead subtracts the
log density of `Exp(N(0,I))`, evaluated at the sampled latent rotation. These
Monte Carlo estimates can occasionally be negative without violating the
nonnegativity of the true expected KL; they are not clamped.
The infinite sum is approximated with `k=-3,...,3`; the VAE limits posterior
standard deviations to at most 1, for which omitted Gaussian tails are tiny.
The helper accepts more wraps for wider distributions. The wrapped density is
singular exactly at identity; evaluation uses a small finite-radius convention
there. The point has zero probability under continuous posterior sampling.

The architecture combines this manifold posterior with MotionSiT's temporal
motion compression. The mathematical sources and the scope of the adaptation
are listed in Acknowledgements.

## Geometric factorized interpolant (GFI)

The equation implemented from FlashMo Section 3.2 is

```text
a(t)[tau,s] = alpha_T(t)[tau] alpha_S(t)[s] Log(R_data[tau,s])
             + sigma(t) epsilon[tau,s]
R(t)[tau,s] = Exp(a(t)[tau,s]),  epsilon ~ N(0,I).
```

The GFI implementation uses **0 for noise and 1 for data**.
The configurable schedules are

| Setting | `alpha_T` | `alpha_S` | `sigma / noise_scale` |
| --- | --- | --- | --- |
| `linear` (default) | `t^p_T` | `t^p_S` | `1-t` |
| `paper_sqrt` | `t^p_T` | `t^p_S` | `sqrt(1-t)` |
| `gvp` | `sin(pi t/2)^p_T` | `sin(pi t/2)^p_S` | `cos(pi t/2)` |

Temporal/spatial powers can be scalars or vectors over their axes; defaults are
both 1. **The default signal coefficient is therefore `t^2`**, not `t`.
This is the direct product of the two linear schedules. Scalar defaults impose separable but
identical schedules within each axis; distinct per-frame/per-factor schedules
can be configured explicitly. Schedules are fixed configuration parameters;
the paper does not report fitted values or learned schedules.

The paper's Appendix A.1 is inconsistent about time direction and calls
`alpha=1-t, sigma=sqrt(t)` a linear interpolant; the implementation's default
uses `alpha_T=alpha_S=t, sigma=1-t` for noise-to-data generation. Both interpretations
are exposed. `paper_sqrt` reverses the appendix time direction and has a
singular velocity at the data endpoint; training excludes `t=1`, and the
implemented samplers evaluate strictly before that endpoint. Factorizing the
GVP signal coefficients does not preserve total variance; the code does not
assert otherwise.

The Euclidean root branch uses the corresponding product schedule without
Exp/Log. When per-spatial-factor powers are configured, its spatial power is the
first factor's (the designated root) power.

This exponential-coordinate interpolant stays on SO(3) but is **not generally a
shortest geodesic between a noisy rotation and a target rotation**. The latter
would be `R0 Exp(t Log(R0^T R1))`, a different path from the GFI equation
in FlashMo. The Gaussian
prior pushed through Exp is also not Haar-uniform.

## Velocity targets and noncommutativity

Simply using `a_dot` as body angular velocity is incorrect in general because
`a` and `a_dot` need not commute. The exact differential is

```text
a_dot = (alpha_T_dot alpha_S + alpha_T alpha_S_dot) Log(R_data)
        + sigma_dot epsilon
omega_target = J_r(a) a_dot
J_r(a) = I - (1-cos(theta))/theta^2 hat(a)
           + (theta-sin(theta))/theta^3 hat(a)^2.
```

`theta=||a||`; Taylor expansions define the zero-angle limits. The model
predicts `rotation_velocity` in body coordinates and `translation_velocity`
in root-latent Euclidean coordinates. Training minimizes their masked squared
errors. GFI velocity matching learns the conditional mean velocity field of the
interpolant's marginal distribution directly in body angular coordinates.

## Sampling

The initial state matches training:

```text
R_0 = Exp(noise_scale * epsilon),  x_0 = noise_scale * eta.
```

The default Lie-Euler update is

```text
R_{n+1} = R_n Exp(dt * omega_theta(R_n,x_n,t_n,c))
x_{n+1} = x_n + dt * v_theta(R_n,x_n,t_n,c).
```

All rotation iterates stay in SO(3) to floating-point precision, without an SVD
projection or treating matrix entries as Euclidean coordinates. Ten steps mean
ten model evaluations. The optional Lie-midpoint method uses two evaluations
per step. It evaluates the model at a half-step group update before applying
the full-step velocity. The sampler is deterministic given initial noise and
conditioning, uses `no_grad`, and does not add arbitrary Langevin noise: a
consistent manifold SDE would require a correctly trained manifold score.

## Verification and limits

Tests compare Exp/Log with SciPy at identity, generic angles, and near pi;
differentiate them numerically; integrate the wrapped posterior density to
unit mass; and verify its invariance to left multiplication. Finite differences
of noncommuting paths verify the right-Jacobian body velocity, including
factorized axis-specific schedules and the noise endpoint. Sampler tests check
group membership, exact constant body-velocity evolution, evaluation counts,
and midpoint improvement for time-varying noncommuting velocities.

These checks establish mathematical and software behavior. They do not
establish the paper's FID, timing, or text-to-motion quality; reproducing those
requires the licensed complete dataset, full training, and the same benchmark
evaluator and preprocessing.

## Acknowledgements

- The [FlashMo discussion and author responses](https://openreview.net/forum?id=B9T5kKcKTy&noteId=55DfS6aqjp)
  clarify the SO(3) VAE posterior and training on original AMASS rotations.
- The SO(3) posterior and wrapped density follow
  [Falorsi et al., Explorations in Homeomorphic Variational Auto-Encoding](https://arxiv.org/abs/1807.04689),
  especially equations 6–9 and Sections 3–4. The uniform Haar prior option
  matches that work's experiment. This motion adaptation does not reproduce
  its image decoder or Wigner-D group-action architecture.
- The velocity-matching organization and default noise schedule are adapted
  from the official [SiT coupling plans](https://github.com/willisma/SiT/blob/main/transport/path.py)
  and [transport implementation](https://github.com/willisma/SiT/blob/main/transport/transport.py).
  Their linear path uses `alpha=t, sigma=1-t`; MotionSiT's GFI applies separate
  temporal and spatial signal factors, whose default product is `t^2`.
  Source revisions and licenses are retained in [upstream provenance](UPSTREAM.md).
