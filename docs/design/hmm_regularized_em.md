# Regularized EM for a discretized scalar-state HMM

Design / feasibility analysis for `linsys.hmm`.

**Question (Pablo).** For a discrete-state Markov chain with *scalar* state and
*scalar* observations, can we learn the transition matrix `T` and observation
matrix `O` by EM? Learning independent `N²` entries (`N` = number of states) is
not practical from one short scalar sequence. What regularization makes it
work — banded `T`, banded/parametric `O`, or something simpler?

**Short answer.** Yes, EM (Baum–Welch) runs fine mechanically, but the
*unconstrained* `N×N`/`M×N` fit is badly underdetermined from a single scalar
sequence of length `N_samp ~ 1000–2000` and overfits. The fix is to exploit
that the states are a **discretized continuous scalar**, so dynamics are a
*local random walk / drift–diffusion*. Three regularizers, in increasing
strength, all have closed-form (or low-dimensional) M-steps that drop straight
into the existing forward–backward code:

1. **Banded `T`/`O`** (support `|i−j| ≤ k`): zero outside the band, renormalize
   columns. Cheapest to add, but still `O(Nk)` free params and still overfits
   unless `k` is chosen near the true diffusion scale (too-small `k` is a
   *local-optimum trap* — see experiment).
2. **Toeplitz `T`/`O`** (`T[i,j]=f(i−j)`): pool the Baum–Welch ξ-statistics
   along diagonals → `~2N` params. Robust, train ≈ test logL.
3. **Parametric drift–diffusion** (`a, q` for `T`; Gaussian/sigmoid link for
   `O`): `O(1)` params, M-step is a tiny weighted regression. Best held-out
   logL in the experiment, essentially matching the true model.

The conventions below follow `linsys.hmm.helpers`: `T` is `(D,D)`
column-stochastic, `T[i,j] = p(x_{k+1}=i | x_k=j)`; `O` is `(M,D)`
column-stochastic, `O[m,d] = p(y=m | x=d)`; state posteriors are `(D,N)` with
time along columns.

---

## 1. Feasibility of plain EM (full `T`, full `O`)

**Parameter counts vs data.** A free `T` has `D(D−1)` free parameters
(`D²` minus one normalization per column); a free `O` has `M(D−1)`. For the
realistic motor-adaptation grid (`discretizeObs` with `N≈50–101` bins, e.g. the
`[-200:4:200]` PSE grid in `testStationaryHMM.m`, `D=M≈101`) that is
`~10⁴ + 10⁴ ≈ 2×10⁴` parameters. The data is **one** scalar sequence: `N_samp`
observations and `N_samp−1` state *transitions*. With `N_samp ~ 1000–2000` we
are estimating `~10⁴` numbers from `~2×10³` events: an order of magnitude
underdetermined.

**Why it overfits / is underdetermined.**
- The expected transition count into column `j` is `Σ_k γ_k[j]`, the expected
  time spent in state `j`. With `D=50` and `N_samp=2000`, that is on average
  ~40 visits per state — but a diffusing scalar spends almost all its time in a
  handful of states and *never visits the rest*. Unvisited (or rarely visited)
  columns of `T` and `O` are estimated from ~0 effective samples → their
  entries are set by the prior/initialization, not the data ("mass leaking to
  unvisited states").
- Each ξ-statistic `ξ_k[i,j]` is spread over `D²` cells; only the few cells
  near the diagonal accumulate real mass. The off-band cells fit noise.

**Symptoms to expect.**
- **Overfitting:** train logL climbs *above* the true-model logL while held-out
  logL is *worse* than the truth (confirmed: §5, unconstrained train
  −2.40 vs true −2.56, test −2.78).
- **Spiky / non-smooth `T`** — recovered columns have large, noisy off-diagonal
  entries; Frobenius error to the true `T` is large (rel 1.29).
- **State-permutation (label-switching) ambiguity:** the complete-data
  likelihood is invariant to any joint relabeling of states applied to the
  columns of `T`, the rows/cols of `O` permutation, and `p0`. Different inits
  converge to different permutations. (Scalar *ordered* observations damp this
  if `O` is constrained — see §4.)
- **Slow/!monotone in practice** because the package M-step
  (`hmm_matrix_estim`) uses the *marginal-product* approximation of ξ, not the
  exact pairwise posterior — so even "plain EM" here is only approximately
  monotone (see §6 "exact ξ").

**Verdict.** Mechanically fine; statistically hopeless for `N ≳ 20` from one
short sequence without structure.

---

## 2. Regularization options and their M-steps

EM background (exact). E-step: forward–backward gives smoothed marginals
`γ_k[d] = p(x_k=d | Y)` and pairwise posteriors
`ξ_k[i,j] = p(x_k=j, x_{k+1}=i | Y)`. The expected complete-data log-likelihood
to be maximized in the M-step is

```
Q = Σ_d p0-terms + Σ_{i,j} Ξ[i,j] log T[i,j] + Σ_{m,d} Γ_obs[m,d] log O[m,d]
```

with the **accumulated statistics**

```
Ξ[i,j]      = Σ_{k=0}^{N-2} ξ_k[i,j]              # expected # of j -> i transitions
Γ_obs[m,d]  = Σ_{k: y_k=m} γ_k[d]                 # expected # of (state d, symbol m)
```

Unconstrained maximizer: `T = colnorm(Ξ)`, `O = colnorm(Γ_obs)`. Every
regularizer below changes only how `Ξ`/`Γ_obs` are *pooled or projected* before
normalization — the E-step is untouched.

> **Note on the current package M-step.** `hmm_matrix_estim` forms
> `joint = P[:,1:] @ P[:,:-1].T` (product of smoothed *marginals*), which is the
> approximation `ξ_k[i,j] ≈ γ_{k+1}[i] γ_k[j]`. It is exact only for
> near-deterministic posteriors. All structured M-steps below are *more*
> sensitive to getting ξ right, so a regularized implementation should compute
> the **exact** `Ξ` (cheap: one `(D×(N−1))·((N−1)×D)` matmul, see §5/§6).

### 2a. Banded `T` (hard support, bandwidth `k`)

Constraint: `T[i,j] = 0` for `|i−j| > k`. The M-step that maximizes `Q` subject
to this support is simply masking then renormalizing columns:

```
T = colnorm( Ξ ⊙ 1[|i−j| ≤ k] )
```

Closed form, valid EM (it is the exact constrained maximizer of `Q`). Same for
`O` with its own bandwidth. Params: `~N(2k+1)`.

Caveat (important, from the experiment): choosing `k` *smaller* than the true
diffusion scale (`k < ~2.5·√q`) both misspecifies the model and creates a
**local-optimum trap** — from a diffuse init, banded-`k=3` converged to a logL
*worse than the init* (−4.08 vs init −3.27). With `k=5 ≈ 2.5√q` it behaves like
the unconstrained fit (and still overfits). Banded alone is the weakest
regularizer: it removes off-band noise but leaves `O(Nk)` loosely-determined
in-band parameters.

### 2b. Toeplitz / translation-invariant `T` (`T[i,j] = f(i−j)`)

If dynamics are spatially homogeneous (drift–diffusion with constant
coefficients), the transition kernel depends only on the offset `d = i−j`. Pool
ξ along diagonals:

```
f(d) = Σ_{(i,j): i-j = d} Ξ[i,j]       # = np.trace(Ξ, offset=d)
T[i,j] = f(i-j),  then colnorm
```

This is the exact maximizer of `Q` over the Toeplitz family (each `f(d)` is a
multinomial cell pooled across all columns sharing that offset). Params: `~2N−1`
non-zero offsets, or `2k+1` if also banded.

**Relation to the discretized random walk.** `f(·)` *is* the discretized
process-noise kernel: for `x_{k+1} = x_k + w`, `f(d) ≈ N(d; drift, q)`. This is
exactly what `linearTransitionMatrix.m` /
`linsys.hmm.linear_transition_matrix` build analytically (a Gaussian-ish kernel
`exp((x_i − a x_j − b u)/(2√q))`; note the MATLAB port is an *unfinished*
kernel — the residual is not squared, see the docstring — the experiment uses
the corrected `exp(−(x_i − a x_j)²/2q)`).

**Boundary conditions break Toeplitz at the edges.** A bounded scalar (the PSE
grid is clipped to `[-200,200]`) has *absorbing* or *reflecting* edges:
probability that would leave the grid is either lost (absorbing) or folded back
(reflecting). Either way the first/last `~k` columns are **not** translates of
`f`. Practical handling: estimate the homogeneous `f(·)` from the interior
diagonals only, then treat the `≤k` boundary columns as free (re-normalized)
multinomials, or impose reflecting BCs by adding the truncated tail mass back
onto the edge state. The experiment uses truncation + per-column renormalization
(a reflecting-like BC), which is why Toeplitz is near-exact in the bulk.

### 2c. Parametric `T`: discretized AR(1) / drift–diffusion (`a, q`)

Model `x_{k+1} = μ + a(x_k − μ) + w`, `w ~ N(0,q)`, discretized onto the grid
`s = 0..D−1`:

```
T[i,j] ∝ exp( -(s_i - (μ + a(s_j - μ)))² / (2q) ),  colnorm
```

This is **the discretized version of the scalar LDS `x[k+1]=a x[k]+w`** that
this very package estimates in continuous form (`linsys` Kalman/EM). The M-step
maximizes `Q(a,q)` over 2 scalars. Treat each cell `Ξ[i,j]` as `Ξ[i,j]`
"observations" of the pair `(prev=s_j, next=s_i)` and do **weighted least
squares** (moment matching), ignoring the mild dependence of the per-column
normalizer on `(a,q)` in the interior (this makes it a *generalized* EM step —
it increases `Q` to first order; one can add a line-search to guarantee
monotonicity):

```
u = s_j - μ,  v = s_i - μ
a = Σ_{ij} Ξ[i,j] u v / Σ_{ij} Ξ[i,j] u²
q = Σ_{ij} Ξ[i,j] (v - a u)² / Σ_{ij} Ξ[i,j]
```

Params: 2 (+1 if `μ` free; +1 for an input/drift gain `b·u_k` matching
`linear_transition_matrix`'s input term — that gives a non-stationary,
input-driven kernel, reusing `hmm_nonstationary_inference_alt`).

### 2d. Observation model `O` — same menu

`O[m,d] = p(y=m | x=d)` with **ordered** symbols and states. Options:
- **Banded:** `O[m,d]=0` for `|m−d|>k_O`; mask + renormalize `Γ_obs`.
- **Toeplitz:** emission kernel depends only on `m−d`; pool `Γ_obs` along
  diagonals. (Use when the observation is `y = x + noise` with homogeneous
  noise.)
- **Parametric Gaussian link:** `O[m,d] ∝ exp(-(s_m - (β0 + β1 s_d))²/(2r))`.
  M-step = weighted regression of the observed symbol `y_k` on the state, with
  weights `γ_k[d]`: closed-form `β0, β1, r` (see `mstep_O_parametric`). One
  width `r` + an affine link = 3 params.
- **Parametric monotone / sigmoid link:** the motor-adaptation use case in
  `testStationaryHMM.m` uses a **logistic** emission
  `p(y=1|x) = 1/(1+exp((x+bias)/σ))` (binary choice from a continuous internal
  PSE). M-step = weighted logistic regression (1–2 IRLS steps inside EM, still
  GEM-valid). This is the natural `O` for 2-alternative-forced-choice data.

### 2e. Soft penalties instead of hard structure (MAP-EM)

- **Dirichlet prior on each column** (`Dir(α)`): MAP M-step just **adds
  pseudocounts**, `T = colnorm(Ξ + (α−1))`. With `α>1` this smooths/keeps mass
  on unvisited states; cheapest possible regularizer, one line, fully monotone
  MAP-EM. A *band-structured* `α` (large near diagonal, ~1 off) softly
  encourages locality without a hard cut.
- **Smoothness / total-variation along diagonals:** penalize
  `Σ_d (f(d+1)−f(d))²` (ridge on the Toeplitz kernel) or `Σ |f(d+1)−f(d)|` (TV,
  keeps a sharp peak but flat tails). Adds a small convex sub-problem per M-step
  (closed-form for ridge, prox/`scipy` for TV).
- **Entropic regularization** of columns nudges toward the uniform; rarely what
  you want for a peaked random walk (prefer Dirichlet with band-structured α).

Recommended default for the package: **MAP Dirichlet pseudocounts** (always on,
tiny `α`) *plus* one hard-structure choice (`toeplitz` or `parametric`).

---

## 3. Sample-complexity intuition

| model | free params | rule of thumb |
|---|---|---|
| full `T` | `D(D−1) ≈ D²` | needs `≫ D²` transitions; hopeless for `D≳20`, `N_samp~2k` |
| banded (bw `k`) | `~2kD` | needs `≫ 2kD`; with `D=50,k=5` that's ~500 ≈ `N_samp`/4 — borderline, overfits |
| Toeplitz | `~2k`–`2D` | `~10²`; comfortable at `N_samp~2k` |
| parametric (a,q) | `2–4` | trivial; works at `N_samp~10²` |

The binding quantity is **effective transitions per estimated parameter**. A
diffusing scalar only explores `O(√(N_samp·q))` distinct states, so the
*effective* `D` is smaller than the grid, but the *unvisited* columns still need
a prior. **Realistic grid sizes for `N_samp≈1000–2000`:** full `T` only for
`D ≤ ~10`; banded for `D ≤ ~30` (and only with `k` tuned and a prior); Toeplitz
and parametric scale to `D = 50–101+` (the PSE grid) with room to spare —
their cost is essentially independent of `D`, so pick `D` for *discretization
fidelity*, not for statistics.

---

## 4. Identifiability

**Label-switching.** The likelihood is invariant under a permutation `π` applied
jointly to states (`T → P_π T P_πᵀ`, `O → O P_πᵀ`, `p0 → P_π p0`). With a free
`O` this is a genuine `D!` symmetry: EM lands on an arbitrary relabeling.

**Why ordered/banded `O` removes it.** If `O` is constrained so that the
emission mean is **monotone** in the state index (banded around the diagonal,
Toeplitz `f(m−d)`, or a parametric link with `β1 > 0`), then a permutation of
states that preserved the likelihood would have to preserve that monotone
ordering — and the only order-preserving permutation is the identity. So
**ordered scalar observations + ordered `O` ⇒ the discrete label-switching
symmetry collapses to a unique labeling.** This is the key reason the scalar
setting is well-behaved where general HMMs are not.

**Residual (continuous) degeneracy.** Ordering kills the *discrete* symmetry but
a *continuous* reparametrization survives: stretching the latent state axis
`s → c·s` and compensating in `O`'s gain and in `q` leaves the data
distribution (almost) unchanged. The experiment shows this directly — the
parametric fit recovers `a=0.92` (true 0.90, scale-invariant, recovered well)
but `q=9.6` and `O_slope=0.60` instead of `(q=4, slope=1)`: the latent axis was
rescaled, `q` and the observation gain traded off, **yet held-out logL is
essentially optimal**. Fix by *anchoring the scale*: fix `O`'s slope to 1
(observation grid = state grid, the usual `discretizeObs` setup), or fix the
state grid spacing. Then `q` is identified. Without *any* `O` constraint you get
both the discrete label-switching *and* this scale degeneracy.

---

## 5. Numerical experiment

Script: `docs/design/exp_regularized_em.py` (self-contained; imports only
`hmm_logl`, `column_normalize` from `linsys.hmm`; implements its own exact
forward–backward-with-ξ and the four M-steps). Runtime **~12 s**.

Setup: discretized AR(1) drift–diffusion, `N=50` states, `M=50` symbols,
true `(a,q,r)=(0.9, 4.0, 4.0)`, Gaussian emission `y = x + noise`. Train on
`N_samp=2000`, evaluate held-out logL on a fresh `N=1000` sequence. **Same
diffuse init** for all variants. Banded uses `k=5 ≈ 2.5√q`.

```
true-model  test logL/sample = -2.5642     (oracle upper bound)
init        test logL/sample = -3.2665

variant        iters   train LL/n    test LL/n   ||T-T*||_F   rel    recovered params
unconstrained     40      -2.3984      -2.7790      3.4458    1.29   -
banded(k=5)       40      -2.4334      -2.7511      3.7884    1.42   -
toeplitz          40      -2.5784      -2.5898      1.5489    0.58   -
parametric        40      -2.5619      -2.5667      0.8844    0.33   a=0.920, q=9.647,
                                                                    O_slope=0.604, O_int=9.024, O_r=4.290
```

(`rel` = Frobenius error to the true `T`, normalized by `‖T*‖_F`.)

**Reading the table.**
- **unconstrained** overfits textbook-style: train logL (−2.40) *exceeds* the
  true model (−2.56) while held-out (−2.78) is the *worst* of all and `T` is far
  from truth (rel 1.29). Free `M×N` `O` also drifts.
- **banded(k=5)** is barely better than unconstrained on held-out (−2.75) and
  its `T` is *not* closer (rel 1.42): with `k=5` there are still 550 in-band
  params fitting noise. (And `k=3 < √q·2.5` is a local-optimum **trap**: it
  converges to logL −4.08, worse than the init — see the bandwidth-sensitivity
  note in the script's commit message / §2a.) Banded is the weakest regularizer.
- **toeplitz** essentially eliminates overfitting: train (−2.578) ≈ test
  (−2.590) ≈ true (−2.564), and `T` error halves (rel 0.58). ~100 params.
- **parametric** is best: held-out −2.567, statistically indistinguishable from
  the oracle, `T` rel-error 0.33, and it returns interpretable `a≈0.92`
  (true 0.90). The `(q, slope)` mismatch is the §4 scale degeneracy, not a
  fitting failure — anchoring `O_slope=1` would identify `q`.

**Takeaway:** structure helps monotonically with its strength, and the two
*low-dimensional, smoothness-inducing* structures (Toeplitz, parametric) are the
ones that actually recover `T` and generalize; hard banding alone does not.

**API friction discovered (useful design feedback).** The existing inference
(`hmm_stationary_inference`) returns only the *normalized smoothed marginals*
`p_smoothed`; it exposes neither the scaling factors `c_k = p(y_k|y_{<k})` nor
the pairwise posteriors `ξ_k`. An exact (monotone) Baum–Welch M-step needs the
`ξ` statistic, so the experiment had to **re-implement scaled forward–backward
from scratch**. The package M-step (`hmm_matrix_estim`) sidesteps this with the
marginal-product approximation `ξ ≈ γ⊗γ`, which is *not* exact and makes EM only
approximately monotone (the existing test even allows "tiny dips"). A regularized
implementation should add an inference path that returns `Ξ` (and `c_k` for a
cheap logL) — see §6.

---

## 6. Implementation plan for `linsys.hmm`

**(i) Add exact-ξ inference.** New internal routine (or extend
`_forward_backward`) that returns the accumulated `Ξ` and the scale factors,
without breaking the existing `HMMInferenceResult` API. Sketch:

```python
def forward_backward_stats(obs, T, O, p0):
    # scaled alpha/beta; returns gamma (D,N), Xi (D,D), loglik
    # Xi = T * (A @ B.T)  with A=(O[obs[1:]].T*beta[:,1:])/c[1:], B=alpha[:,:-1]
    ...
    return gamma, Xi, loglik
```

This is ~25 lines and is the single most valuable fix (also makes `hmm_em`
exactly monotone). Reuses `column_normalize`; replaces the
`hmm_matrix_estim` marginal-product approximation when `structure != None`.

**(ii) Strategy objects for the M-step.** Prefer a small strategy/protocol over
a pile of kwargs, but offer string shortcuts:

```python
class TransitionModel(Protocol):
    def mstep(self, Xi) -> np.ndarray: ...      # returns (D,D) colnorm T

class ObservationModel(Protocol):
    def mstep(self, gamma, obs, M) -> np.ndarray: ...

# built-ins
FullTransition()                 # colnorm(Xi)         [= current behavior]
BandedTransition(k, alpha=0.0)   # mask + Dirichlet pseudocounts
ToeplitzTransition(k=None)       # pool diagonals (+ optional reflecting BC)
DriftDiffusionTransition(mu=None, fixed_a=None)   # parametric (a,q)
# observation analogues: Full/Banded/Toeplitz/GaussianLink/LogisticLink
```

Public entry point, backward compatible (defaults reproduce today's `hmm_em`):

```python
def hmm_em(observations, p0, observation_matrix=None, transition_matrix=None,
           n_symbols=None, max_iter=100, tol=1e-8, rng=None,
           transition_structure="full",     # or "banded"/"toeplitz"/"drift_diffusion"/callable/TransitionModel
           observation_structure="full",    # or "banded"/"toeplitz"/"gaussian"/"logistic"/callable
           bandwidth=None, prior_counts=0.0):  # Dirichlet alpha (MAP-EM)
    ...
```

`prior_counts` implements MAP Dirichlet smoothing (`colnorm(Ξ + alpha)`) and is
orthogonal to the structure choice. `transition_structure=callable` lets a user
pass a custom `mstep(Xi)->T`.

**(iii) Reuse.**
- E-step: `hmm_stationary_inference` (+ new `forward_backward_stats`).
- Kernels: `linear_transition_matrix` (fix the unsquared-residual kernel, or add
  `gaussian_transition`) for `DriftDiffusionTransition` and for building inits.
  The input-driven term `b·u` reuses `hmm_nonstationary_inference_alt`.
- `column_normalize`, `hmm_logl` unchanged. `discretize_obs` for the
  preprocessing front-end (continuous scalar → symbols).
- Connection to the rest of `linsys`: `DriftDiffusionTransition` is the
  discretized `x[k+1]=a x[k]+w` already estimated continuously by the package's
  Kalman/EM — could share parameter conventions / provide a "continuous-init →
  discretize" helper.

**(iv) Test plan** (mirror `tests/test_hmm.py` style):
- *Exact ξ monotonicity*: `hmm_em` logL is non-decreasing to machine tol (no
  "allow tiny dips") once exact `Ξ` is used.
- *Banded M-step*: recovered `T` has zero mass outside the band; columns sum 1.
- *Toeplitz M-step*: recovered `T` is Toeplitz in the interior; on Toeplitz-true
  data, recovers within tolerance and beats full-`T` held-out logL.
- *Parametric*: on drift–diffusion-true data with `O_slope` fixed to 1,
  recovers `(a,q)` within tolerance; held-out logL ≈ oracle.
- *MAP Dirichlet*: `prior_counts>0` keeps strictly positive mass on an
  unvisited state.
- *Regression*: `structure="full", prior_counts=0` reproduces current `hmm_em`
  outputs (within the exact-vs-approx ξ difference — document it).
- Reuse the existing simulate/`make_hmm` harness; add a `simulate_drift_diffusion`
  helper.

**Effort estimate.** Exact-ξ + banded/Toeplitz/Dirichlet: ~half a day.
Parametric (Gaussian + logistic links) + tests: ~1 day. No breaking changes;
all new behavior is opt-in via the two `*_structure` kwargs.
