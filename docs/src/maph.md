```@meta
CurrentModule = PhaseTypeDistributions
```

# MAPH distributions

A **multi-absorbing phase-type** distribution is the joint distribution of the
pair `(τ, κ)`, where `τ` is the absorption time of an `m`-phase
continuous-time Markov chain and `κ ∈ {1, …, n}` indexes the absorbing state
reached. When `n = 1` it reduces to a PH distribution; for `n > 1` it is the
natural model for *competing risks*.

[`MAPHDist`](@ref) is parameterized by:

- `α`   — `m`-vector initial distribution over transient phases (`sum(α) = 1`),
- `T`   — `m × m` sub-generator of the transient phases,
- `D`   — `m × n` non-negative matrix of absorbing rates,

with the constraint `T · 𝟙_m + D · 𝟙_n = 0` (every row of the full generator
sums to 0).

```@example maph
using PhaseTypeDistributions, Distributions, LinearAlgebra, Random, Statistics
nothing # hide
```

## Construction

### From `(α, T, D)`

```@example maph
α = [0.6, 0.4]
T = [-3.0 1.0; 0.5 -2.0]
D = [1.5 0.5; 1.0 0.5]
d = MAPHDist(α, T, D)
```

The constructor validates the row-sum constraint:

```@example maph
sum(T, dims=2) + sum(D, dims=2)
```

### From per-category PH distributions

Given a PH distribution for each absorbing state and a marginal probability
vector `π`, build a block-diagonal MAPH whose conditional `τ | κ = k` is
exactly the supplied PH:

```@example maph
he = HyperExponentialDist([0.4, 0.6], [2.0, 5.0])
er = ErlangPHDist(3, 2.0)
d_blk = MAPHDist(AbstractPHDist[he, er], [0.3, 0.7])
nphases(d_blk), nabsorbing(d_blk), marginal_absorption(d_blk)
```

### Moment-matched from `(π, μ, σ²)`

Given target marginal absorption probabilities, conditional means, and
conditional variances, build a small MAPH that matches them. Each category is
realized by:

| Conditional `c²_k = σ²_k / μ_k²` | Realization                     |
|-----------------------------------|---------------------------------|
| `c²_k > 1`                       | 2-phase [`HyperExponentialDist`](@ref) |
| `c²_k = 1`                       | single-phase exponential ([`CoxianDist`](@ref)) |
| `c²_k < 1`                       | [`HypoExponentialDist`](@ref) by (mean, scv) |

```@example maph
π_t  = [0.25, 0.45, 0.30]
μ_t  = [1.0,  3.0,  5.0]
σ²_t = [0.5, 12.0,  4.0]
d_mm = MAPHDist(π_t, μ_t, σ²_t)
marginal_absorption(d_mm)
```

### Alternative `(α, q, R, U)` parameterization

```@example maph
R = absorption_probs(d)
q = -diag(subgenerator(d))
U = zeros(2, 2)
for i in 1:2, j in 1:2
    U[i, j] = i == j ? 0.0 : (T[i, j] / q[i]) * R[j, 1] / R[i, 1]
end
d_alt = MAPHDist(α, q, R, U)          # recovers the same MAPH
isapprox(d_alt, d)
```

### Embed a PH as a 1-absorbing-state MAPH

```@example maph
ph = HyperExponentialDist([0.4, 0.6], [2.0, 5.0])
MAPHDist(ph)
```

## Accessors

```@example maph
initial_prob(d), subgenerator(d), exit_rate_matrix(d)
```

```@example maph
nphases(d), nabsorbing(d), exit_rates(d)
```

```@example maph
params(d)             # (α, T, D)
```

## Joint distribution API

The pair `(τ, κ)` has a *joint sub-density* `f(u, k)` and a *joint cdf*
`F(u, k)`. Integrating `f(·, k)` over `u ∈ (0, ∞)` gives the marginal
absorption probability `π_k`:

```@example maph
pdf(d, 1.0, 1)            # f(1.0, 1)
cdf(d, 1.0, 1)            # P(τ ≤ 1.0, κ = 1)
ccdf(d, 1.0, 1)           # P(τ > 1.0, κ = 1)
```

`cdf(d, u, k) + ccdf(d, u, k) = π_k` for all `u ≥ 0`. As `u → ∞`,
`cdf(d, u, k) → π_k` and `ccdf(d, u, k) → 0`.

### Marginal (all-cause) law of `τ`

Dropping the cause index gives the marginal law of the absorption time. The
survival `S(u) = P(τ > u) = α' exp(Tu) 1` is the likelihood of a record known
only to have survived past `u` — a **right-censored** observation, whose
eventual cause is unobserved:

```@example maph
ccdf(d, 1.0)              # S(1.0) = P(τ > 1.0)
```

```@example maph
cdf(d, 1.0), logccdf(d, 1.0)
```

It sums the joint survivals over causes, and coincides with the marginal PH law
of `τ` (see [Bridge to PH](#Bridge-to-PH)):

```@example maph
ccdf(d, 1.0) ≈ sum(ccdf(d, 1.0, k) for k in 1:nabsorbing(d)) ≈ ccdf(PHDist(d), 1.0)
```

### Absorption probabilities

`R[i, k] = P(κ = k | start in phase i) = (-T⁻¹D)[i, k]`. Rows sum to 1 for
non-degenerate MAPHs.

```@example maph
absorption_probs(d)
```

```@example maph
sum(absorption_probs(d), dims=2)
```

`π = α · R` gives the marginal:

```@example maph
marginal_absorption(d)
```

### Joint moments

`E[τ^j · 𝟙{κ = k}] = (-1)^{j+1} · j! · α · T^{-(j+1)} · D[:, k]`:

```@example maph
kth_joint_moment(d, 1, 1)      # E[τ · 𝟙{κ=1}]
kth_joint_moment(d, 1, 2)      # E[τ² · 𝟙{κ=1}]
```

Summing across `k` at `j = 1` recovers the marginal mean:

```@example maph
sum(kth_joint_moment(d, k, 1) for k in 1:nabsorbing(d)) ≈ mean(PHDist(d))
```

## Sampling

`rand` returns a `Tuple{Float64, Int}` of `(τ, κ)`:

```@example maph
rng = Random.MersenneTwister(42)
rand(rng, d)
```

```@example maph
samples = rand(rng, d, 5)
```

Samples agree with the marginal absorption probabilities:

```@example maph
n = 50_000
sims = rand(rng, d, n)
[sum(s -> s[2] == k, sims) / n for k in 1:nabsorbing(d)]
```

### Right-censored sampling

`rand_censored` draws subjects that may be right-censored before their event.
The censoring argument is either a fixed administrative horizon or a
distribution for the per-subject censoring time; either way it is drawn
independently of `(τ, κ)`, so the censoring is non-informative:

```@example maph
sim = rand_censored(rng, d, 5_000, 0.4)     # horizon at u = 0.4
length(sim.events), length(sim.censored)
```

The censored fraction matches the survival at the horizon:

```@example maph
length(sim.censored) / 5_000, ccdf(d, 0.4)
```

The returned `(events, censored)` pair is shaped to feed the censored-data
fitting interface of
[PhaseTypeDistributionsFitting.jl](https://julia-matrix-analytic-probability.github.io/PhaseTypeDistributionsFitting.jl/)
directly, as `fit_mle(MAPHDist, sim.events; m = 3, censored = sim.censored)`.

## Bridge to PH

Every MAPH has a marginal `τ`, which is a PH distribution:

```@example maph
PHDist(d)                   # PH(α, T)
```

Conditional on the absorbing state reached, `τ | κ = k` is also PH:

```@example maph
ph1 = PHDist(d, 1)          # or: conditional_time(d, 1)
mean(ph1)
```

The conditional reweights the initial distribution and transient transitions
by `ρ_{i,k}`, the per-phase absorption probability. Phases with `ρ_{i,k} = 0`
cannot reach absorbing state `k` and are dropped from the conditional.

```@example maph
mean(ph1) ≈ kth_joint_moment(d, 1, 1) / marginal_absorption(d)[1]
```

## Reference — type

```@docs
AbstractMAPHDist
MAPHDist
```

## Reference — accessors and joint distribution

```@docs
exit_rate_matrix
nabsorbing
absorption_probs
marginal_absorption
kth_joint_moment
conditional_time
Distributions.cdf(::MAPHDist, ::Real, ::Integer)
Distributions.ccdf(::MAPHDist, ::Real, ::Integer)
Distributions.pdf(::MAPHDist, ::Real, ::Integer)
Distributions.cdf(::MAPHDist, ::Real)
Distributions.ccdf(::MAPHDist, ::Real)
Distributions.logccdf(::MAPHDist, ::Real)
Base.rand(::Random.AbstractRNG, ::MAPHDist)
rand_censored
```
