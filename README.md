# PhaseTypeDistributions.jl

A Julia package for working with [Phase-Type (PH) distributions](https://en.wikipedia.org/wiki/Phase-type_distribution). PH distributions are implemented as subtypes of `ContinuousUnivariateDistribution` from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), providing full integration with the Julia statistics ecosystem.

## Supported Distribution Types

[Phase-Type distributions](https://en.wikipedia.org/wiki/Phase-type_distribution) model non-negative random variables as the absorption time of a continuous-time Markov chain (CTMC). They are a versatile semi-parametric family that can approximate any non-negative distribution and are widely used in queueing theory, survival analysis, and reliability engineering.

Currently supported: **continuous-time, finite state space** PH distributions with fully observed absorption times.

### Type Hierarchy

All PH distribution types subtype `AbstractPHDist <: ContinuousUnivariateDistribution`:

- **`PHDist`** — General PH distribution parameterized by initial probability vector `α` and sub-generator matrix `T`
- **`HyperExponentialDist`** — Mixture of exponentials (diagonal `T`). SCV >= 1.
- **`HypoExponentialDist`** — Convolution of exponentials with distinct rates (bidiagonal `T`). SCV <= 1.
- **`ErlangPHDist`** — `k` phases with equal rate `λ` (special hypoexponential)
- **`CoxianDist`** — Sequential phases with exit probabilities at each stage

Each subtype stores its natural parameters and provides efficient specialized implementations for `mean`, `var`, `pdf`, `cdf`, `rand`, etc. All subtypes can be converted to the general `PHDist(α, T)` form via `PHDist(d)`.

### Conversions from Distributions.jl

```julia
PHDist(Exponential(2.0))   # 1-phase PH
PHDist(Erlang(3, 0.5))     # 3-phase PH
```

## Features

- **Distributions.jl integration**: All types are `ContinuousUnivariateDistribution` subtypes — `pdf`, `cdf`, `rand`, `mean`, `var`, `logpdf`, `ccdf` all work
- **Efficient specializations**: Each subtype uses closed-form formulas where possible (no matrix exponentials for hyperexponential, Erlang, etc.)
- **Moments**: `mean`, `var`, `scv` (squared coefficient of variation), `kth_moment`, `mgf`
- **Accessor functions**: `initial_prob(d)`, `subgenerator(d)`, `exit_rates(d)`, `nphases(d)` provide the underlying (α, T) representation for any subtype
- **Non-identifiability helpers**: PH distributions are not identifiable — different (α, T) can represent the same distribution. The package provides:
  - `moments_isapprox(d1, d2)` — compare by moments (necessary but not sufficient)
  - `distribution_isapprox(d1, d2)` — compare by CDF values (stronger test)
  - `moment_vector(d, order)` — extract raw moments for comparison

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yoninazarathy/PhaseTypeDistributions.jl")
```

## Basic Examples

### Constructing distributions

```julia
using PhaseTypeDistributions
using Distributions

# General PH distribution from (α, T)
ph = PHDist([0.6, 0.4], [-3.0 1.0; 0.0 -2.0])

# Hyperexponential — mixture of exponentials (SCV ≥ 1)
he = HyperExponentialDist([0.4, 0.6], [2.0, 5.0])

# ...or from a desired mean and SCV
he2 = HyperExponentialDist(2.0, 3.0)   # mean=2, scv=3

# Hypoexponential — convolution of exponentials with distinct rates (SCV ≤ 1)
ho = HypoExponentialDist([3.0, 5.0])
ho2 = HypoExponentialDist(2.0, 0.5)    # mean=2, scv≈0.5

# Erlang — k phases, equal rate λ
er = ErlangPHDist(3, 2.0)

# Coxian — rates and exit probabilities (last phase always absorbs)
cox = CoxianDist([3.0, 4.0, 5.0], [0.2, 0.3])

# From a Distributions.jl distribution
ph_exp = PHDist(Exponential(2.0))      # 1-phase PH
ph_erl = PHDist(Erlang(3, 0.5))        # 3-phase PH
```

### Distributions.jl API

Every PH type is a `ContinuousUnivariateDistribution`, so the standard API works:

```julia
mean(er)              # 1.5
var(er)               # 0.75
std(er)               # 0.866...
pdf(er, 1.0)          # density at 1.0
logpdf(er, 1.0)       # log-density (numerically stable for Erlang)
cdf(er, 1.0)          # P(X ≤ 1)
ccdf(er, 1.0)         # 1 - cdf = P(X > 1)
minimum(er), maximum(er)   # 0.0, Inf
insupport(er, 1.0)    # true

# Sampling
x  = rand(er)                 # one sample
xs = rand(er, 10_000)         # vector of samples

using Random
rng = MersenneTwister(42)
rand(rng, er, 1000)           # reproducible samples
```

### PH-specific quantities

```julia
# Moments
scv(he)                # squared coefficient of variation: Var/E[X]²
kth_moment(he, 3)      # E[X³]
mgf(he, 0.5)           # moment generating function at t=0.5

# Accessors — the underlying (α, T) representation
initial_prob(cox)      # α
subgenerator(cox)      # T
exit_rates(cox)        # t⁰ = -T·1
nphases(cox)           # 3
```

### Conversions and comparison

```julia
# Any subtype can be converted to the general (α, T) form
ph_from_he = PHDist(he)

# PH distributions are not identifiable — (α, T) is not unique.
# These helpers compare on moments and on CDF values:
moments_isapprox(he, ph_from_he)        # true — matching moments
distribution_isapprox(he, ph_from_he)   # true — matching CDFs
moment_vector(he, 4)                    # [E[X], E[X²], E[X³], E[X⁴]]
```

## Not Yet Supported

The following are not currently supported but may be added in the future:

- **Discrete-time PH distributions** (matrix-geometric distributions)
- **Censored observations** (right-censored, interval-censored data in EM fitting)
- **Infinite state space** PH distributions
- **Matrix-exponential distributions** (the broader class that generalizes PH distributions — every PH distribution is matrix-exponential, but not vice versa)
- **Point mass at zero** (defective initial distributions where `sum(α) < 1`)
- **Markovian Arrival Processes (MAP)** and **Batch MAP (BMAP)**
- **General multivariate PH distributions** (e.g. the MPH* class of Bladt and Nielsen)
- **MAPH distributions** (multi-absorbing PH for competing risks) — under development

## Accompanying Paper

This package accompanies a paper currently in preparation:

> Zhihao Qiao, Budhi Surya, Azam Asanjarani, Yoni Nazarathy. *Inference for Multi-Absorbing Phase Type Distributions, Algorithms, and Applications*. (In preparation.)
