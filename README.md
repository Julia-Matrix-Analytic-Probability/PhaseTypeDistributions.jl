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

## Quick Example

```julia
using PhaseTypeDistributions
using Distributions

# General PH distribution
ph = PHDist([0.6, 0.4], [-3.0 1.0; 0.0 -2.0])
println("Mean: ", mean(ph))
println("PDF at 1.0: ", pdf(ph, 1.0))
println("CDF at 1.0: ", cdf(ph, 1.0))
samples = rand(ph, 1000)

# Hyperexponential (SCV > 1)
he = HyperExponentialDist([0.4, 0.6], [2.0, 5.0])
println("HyperExp mean: ", mean(he))

# Erlang (k phases, rate λ)
er = ErlangPHDist(3, 2.0)
println("Erlang mean: ", mean(er), " var: ", var(er))

# Coxian with exit probabilities
cox = CoxianDist([3.0, 4.0, 5.0], [0.2, 0.3])
println("Coxian mean: ", mean(cox))

# Convert any subtype to general (α, T) form
ph_from_he = PHDist(he)

# Compare distributions (non-identifiability)
moments_isapprox(he, ph_from_he)       # true — same moments
distribution_isapprox(he, ph_from_he)  # true — same CDF
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
