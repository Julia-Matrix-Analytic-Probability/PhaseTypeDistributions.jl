```@meta
CurrentModule = PhaseTypeDistributions
```

# PhaseTypeDistributions.jl

A Julia package for working with [phase-type (PH) distributions](https://en.wikipedia.org/wiki/Phase-type_distribution)
and their multi-absorbing generalization (MAPH). PH distributions are
implemented as subtypes of `Distributions.ContinuousUnivariateDistribution`,
integrating with the Julia statistics ecosystem.

The package accompanies the paper *Inference for Multi-Absorbing Phase Type
Distributions, Algorithms, and Applications* (Qiao, Surya, Asanjarani,
Nazarathy; in preparation).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributions.jl")
```

## Scope

Currently supported: **continuous-time, finite state space** PH distributions
with fully observed absorption times, and their multi-absorbing generalization.

| Type                    | Description                                     |
|-------------------------|-------------------------------------------------|
| [`PHDist`](@ref)                | General PH from `(α, T)`                        |
| [`HyperExponentialDist`](@ref)  | Mixture of exponentials (SCV ≥ 1)               |
| [`HypoExponentialDist`](@ref)   | Convolution of exponentials (SCV ≤ 1)           |
| [`ErlangPHDist`](@ref)          | `k` phases with equal rate `λ`                  |
| [`CoxianDist`](@ref)            | Sequential phases with per-phase exit probs     |
| [`MAPHDist`](@ref)              | Multi-absorbing PH: joint distribution of (τ,κ) |

## Quick start

```julia
using PhaseTypeDistributions
using Distributions

# General PH
ph = PHDist([0.6, 0.4], [-3.0 1.0; 0.0 -2.0])
mean(ph); var(ph); pdf(ph, 1.0); cdf(ph, 1.0); rand(ph, 1000)

# Specialized subtypes
he  = HyperExponentialDist(2.0, 3.0)    # mean=2, SCV=3
ho  = HypoExponentialDist(2.0, 0.5)     # mean=2, SCV=0.5
er  = ErlangPHDist(3, 2.0)
cox = CoxianDist([3.0, 4.0, 5.0], [0.2, 0.3])

# Conversion to general PH
PHDist(he)                              # (α, T) form

# MAPH — joint distribution of (absorption time, absorbing state)
α = [0.6, 0.4]
T = [-3.0 1.0; 0.5 -2.0]
D = [1.5 0.5; 1.0 0.5]
d = MAPHDist(α, T, D)
τ, κ = rand(d)                          # one sample
marginal_absorption(d)                  # P(κ = k) vector
PHDist(d)                               # marginal τ  ~  PH
PHDist(d, 1)                            # conditional τ | κ = 1
```

## Navigation

- [PH distributions](ph.md) — walk through the PH types and API.
- [MAPH distributions](maph.md) — MAPH type and its interface with PH.
- [API reference](api.md) — exported names with full docstrings.
