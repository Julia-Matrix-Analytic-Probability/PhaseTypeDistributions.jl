```@meta
CurrentModule = PhaseTypeDistributions
```

# PhaseTypeDistributions.jl

A Julia package for working with
[phase-type (PH) distributions](https://en.wikipedia.org/wiki/Phase-type_distribution)
and their multi-absorbing generalization (MAPH). PH distributions are
implemented as subtypes of `Distributions.ContinuousUnivariateDistribution`,
integrating with the Julia statistics ecosystem.

The package accompanies the paper *Inference for Multi-Absorbing Phase Type
Distributions, Algorithms, and Applications* (Qiao, Surya, Asanjarani,
Nazarathy; in preparation).

## Background

A continuous-time PH distribution is the distribution of the absorption time
of a finite-state continuous-time Markov chain with a single absorbing state.
Concretely, given an initial distribution `α` over `m` transient phases and a
sub-generator matrix `T`, the absorption time `τ` has

```math
F_τ(x) = 1 - α^⊤ \exp(Tx) \mathbf{1}, \qquad
f_τ(x) = α^⊤ \exp(Tx) t^0, \qquad
t^0 = -T \mathbf{1}.
```

The MAPH generalization replaces the single absorbing state with `n` distinct
ones; the joint distribution of `(τ, κ)` (absorption time and the index of
the absorbing state reached) is the natural model for *competing risks*.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributions.jl")
```

## Scope

Continuous-time, finite state space, fully observed absorption times — for PH
and MAPH.

| Type                            | Description                                     |
|---------------------------------|-------------------------------------------------|
| [`PHDist`](@ref)                | General PH from `(α, T)`                        |
| [`HyperExponentialDist`](@ref)  | Mixture of exponentials (SCV ≥ 1)               |
| [`HypoExponentialDist`](@ref)   | Convolution of exponentials (SCV ≤ 1)           |
| [`ErlangPHDist`](@ref)          | `k` phases with equal rate `λ`                  |
| [`CoxianDist`](@ref)            | Sequential phases with per-phase exit probs     |
| [`MAPHDist`](@ref)              | Multi-absorbing PH: joint distribution of (τ,κ) |

## Quick start

```@example quick
using PhaseTypeDistributions
using Distributions

ph  = PHDist([0.6, 0.4], [-3.0 1.0; 0.0 -2.0])
mean(ph), var(ph), pdf(ph, 1.0), cdf(ph, 1.0)
```

```@example quick
he  = HyperExponentialDist(2.0, 3.0)        # by mean and SCV
ho  = HypoExponentialDist(2.0, 0.5)
er  = ErlangPHDist(3, 2.0)
cox = CoxianDist([3.0, 4.0, 5.0], [0.2, 0.3])

mean.([he, ho, er, cox])
```

```@example quick
α = [0.6, 0.4]
T = [-3.0 1.0; 0.5 -2.0]
D = [1.5 0.5; 1.0 0.5]
d = MAPHDist(α, T, D)
marginal_absorption(d), rand(d)
```

## Navigation

- [PH distributions](ph.md) — walk through every PH type and the full API.
- [MAPH distributions](maph.md) — MAPH type, joint distribution API, bridges to PH.
- [API reference](api.md) — index of every documented name.

## Not yet supported

- Discrete-time PH distributions (matrix-geometric).
- Censored observations.
- Infinite state space.
- Matrix-exponential distributions (the broader class generalizing PH).
- Point mass at zero (defective initial distributions where `sum(α) < 1`).
- Markovian Arrival Processes (MAP, BMAP).
- General multivariate PH distributions (e.g. the MPH* class).
