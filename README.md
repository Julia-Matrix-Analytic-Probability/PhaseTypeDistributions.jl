# PhaseTypeDistributions.jl

A Julia package for working with [Phase-Type (PH) distributions](https://en.wikipedia.org/wiki/Phase-type_distribution) and their multi-absorbing generalization (MAPH). The package provides distribution construction, statistical functions, random sampling, and parameter estimation via the EM algorithm, with an interface to [Distributions.jl](https://github.com/JuliaStats/Distributions.jl).

## Supported Distribution Types

### Phase-Type (PH) Distributions

[Phase-Type distributions](https://en.wikipedia.org/wiki/Phase-type_distribution) model non-negative random variables as the absorption time of a continuous-time Markov chain (CTMC). They are a versatile semi-parametric family that can approximate any non-negative distribution and are widely used in queueing theory, survival analysis, and reliability engineering.

Currently supported: **continuous-time** PH distributions. Discrete-time PH distributions and censored observations are not yet supported.

The package provides convenience constructors for common PH sub-classes:
- **Exponential** distributions (`exp_dist`)
- **Hyperexponential** distributions (`hyper_exp_dist`) for squared coefficient of variation > 1
- **Hypoexponential** distributions (`hypo_exp_dist`) for squared coefficient of variation < 1

### Multi-Absorbing Phase-Type (MAPH) Distributions

MAPH distributions generalize PH distributions to the competing risk setting. Where a standard PH distribution models a single absorption time, an MAPH distribution models a bivariate random variable: the time until absorption together with the cause of absorption from one of several possible types. This is constructed via a CTMC with multiple absorbing states, where the distribution records both *when* absorption occurs and *which* absorbing state is reached.

An MAPH distribution of order (m, n) has m transient states (phases) and n absorbing states, and is parameterized by an initial distribution vector, a transient rate matrix, and an exit rate matrix to the absorbing states. When n = 1, it reduces to a standard PH distribution. MAPH distributions are useful for modelling competing risks in biostatistics (e.g. multiple causes of failure), reliability analysis, and other domains where both the timing and type of an event matter.

## Features

- **Construction**: Create MAPH distributions from parameters using multiple parameterizations: (alpha, T, D) or (alpha, q, R, U)
- **Moments**: `mean`, `var`, `scv` (squared coefficient of variation), `kth_moment`
- **Density and distribution**: `sub_pdf`, `sub_distribution` (conditional on absorbing state), `mgf`
- **Absorption probabilities**: `absorption_probs`
- **Random sampling**: `rand` generates full trajectory observations from the underlying CTMC
- **Parameter estimation**: `EM_fit` implements the EM algorithm for fitting MAPH parameters from observed (time, absorbing state) data
- **Initialization**: `maph_initialization` provides a moment-based heuristic for constructing initial parameter estimates
- **Distributions.jl integration**: Overloads `mean`, `var`, `rand` for MAPH types

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yoninazarathy/PhaseTypeDistributions.jl")
```

## Quick Example

```julia
using MAPHDistributions

# Create an MAPH(2,2) distribution: 2 transient states, 2 absorbing states
alpha = [0.5, 0.5]
T = [-3.0 1.0; 0.5 -2.0]
D = [1.5 0.5; 0.5 1.0]
maph = MAPHDist(alpha, T, D)

# Compute statistics
println("Mean: ", mean(maph))
println("Variance: ", var(maph))
println("Absorption probabilities: ", absorption_probs(maph))

# Sample observations
obs = [rand(maph) for _ in 1:1000]

# Fit a model from observations using the EM algorithm
maph_init = maph_initialization(obs, 2)
fitted = EM_fit(obs, maph_init)
```

## Accompanying Paper

This package accompanies the paper:

> Zhihao Qiao, Budhi Surya, Azam Asanjarani, Yoni Nazarathy. *Inference for Multi-Absorbing Phase Type Distributions, Algorithms, and Applications*.

See the [paper repository](https://github.com/yoninazarathy/maph-fitting-paper) for details.
