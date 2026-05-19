```@meta
CurrentModule = PhaseTypeDistributions
```

# PH distributions

All PH distribution types subtype `AbstractPHDist <: ContinuousUnivariateDistribution`,
so the full [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
interface (`pdf`, `cdf`, `ccdf`, `logpdf`, `quantile`, `mean`, `var`, `std`,
`skewness`, `kurtosis`, `rand`, `minimum`, `maximum`, `insupport`, `params`)
works on every subtype. Specialized closed-form implementations are used where
available; the matrix-exponential fallback is used otherwise.

## Constructors

```julia
using PhaseTypeDistributions, Distributions

# General PH(α, T)
ph  = PHDist([0.6, 0.4], [-3.0 1.0; 0.0 -2.0])

# Hyperexponential — mixture of exponentials (SCV ≥ 1)
he  = HyperExponentialDist([0.4, 0.6], [2.0, 5.0])
he2 = HyperExponentialDist(2.0, 3.0)       # by mean and SCV

# Hypoexponential — convolution of exponentials (SCV ≤ 1)
ho  = HypoExponentialDist([3.0, 5.0])
ho2 = HypoExponentialDist(2.0, 0.5)        # by mean and SCV

# Erlang — k equal phases
er  = ErlangPHDist(3, 2.0)

# Coxian — sequential phases with per-phase exit probabilities
cox = CoxianDist([3.0, 4.0, 5.0], [0.2, 0.3])

# From a Distributions.jl distribution
ph_exp = PHDist(Exponential(2.0))          # 1-phase PH
ph_erl = PHDist(Erlang(3, 0.5))            # 3-phase PH
```

## Standard interface

```julia
mean(er); var(er); std(er)
pdf(er, 1.0); logpdf(er, 1.0)
cdf(er, 1.0); ccdf(er, 1.0)
quantile(er, 0.95); median(er)
skewness(er); kurtosis(er)         # excess kurtosis
minimum(er), maximum(er)           # 0.0, Inf
rand(er); rand(er, 1000)
```

## PH-specific quantities

```julia
scv(he)                # squared coefficient of variation
kth_moment(he, 3)      # E[X³]
mgf(he, 0.5)           # MGF at t=0.5
initial_prob(cox)      # α
subgenerator(cox)      # T
exit_rates(cox)        # t⁰ = -T·1
nphases(cox)
```

## Conversion and comparison

Any subtype can be converted to the general `(α, T)` form, and PH distributions
are non-identifiable, so the package provides comparison helpers:

```julia
PHDist(he)                              # to general form
moments_isapprox(he, PHDist(he))        # by moments
distribution_isapprox(he, PHDist(he))   # by CDF on an adaptive grid
moment_vector(he, 4)                    # [E[X], …, E[X⁴]]
```

## Type reference

```@docs
AbstractPHDist
PHDist
HyperExponentialDist
HypoExponentialDist
ErlangPHDist
CoxianDist
```

## PH function reference

```@docs
initial_prob
subgenerator
exit_rates
nphases
scv
kth_moment
Distributions.mgf(::AbstractPHDist, ::Real)
moments_isapprox
distribution_isapprox
moment_vector
```
