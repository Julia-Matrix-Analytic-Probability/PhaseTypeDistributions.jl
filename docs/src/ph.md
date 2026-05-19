```@meta
CurrentModule = PhaseTypeDistributions
```

# PH distributions

All PH distributions in this package subtype
`AbstractPHDist <: ContinuousUnivariateDistribution`, so the full
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) interface
works on every subtype. Specialized subtypes use closed-form implementations
where available; the general [`PHDist`](@ref) falls back to matrix-exponential
formulas.

```@example ph
using PhaseTypeDistributions, Distributions, Random, Statistics
nothing # hide
```

## Constructors

### General `PHDist(α, T)`

```@example ph
ph = PHDist([0.6, 0.4], [-3.0 1.0; 0.0 -2.0])
```

### [`HyperExponentialDist`](@ref) — mixture of exponentials

A diagonal sub-generator: SCV is always ≥ 1.

```@example ph
he  = HyperExponentialDist([0.4, 0.6], [2.0, 5.0])
he2 = HyperExponentialDist(2.0, 3.0)        # by mean and SCV
mean(he2), scv(he2)
```

### [`HypoExponentialDist`](@ref) — convolution of exponentials

A bidiagonal sub-generator with `α = [1, 0, …, 0]`: SCV is always ≤ 1. Repeated
rates fall back to the matrix-exponential form (partial fractions diverge for
equal rates).

```@example ph
ho  = HypoExponentialDist([3.0, 5.0])
ho2 = HypoExponentialDist(2.0, 0.5)         # by mean and SCV
mean(ho2), scv(ho2)
```

### [`ErlangPHDist`](@ref) — `k` equal phases

```@example ph
er = ErlangPHDist(3, 2.0)
mean(er), var(er)
```

### [`CoxianDist`](@ref) — sequential phases with exit probabilities

```@example ph
cox = CoxianDist([3.0, 4.0, 5.0], [0.2, 0.3])
mean(cox), var(cox)
```

### From Distributions.jl

```@example ph
ph_exp = PHDist(Exponential(2.0))           # 1-phase
ph_erl = PHDist(Erlang(3, 0.5))             # 3-phase
mean(ph_exp), mean(ph_erl)
```

## Standard Distributions.jl interface

Every PH type supports the standard density / cdf / sampling API:

```@example ph
pdf(er, 1.0), logpdf(er, 1.0), cdf(er, 1.0), ccdf(er, 1.0)
```

`ccdf` is the natively-computed quantity for PH distributions
(`α' exp(Tx) 𝟙`) — `cdf` is derived from it — so tail precision is preserved
into the deep tail:

```@example ph
ccdf(er, 50.0)        # ≪ 1, but still finite
```

Support is `[0, ∞)`:

```@example ph
minimum(er), maximum(er), insupport(er, 1.0), insupport(er, -1.0)
```

### Moments and shape

```@example ph
mean(he), var(he), std(he), scv(he)
```

```@example ph
skewness(he), kurtosis(he)      # excess kurtosis
```

```@example ph
kth_moment(he, 3), mgf(he, 0.5)
```

`mgf` extends `Distributions.mgf`. It is defined for `t < min(-diag(T))`; the
linear solve diverges otherwise.

### Quantile / median

```@example ph
quantile(er, 0.95), median(er)
```

`quantile` for the general `AbstractPHDist` uses bisection; specialized
subtypes that are exactly `Distributions.Erlang` etc. could use closed forms
but currently also bisect via the fallback.

### Sampling

```@example ph
rng = Random.MersenneTwister(42)
rand(rng, er)
```

```@example ph
xs = rand(rng, er, 1000);
length(xs), extrema(xs), Statistics.mean(xs)
```

Internally, sampling simulates the underlying CTMC until absorption. Each
specialized subtype overrides `rand` with its natural recipe — e.g.
`HyperExponentialDist` picks a component and draws one exponential;
`ErlangPHDist` sums `k` independent exponentials.

## Accessors — the underlying `(α, T)` representation

Every subtype exposes its parameters in the canonical PH form:

```@example ph
initial_prob(cox), subgenerator(cox), exit_rates(cox), nphases(cox)
```

```@example ph
params(cox)                     # natural parameters: (rates, exit_probs)
params(he)                      # (probs, rates)
params(er)                      # (shape, rate)
params(ph)                      # (α, T)
```

## Conversion to the general form

Any subtype converts to the general `(α, T)` form via `PHDist(d)`:

```@example ph
ph_from_he = PHDist(he)
ph_from_er = PHDist(er)
nphases(ph_from_he), nphases(ph_from_er)
```

`PHDist(d::PHDist)` is a no-op.

## Comparison helpers for non-identifiable distributions

Two different `(α, T)` representations can describe the same distribution.
The package provides comparison routines that work *across* representations:

```@example ph
moments_isapprox(he, ph_from_he)        # by moments
distribution_isapprox(he, ph_from_he)   # by CDF on an adaptive grid
moment_vector(he, 4)                    # [E[X], E[X²], E[X³], E[X⁴]]
```

`moments_isapprox` matches a fixed number of moments (necessary but not
sufficient); `distribution_isapprox` evaluates the CDF on a grid spanning out
to several standard deviations and so is a stronger check.

## Reference — types

```@docs
AbstractPHDist
PHDist
HyperExponentialDist
HypoExponentialDist
ErlangPHDist
CoxianDist
```

## Reference — accessors and moments

```@docs
initial_prob
subgenerator
exit_rates
nphases
scv
kth_moment
Distributions.mgf(::AbstractPHDist, ::Real)
```

## Reference — comparison helpers

```@docs
moments_isapprox
distribution_isapprox
moment_vector
```
