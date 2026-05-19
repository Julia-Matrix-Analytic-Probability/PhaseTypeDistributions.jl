```@meta
CurrentModule = PhaseTypeDistributions
```

# MAPH distributions

A **multi-absorbing phase-type** distribution is the joint distribution of the
pair `(τ, κ)`, where `τ` is the absorption time of a continuous-time Markov
chain and `κ ∈ {1, …, n}` is the index of the absorbing state reached. It
reduces to a standard PH distribution when `n = 1`, and is the natural model
for *competing risks*.

[`MAPHDist`](@ref) is parameterized by an initial distribution `α` over
transient phases, an `m × m` sub-generator `T`, and an `m × n` non-negative
matrix `D` of absorbing rates, with the constraint `T · 1_m + D · 1_n = 0`.

## Basic usage

```julia
using PhaseTypeDistributions

α = [0.6, 0.4]
T = [-3.0 1.0; 0.5 -2.0]
D = [1.5 0.5; 1.0 0.5]
d = MAPHDist(α, T, D)

pdf(d, 1.0, 1)              # joint sub-density f(1.0, κ=1)
cdf(d, 1.0, 1)              # P(τ ≤ 1, κ = 1)
ccdf(d, 1.0, 1)             # P(τ > 1, κ = 1)
marginal_absorption(d)      # vector of P(κ = k)
absorption_probs(d)         # matrix ρ_{ik} = P(κ=k | start i)
kth_joint_moment(d, 1, 2)   # E[τ² · 𝟙{κ=1}]

(τ, k) = rand(d)            # sample the pair
samples = rand(d, 1000)     # vector of (τ, κ) tuples
```

## Interface with PH

Every MAPH decomposes into PH distributions, and every PH embeds as an MAPH
with `n = 1`:

```julia
PHDist(d)               # marginal τ ~ PH(α, T)
PHDist(d, 1)            # conditional τ | κ = 1
conditional_time(d, 1)  # alias for PHDist(d, 1)

MAPHDist(ph)            # embed any AbstractPHDist as a 1-absorbing-state MAPH
```

## Construction from per-category PHs

Build an MAPH by supplying a PH distribution for each absorbing state and a
marginal probability vector `π`. The resulting MAPH is block-diagonal and the
conditional `τ | κ = k` is exactly the supplied PH:

```julia
he = HyperExponentialDist([0.4, 0.6], [2.0, 5.0])
er = ErlangPHDist(3, 2.0)
MAPHDist(AbstractPHDist[he, er], [0.3, 0.7])
```

## Moment-matched construction

Given target marginal absorption probabilities `π`, conditional means `μ`, and
conditional variances `σ²`, construct an MAPH that matches them. Each category
is realized by a 2-phase hyperexponential (`c² > 1`), hypoexponential
(`c² < 1`), or exponential (`c² = 1`), per the accompanying paper.

```julia
π  = [0.25, 0.45, 0.30]
μ  = [1.0,  3.0,  5.0]
σ² = [0.5, 12.0,  4.0]
MAPHDist(π, μ, σ²)
```

## Alternative parameterization `(α, q, R, U)`

```julia
MAPHDist(α, q, R, U)
```

with `q_i = -T_ii` the per-phase exit rates, `R` the `m × n` absorption-
probability matrix, and `U` the matrix of conditional one-step transition
probabilities. See the type reference below for the precise recovery formula.

## Type reference

```@docs
AbstractMAPHDist
MAPHDist
```

## MAPH function reference

```@docs
exit_rate_matrix
nabsorbing
absorption_probs
marginal_absorption
kth_joint_moment
conditional_time
Distributions.ccdf(::MAPHDist, ::Real, ::Integer)
```
