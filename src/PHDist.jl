# Abstract type for all phase-type distributions
abstract type AbstractPHDist <: ContinuousUnivariateDistribution end

# --- Accessor functions (subtypes must implement) ---

"""Return the initial probability vector α."""
initial_prob(d::AbstractPHDist) = error("initial_prob not implemented for $(typeof(d))")

"""Return the sub-generator matrix T."""
subgenerator(d::AbstractPHDist) = error("subgenerator not implemented for $(typeof(d))")

"""Return the exit rate vector t⁰ = -T * 1."""
exit_rates(d::AbstractPHDist) = -subgenerator(d) * ones(nphases(d))

"""Return the number of transient states (phases)."""
nphases(d::AbstractPHDist) = length(initial_prob(d))

# --- General PH distribution: (α, T) matrix representation ---

struct PHDist <: AbstractPHDist
    α::Vector{Float64}
    T::Matrix{Float64}

    function PHDist(α::Vector{<:Real}, T::Matrix{<:Real})
        m = length(α)
        size(T) == (m, m) || throw(DimensionMismatch("T must be $m × $m, got $(size(T))"))
        all(α .>= 0) || throw(ArgumentError("α must be non-negative"))
        isapprox(sum(α), 1.0; atol=1e-10) || throw(ArgumentError("α must sum to 1, got $(sum(α))"))
        new(Float64.(α), Float64.(T))
    end
end

initial_prob(d::PHDist) = d.α
subgenerator(d::PHDist) = d.T

# --- Generic Distributions.jl interface (fallbacks using matrix operations) ---

Distributions.minimum(d::AbstractPHDist) = 0.0
Distributions.maximum(d::AbstractPHDist) = Inf
Distributions.insupport(d::AbstractPHDist, x::Real) = x >= 0.0

function Distributions.pdf(d::AbstractPHDist, x::Real)
    x < 0 && return 0.0
    α = initial_prob(d)
    T = subgenerator(d)
    t0 = exit_rates(d)
    return α' * exp(T * x) * t0
end

function Distributions.logpdf(d::AbstractPHDist, x::Real)
    p = pdf(d, x)
    return p > 0 ? log(p) : -Inf
end

# For PH distributions, ccdf is the natively-computed quantity: α' exp(Tx) 1.
# cdf derives from it, so tail precision is preserved. Subtypes override ccdf.
function Distributions.ccdf(d::AbstractPHDist, x::Real)
    x <= 0 && return 1.0
    α = initial_prob(d)
    T = subgenerator(d)
    m = nphases(d)
    return α' * exp(T * x) * ones(m)
end

function Distributions.cdf(d::AbstractPHDist, x::Real)
    x < 0 && return 0.0
    return 1.0 - ccdf(d, x)
end

function Statistics.mean(d::AbstractPHDist)
    α = initial_prob(d)
    T = subgenerator(d)
    m = nphases(d)
    return -(α' * (T \ ones(m)))
end

function Statistics.var(d::AbstractPHDist)
    α = initial_prob(d)
    T = subgenerator(d)
    m = nphases(d)
    Tinv_ones = T \ ones(m)
    m1 = -(α' * Tinv_ones)
    m2 = 2.0 * α' * (T \ Tinv_ones)
    return m2 - m1^2
end

"""Squared coefficient of variation: Var(X) / E[X]²."""
function scv(d::AbstractPHDist)
    μ = mean(d)
    σ2 = var(d)
    return σ2 / μ^2
end

"""Compute the k-th raw moment E[X^k]."""
function kth_moment(d::AbstractPHDist, k::Int)
    k >= 1 || throw(ArgumentError("k must be >= 1"))
    α = initial_prob(d)
    T = subgenerator(d)
    m = nphases(d)
    return Float64(factorial(k)) * ((-1)^k) * α' * (T^(-k)) * ones(m)
end

"""Moment generating function E[exp(t*X)], defined for t < min(-diag(T))."""
function mgf(d::AbstractPHDist, t::Real)
    α = initial_prob(d)
    T = subgenerator(d)
    t0 = exit_rates(d)
    return -(α' * ((T + t * I) \ t0))
end

# Skewness: E[(X-μ)³] / σ³
function Distributions.skewness(d::AbstractPHDist)
    μ = mean(d)
    m2 = kth_moment(d, 2)
    m3 = kth_moment(d, 3)
    σ2 = m2 - μ^2
    σ = sqrt(σ2)
    return (m3 - 3μ * m2 + 2μ^3) / σ^3
end

# Excess kurtosis: E[(X-μ)⁴] / σ⁴ - 3
function Distributions.kurtosis(d::AbstractPHDist)
    μ = mean(d)
    m2 = kth_moment(d, 2)
    m3 = kth_moment(d, 3)
    m4 = kth_moment(d, 4)
    σ2 = m2 - μ^2
    return (m4 - 4μ * m3 + 6μ^2 * m2 - 3μ^4) / σ2^2 - 3
end

# Quantile via bisection. PH support is [0, ∞), so we grow hi from the mean.
function Distributions.quantile(d::AbstractPHDist, p::Real)
    0 <= p <= 1 || throw(DomainError(p, "quantile p must be in [0, 1]"))
    p == 0 && return 0.0
    p == 1 && return Inf
    hi = max(mean(d), 1.0)
    while cdf(d, hi) < p
        hi *= 2
        hi > 1e300 && return hi
    end
    lo = 0.0
    tol = 1e-12 * max(1.0, hi)
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        if cdf(d, mid) < p
            lo = mid
        else
            hi = mid
        end
        (hi - lo) < tol && break
    end
    return 0.5 * (lo + hi)
end

# Distributions.jl convention: params returns a tuple of the defining parameters.
Distributions.params(d::PHDist) = (d.α, d.T)

function Base.show(io::IO, d::PHDist)
    print(io, "PHDist(α=", d.α, ", T=", d.T, ")")
end

function Random.rand(rng::AbstractRNG, d::AbstractPHDist)
    α = initial_prob(d)
    T = subgenerator(d)
    t0 = exit_rates(d)
    m = nphases(d)

    # Choose initial state from α
    u = rand(rng)
    cumprob = 0.0
    state = 1
    for i in 1:m
        cumprob += α[i]
        if u <= cumprob
            state = i
            break
        end
    end

    # Simulate CTMC until absorption
    time = 0.0
    while true
        rate = -T[state, state]
        time += randexp(rng) / rate

        # Absorb or move to another transient state
        u = rand(rng)
        p_absorb = t0[state] / rate
        if u < p_absorb
            return time
        end

        # Move to another transient state
        cumprob = 0.0
        for j in 1:m
            j == state && continue
            cumprob += T[state, j] / rate
            if u - p_absorb <= cumprob
                state = j
                break
            end
        end
    end
end
