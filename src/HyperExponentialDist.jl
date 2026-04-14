"""
Hyperexponential distribution — mixture of exponentials.
Represents a PH distribution with diagonal sub-generator matrix T.
SCV is always ≥ 1.
"""
struct HyperExponentialDist <: AbstractPHDist
    probs::Vector{Float64}   # mixing probabilities (sum to 1)
    rates::Vector{Float64}   # exponential rates (all > 0)

    function HyperExponentialDist(probs::Vector{<:Real}, rates::Vector{<:Real})
        length(probs) == length(rates) || throw(DimensionMismatch("probs and rates must have same length"))
        all(probs .>= 0) || throw(ArgumentError("probs must be non-negative"))
        isapprox(sum(probs), 1.0; atol=1e-10) || throw(ArgumentError("probs must sum to 1, got $(sum(probs))"))
        all(rates .> 0) || throw(ArgumentError("rates must be positive"))
        new(Float64.(probs), Float64.(rates))
    end
end

"""Construct a HyperExponentialDist from desired mean and SCV (must be > 1)."""
function HyperExponentialDist(mean_desired::Real, scv_desired::Real)
    scv_desired > 1.0 || throw(ArgumentError("SCV must be > 1 for hyperexponential, got $scv_desired"))
    μ1 = 1 / (scv_desired + 1)
    p = (scv_desired - 1) / (scv_desired + 1 + 2 / μ1^2 - 4 / μ1)
    μ2 = (1 - p) / (1 - p / μ1)
    probs = [p, 1 - p]
    rates = [μ1 / mean_desired, μ2 / mean_desired]
    return HyperExponentialDist(probs, rates)
end

# Accessors
nphases(d::HyperExponentialDist) = length(d.rates)
initial_prob(d::HyperExponentialDist) = d.probs
subgenerator(d::HyperExponentialDist) = diagm(-d.rates)
exit_rates(d::HyperExponentialDist) = d.rates

# Efficient specializations
function Statistics.mean(d::HyperExponentialDist)
    return sum(d.probs ./ d.rates)
end

function Statistics.var(d::HyperExponentialDist)
    m1 = sum(d.probs ./ d.rates)
    m2 = 2.0 * sum(d.probs ./ (d.rates .^ 2))
    return m2 - m1^2
end

function Distributions.pdf(d::HyperExponentialDist, x::Real)
    x < 0 && return 0.0
    return sum(d.probs .* d.rates .* exp.(-d.rates .* x))
end

function Distributions.cdf(d::HyperExponentialDist, x::Real)
    x < 0 && return 0.0
    return 1.0 - sum(d.probs .* exp.(-d.rates .* x))
end

function Random.rand(rng::AbstractRNG, d::HyperExponentialDist)
    # Pick component from mixing distribution
    u = rand(rng)
    cumprob = 0.0
    for i in eachindex(d.probs)
        cumprob += d.probs[i]
        if u <= cumprob
            return randexp(rng) / d.rates[i]
        end
    end
    return randexp(rng) / d.rates[end]
end

function kth_moment(d::HyperExponentialDist, k::Int)
    k >= 1 || throw(ArgumentError("k must be >= 1"))
    return Float64(factorial(k)) * sum(d.probs ./ (d.rates .^ k))
end

function mgf(d::HyperExponentialDist, t::Real)
    return sum(d.probs .* d.rates ./ (d.rates .- t))
end
