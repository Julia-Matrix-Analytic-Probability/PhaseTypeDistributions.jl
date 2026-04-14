"""
Hypoexponential distribution — convolution of exponentials with distinct rates.
Represents a PH distribution with bidiagonal sub-generator matrix T and α = [1, 0, ..., 0].
SCV is always ≤ 1 (when rates are distinct and positive).
"""
struct HypoExponentialDist <: AbstractPHDist
    rates::Vector{Float64}   # phase rates (all > 0, should be distinct for closed-form)

    function HypoExponentialDist(rates::Vector{<:Real})
        length(rates) >= 1 || throw(ArgumentError("must have at least one rate"))
        all(rates .> 0) || throw(ArgumentError("rates must be positive"))
        new(Float64.(rates))
    end
end

"""Construct a HypoExponentialDist from desired mean and SCV (must be < 1)."""
function HypoExponentialDist(mean_desired::Real, scv_desired::Real)
    scv_desired < 1.0 || throw(ArgumentError("SCV must be < 1 for hypoexponential, got $scv_desired"))
    n = Int(ceil(1 / scv_desired))
    ν1 = n / (1 + sqrt((n - 1) * (n * scv_desired - 1)))
    ν2 = ν1 * (n - 1) / (ν1 - 1)
    rates = vcat([ν1 / mean_desired], fill(ν2 / mean_desired, n - 1))
    return HypoExponentialDist(rates)
end

# Accessors
nphases(d::HypoExponentialDist) = length(d.rates)

function initial_prob(d::HypoExponentialDist)
    m = nphases(d)
    α = zeros(m)
    α[1] = 1.0
    return α
end

function subgenerator(d::HypoExponentialDist)
    m = nphases(d)
    T = zeros(m, m)
    for i in 1:m
        T[i, i] = -d.rates[i]
        if i < m
            T[i, i+1] = d.rates[i]
        end
    end
    return T
end

function exit_rates(d::HypoExponentialDist)
    m = nphases(d)
    t0 = zeros(m)
    t0[m] = d.rates[m]  # only the last phase can absorb
    return t0
end

# Efficient specializations
function Statistics.mean(d::HypoExponentialDist)
    return sum(1.0 ./ d.rates)
end

function Statistics.var(d::HypoExponentialDist)
    return sum(1.0 ./ (d.rates .^ 2))
end

function Distributions.pdf(d::HypoExponentialDist, x::Real)
    x < 0 && return 0.0
    λ = d.rates
    n = length(λ)
    n == 1 && return λ[1] * exp(-λ[1] * x)

    # Partial fraction expansion (requires distinct rates)
    result = 0.0
    for i in 1:n
        coeff = λ[i]
        for j in 1:n
            j == i && continue
            coeff *= λ[j] / (λ[j] - λ[i])
        end
        result += coeff * exp(-λ[i] * x)
    end
    return result
end

function Distributions.cdf(d::HypoExponentialDist, x::Real)
    x < 0 && return 0.0
    λ = d.rates
    n = length(λ)
    n == 1 && return 1.0 - exp(-λ[1] * x)

    # Partial fraction expansion
    result = 1.0
    for i in 1:n
        coeff = 1.0
        for j in 1:n
            j == i && continue
            coeff *= λ[j] / (λ[j] - λ[i])
        end
        result -= coeff * exp(-λ[i] * x)
    end
    return result
end

function Random.rand(rng::AbstractRNG, d::HypoExponentialDist)
    return sum(randexp(rng) / λ for λ in d.rates)
end

function kth_moment(d::HypoExponentialDist, k::Int)
    # For a sum of independent exponentials, use the general PH formula
    # but mean and variance have simple forms
    k == 1 && return mean(d)
    k == 2 && return var(d) + mean(d)^2
    # Fall back to general formula for higher moments
    return invoke(kth_moment, Tuple{AbstractPHDist, Int}, d, k)
end

function mgf(d::HypoExponentialDist, t::Real)
    return prod(d.rates ./ (d.rates .- t))
end
