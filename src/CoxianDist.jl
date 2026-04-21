"""
Coxian PH distribution — a sequential phase-type distribution where at each phase
the process either absorbs (with probability pᵢ) or advances to the next phase.
The last phase always absorbs (pₖ = 1 is implied and not stored).

Generalizes hypoexponential (all exit_probs = 0 except last) and Erlang
(all rates equal, all exit_probs = 0 except last).
"""
struct CoxianDist <: AbstractPHDist
    rates::Vector{Float64}       # phase rates λ₁, ..., λₖ (all > 0)
    exit_probs::Vector{Float64}  # absorption probability at each phase, length k-1 (pₖ=1 implied)

    function CoxianDist(rates::Vector{<:Real}, exit_probs::Vector{<:Real})
        k = length(rates)
        k >= 1 || throw(ArgumentError("must have at least one phase"))
        all(rates .> 0) || throw(ArgumentError("rates must be positive"))
        length(exit_probs) == k - 1 || throw(DimensionMismatch(
            "exit_probs must have length $(k-1) (last phase always absorbs), got $(length(exit_probs))"))
        all(0 .<= exit_probs .<= 1) || throw(ArgumentError("exit_probs must be in [0, 1]"))
        new(Float64.(rates), Float64.(exit_probs))
    end
end

"""Construct a single-phase Coxian (i.e., Exponential)."""
CoxianDist(rate::Real) = CoxianDist([Float64(rate)], Float64[])

# Accessors
nphases(d::CoxianDist) = length(d.rates)

function initial_prob(d::CoxianDist)
    α = zeros(nphases(d))
    α[1] = 1.0
    return α
end

function subgenerator(d::CoxianDist)
    k = nphases(d)
    T = zeros(k, k)
    for i in 1:k
        T[i, i] = -d.rates[i]
        if i < k
            # Transition to next phase: rate * (1 - exit_prob)
            T[i, i+1] = d.rates[i] * (1.0 - d.exit_probs[i])
        end
    end
    return T
end

function exit_rates(d::CoxianDist)
    k = nphases(d)
    t0 = zeros(k)
    for i in 1:k
        if i < k
            t0[i] = d.rates[i] * d.exit_probs[i]
        else
            t0[i] = d.rates[i]  # last phase always absorbs
        end
    end
    return t0
end

# Efficient specializations

function Statistics.mean(d::CoxianDist)
    k = nphases(d)
    # E[X] = Σᵢ (1/λᵢ) * P(reach phase i)
    # P(reach phase 1) = 1
    # P(reach phase i) = Π_{j=1}^{i-1} (1 - pⱼ)
    μ = 1.0 / d.rates[1]
    prob_reach = 1.0
    for i in 2:k
        prob_reach *= (1.0 - d.exit_probs[i-1])
        μ += prob_reach / d.rates[i]
    end
    return μ
end

function Statistics.var(d::CoxianDist)
    m2 = kth_moment(d, 2)
    m1 = mean(d)
    return m2 - m1^2
end

function kth_moment(d::CoxianDist, k_moment::Int)
    k_moment >= 1 || throw(ArgumentError("k must be >= 1"))
    # Use the general PH formula via (α, T) for moments > 1
    # For moment 1, we have the efficient recursive formula via mean()
    if k_moment == 1
        return mean(d)
    end
    # Fall back to matrix formula
    return invoke(kth_moment, Tuple{AbstractPHDist, Int}, d, k_moment)
end

function Distributions.pdf(d::CoxianDist, x::Real)
    x < 0 && return 0.0
    # Use the general matrix exponential formula
    α = initial_prob(d)
    T = subgenerator(d)
    t0 = exit_rates(d)
    return α' * exp(T * x) * t0
end

# ccdf and cdf inherited from AbstractPHDist (matrix exponential form).

function Random.rand(rng::AbstractRNG, d::CoxianDist)
    k = nphases(d)
    time = 0.0
    for i in 1:k
        # Sojourn time in phase i
        time += randexp(rng) / d.rates[i]
        # Absorb or continue?
        if i < k
            if rand(rng) < d.exit_probs[i]
                return time  # absorb from phase i
            end
            # else continue to phase i+1
        end
    end
    return time  # absorb from last phase
end

function mgf(d::CoxianDist, t::Real)
    α = initial_prob(d)
    T = subgenerator(d)
    t0 = exit_rates(d)
    return -α' * ((T - t * I) \ t0)
end

Distributions.params(d::CoxianDist) = (d.rates, d.exit_probs)

function Base.show(io::IO, d::CoxianDist)
    print(io, "CoxianDist(rates=", d.rates, ", exit_probs=", d.exit_probs, ")")
end
