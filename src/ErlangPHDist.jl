"""
Erlang PH distribution — k phases with equal rate λ.
A special case of hypoexponential where all rates are identical.
Equivalent to Gamma(k, 1/λ) or Distributions.Erlang(k, 1/λ).
"""
struct ErlangPHDist <: AbstractPHDist
    shape::Int       # k — number of phases
    rate::Float64    # λ — rate of each phase

    function ErlangPHDist(shape::Int, rate::Real)
        shape >= 1 || throw(ArgumentError("shape must be >= 1, got $shape"))
        rate > 0 || throw(ArgumentError("rate must be positive, got $rate"))
        new(shape, Float64(rate))
    end
end

# Accessors
nphases(d::ErlangPHDist) = d.shape

function initial_prob(d::ErlangPHDist)
    α = zeros(d.shape)
    α[1] = 1.0
    return α
end

function subgenerator(d::ErlangPHDist)
    k = d.shape
    T = zeros(k, k)
    for i in 1:k
        T[i, i] = -d.rate
        if i < k
            T[i, i+1] = d.rate
        end
    end
    return T
end

function exit_rates(d::ErlangPHDist)
    t0 = zeros(d.shape)
    t0[d.shape] = d.rate
    return t0
end

# Efficient specializations — closed-form Erlang formulas
function Statistics.mean(d::ErlangPHDist)
    return d.shape / d.rate
end

function Statistics.var(d::ErlangPHDist)
    return d.shape / d.rate^2
end

function Distributions.pdf(d::ErlangPHDist, x::Real)
    x < 0 && return 0.0
    k = d.shape
    λ = d.rate
    # f(x) = λ^k * x^(k-1) * exp(-λx) / (k-1)!
    return (λ^k * x^(k-1) * exp(-λ * x)) / factorial(k - 1)
end

function Distributions.logpdf(d::ErlangPHDist, x::Real)
    x < 0 && return -Inf
    k = d.shape
    λ = d.rate
    return k * log(λ) + (k - 1) * log(x) - λ * x - first(logabsgamma(k))
end

function Distributions.cdf(d::ErlangPHDist, x::Real)
    x < 0 && return 0.0
    k = d.shape
    λ = d.rate
    # CDF = 1 - Σ_{i=0}^{k-1} (λx)^i * exp(-λx) / i!
    s = 0.0
    term = 1.0  # (λx)^0 / 0! = 1
    lx = λ * x
    for i in 0:(k-1)
        s += term
        term *= lx / (i + 1)
    end
    return 1.0 - s * exp(-lx)
end

function Random.rand(rng::AbstractRNG, d::ErlangPHDist)
    # Sum of k iid Exponential(λ) random variables
    s = 0.0
    for _ in 1:d.shape
        s += randexp(rng) / d.rate
    end
    return s
end

function kth_moment(d::ErlangPHDist, k_moment::Int)
    k_moment >= 1 || throw(ArgumentError("k must be >= 1"))
    n = d.shape
    λ = d.rate
    # E[X^k] = Γ(n+k) / (λ^k * Γ(n)) = (n+k-1)! / (λ^k * (n-1)!)
    return exp(first(logabsgamma(n + k_moment)) - first(logabsgamma(n))) / λ^k_moment
end

function mgf(d::ErlangPHDist, t::Real)
    return (d.rate / (d.rate - t))^d.shape
end
