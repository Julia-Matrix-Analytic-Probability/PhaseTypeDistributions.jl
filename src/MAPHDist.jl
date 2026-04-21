# Multi-Absorbing Phase-Type (MAPH) distribution.
# Joint distribution of the pair (τ, κ), where τ is the absorption time of an
# underlying CTMC and κ ∈ {1,...,n} indexes the absorbing state reached.
# When n = 1, MAPH reduces to a standard phase-type distribution.

"""
Abstract supertype for multi-absorbing phase-type (MAPH) distributions.
Not a Distributions.jl `Distribution` — the value support is bivariate with
a continuous and a discrete component.
"""
abstract type AbstractMAPHDist end

"""
    MAPHDist(α, T, D)

MAPH distribution parameterized by

- `α`   m-vector: initial distribution over transient phases (sums to 1)
- `T`   m×m sub-generator of the transient phases
- `D`   m×n non-negative matrix of absorbing rates

The pair must satisfy `T·1_m + D·1_n = 0` (row sums of the full generator vanish).
"""
struct MAPHDist <: AbstractMAPHDist
    α::Vector{Float64}
    T::Matrix{Float64}
    D::Matrix{Float64}

    function MAPHDist(α::AbstractVector{<:Real}, T::AbstractMatrix{<:Real}, D::AbstractMatrix{<:Real})
        m = length(α)
        size(T) == (m, m) || throw(DimensionMismatch("T must be $m × $m, got $(size(T))"))
        size(D, 1) == m   || throw(DimensionMismatch("D must have $m rows, got $(size(D,1))"))
        size(D, 2) >= 1   || throw(ArgumentError("D must have at least one column"))
        all(α .>= 0)      || throw(ArgumentError("α must be non-negative"))
        isapprox(sum(α), 1.0; atol=1e-10) || throw(ArgumentError("α must sum to 1, got $(sum(α))"))
        all(D .>= 0)      || throw(ArgumentError("D must be non-negative"))
        rowsum = vec(sum(T; dims=2) .+ sum(D; dims=2))
        all(abs.(rowsum) .< 1e-8) || throw(ArgumentError(
            "T·1_m + D·1_n must equal 0 (max |rowsum| = $(maximum(abs.(rowsum))))"))
        new(Float64.(α), Float64.(T), Float64.(D))
    end
end

# ---- Accessors ----

initial_prob(d::MAPHDist)     = d.α
subgenerator(d::MAPHDist)     = d.T
exit_rate_matrix(d::MAPHDist) = d.D
nphases(d::MAPHDist)          = length(d.α)
nabsorbing(d::MAPHDist)       = size(d.D, 2)
exit_rates(d::MAPHDist)       = -diag(d.T)

"""
    absorption_probs(d::MAPHDist) -> Matrix{Float64}

The m×n matrix R with entries ρ_{ik} = P(κ = k | start in phase i) = (-T⁻¹D)_{ik}.
Row sums are 1 when the MAPH is non-degenerate.
"""
absorption_probs(d::MAPHDist) = -d.T \ d.D

"""
    marginal_absorption(d::MAPHDist) -> Vector{Float64}

Marginal probabilities π_k = P(κ = k) = (α R)_k.
"""
marginal_absorption(d::MAPHDist) = vec(d.α' * absorption_probs(d))

# ---- Alternative parameterization (α, q, R, U) ----

"""
    MAPHDist(α, q, R, U)

Construct from the `(α, q, R, U)` parameterization described in the paper:

- `α`   m-vector initial distribution
- `q`   m-vector of total exit rates from each transient phase (q_i = -T_ii)
- `R`   m×n matrix of absorption probabilities ρ_{ik}
- `U`   m×m matrix of conditional one-step transition probabilities `p^λ_{ij|1}`

Recovers `T_ij = q_i · U_ij · R_{i1} / R_{j1}` for i≠j and `T_ii = -q_i`, then
`D = -T R`.
"""
function MAPHDist(α::AbstractVector{<:Real}, q::AbstractVector{<:Real},
                  R::AbstractMatrix{<:Real}, U::AbstractMatrix{<:Real})
    m = length(α)
    length(q) == m     || throw(DimensionMismatch("q must have length $m"))
    size(R, 1) == m    || throw(DimensionMismatch("R must have $m rows"))
    size(U) == (m, m)  || throw(DimensionMismatch("U must be $m × $m"))
    all(q .> 0)        || throw(ArgumentError("q must be positive"))
    all(R[:, 1] .> 0)  || throw(ArgumentError("first column of R must be strictly positive"))

    T = zeros(m, m)
    for i in 1:m, j in 1:m
        T[i, j] = i == j ? -q[i] : q[i] * U[i, j] * R[i, 1] / R[j, 1]
    end
    D = -T * R
    return MAPHDist(α, T, D)
end

# ---- Distribution-like API (bivariate: u continuous, k discrete) ----

function Distributions.pdf(d::MAPHDist, u::Real, k::Integer)
    1 <= k <= nabsorbing(d) || throw(ArgumentError("k = $k out of range 1..$(nabsorbing(d))"))
    u < 0 && return 0.0
    return d.α' * exp(d.T * u) * d.D[:, k]
end

function Distributions.cdf(d::MAPHDist, u::Real, k::Integer)
    1 <= k <= nabsorbing(d) || throw(ArgumentError("k = $k out of range 1..$(nabsorbing(d))"))
    u <= 0 && return 0.0
    return d.α' * (I - exp(d.T * u)) * (d.T \ d.D[:, k])
end

"""
    ccdf(d::MAPHDist, u::Real, k::Integer)

Joint survival `P(τ > u, κ = k) = π_k - F(u, k)`.
"""
function Distributions.ccdf(d::MAPHDist, u::Real, k::Integer)
    1 <= k <= nabsorbing(d) || throw(ArgumentError("k = $k out of range 1..$(nabsorbing(d))"))
    u <= 0 && return marginal_absorption(d)[k]
    return marginal_absorption(d)[k] - cdf(d, u, k)
end

function _sample_categorical(rng::AbstractRNG, p::AbstractVector)
    u = rand(rng)
    cum = 0.0
    for i in eachindex(p)
        cum += p[i]
        u <= cum && return i
    end
    return length(p)
end

function Random.rand(rng::AbstractRNG, d::MAPHDist)
    T, D = d.T, d.D
    m, n = nphases(d), nabsorbing(d)
    state = _sample_categorical(rng, d.α)
    time = 0.0
    while true
        q_i = -T[state, state]
        time += randexp(rng) / q_i
        u = rand(rng)
        cum = 0.0
        # Transient moves
        moved = false
        next_state = state
        for j in 1:m
            j == state && continue
            cum += T[state, j] / q_i
            if u <= cum
                next_state = j
                moved = true
                break
            end
        end
        if moved
            state = next_state
            continue
        end
        # Absorbing moves
        for k in 1:n
            cum += D[state, k] / q_i
            if u <= cum
                return (time, k)
            end
        end
        # Numerical drift fallback: pick the most probable absorbing state
        return (time, argmax(view(D, state, :)))
    end
end

Random.rand(d::MAPHDist) = rand(Random.default_rng(), d)
Random.rand(rng::AbstractRNG, d::MAPHDist, n::Integer) = [rand(rng, d) for _ in 1:n]
Random.rand(d::MAPHDist, n::Integer) = rand(Random.default_rng(), d, n)

# ---- Moments ----

"""
    kth_joint_moment(d::MAPHDist, k::Integer, j::Integer)

`E[τ^j · 𝟙{κ=k}] = (-1)^{j+1} · j! · α · T^{-(j+1)} · D[:, k]`.
"""
function kth_joint_moment(d::MAPHDist, k::Integer, j::Integer)
    j >= 1 || throw(ArgumentError("j must be >= 1"))
    1 <= k <= nabsorbing(d) || throw(ArgumentError("k = $k out of range 1..$(nabsorbing(d))"))
    x = d.D[:, k]
    for _ in 1:(j + 1)
        x = d.T \ x
    end
    sign_ = isodd(j) ? 1.0 : -1.0  # (-1)^{j+1}
    return sign_ * Float64(factorial(j)) * (d.α' * x)
end

# ---- Bridges to/from PH ----

"""
    PHDist(d::MAPHDist) -> PHDist

Marginal distribution of τ, which is `PH(α, T)`.
"""
PHDist(d::MAPHDist) = PHDist(d.α, d.T)

"""
    PHDist(d::MAPHDist, k::Integer) -> PHDist
    conditional_time(d::MAPHDist, k::Integer) -> PHDist

Conditional distribution of τ given κ = k, obtained as a Doob h-transform:

    α^(k)_i  = α_i · ρ_{ik} / ρ_k
    T^(k)_ij = T_ij · ρ_{jk} / ρ_{ik}   (i ≠ j),   T^(k)_ii = T_ii

Phases `i` with `ρ_{ik} = 0` cannot reach absorbing state `k` and are dropped.
"""
function PHDist(d::MAPHDist, k::Integer)
    1 <= k <= nabsorbing(d) || throw(ArgumentError("k = $k out of range 1..$(nabsorbing(d))"))
    R = absorption_probs(d)
    ρ_col = R[:, k]
    ρ_k = dot(d.α, ρ_col)
    ρ_k > 0 || throw(ArgumentError("P(κ = $k) = 0, conditional distribution undefined"))

    keep = findall(>(0), ρ_col)
    isempty(keep) && throw(ArgumentError("no transient state can reach absorbing state $k"))

    α_cond = d.α[keep] .* ρ_col[keep] ./ ρ_k
    α_cond ./= sum(α_cond)   # guard against floating-point drift
    T_cond = Matrix{Float64}(undef, length(keep), length(keep))
    for (ii, i) in enumerate(keep), (jj, j) in enumerate(keep)
        T_cond[ii, jj] = ii == jj ? d.T[i, i] : d.T[i, j] * ρ_col[j] / ρ_col[i]
    end
    return PHDist(α_cond, T_cond)
end

conditional_time(d::MAPHDist, k::Integer) = PHDist(d, k)

# ---- PH → MAPH embedding ----

"""
    MAPHDist(ph::AbstractPHDist) -> MAPHDist

Embed a phase-type distribution as an MAPH with a single absorbing state (n=1).
"""
function MAPHDist(ph::AbstractPHDist)
    α = collect(initial_prob(ph))
    T = Matrix{Float64}(subgenerator(ph))
    D = reshape(Float64.(exit_rates(ph)), :, 1)
    return MAPHDist(α, T, D)
end

# ---- Block-diagonal construction from per-category PHs ----

"""
    MAPHDist(phs::Vector{<:AbstractPHDist}, π::AbstractVector{<:Real}) -> MAPHDist

Block-diagonal MAPH where `phs[k]` feeds absorbing state `k` with marginal
probability `π[k]`. Resulting MAPH has `m = Σ nphases(phs[k])` transient phases
and `n = length(π)` absorbing states; the conditional τ | κ = k is exactly `phs[k]`.
"""
function MAPHDist(phs::AbstractVector{<:AbstractPHDist}, π::AbstractVector{<:Real})
    length(phs) == length(π) || throw(DimensionMismatch("phs and π must have same length"))
    all(π .>= 0)             || throw(ArgumentError("π must be non-negative"))
    isapprox(sum(π), 1.0; atol=1e-10) || throw(ArgumentError("π must sum to 1"))
    n = length(phs)
    ms = [nphases(p) for p in phs]
    m = sum(ms)
    offsets = [0; cumsum(ms)]

    α = zeros(m)
    T = zeros(m, m)
    D = zeros(m, n)

    for k in 1:n
        rows = (offsets[k] + 1):offsets[k + 1]
        α[rows]     = π[k] .* initial_prob(phs[k])
        T[rows, rows] = subgenerator(phs[k])
        D[rows, k]  = exit_rates(phs[k])
    end

    return MAPHDist(α, T, D)
end

# ---- Moment-matched construction ----

"""
    MAPHDist(π, μ, σ²) -> MAPHDist

Moment-matched MAPH: the marginal absorption probabilities equal `π`, and the
conditional τ | κ = k has mean `μ[k]` and variance `σ²[k]`. Each category is
realized by a 2-phase hyperexponential when c²_k = σ²_k/μ_k² > 1, by a
hypoexponential when c²_k < 1, and by an exponential when c²_k = 1.
"""
function MAPHDist(π::AbstractVector{<:Real}, μ::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
    length(π) == length(μ) == length(σ²) ||
        throw(DimensionMismatch("π, μ, σ² must have the same length"))
    all(μ .> 0) || throw(ArgumentError("μ must be positive"))
    all(σ² .>= 0) || throw(ArgumentError("σ² must be non-negative"))

    phs = AbstractPHDist[]
    for k in eachindex(π)
        c² = σ²[k] / μ[k]^2
        ph = if isapprox(c², 1.0; atol=1e-8)
            CoxianDist(1 / μ[k])                   # single-phase exponential
        elseif c² > 1.0
            HyperExponentialDist(μ[k], c²)
        else
            HypoExponentialDist(μ[k], c²)
        end
        push!(phs, ph)
    end
    return MAPHDist(phs, collect(Float64, π))
end

# ---- params, show, isapprox ----

Distributions.params(d::MAPHDist) = (d.α, d.T, d.D)

function Base.show(io::IO, d::MAPHDist)
    print(io, "MAPHDist{m=", nphases(d), ", n=", nabsorbing(d), "}(α=", d.α,
              ", T=", d.T, ", D=", d.D, ")")
end

function Base.isapprox(d1::MAPHDist, d2::MAPHDist; kwargs...)
    return isapprox(d1.α, d2.α; kwargs...) &&
           isapprox(d1.T, d2.T; kwargs...) &&
           isapprox(d1.D, d2.D; kwargs...)
end
