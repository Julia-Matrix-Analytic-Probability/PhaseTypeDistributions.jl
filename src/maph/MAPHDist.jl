"""
$(TYPEDEF)
The mutable struct which representing the MAPH distribution in different parameterizations
"""
struct MAPHDist 
    """m × 1 Initial probability distribution across the phases. Elements should sum to 1"""
    α::Vector{<:Real}

    """m x m transition rate matrix for the MAPH states"""    
    T::Matrix{<:Real}

    """m x n abosrbing rate matrix for the MAPH states."""
    D::Matrix{<:Real}

    """m × 1 vector of transition rates from the MAPH states to the absorbing states."""
    q::Vector{<:Real}

    """m x n absorbing probabilities matrix."""
    R::Matrix{<:Real}

    """Vector of n copies of m x m matrices, representing the transition probabilities between the MAPH states given absorbing states."""
    U::Matrix{<:Real}

    function MAPHDist(α::Vector{<:Real}, T::Matrix{<:Real}, D::Matrix{<:Real}, q::Vector{<:Real}, R::Matrix{<:Real}, U::Matrix{<:Real})
        @assert length(α) == size(T,1) "must have the same num of phases"
        @assert size(T, 1) == size(T, 2) "T must be a square matrix"
        @assert size(q, 1) == size(T,1) "q must be the same length as the number of rows in T"
        @assert size(D, 1) == size(T, 1) "T0 must be a mxn matrix"
        @assert length(α) == size(T, 1) "The length of α must be equal to the number of rows in T"
        @assert size(D) == size(R) "prob and rate matrix must have the same dimension"
        @assert isapprox(sum(α), 1.0) "initial prob must sum to 1"
        @assert all(isapprox.(sum(R, dims = 2) , 1.0; atol = 1e-3))
        if !all(isapprox.((sum(T, dims = 2) + sum(D, dims = 2)) , 0.0; atol = 1e-3 ))
            @show (sum(T, dims = 2) + sum(D, dims = 2))
            @assert false
        end
        @assert satisfies_constraint_U(R, U)
        return new(α, T, D, q, R, U)
    end
end

function MAPHDist(α::Vector{<:Real}, T:: Matrix{<:Real}, D::Matrix{<:Real})
    q, R, U = if is_degenerate_maph(T)
        valid_states = abs.(diag(T)) .> sqrt(eps())
        new_α = α[:, valid_states]
        new_T = T[valid_states, valid_states]
        new_D = D[valid_states, valid_states]
        R_U_from_T_D(new_T, new_D)
    else
        R_U_from_T_D(T, D)
    end    

    return MAPHDist(α, T, D, q, R, U)
end

function MAPHDist(α::Vector{<:Real}, q::Vector{<:Real}, R::Matrix{<:Real}, U::Matrix{<:Real})
    T, D = T_D_from_R_U_q(q, R, U)
    return MAPHDist(α, T, D, q, R, U)
end

function is_degenerate_maph(T::Matrix{<:Real})
    return !all(abs.(diag(T)) .> sqrt(eps()))
end

function R_U_from_T_D(T::Matrix{<:Real}, D::Matrix{<:Real})
    m, _  = size(D)
    q = -diag(T)
    R = -inv(T)*D
    k = 1
    U = [i != j ? -T[i,j] * R[j,k] / T[i,i] / R[i,k] : 0.0 for i in 1:m, j in 1:m]

    return q, R, U
end


function T_D_from_R_U_q(q::Vector{<:Real}, R::Matrix{<:Real}, U::Matrix{<:Real})
    #add equation 7 check the conditon 
    #add a helper function to check the conditions and fix P 
    m = length(q)

    T =  [i==j ? -q[i] : q[i] * U[i,j] * R[i,1] / R[j,1] for i in 1:m, j in 1:m]

    D = -T * R

    return T, D
end

function satisfies_constraint_U(R::Matrix, U::Matrix)
    m, n = size(R)
    for i in 1:m, k in 1:n
        lhs = sum((R[j, k] / R[j, 1]) * U[i, j] for j in 1:m)
        rhs = R[i, k] / R[i, 1]
        (lhs > rhs) && (return false)
    end
    return true
end

function Base.isapprox(maph1::MAPHDist, maph2::MAPHDist)
    return isapprox(maph1.α, maph2.α) && isapprox(maph1.T, maph2.T)&& isapprox(maph1.D, maph2.D)&& isapprox(maph1.q, maph2.q)&& isapprox(maph1.R, maph2.R) && isapprox(maph1.U, maph2.U)
end