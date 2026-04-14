"""
$(TYPEDEF)

Defines a coninutous time markov chain
"""
struct ContinuousTimeMarkovChain
    "the transient states"
    transient_states::Vector{Int}

    "the absorbing states"
    absorbing_states::Vector{Int}

    "jumping chain"
    P_jump::Matrix{<:Real}

    "diagonal rate"
    Λ::Vector{<:Real}

end

function MAPH_TO_CTMC(maph::MAPHDist)
    m, n  = model_size(maph)
    transient_states = 1:m
    absorbing_states = (m+1):(m+n)
    P_λ = I - inv(Diagonal(maph.T))*maph.T
    P_μ = -inv(Diagonal(maph.T))*maph.D
    P1 = hcat(P_λ, P_μ)
    P2 = hcat(zeros(n,m), Matrix(Diagonal(ones(n))))
    P_jump = [P1; P2]
    Λ = vcat(-diag(maph.T), zeros(n))
    return ContinuousTimeMarkovChain(transient_states, absorbing_states, P_jump, Λ)
end

struct SingleObservation
    "absorbing state"
    a::Int

    "absorbing time"
    y::Real

    "sojourn_times"
    sojourn_times::Vector{<:Real}

    "all_states"
    all_states::Vector{Int}
end

function rand(maph::MAPHDist; full_trace = true)
    m, n = model_size(maph)
    all_states = 1:(m+n)
    ctmc = MAPH_TO_CTMC(maph) 

    states = Int[]
    sojourn_times = Real[]
    
    state = sample(ctmc.transient_states, weights(maph.α))
    t = 0.0
    while state ∈ ctmc.transient_states
        sojourn_time = rand(Exponential(1/ctmc.Λ[state]))
        if full_trace
            push!(states, state)
            push!(sojourn_times, sojourn_time)
        end
        t += sojourn_time
        state = sample(all_states, weights(ctmc.P_jump[state, :]))
    end
    push!(states, state)
    
    return SingleObservation(state, t, sojourn_times, states)
end

"""
$(TYPEDEF)
"""
struct MAPHSufficientStats
    "initial starts"
    B::Vector{<:Real} 

    "time spent"
    Z::Vector{<:Real} 

    "transitions between transient states"
    M::Matrix{<:Real} 

    "transitions between transient to abosrbing states"
    E::Vector{<:Real} 

    "total number of transitions leaving state i"
    N::Vector{<:Real}

    function MAPHSufficientStats(B::Vector{<:Real} , Z::Vector{<:Real} , M::Matrix{<:Real} , E::Vector{<:Real} , N::Vector{<:Real})
        @assert all(Z .>= 0.0)
        return new(B, Z, M, E, N)
    end
end



+(ss1::MAPHSufficientStats, ss2::MAPHSufficientStats) = MAPHSufficientStats(ss1.B+ss2.B, ss1.Z+ss2.Z, ss1.M+ ss2.M, ss1.E+ss2.E, ss1.N+ss2.N)
/(ss::MAPHSufficientStats,n::Real) = MAPHSufficientStats(ss.B/n, ss.Z/n, ss.M/n, ss.E/n, ss.N/n)
/(ss1::MAPHSufficientStats,ss2::MAPHSufficientStats) = MAPHSufficientStats(ss1.B ./ ss2.B, ss1.Z ./  ss2.Z, ss1.M ./  ss2.M,  ss1.E ./  ss2.E, ss1.N ./ ss2.N )
-(ss1::MAPHSufficientStats, ss2::MAPHSufficientStats) = MAPHSufficientStats(ss1.B-ss2.B, ss1.Z-ss2.Z, ss1.M - ss2.M, ss1.E - ss2.E, ss1.N - ss2.N)
*(n::Real, ss::MAPHSufficientStats) = MAPHSufficientStats(ss.B *n, ss.Z*n, ss.M*n, ss.E*n, ss.N*n)

function very_crude_c_solver(y::Real, i::Int, j::Int, k::Int, maph::MAPHDist)
    max(quadgk(u -> (maph.α' * exp(maph.T*u))[i] * (exp(maph.T*(y-u))*maph.D[:,k])[j] , 0, y, rtol=1e-3) |> first, 0)
end


function compute_sufficient_stats(observation::SingleObservation, 
                            maph::MAPHDist; 
                            c_solver = very_crude_c_solver)

    m, n = model_size(maph)
    a(y::Real) = maph.α' * exp(maph.T*y)
    b(y::Real, k::Int) = exp(maph.T*y) * maph.D[:,k]
    c(y::Real, i::Int, j::Int, k::Int) = very_crude_c_solver(y, i, j, k, maph)

    non_degenerate_condtion(y::Real, k::Int) = (sum(b(y,k)) != 0)
    EB(y::Real, i::Int, k::Int) = non_degenerate_condtion(y, k) ? (maph.α[i] * b(y, k)[i] / reduce(vcat, (maph.α' * b(y, k)))) : maph.α[i]
    EZ(y::Real, i::Int, k::Int) = non_degenerate_condtion(y, k) ? c(y, i, i, k) / reduce(vcat, (maph.α' * b(y,k))) : 0
    ENT_non_diagonal(y::Real, i::Int, j::Int, k::Int) = non_degenerate_condtion(y,k) ? maph.T[i,j] .* c(y, i, j, k) / reduce(vcat, (maph.α' * b(y,k))) : 0
    ENT(y::Real, i::Int, j::Int, k::Int) = i != j ?  ENT_non_diagonal(y::Real, i::Int, j::Int, k::Int) : 0
    ENA(y::Real, i::Int, k::Int) =  non_degenerate_condtion(y,k) ? a(y)[i] * maph.D[i,k] / reduce(vcat, (maph.α' * b(y,k))) : 0.0
    
    B = map(i -> EB(observation.y, i, observation.a - m), 1:m)
    Z = map(i -> EZ(observation.y, i, observation.a - m), 1:m)
    M = reduce(hcat, map(i ->  map(j -> ENT(observation.y, i, j, observation.a - m), 1:m), 1:m))
    E = [ENA(observation.y, i, observation.a - m) for i =1:m]
    N = vec(sum(M, dims = 2)) + E    
    return MAPHSufficientStats(B, Z, M ,E, N)
end


function data_filter(all_obs::Vector{SingleObservation}, lower_quantile = 0.05, upper_quantile = 0.95; censoring::Bool = false)
    all_absorbing_states = map(obs -> obs.a, all_obs)
    all_absorbing_times = map(obs -> obs.y, all_obs)

    all_obs = if censoring
            min_time = quantile(all_absorbing_times, lower_quantile)
            max_time = quantile(all_absorbing_times, upper_quantile)

            #filter out those ones with abnormal time
            filtered_obs = filter(obs -> (min_time < obs.y < max_time),  all_obs)
    else
        all_obs
    end
    unique_absorbing_state = sort(unique(all_absorbing_states))

    data_dict = Dict{Int, Any}()
    for k in unique_absorbing_state
        data = filter(obs -> obs.a == k, all_obs)
        @assert !isempty(data) "Boom we have no observation in the absorbing state $k"
        data_dict[k] = data
    end

    return data_dict

end


function get_emperical_absorb_prob(all_obs::Vector{SingleObservation})
    data_dict = data_filter(all_obs)
    num_ob = length(reduce(vcat, values(data_dict)))
    emprical_prob_dict = Dict{Int, Any}()
    for k in keys(data_dict)
        emprical_prob_dict[k] = length(data_dict[k]) / num_ob 
    end

    return emprical_prob_dict
end


function get_emperical_statistics(all_obs::Vector{SingleObservation},  ω::Real)
    _, data = data_filter(all_obs)
    mean_per_state  = Dict()
    scv_per_state = Dict()
    for k ∈ eachindex(data)
        absorbing_time = map(dat -> dat.y, data[k])
        emperical_mean= mean(absorbing_time)
        emperical_scv = var(absorbing_time) / mean(absorbing_time)^2

        mean_per_state[data[k][1].a] = emperical_mean - 1 / ω
        scv_per_state[data[k][1].a] = (emperical_scv * (ω * emperical_mean)^2 -1) / (ω * emperical_mean)^2
    end

    return mean_per_state, scv_per_state

end



