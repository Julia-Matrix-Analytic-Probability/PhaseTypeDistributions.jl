function E_step(all_obs::Vector{SingleObservation}, maph::MAPHDist)
    data_dict = data_filter(all_obs)
    data_length = length(reduce(vcat,values(data_dict)))
    stats_dict = Dict{Int, Any}()

    for k in keys(data_dict)
        stats_dict[k] = sum(compute_sufficient_stats.(data_dict[k], Ref(maph)))
    end

    total_stats = sum(reduce(vcat, values(stats_dict)))

    emprical_prob_dict = get_emperical_absorb_prob(all_obs)
    highest_prob_state = argmax(emprical_prob_dict)

    return data_length, stats_dict, total_stats, highest_prob_state
end

function M_step(data_length::Int, stats::Dict{Int, Any}, sum_stats::MAPHSufficientStats, highest_prob_state::Int,  maph::MAPHDist)
    m, n = model_size(maph)
    absorbing_states = sort(collect(keys(stats)))
    @assert length(absorbing_states) == n "we dont have enough data!"
    number_times_exit = [sum_stats.N[i] for i ∈ 1:m]
    number_times_exit_per_state = [[stats[k].N[i]  for i ∈ 1:m] for k ∈ absorbing_states]
    α_est =  sum_stats.B ./data_length
    q_est = [number_times_exit[i] / sum_stats.Z[i] for i ∈ 1:m]
    R_est = [stats[k].B[i] / sum_stats.B[i] for i ∈ 1:m, k ∈ absorbing_states]
    U_est = stats[highest_prob_state].M ./ stats[highest_prob_state].N 

    if !satisfies_constraint_U(R_est, U_est)
        @info "we are going to project a new U matrix here since the fitted one does not statisfy the constraint"
        U_est = project_U_step(R_est, U_est)
    end

    return MAPHDist(α_est, q_est, R_est, U_est)

end

function project_U_step(R::Matrix, target_U::Matrix)
    m, n = size(R)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, u[i=1:m, j = 1:m] >= 0)
    @variable(model, d[i=1:m, j =1:m] >= 0)
    
    @constraint(model, [i=1:m, j = 1:m], d[i,j] >= u[i,j] - target_U[i,j])
    @constraint(model, [i=1:m, j = 1:m], d[i,j] >= -(u[i,j] - target_U[i,j]))
    @constraint(model, [i=1:m, k = 1:n], sum((R[j,k]/R[j,1]) * u[i,j] for j = 1:m) <= (R[i,k]/R[i,1]))

    @objective(model, Min, sum(d[i,j] for i in 1:m, j in 1:m))
    optimize!(model)

    model_status = termination_status(model)
    uvals = [value(u[i,j]) for i in 1:m, j in 1:m]
    U = 
    if model_status == MOI.OPTIMAL
        @info "Optimal solution found!"
        uvals
    elseif model_status in [MOI.FEASIBLE_POINT, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL]
        @info "Suboptimal solution found. Extracting best known values."
        uvals
    elseif model_status == MOI.INFEASIBLE 
        @info "The problem is infeasible. No feasible solution exists."
        nothing  # No valid solution
    elseif model_status == MOI.UNBOUNDED
        @info "The problem is unbounded. The objective function can decrease indefinitely."
        nothing
    elseif model_status == MOI.INFEASIBLE_OR_UNBOUNDED
        @info "The problem is either infeasible or unbounded."
        nothing
    else
        @info "Optimization did not fully converge. Status: $model_status"
        nothing  # Handle unexpected cases
    end
    return U

end


function EM_fit(all_obs::Vector{SingleObservation}, maph::MAPHDist, iterations::Int = 300)
    maph_out = deepcopy(maph)
    means = []
    for k ∈ 1:iterations
        @time data_length, stats, s_stats, highest_prob_state = E_step(all_obs, maph_out)
        @time maph_out = M_step(data_length, stats, s_stats, highest_prob_state, maph_out)
        @show mean(maph_out)
        push!(means, mean(maph_out))
    end
    return maph_out
end

   