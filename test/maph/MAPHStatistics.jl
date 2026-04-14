using Random
Random.seed!(0)

# Λ₄, λ45, λ54, Λ₅ = 15.0, 5.0, 7.0, 16.0
# μ41, μ42, μ43, μ51, μ52, μ53 = 4.0, 3.0, 3.0, 1.0, 7.0, 1.0 
# T = [-Λ₄ λ45; λ54 -Λ₅]
# D = [μ41 μ42 μ43; μ51 μ52 μ53]
# α = [0.5, 0.5]

T = [-16.5 0.5 0.0 1.0;
     2.6 -7.8 0.6 2.6;
     12.3 2.9 -20.8 1.6;
     11.2 0.1 5.7  -23]

D = [12.0 1.0 2.0;
     0.9 0.1 1.0;
     2.9 0.1 1.0;
     2.9 0.1 3.0]
    
α = [0.75, 0.05, 0.15, 0.05]

ground_truth_maph = MAPHDist(α, T, D)
# single_ob = rand(maph)

# @show MAPHDistributions.compute_sufficient_stats(single_ob, maph)
all_obs = [rand(ground_truth_maph) for _ in 1:3000]
# @show MAPHDistributions.data_filter(all_obs)

# data_length, stats_dict, total_stats, highest_prob_state = MAPHDistributions.E_step(all_obs, maph)

# @show total_stats

# @show MAPHDistributions.M_step(data_length, stats_dict, total_stats, highest_prob_state,  maph)


T = [-16.5 0.5 0.0 1.0;
     2.6 -7.8 0.6 2.6;
     12.3 2.9 -20.8 1.6;
     11.2 0.1 5.7  -23]

D = [12.0 1.0 2.0;
     0.9 0.1 1.0;
     2.9 0.1 1.0;
     2.9 0.1 3.0]

T = T + [-0.5 0.1 0.1 0.3;
          0.2 -0.6 0.2 0.2;
          0.5 0.25 -1 0.25;
          1    1    1  -3]

D = D 
α = α

init_maph = MAPHDist(α, T, D)

maph_new = MAPHDistributions.EM_fit(all_obs, init_maph)
@show mean(maph_new)
@show mean(maph)
# @show maph_new
# maph_new = MAPHDist([0.44377420920253696, 0.5562257907974631], [-16.57111742766177 14.445195231489837; 1.5737259906465026 -17.565226110195773], [2.1259221961719352 -5.329070518200751e-15; 0.5376222958710618 15.45387782367821], [16.57111742766177, 17.565226110195773], [0.16809986737663468 0.8319001326233653; 0.04566781100075241 0.9543321889992475], [0.0 0.23681785613801207; 0.32978625315812626 0.0])


# ob = SingleObservation(5, 0.0011776386581249302, Real[0.0011776386581249302], [1, 5])
# @show MAPHDistributions.compute_sufficient_stats(ob, maph_new)