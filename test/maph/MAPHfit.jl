Λ₄, λ45, λ54, Λ₅ = 15.0, 5.0, 7.0, 16.0
μ41, μ42, μ43, μ51, μ52, μ53 = 4.0, 3.0, 3.0, 1.0, 7.0, 1.0 
T = [-Λ₄ λ45; λ54 -Λ₅]
D = [μ41 μ42 μ43; μ51 μ52 μ53]
α = [0.5, 0.5]

@testset "test the optimizer" begin
    q, R, U = MAPHDistributions.R_U_from_T_D(T,D)
    bad_U = rand(size(U)...)
    bad_U ./= sum(bad_U, dims =1)

    @test !MAPHDistributions.satisfies_constraint_U(R, bad_U)
    new_U = MAPHDistributions.project_U_step(R, U)
    @test isapprox(new_U, U)
end


maph = MAPHDist(α, T, D)


all_obs = [rand(maph) for _ in 1:10]
@show E_step(all_obs, maph)