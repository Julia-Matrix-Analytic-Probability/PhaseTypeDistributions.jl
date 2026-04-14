Λ₄, λ45, λ54, Λ₅ = 15.0, 5.0, 7.0, 16.0
μ41, μ42, μ43, μ51, μ52, μ53 = 4.0, 3.0, 3.0, 1.0, 7.0, 1.0 
T = [-Λ₄ λ45; λ54 -Λ₅]
D = [μ41 μ42 μ43; μ51 μ52 μ53]
α = [0.5, 0.5]

maph = MAPHDist(α, T, D)

@testset "testing MAPH DIST FIT" begin
    q, R, U = MAPHDistributions.R_U_from_T_D(T,D)
    @test q == maph.q
    @test R == maph.R
    @test U == maph.U

    new_T , new_D = MAPHDistributions.T_D_from_R_U_q(q, R, U)
    @test isapprox(new_T, maph.T)
    @test isapprox(new_D, maph.D)

    @test MAPHDistributions.satisfies_constraint_U(R, U)
end

@testset "test maph multiple dispatch" begin
    q, R, U = MAPHDistributions.R_U_from_T_D(T,D)
    maph2 = MAPHDist(α, q, R, U)
    maph3 = MAPHDist(α, T, D, q, R, U)

    @test isapprox(maph, maph2)
    @test isapprox(maph, maph3)
    @test isapprox(maph2, maph3)
end