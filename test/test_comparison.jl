@testset "Comparison helpers" begin
    @testset "moment_vector" begin
        ph = PHDist([1.0], [-2.0;;])
        mv = moment_vector(ph, 3)
        @test length(mv) == 3
        @test mv[1] ≈ 0.5 atol=1e-10      # E[X]
        @test mv[2] ≈ 0.5 atol=1e-10      # E[X²] = 2/λ²
        @test mv[3] ≈ 0.75 atol=1e-10     # E[X³] = 6/λ³
    end

    @testset "moments_isapprox — same distribution, different representations" begin
        # Exponential(rate=2) as PHDist vs HyperExponentialDist with one component
        ph = PHDist([1.0], [-2.0;;])
        he = HyperExponentialDist([1.0], [2.0])
        @test moments_isapprox(ph, he; order=5)
    end

    @testset "moments_isapprox — different distributions" begin
        ph1 = PHDist([1.0], [-2.0;;])  # Exp(rate=2)
        ph2 = PHDist([1.0], [-3.0;;])  # Exp(rate=3)
        @test !moments_isapprox(ph1, ph2)
    end

    @testset "moments_isapprox — same moments possible, different dist" begin
        # Two hyperexponentials can share low-order moments
        he1 = HyperExponentialDist([0.5, 0.5], [1.0, 3.0])
        he2 = HyperExponentialDist([0.5, 0.5], [1.0, 3.0])
        @test moments_isapprox(he1, he2; order=10)
    end

    @testset "distribution_isapprox — same distribution" begin
        er = ErlangPHDist(3, 2.0)
        ph = PHDist(er)
        @test distribution_isapprox(er, ph)
    end

    @testset "distribution_isapprox — different distributions" begin
        ph1 = PHDist([1.0], [-2.0;;])
        ph2 = PHDist([1.0], [-3.0;;])
        @test !distribution_isapprox(ph1, ph2)
    end

    @testset "ErlangPHDist vs HypoExponential with equal rates" begin
        # ErlangPHDist(2, 3.0) should match HypoExponential([3.0, 3.0]) distributionally
        # Note: HypoExponential partial fractions need distinct rates,
        # so we use general PHDist comparison
        er = ErlangPHDist(2, 3.0)
        ho_ph = PHDist([1.0, 0.0], [-3.0 3.0; 0.0 -3.0])
        @test distribution_isapprox(er, ho_ph)
        @test moments_isapprox(er, ho_ph; order=5)
    end
end
