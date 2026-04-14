@testset "Conversions" begin
    @testset "PHDist from Exponential" begin
        exp_d = Exponential(2.0)
        ph = PHDist(exp_d)
        @test nphases(ph) == 1
        @test mean(ph) ≈ 2.0 atol=1e-10
        @test var(ph) ≈ 4.0 atol=1e-10
        for x in [0.1, 1.0, 3.0]
            @test pdf(ph, x) ≈ pdf(exp_d, x) atol=1e-10
        end
    end

    @testset "PHDist from Erlang" begin
        er_d = Erlang(3, 0.5)  # shape=3, scale=0.5
        ph = PHDist(er_d)
        @test nphases(ph) == 3
        @test mean(ph) ≈ Distributions.mean(er_d) atol=1e-10
        @test var(ph) ≈ Distributions.var(er_d) atol=1e-10
        for x in [0.1, 0.5, 1.0, 2.0]
            @test pdf(ph, x) ≈ pdf(er_d, x) atol=1e-10
        end
    end

    @testset "PHDist from subtypes" begin
        he = HyperExponentialDist([0.3, 0.7], [1.0, 4.0])
        ph_he = PHDist(he)
        @test nphases(ph_he) == 2
        @test mean(ph_he) ≈ mean(he) atol=1e-10

        ho = HypoExponentialDist([2.0, 3.0])
        ph_ho = PHDist(ho)
        @test nphases(ph_ho) == 2
        @test mean(ph_ho) ≈ mean(ho) atol=1e-10

        er = ErlangPHDist(4, 3.0)
        ph_er = PHDist(er)
        @test nphases(ph_er) == 4
        @test mean(ph_er) ≈ mean(er) atol=1e-10

        cox = CoxianDist([2.0, 3.0], [0.5])
        ph_cox = PHDist(cox)
        @test nphases(ph_cox) == 2
        @test mean(ph_cox) ≈ mean(cox) atol=1e-10
    end

    @testset "PHDist(PHDist) is no-op" begin
        ph = PHDist([1.0], [-3.0;;])
        @test PHDist(ph) === ph
    end
end
