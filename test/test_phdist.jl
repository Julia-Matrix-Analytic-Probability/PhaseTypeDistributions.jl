@testset "PHDist — general PH distribution" begin
    # 1-phase PH = Exponential(rate=2, mean=0.5)
    ph1 = PHDist([1.0], [-2.0;;])
    exp_dist = Exponential(0.5)

    @testset "basic properties" begin
        @test nphases(ph1) == 1
        @test initial_prob(ph1) == [1.0]
        @test subgenerator(ph1) == [-2.0;;]
        @test exit_rates(ph1) ≈ [2.0]
    end

    @testset "Distributions.jl interface" begin
        @test minimum(ph1) == 0.0
        @test maximum(ph1) == Inf
        @test insupport(ph1, 1.0) == true
        @test insupport(ph1, -1.0) == false
    end

    @testset "pdf/cdf/ccdf match Exponential" begin
        for x in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
            @test pdf(ph1, x) ≈ pdf(exp_dist, x) atol=1e-10
            @test cdf(ph1, x) ≈ cdf(exp_dist, x) atol=1e-10
            @test ccdf(ph1, x) ≈ ccdf(exp_dist, x) atol=1e-10
            @test cdf(ph1, x) + ccdf(ph1, x) ≈ 1.0 atol=1e-10
        end
        @test pdf(ph1, -1.0) == 0.0
        @test cdf(ph1, -1.0) == 0.0
        @test ccdf(ph1, -1.0) == 1.0
    end

    @testset "logpdf" begin
        for x in [0.1, 0.5, 1.0, 5.0]
            @test logpdf(ph1, x) ≈ log(pdf(ph1, x)) atol=1e-10
        end
        @test logpdf(ph1, -1.0) == -Inf
    end

    @testset "ccdf tail stability" begin
        # In the deep tail, 1 - cdf loses precision; direct ccdf should not.
        x_far = 50.0
        @test ccdf(ph1, x_far) > 0
        @test ccdf(ph1, x_far) ≈ exp(-2.0 * x_far) atol=1e-30
    end

    @testset "mean/var/std" begin
        @test mean(ph1) ≈ 0.5 atol=1e-10
        @test var(ph1) ≈ 0.25 atol=1e-10
        @test std(ph1) ≈ 0.5 atol=1e-10
        @test scv(ph1) ≈ 1.0 atol=1e-10
    end

    @testset "skewness/kurtosis" begin
        # Exp(λ): skewness = 2, excess kurtosis = 6
        @test skewness(ph1) ≈ 2.0 atol=1e-8
        @test kurtosis(ph1) ≈ 6.0 atol=1e-8
    end

    @testset "moments" begin
        @test kth_moment(ph1, 1) ≈ 0.5 atol=1e-10
        @test kth_moment(ph1, 2) ≈ 0.5 atol=1e-10  # 2!/λ² = 2/4
    end

    @testset "mgf" begin
        # MGF of Exp(λ) = λ/(λ-t)
        @test mgf(ph1, 0.0) ≈ 1.0 atol=1e-10
        @test mgf(ph1, 1.0) ≈ 2.0 atol=1e-10  # 2/(2-1)
    end

    @testset "quantile/median" begin
        # For Exp(rate=2): quantile(p) = -log(1-p)/2
        for p in [0.1, 0.25, 0.5, 0.75, 0.99]
            @test quantile(ph1, p) ≈ -log(1 - p) / 2.0 atol=1e-8
            # Round-trip: cdf(quantile(p)) ≈ p
            @test cdf(ph1, quantile(ph1, p)) ≈ p atol=1e-8
        end
        @test quantile(ph1, 0.0) == 0.0
        @test quantile(ph1, 1.0) == Inf
        @test median(ph1) ≈ log(2) / 2.0 atol=1e-8
        @test_throws DomainError quantile(ph1, -0.1)
        @test_throws DomainError quantile(ph1, 1.1)
    end

    @testset "params" begin
        α, T = params(ph1)
        @test α == [1.0]
        @test T == [-2.0;;]
    end

    @testset "show" begin
        s = sprint(show, ph1)
        @test occursin("PHDist", s)
        @test occursin("α", s)
        @test occursin("T", s)
    end

    @testset "random sampling" begin
        rng = StableRNG(42)
        samples = [rand(rng, ph1) for _ in 1:10_000]
        @test all(s -> s >= 0, samples)
        @test isapprox(Statistics.mean(samples), 0.5; atol=0.05)
        @test isapprox(Statistics.var(samples), 0.25; atol=0.05)

        # Vectorized rand(d, n)
        xs = rand(rng, ph1, 1000)
        @test length(xs) == 1000
        @test all(x -> x >= 0, xs)
    end

    # 2-phase PH (Erlang-like: α=[1,0], T=[-2 2; 0 -2])
    ph2 = PHDist([1.0, 0.0], [-2.0 2.0; 0.0 -2.0])

    @testset "2-phase PH matches Erlang(2, 0.5)" begin
        erlang = Erlang(2, 0.5)
        @test mean(ph2) ≈ Distributions.mean(erlang) atol=1e-10
        @test var(ph2) ≈ Distributions.var(erlang) atol=1e-10
        for x in [0.1, 0.5, 1.0, 2.0]
            @test pdf(ph2, x) ≈ pdf(erlang, x) atol=1e-10
            @test cdf(ph2, x) ≈ cdf(erlang, x) atol=1e-10
            @test ccdf(ph2, x) ≈ ccdf(erlang, x) atol=1e-10
        end
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError PHDist([0.5, 0.3], [-1.0 0.0; 0.0 -1.0])  # doesn't sum to 1
        @test_throws DimensionMismatch PHDist([1.0], [-1.0 0.0; 0.0 -1.0])   # dimension mismatch
        @test_throws ArgumentError PHDist([-0.5, 1.5], [-1.0 0.0; 0.0 -1.0]) # negative α
    end
end
